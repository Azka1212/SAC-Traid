from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


def mlp(in_dim, hidden, out_dim, act=nn.ReLU):
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

def gumbel_softmax(logits: torch.Tensor, tau: float = 0.5, eps: float = 1e-8):
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    return F.softmax((logits + g) / tau, dim=-1)

def to01(x: torch.Tensor) -> torch.Tensor:
    return (torch.tanh(x) * 0.5) + 0.5


class Actor(nn.Module):
    def __init__(self, obs_dim: int, num_ops: int, cont_dim: int, hidden=(256, 256)):
        super().__init__()
        self.backbone = mlp(obs_dim, hidden, 256)
        self.logits = nn.Linear(256, num_ops)
        self.mu     = nn.Linear(256, cont_dim)
        self.logstd = nn.Linear(256, cont_dim)

    def forward(self, obs: torch.Tensor):
        z = self.backbone(obs)
        logits = self.logits(z)
        mu = self.mu(z)
        logstd = torch.clamp(self.logstd(z), -5, 2)
        return logits, mu, logstd

class Critic(nn.Module):
    def __init__(self, obs_dim: int, num_ops: int, cont_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q = mlp(obs_dim + num_ops + cont_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, op_onehot: torch.Tensor, cont01: torch.Tensor):
        x = torch.cat([obs, op_onehot, cont01], dim=-1)
        return self.q(x)


@dataclass
class SACCfg:
    total_steps: int = 10000
    learning_starts: int = 1000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2
    tau_gumbel: float = 0.5
    seed: int = 1
    buffer_size: int = 50000
    save_every: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class UniformReplay:
    def __init__(self, size: int, obs_dim: int, cont_dim: int):
        self.size=size; self.ptr=0; self.full=False
        self.obs=np.zeros((size,obs_dim),np.float32)
        self.op=np.zeros((size,1),np.int64)
        self.cont=np.zeros((size,cont_dim),np.float32)
        self.rew=np.zeros((size,1),np.float32)
        self.nobs=np.zeros((size,obs_dim),np.float32)
        self.done=np.zeros((size,1),np.float32)

    def add(self, obs, op, cont, rew, nobs, done, bucket: int = 0):
        i=self.ptr
        self.obs[i]=obs; self.op[i,0]=op; self.cont[i]=cont
        self.rew[i,0]=rew; self.nobs[i]=nobs; self.done[i,0]=float(done)
        self.ptr=(self.ptr+1)%self.size
        if self.ptr==0: self.full=True

    def sample(self, batch: int):
        n=self.size if self.full else self.ptr
        idx=np.random.randint(0,n,size=batch)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.op[idx]),
            torch.as_tensor(self.cont[idx]),
            torch.as_tensor(self.rew[idx]),
            torch.as_tensor(self.nobs[idx]),
            torch.as_tensor(self.done[idx]),
        )


class StratifiedReplay(UniformReplay):
    """
    4 buckets:
      0: fail-low, 1: fail-high, 2: succ-low, 3: succ-high
    Use equal sampling across non-empty buckets.
    """
    def __init__(self, size: int, obs_dim: int, cont_dim: int):
        super().__init__(size, obs_dim, cont_dim)
        self.bucket = np.zeros((size,1), np.int64)

    def add(self, obs, op, cont, rew, nobs, done, bucket: int = 0):
        i=self.ptr
        super().add(obs, op, cont, rew, nobs, done, bucket=bucket)
        self.bucket[i,0]=int(bucket)

    def sample(self, batch: int):
        n=self.size if self.full else self.ptr
        if n <= 0:
            return super().sample(batch)

        # build indices per bucket
        idxs = [np.where(self.bucket[:n,0]==b)[0] for b in range(4)]
        nonempty = [b for b in range(4) if len(idxs[b])>0]
        if not nonempty:
            return super().sample(batch)

        per = max(1, batch // len(nonempty))
        out_idx = []
        for b in nonempty:
            take = min(per, len(idxs[b]))
            out_idx.append(np.random.choice(idxs[b], size=take, replace=(take>len(idxs[b]))))
        out_idx = np.concatenate(out_idx, axis=0)

        # top-up
        if len(out_idx) < batch:
            extra = np.random.randint(0, n, size=(batch-len(out_idx)))
            out_idx = np.concatenate([out_idx, extra], axis=0)

        return (
            torch.as_tensor(self.obs[out_idx]),
            torch.as_tensor(self.op[out_idx]),
            torch.as_tensor(self.cont[out_idx]),
            torch.as_tensor(self.rew[out_idx]),
            torch.as_tensor(self.nobs[out_idx]),
            torch.as_tensor(self.done[out_idx]),
        )


def train_sac_b3(
    *,
    env_id: str,
    env_kwargs: Dict[str, Any],
    cfg: SACCfg,
    out_dir: str,
    stratified: bool,
) -> Dict[str, Any]:
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run_spec.json").write_text(json.dumps({
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "sac_cfg": asdict(cfg),
        "stratified_replay": bool(stratified),
    }, indent=2), encoding="utf-8")

    env = gym.make(env_id, **env_kwargs)
    obs, info = env.reset(seed=cfg.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    num_ops = int(env.action_space["op"].n)
    cont_dim = int(env.action_space["cont"].shape[0])

    device = torch.device(cfg.device)
    actor = Actor(obs_dim, num_ops, cont_dim).to(device)
    q1 = Critic(obs_dim, num_ops, cont_dim).to(device)
    q2 = Critic(obs_dim, num_ops, cont_dim).to(device)
    q1_t = Critic(obs_dim, num_ops, cont_dim).to(device)
    q2_t = Critic(obs_dim, num_ops, cont_dim).to(device)
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())

    opt_actor = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    opt_q1 = torch.optim.Adam(q1.parameters(), lr=cfg.lr)
    opt_q2 = torch.optim.Adam(q2.parameters(), lr=cfg.lr)

    buf = StratifiedReplay(cfg.buffer_size, obs_dim, cont_dim) if stratified else UniformReplay(cfg.buffer_size, obs_dim, cont_dim)

    def act(o_np):
        with torch.no_grad():
            o = torch.as_tensor(o_np, device=device).unsqueeze(0)
            logits, mu, logstd = actor(o)
            op = torch.argmax(logits, dim=-1).item()
            cont = to01(mu).squeeze(0).cpu().numpy()
        return op, cont

    def bucket_from_info(info: Dict[str, Any], reward: float) -> int:
        succ = 0.0
        scores = info.get("scores") or {}
        try:
            succ = float(scores.get("Success", 0.0))
        except Exception:
            succ = 0.0
        is_succ = succ >= float(env_kwargs.get("success_threshold", 0.65))
        is_high = float(reward) >= 0.5
        if (not is_succ) and (not is_high): return 0
        if (not is_succ) and is_high: return 1
        if is_succ and (not is_high): return 2
        return 3

    for step in range(cfg.total_steps):
        if step < cfg.learning_starts:
            op = np.random.randint(0, num_ops)
            cont = np.random.rand(cont_dim).astype(np.float32)
        else:
            op, cont = act(obs)

        next_obs, reward, terminated, truncated, info = env.step({"op": op, "cont": cont})
        done = bool(terminated or truncated)

        b = bucket_from_info(info, float(reward))
        buf.add(obs, op, cont, float(reward), next_obs, done, bucket=b)

        obs = next_obs
        if done:
            obs, info = env.reset()

        if step >= cfg.learning_starts:
            b_obs, b_op, b_cont, b_rew, b_nobs, b_done = buf.sample(cfg.batch_size)
            b_obs=b_obs.to(device); b_nobs=b_nobs.to(device)
            b_op=b_op.to(device).squeeze(-1)
            b_cont=b_cont.to(device)
            b_rew=b_rew.to(device).squeeze(-1)
            b_done=b_done.to(device).squeeze(-1)

            with torch.no_grad():
                logits_n, mu_n, logstd_n = actor(b_nobs)
                pi_op_soft = gumbel_softmax(logits_n, tau=cfg.tau_gumbel)
                std_n = logstd_n.exp()
                eps = torch.randn_like(std_n)
                cont_pre = mu_n + std_n * eps
                cont01_n = to01(cont_pre)

                dist = torch.distributions.Normal(mu_n, std_n)
                logp_cont = dist.log_prob(cont_pre) - torch.log(1 - torch.tanh(cont_pre).pow(2) + 1e-6)
                logp_cont = logp_cont.sum(-1)
                ent_op = -(pi_op_soft * (pi_op_soft.clamp(1e-8,1.0).log())).sum(-1)

                q1n = q1_t(b_nobs, pi_op_soft, cont01_n).squeeze(-1)
                q2n = q2_t(b_nobs, pi_op_soft, cont01_n).squeeze(-1)
                qn = torch.min(q1n, q2n)
                target = b_rew + (1.0 - b_done) * cfg.gamma * (qn + cfg.alpha * (ent_op - logp_cont))

            op_onehot = F.one_hot(b_op, num_classes=num_ops).float()
            q1_pred = q1(b_obs, op_onehot, b_cont).squeeze(-1)
            q2_pred = q2(b_obs, op_onehot, b_cont).squeeze(-1)
            loss_q1 = F.mse_loss(q1_pred, target)
            loss_q2 = F.mse_loss(q2_pred, target)
            opt_q1.zero_grad(); loss_q1.backward(); opt_q1.step()
            opt_q2.zero_grad(); loss_q2.backward(); opt_q2.step()

            logits, mu, logstd = actor(b_obs)
            pi_op_soft = gumbel_softmax(logits, tau=cfg.tau_gumbel)
            std = logstd.exp()
            eps = torch.randn_like(std)
            cont_pre = mu + std * eps
            cont01 = to01(cont_pre)

            dist = torch.distributions.Normal(mu, std)
            logp_cont = dist.log_prob(cont_pre) - torch.log(1 - torch.tanh(cont_pre).pow(2) + 1e-6)
            logp_cont = logp_cont.sum(-1)
            ent_op = -(pi_op_soft * (pi_op_soft.clamp(1e-8,1.0).log())).sum(-1)

            q1_pi = q1(b_obs, pi_op_soft, cont01).squeeze(-1)
            q2_pi = q2(b_obs, pi_op_soft, cont01).squeeze(-1)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (cfg.alpha * (ent_op - logp_cont) - q_pi).mean()

            opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()

            with torch.no_grad():
                for p, pt in zip(q1.parameters(), q1_t.parameters()):
                    pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
                for p, pt in zip(q2.parameters(), q2_t.parameters()):
                    pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

        if (step + 1) % cfg.save_every == 0:
            ck = out / "checkpoints" / f"step_{step+1:06d}.pt"
            torch.save({
                "step": step+1,
                "actor": actor.state_dict(),
                "q1": q1.state_dict(),
                "q2": q2.state_dict(),
                "q1_t": q1_t.state_dict(),
                "q2_t": q2_t.state_dict(),
                "cfg": asdict(cfg),
                "stratified_replay": bool(stratified),
            }, ck)
            torch.save({"actor": actor.state_dict()}, out / "checkpoints" / "actor_latest.pt")

    env.close()
    return {"out_dir": str(out)}
