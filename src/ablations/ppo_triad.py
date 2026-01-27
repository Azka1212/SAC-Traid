# src/ablations/ppo_triad.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import random

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    total_steps: int = 20000
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_epochs: int = 8
    mini_batch_size: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_len: int = 1024
    device: str = "cuda"


class TriadActorCritic(nn.Module):
    """
    Simple actor-critic:
      - discrete head over operator ids ("op")
      - continuous head over temperature (0–1)
    """

    def __init__(self, obs_dim: int, num_ops: int):
        super().__init__()
        hidden = 256

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # operator logits
        self.op_head = nn.Linear(hidden, num_ops)

        # temperature Gaussian head
        self.temp_mean = nn.Linear(hidden, 1)
        self.temp_log_std = nn.Parameter(torch.zeros(1))

        # critic
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.shared(obs)
        logits = self.op_head(x)
        temp_mean = self.temp_mean(x)
        temp_log_std = self.temp_log_std.expand_as(temp_mean)
        value = self.v_head(x)
        return {
            "logits": logits,
            "temp_mean": temp_mean,
            "temp_log_std": temp_log_std,
            "value": value,
        }


# -------------------------------------------------------------------
# ENV HOOK – wire to existing Triad env
# -------------------------------------------------------------------

from ..rl.envs import PromptHybridEnv


def _build_env(
    stack_config_path: str,
    runs_model: str,
    target_model: str,
    dataset_tag: str,
    seeds_path_override: str | None,
) -> Any:
    """
    Construct the SAME env that SAC uses for train/eval.
    This must match your train-sac pipeline.
    """
    env = PromptHybridEnv(
        config_path=stack_config_path,
        runs_model=runs_model,
        rewriter_model="ollama/yi:9b",          # same as SAC
        judge_model="ollama/llama3:instruct",   # same judge as SAC
        target_model=target_model,
        dataset_tag=dataset_tag,
        seeds_path_override=seeds_path_override,
        use_target_echo=False,
    )
    return env


# -------------------------------------------------------------------
# PPO core
# -------------------------------------------------------------------


def _sample_action(
    ac: TriadActorCritic,
    obs: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Sample (op, temp) and return both action + logprobs.
    """
    obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    out = ac(obs)
    logits = torch.nan_to_num(out["logits"], nan=0.0, posinf=1.0, neginf=-1.0)
    temp_mean = out["temp_mean"]
    temp_log_std = out["temp_log_std"]

    # operator
    op_dist = torch.distributions.Categorical(logits=logits)
    op = op_dist.sample()
    op_logprob = op_dist.log_prob(op)

    # temperature: Gaussian -> sigmoid
    std = torch.exp(temp_log_std)
    temp_dist = torch.distributions.Normal(temp_mean, std)
    temp_raw = temp_dist.rsample()
    temp = torch.sigmoid(temp_raw)
    temp_logprob = temp_dist.log_prob(temp_raw).sum(-1)

    action = {"op": op, "temp": temp}
    logprob = {"op_logprob": op_logprob, "temp_logprob": temp_logprob}
    return action, logprob


def _ppo_update(
    ac: TriadActorCritic,
    optimizer: optim.Optimizer,
    cfg: PPOConfig,
    batch: Dict[str, torch.Tensor],
) -> None:
    obs = batch["obs"]
    if obs.numel() == 0:
        return

    ops = batch["ops"]
    temps = batch["temps"]
    returns = batch["returns"]
    advantages = batch["advantages"]
    old_op_logp = batch["op_logp"]
    old_temp_logp = batch["temp_logp"]

    n = obs.size(0)
    idx = torch.arange(n, device=obs.device)

    for _ in range(cfg.train_epochs):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, cfg.mini_batch_size):
            end = start + cfg.mini_batch_size
            mb_idx = perm[start:end]
            if mb_idx.numel() == 0:
                continue

            mb_obs = obs[mb_idx]
            # sanitize observations (avoid NaN/Inf propagating)
            mb_obs = torch.nan_to_num(mb_obs, nan=0.0, posinf=1.0, neginf=-1.0)

            mb_ops = ops[mb_idx]
            mb_temps = temps[mb_idx]
            # keep temperatures strictly inside (0, 1) for inverse-sigmoid
            mb_temps = mb_temps.clamp(1e-4, 1.0 - 1e-4)

            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_old_op_logp = old_op_logp[mb_idx]
            mb_old_temp_logp = old_temp_logp[mb_idx]

            out = ac(mb_obs)
            logits = torch.nan_to_num(out["logits"], nan=0.0, posinf=1.0, neginf=-1.0)
            temp_mean = out["temp_mean"]
            temp_log_std = out["temp_log_std"]
            values = out["value"].squeeze(-1)
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            # bail out if something is still broken
            if not torch.isfinite(logits).all():
                print("[PPO] Warning: non-finite logits encountered, skipping minibatch")
                continue

            # new logprobs
            op_dist = torch.distributions.Categorical(logits=logits)
            new_op_logp = op_dist.log_prob(mb_ops)

            std = torch.exp(temp_log_std)
            temp_dist = torch.distributions.Normal(temp_mean, std)

            # inverse-sigmoid to recover raw temperature
            temp_raw = torch.log(mb_temps) - torch.log(1.0 - mb_temps + 1e-8)
            new_temp_logp = temp_dist.log_prob(temp_raw).sum(-1)

            # combine ratios
            op_ratio = torch.exp(new_op_logp - mb_old_op_logp)
            temp_ratio = torch.exp(new_temp_logp - mb_old_temp_logp)
            ratio = 0.5 * (op_ratio + temp_ratio)

            # clipped objective
            adv_std = mb_adv.std()
            if adv_std > 1e-8:
                mb_adv = (mb_adv - mb_adv.mean()) / (adv_std + 1e-8)
            else:
                mb_adv = mb_adv - mb_adv.mean()
            # avoid extremely huge advantages
            mb_adv = torch.clamp(mb_adv, -10.0, 10.0)

            unclipped = ratio * mb_adv
            clipped = torch.clamp(
                ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
            ) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            # value / entropy
            value_loss = (mb_returns - values).pow(2).mean()
            ent = op_dist.entropy().mean() + temp_dist.entropy().mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
            optimizer.step()


# -------------------------------------------------------------------
# Public functions used by algorithms.py
# -------------------------------------------------------------------


def train_ppo_triad(
    stack_config_path: str,
    runs_model: str,
    target_model: str,
    dataset_tag: str,
    total_steps: int,
    seed: int,
    out_dir: Path,
) -> Path:
    """
    Train PPO-Triad on SORRY-Bench.

    Returns:
        Path to PPO checkpoint (.pt).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    cfg = PPOConfig(total_steps=total_steps)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    env = _build_env(
        stack_config_path=stack_config_path,
        runs_model=runs_model,
        target_model=target_model,
        dataset_tag=dataset_tag,
        seeds_path_override=None,
    )

    # infer obs_dim & num_ops
    obs, info = env.reset(seed=seed)
    if isinstance(obs, dict):
        raise ValueError("ppo_triad assumes flat tensor obs, not dict obs.")

    obs_dim = obs.shape[-1]
    num_ops = env.action_space["op"].n  # assumes dict action space with 'op'

    ac = TriadActorCritic(obs_dim=obs_dim, num_ops=num_ops).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=cfg.lr)

    # buffers
    obs_buf = []
    op_buf = []
    temp_buf = []
    reward_buf = []
    done_buf = []
    op_logp_buf = []
    temp_logp_buf = []
    value_buf = []

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    step = 0

    while step < cfg.total_steps:
        # rollout
        for _ in range(cfg.rollout_len):
            with torch.no_grad():
                action, logp = _sample_action(ac, obs_t.unsqueeze(0))
                out = ac(obs_t.unsqueeze(0))
                value = out["value"].squeeze(0)

            op = action["op"].item()
            temp = action["temp"].item()

            # Map scalar temp -> 3-d continuous control for env ("cont")
            env_action = {
                "op": op,
                "cont": [0.5, float(temp), 0.5],  # length, temperature, persona
            }

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)

            obs_buf.append(obs_t.cpu())
            op_buf.append(op)
            temp_buf.append(float(temp))
            reward_buf.append(float(reward))
            done_buf.append(done)
            op_logp_buf.append(float(logp["op_logprob"]))
            temp_logp_buf.append(float(logp["temp_logprob"]))
            value_buf.append(value.cpu())

            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            step += 1

            if done or step >= cfg.total_steps:
                obs_t, info = env.reset()
                obs_t = torch.as_tensor(obs_t, dtype=torch.float32, device=device)
                break

        if len(reward_buf) == 0:
            continue

        # GAE / returns
        rewards = torch.tensor(reward_buf, dtype=torch.float32, device=device)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        dones = torch.tensor(done_buf, dtype=torch.float32, device=device)
        values = torch.stack(value_buf).squeeze(-1).to(device)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_gae_lam = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * next_value * nonterminal - values[t]
            last_gae_lam = delta + cfg.gamma * cfg.lam * nonterminal * last_gae_lam
            advantages[t] = last_gae_lam
            next_value = values[t]
        returns = advantages + values

        batch = {
            "obs": torch.stack(obs_buf).to(device),
            "ops": torch.tensor(op_buf, dtype=torch.long, device=device),
            "temps": torch.tensor(temp_buf, dtype=torch.float32, device=device),
            "returns": returns.detach(),
            "advantages": advantages.detach(),
            "op_logp": torch.tensor(op_logp_buf, dtype=torch.float32, device=device),
            "temp_logp": torch.tensor(temp_logp_buf, dtype=torch.float32, device=device),
        }

        _ppo_update(ac, optimizer, cfg, batch)

        # clear buffers
        obs_buf.clear()
        op_buf.clear()
        temp_buf.clear()
        reward_buf.clear()
        done_buf.clear()
        op_logp_buf.clear()
        temp_logp_buf.clear()
        value_buf.clear()

    ckpt_path = out_dir / f"ppo_triad_seed{seed:03d}.pt"
    torch.save({"model_state_dict": ac.state_dict()}, ckpt_path)
    return ckpt_path


def eval_ppo_triad(
    stack_config_path: str,
    runs_model: str,
    target_model: str,
    checkpoint_path: str,
    seeds_path_override: str,
    dataset_tag: str,
    episodes: int,
) -> None:
    """
    Evaluate PPO-Triad on JBB-OOD.

    Assumes:
      - env writes events into RunStore/events.jsonl with dataset_tag.
      - compute_metrics() will later pick them up (same as SAC).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = _build_env(
        stack_config_path=stack_config_path,
        runs_model=runs_model,
        target_model=target_model,
        dataset_tag=dataset_tag,
        seeds_path_override=seeds_path_override,
    )

    obs, info = env.reset()
    obs_dim = obs.shape[-1]
    num_ops = env.action_space["op"].n

    ac = TriadActorCritic(obs_dim=obs_dim, num_ops=num_ops).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    ac.load_state_dict(state["model_state_dict"])
    ac.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        while not done:
            with torch.no_grad():
                action, logp = _sample_action(ac, obs_t.unsqueeze(0))

            op = action["op"].item()
            temp = action["temp"].item()

            env_action = {
                "op": op,
                "cont": [0.5, float(temp), 0.5],
            }

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

    print(f"[PPO-Triad][EVAL] Completed {episodes} episodes on dataset {dataset_tag}")
