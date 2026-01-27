# src/rl/hybrid_sac.py
# ---- tracing helpers ----
import os as _os, time as _time

def _t(): return _time.monotonic()
def _ms(dt): return f"{dt*1000:.1f} ms"
def _trace_on(env_kwargs): 
    try:
        if env_kwargs and bool(env_kwargs.get("trace")): return True
    except Exception:
        pass
    return bool(int(_os.getenv("APP_TRACE", "0")))

def _p(msg: str):
    print(msg, flush=True)
# -------------------------

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from pathlib import Path
import csv
import json
import random
import datetime as _dt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torch.serialization as _ts  # add

# ---------------------------
# Small helpers
# ---------------------------
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
    y = F.softmax((logits + g) / tau, dim=-1)
    return y

def to01(x: torch.Tensor) -> torch.Tensor:
    return (torch.tanh(x) * 0.5) + 0.5

def _preview(s: Optional[str], n: int = 160) -> str:
    if not s:
        return ""
    s = str(s).replace("\n", " ").replace("\r", " ")
    return (s[:n] + "…") if len(s) > n else s


# ---------------------------
# Networks
# ---------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim: int, num_ops: int, cont_dim: int, hidden=(256, 256)):
        super().__init__()
        self.backbone = mlp(obs_dim, hidden, 256)
        self.logits = nn.Linear(256, num_ops)     # categorical over ops
        self.mu     = nn.Linear(256, cont_dim)    # pre-tanh mean for continuous
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


# ---------------------------
# Replay buffer
# ---------------------------
class Replay:
    def __init__(self, size: int, obs_dim: int, cont_dim: int):
        self.size = size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.op  = np.zeros((size, 1), dtype=np.int64)
        self.cont= np.zeros((size, cont_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.nobs= np.zeros((size, obs_dim), dtype=np.float32)
        self.done= np.zeros((size, 1), dtype=np.float32)

    def add(self, obs, op, cont, rew, nobs, done):
        i = self.ptr
        self.obs[i]  = obs
        self.op[i,0] = op
        self.cont[i] = cont
        self.rew[i,0]= rew
        self.nobs[i] = nobs
        self.done[i,0]= float(done)
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch: int):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.op[idx]),
            torch.as_tensor(self.cont[idx]),
            torch.as_tensor(self.rew[idx]),
            torch.as_tensor(self.nobs[idx]),
            torch.as_tensor(self.done[idx]),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "ptr": self.ptr,
            "full": self.full,
            "obs": self.obs,
            "op": self.op,
            "cont": self.cont,
            "rew": self.rew,
            "nobs": self.nobs,
            "done": self.done,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.size = int(sd["size"])
        self.ptr = int(sd["ptr"])
        self.full = bool(sd["full"])
        self.obs[:]  = sd["obs"]
        self.op[:]   = sd["op"]
        self.cont[:] = sd["cont"]
        self.rew[:]  = sd["rew"]
        self.nobs[:] = sd["nobs"]
        self.done[:] = sd["done"]


# ---------------------------
# Config
# ---------------------------
@dataclass
class SACConfig:
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
    log_every: int = 100
    save_every: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# IO helpers
# ---------------------------
def _ensure_out(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out

def _csv_fieldnames() -> List[str]:
    return [
        "runs_model","rewriter_model","judge_model","target_model",
        "total_steps","learning_starts","batch_size","seed",
        "time_iso","step","episode","ep_step","reward","ep_return",
        "op","cont0","cont1","cont2",
        "seed_id","category","operator","dataset_tag",
        "judge_parsed_ok","judge_latency_ms","target_calls","seen_before",
        "prompt_len","prompt_preview","target_preview",
        "score_Success","score_Stealth","score_Novelty","score_Efficiency","score_Impact",
        "error_stage","error",
    ]

def _open_csv_writer(path: Path):
    f = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=_csv_fieldnames(),
        extrasaction="ignore",
    )
    if path.stat().st_size == 0:
        writer.writeheader()
    return f, writer

def _flatten_scores(scores: Dict[str, float]):
    return {
        "score_Success": float(scores.get("Success", 0.0)),
        "score_Stealth": float(scores.get("Stealth", 0.0)),
        "score_Novelty": float(scores.get("Novelty", 0.0)),
        "score_Efficiency": float(scores.get("Efficiency", 0.0)),
        "score_Impact": float(scores.get("Impact", 0.0)),
    }

def _write_run_spec(out: Path, env_id: str, env_kwargs: Dict[str, Any] | None, cfg: SACConfig):
    spec = {
        "created_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "env_id": env_id,
        "env_kwargs": env_kwargs or {},
        "sac_config": asdict(cfg),
        "notes": "Auto-generated by train_hybrid_sac",
        "models": {
            "runs_model": (env_kwargs or {}).get("runs_model"),
            "rewriter_model": (env_kwargs or {}).get("rewriter_model"),
            "judge_model": (env_kwargs or {}).get("judge_model"),
            "target_model": (env_kwargs or {}).get("target_model"),
        },
    }
    (out / "run_spec.json").write_text(json.dumps(spec, indent=2))

def set_global_seed(seed: int):
    import os, random as _py_random
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass


# ---------------------------
# Training
# ---------------------------
def train_hybrid_sac(
    *,
    env_id: str,
    env_kwargs: Optional[Dict[str, Any]],
    cfg: SACConfig,
    out_dir: str,
    resume_from: Optional[str] = None,
):
    TALL = _trace_on(env_kwargs)
    t0_all = _t()
    if TALL:
        _p(f"[SAC] start | env_id={env_id} steps={cfg.total_steps} "
           f"learn_start={cfg.learning_starts} batch={cfg.batch_size} alpha={cfg.alpha} device={cfg.device}")

    # seeding
    if TALL: _p("[SAC] seeding…")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    t_seed = _t()
    set_global_seed(int(cfg.seed))
    if TALL: _p(f"[SAC] set_global_seed in {_ms(_t()-t_seed)}")

    # out dir / CSV
    t_out = _t()
    out = _ensure_out(out_dir)
    if TALL: _p(f"[SAC] ensure out dir in {_ms(_t()-t_out)} -> {out}")

    t_csv = _t()
    metrics_path = out / "metrics.csv"
    f_csv, csvw = _open_csv_writer(metrics_path)
    if TALL: _p(f"[SAC] CSV writer ready in {_ms(_t()-t_csv)} -> {metrics_path}")

    redact_csv = bool((env_kwargs or {}).get("redact_in_csv", False))

    # env registration / creation
    t_reg = _t()
    import src.rl.registry as reg
    reg.register_prompt_hybrid_env()
    if TALL: _p(f"[SAC] registry.register in {_ms(_t()-t_reg)}")

    t_make = _t()
    env: gym.Env = gym.make(env_id, **({**(env_kwargs or {}), "seed": int(cfg.seed)}))
    if TALL: _p(f"[SAC] gym.make in {_ms(_t()-t_make)}")

    t_spec = _t()
    _write_run_spec(out, env_id, env_kwargs, cfg)
    if TALL: _p(f"[SAC] wrote run_spec in {_ms(_t()-t_spec)}")

    t_reset = _t()
    obs, info = env.reset(seed=cfg.seed)
    if TALL: _p(f"[SAC] env.reset in {_ms(_t()-t_reset)}")

    # dims
    obs_dim  = int(np.prod(env.observation_space.shape))
    num_ops  = int(env.action_space["op"].n)
    cont_dim = int(env.action_space["cont"].shape[0])
    if TALL: _p(f"[SAC] dims | obs={obs_dim} ops={num_ops} cont={cont_dim}")

    # nets/opt
    t_models = _t()
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
    if TALL: _p(f"[SAC] models+opt in {_ms(_t()-t_models)} | device={device}")

    # resume?
    start_step = 0
    if resume_from:
        t_load = _t()
        ckpt = _safe_torch_load(resume_from, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        q1.load_state_dict(ckpt["q1"]); q2.load_state_dict(ckpt["q2"])
        q1_t.load_state_dict(ckpt["q1_t"]); q2_t.load_state_dict(ckpt["q2_t"])
        opt_actor.load_state_dict(ckpt["opt_actor"])
        opt_q1.load_state_dict(ckpt["opt_q1"]); opt_q2.load_state_dict(ckpt["opt_q2"])
        start_step = int(ckpt.get("step", 0))
        if TALL: _p(f"[SAC] loaded checkpoint in {_ms(_t()-t_load)} | resume step={start_step}")

        buf = Replay(cfg.buffer_size, obs_dim, cont_dim)
        if "replay" in ckpt:
            try:
                r_sd = ckpt["replay"]
                if int(r_sd.get("size", 0)) == buf.size:
                    buf.load_state_dict(r_sd)
                    if TALL: _p(f"[SAC] replay restored | size={buf.size} ptr={buf.ptr} full={buf.full}")
            except Exception as e:
                if TALL: _p(f"[SAC] replay restore warn: {e}")
        else:
            buf = Replay(cfg.buffer_size, obs_dim, cont_dim)

        # RNG states
        try:
            import random as _py_random
            if "py_random_state" in ckpt:
                _py_random.setstate(ckpt["py_random_state"])
            if "np_random_state" in ckpt:
                np.random.set_state(tuple(ckpt["np_random_state"]))
            if "torch_rng_state" in ckpt:
                torch.set_rng_state(ckpt["torch_rng_state"])
            if torch.cuda.is_available() and "torch_cuda_rng_state" in ckpt:
                torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])
        except Exception as e:
            if TALL: _p(f"[SAC] RNG restore warn: {e}")
    else:
        buf = Replay(cfg.buffer_size, obs_dim, cont_dim)
        if TALL: _p(f"[SAC] replay created | size={buf.size}")

    # static spec columns
    runs_model     = (env_kwargs or {}).get("runs_model")
    rewriter_model = (env_kwargs or {}).get("rewriter_model")
    judge_model    = (env_kwargs or {}).get("judge_model")
    target_model   = (env_kwargs or {}).get("target_model")
    if TALL:
        _p(f"[SAC] env models | rewriter={rewriter_model} judge={judge_model} "
           f"target={target_model} echo={(env_kwargs or {}).get('use_target_echo')}")

    # policy helper
    def act_eval(o_np: np.ndarray):
        with torch.no_grad():
            o = torch.as_tensor(o_np, device=device).unsqueeze(0)
            logits, mu, logstd = actor(o)
            op = torch.argmax(logits, dim=-1).item()
            cont = to01(mu).squeeze(0).cpu().numpy()
        return op, cont

    ep = 0
    ep_return = 0.0
    ep_step = 0

    try:
        for step in range(start_step, cfg.total_steps):
            t_step0 = _t()

            # pick action
            t_a0 = _t()
            if step < cfg.learning_starts:
                op  = np.random.randint(0, num_ops)
                cont= np.random.rand(cont_dim).astype(np.float32)
                picked = "random"
            else:
                op, cont = act_eval(obs)
                picked = "policy"
            dt_act = _t() - t_a0

            # env step
            t_envs0 = _t()
            next_obs, reward, terminated, truncated, info = env.step({"op": op, "cont": cont})
            done = terminated or truncated
            dt_envs = _t() - t_envs0

            # buffer add
            buf.add(obs, op, cont, reward, next_obs, done)

            # per-step CSV row
            tgt_model_used = info.get("target_model") or target_model
            row = {
                "runs_model": runs_model,
                "rewriter_model": rewriter_model,
                "judge_model": judge_model,
                "target_model": tgt_model_used,
                "total_steps": cfg.total_steps,
                "learning_starts": cfg.learning_starts,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
                "time_iso": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "step": step,
                "episode": ep,
                "ep_step": ep_step,
                "reward": float(reward),
                "ep_return": float(ep_return + reward),
                "op": int(op),
                "cont0": float(cont[0]), "cont1": float(cont[1]), "cont2": float(cont[2]),
                "seed_id": info.get("seed_id"),
                "category": info.get("category"),
                "operator": info.get("operator"),
                "dataset_tag": info.get("dataset_tag") or info.get("dataset"),
                "judge_parsed_ok": info.get("judge_parsed_ok"),
                "judge_latency_ms": info.get("judge_latency_ms"),
                "target_calls": info.get("target_calls"),
                "seen_before": info.get("seen_before"),
                "prompt_len": len((info.get("prompt_text") or "")),
                "prompt_preview": "" if redact_csv else _preview(info.get("prompt_text")),
                "target_preview": "" if redact_csv else _preview(info.get("target_text")),
                "error_stage": info.get("error_stage"),
                "error": info.get("error"),
            }
            if isinstance(info.get("scores"), dict):
                row.update(_flatten_scores(info["scores"]))
            csvw.writerow(row)

            if (step + 1) % max(1, int(cfg.log_every)) == 0:
                f_csv.flush()

            # move forward
            ep_return += float(reward)
            ep_step += 1
            obs = next_obs

            if done:
                t_res0 = _t()
                obs, info = env.reset()
                dt_res = _t() - t_res0
                if TALL:
                    _p(f"[SAC][ep {ep}] reset in {_ms(dt_res)} | ep_return={ep_return:.3f} steps={ep_step}")
                ep += 1
                ep_return = 0.0
                ep_step = 0

            # updates
            dt_upd = 0.0
            if step >= cfg.learning_starts:
                t_upd0 = _t()
                b_obs, b_op, b_cont, b_rew, b_nobs, b_done = buf.sample(cfg.batch_size)
                b_obs = b_obs.to(device); b_nobs = b_nobs.to(device)
                b_op  = b_op.to(device).squeeze(-1)
                b_cont= b_cont.to(device)
                b_rew = b_rew.to(device).squeeze(-1)
                b_done= b_done.to(device).squeeze(-1)

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

                # actor update
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

                # target updates
                with torch.no_grad():
                    for p, pt in zip(q1.parameters(), q1_t.parameters()):
                        pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
                    for p, pt in zip(q2.parameters(), q2_t.parameters()):
                        pt.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

                dt_upd = _t() - t_upd0

            # periodic checkpoint
            if (step + 1) % cfg.save_every == 0:
                t_ck0 = _t()
                step_num = step + 1
                ckdir = out / "checkpoints"
                ckdir.mkdir(parents=True, exist_ok=True)
                ck = ckdir / f"step_{step_num:06d}.pt"
                state = {
                    "step": step_num,
                    "actor": actor.state_dict(),
                    "q1": q1.state_dict(), "q2": q2.state_dict(),
                    "q1_t": q1_t.state_dict(), "q2_t": q2_t.state_dict(),
                    "opt_actor": opt_actor.state_dict(),
                    "opt_q1": opt_q1.state_dict(),
                    "opt_q2": opt_q2.state_dict(),
                    "cfg": asdict(cfg),
                    "obs_dim": obs_dim, "num_ops": num_ops, "cont_dim": cont_dim,
                    "replay": buf.state_dict(),
                    "py_random_state": __import__("random").getstate(),
                    "np_random_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                }
                torch.save(state, ck)
                torch.save(state, ckdir / "ckpt_latest.pt")
                f_csv.flush()
                if TALL:
                    _p(f"[SAC] saved ckpt step={step_num} in {_ms(_t()-t_ck0)} -> {ck}")

            if TALL or ((step + 1) % max(1, int(cfg.log_every)) == 0):
                _p(f"[SAC][{step+1}/{cfg.total_steps}] pick={picked} "
                   f"env={_ms(dt_envs)} act={_ms(dt_act)} upd={_ms(dt_upd)} "
                   f"r={reward:.3f} epR={ep_return:.3f} done={bool(done)}")

    finally:
        try:
            f_csv.flush(); f_csv.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        if TALL:
            _p(f"[SAC] total time {_ms(_t()-t0_all)} | out={out.as_posix()}")
        # --- write a light training summary for experiment tracking
        try:
            summary = {
                "out_dir": out.as_posix(),
                "total_steps": cfg.total_steps,
                "seed": cfg.seed,
                "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

    return {"out_dir": out.as_posix()}


# --- add these helpers anywhere above _load_actor_for_env ---
def _allow_numpy_reconstruct():
    """Allow-list numpy reconstruct symbol for PyTorch 2.6 safe unpickler."""
    try:
        fn = getattr(np.core.multiarray, "_reconstruct", None)  # type: ignore[attr-defined]
        if fn is not None and hasattr(_ts, "add_safe_globals"):
            _ts.add_safe_globals([fn])
    except Exception:
        pass



def _safe_torch_load(path: str | Path, *, map_location: torch.device):
    # 1) Preferred: safe loader (no warning)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Optional trace
        if bool(int(_os.getenv("APP_TRACE","0"))):
            _p("[torch.load] weights_only=True failed; trying numpy allowlist…")
        # 2) Add NumPy reconstruct to safe globals and retry safe mode
        _allow_numpy_reconstruct()
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            if bool(int(_os.getenv("APP_TRACE","0"))):
                _p("[torch.load] safe mode still failed; falling back to legacy unpickler (weights_only=False).")
            # 3) Last resort (legacy). Works with old pickles; don’t use untrusted files.
            return torch.load(path, map_location=map_location, weights_only=False)



# ---------------------------
# Evaluation
# ---------------------------
def _load_actor_for_env(ckpt_path: str, env: gym.Env, device: torch.device):
    ck = _safe_torch_load(ckpt_path, map_location=device)
    obs_dim  = int(np.prod(env.observation_space.shape))
    num_ops  = int(env.action_space["op"].n)
    cont_dim = int(env.action_space["cont"].shape[0])
    actor = Actor(obs_dim, num_ops, cont_dim).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def eval_hybrid_sac(
    *,
    checkpoint_path: str,
    env_id: str,
    env_kwargs: Optional[Dict[str, Any]],
    episodes: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # NEW ↓
    progress_every: int = 0,
):
    import src.rl.registry as reg
    reg.register_prompt_hybrid_env()
    env: gym.Env = gym.make(env_id, **(env_kwargs or {}))
    device_t = torch.device(device)
    actor = _load_actor_for_env(checkpoint_path, env, device_t)

    def act(o_np):
        o = torch.as_tensor(o_np, device=device_t).unsqueeze(0)
        logits, mu, logstd = actor(o)
        op = torch.argmax(logits, dim=-1).item()
        cont = to01(mu).squeeze(0).cpu().numpy()
        return op, cont

    rets = []  
    t0 = _time.monotonic()
    for epi in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ret = 0.0
        while not done:
            op, cont = act(obs)
            obs, r, term, trunc, inf = env.step({"op": op, "cont": cont})
            done = term or trunc
            ret += float(r)
        rets.append(ret)

        if progress_every and (epi % progress_every == 0 or epi == episodes):
            dt = _time.monotonic() - t0
            eps_sec = epi / max(1e-6, dt)
            eta_sec = (episodes - epi) / max(1e-6, eps_sec)
            avg = float(np.mean(rets))
            std = float(np.std(rets))
            print(
                f"[EVAL] {epi}/{episodes} | mean={avg:.4f} ± {std:.4f} | "
                f"eps/s={eps_sec:.2f} | ETA≈{eta_sec:.1f}s",
                flush=True,
            )

    env.close()
    arr = np.array(rets, dtype=np.float32)
    return {"episodes": episodes, "mean_reward": float(arr.mean()), "std_reward": float(arr.std())}

# ---------------------------
# Rollout and export prompts
# ---------------------------
@torch.no_grad()
def rollout_and_export(
    *,
    checkpoint_path: str,
    env_id: str,
    env_kwargs: Optional[Dict[str, Any]],
    n: int,
    out_csv: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    import src.rl.registry as reg
    reg.register_prompt_hybrid_env()
    env: gym.Env = gym.make(env_id, **(env_kwargs or {}))
    device_t = torch.device(device)
    actor = _load_actor_for_env(checkpoint_path, env, device_t)

    def act(o_np):
        o = torch.as_tensor(o_np, device=device_t).unsqueeze(0)
        logits, mu, logstd = actor(o)
        op = torch.argmax(logits, dim=-1).item()
        cont = to01(mu).squeeze(0).cpu().numpy()
        return op, cont

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed_id","category","operator","cont0","cont1","cont2",
            "reward","score_Success","score_Stealth","score_Novelty","score_Efficiency","score_Impact",
            "prompt_text","target_text","target_model"
        ])
        writer.writeheader()
        for _ in range(n):
            obs, info = env.reset()
            op, cont = act(obs)
            obs, r, term, trunc, inf = env.step({"op": op, "cont": cont})
            row = {
                "seed_id": inf.get("seed_id"),
                "category": inf.get("category"),
                "operator": inf.get("operator"),
                "cont0": float(cont[0]), "cont1": float(cont[1]), "cont2": float(cont[2]),
                "reward": float(r),
                "prompt_text": inf.get("prompt_text",""),
                "target_text": inf.get("target_text",""),
                "target_model": inf.get("target_model"),
            }
            if isinstance(inf.get("scores"), dict):
                row.update(_flatten_scores(inf["scores"]))
            writer.writerow(row)
    env.close()
    return out_csv
