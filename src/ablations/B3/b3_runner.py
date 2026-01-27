# /home/security/Azka_container/src/ablations/B3/b3_runner.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import yaml

from src.ablations.B3.b3_config import load_yaml, make_b3_paths, write_json
from src.ablations.B3.b3_registry import register_b3_env
from src.ablations.B3.b3_sac import train_sac_b3, SACCfg, Actor, to01
from src.ablations.B3.b3_metrics import compute_b3_metrics


def write_stack_override(base_stack_path: str, out_stack_path: Path, runs_dir: Path) -> str:
    base = Path(base_stack_path)
    cfg = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    cfg.setdefault("paths", {})
    cfg["paths"]["runs"] = str(runs_dir)

    out_stack_path.parent.mkdir(parents=True, exist_ok=True)
    out_stack_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_stack_path.as_posix()


def load_actor_from_checkpoint(
    actor_ckpt_path: Path,
    obs_dim: int,
    num_ops: int,
    cont_dim: int,
    device: torch.device,
) -> Actor:
    if not actor_ckpt_path.exists():
        raise FileNotFoundError(f"Missing actor checkpoint: {actor_ckpt_path}")

    actor = Actor(obs_dim, num_ops, cont_dim).to(device)
    state = torch.load(actor_ckpt_path, map_location=device)
    if "actor" not in state:
        raise KeyError(f"Checkpoint missing 'actor' key: {actor_ckpt_path}")
    actor.load_state_dict(state["actor"])
    actor.eval()
    return actor


def policy_action(actor: Actor, obs_np: np.ndarray, device: torch.device) -> Dict[str, Any]:
    with torch.no_grad():
        obs = torch.as_tensor(obs_np, device=device).float().unsqueeze(0)
        logits, mu, _logstd = actor(obs)
        op = int(torch.argmax(logits, dim=-1).item())
        cont = to01(mu).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return {"op": op, "cont": cont}


def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k.lower()] = _lower_keys(v)
        else:
            out[k.lower()] = v
    return out


def normalize_metrics_for_table(metrics_out: Dict[str, Any]) -> Dict[str, float]:
    m = metrics_out.get("metrics", {}) if isinstance(metrics_out.get("metrics", {}), dict) else {}
    m_l = _lower_keys(m)

    def get_any(*names, default=0.0):
        for n in names:
            if n in m_l:
                try:
                    return float(m_l[n])
                except Exception:
                    return default
        return default

    return {
        "asr": get_any("asr"),
        "qs": get_any("q/s", "q_s", "qs", default=float("inf")),
        "stealth": get_any("stealth"),
        "impact": get_any("impact"),
        "dist1": get_any("dist-1", "dist1"),
        "dist2": get_any("dist-2", "dist2"),
        "dist3": get_any("dist-3", "dist3"),
    }


def main(config_path: str) -> None:
    b3 = load_yaml(config_path)

    artifacts_root = Path(b3["artifacts_root"])
    base_stack = str(b3["stack_config_path"])
    target_model = str(b3["target_model"])

    seeds: List[int] = [int(x) for x in b3["seeds"]]
    budget_k = int(b3["budget_k"])
    success_th = float(b3.get("success_threshold", 0.65))

    train_steps = int(b3["train_steps"])
    learning_starts = int(b3["learning_starts"])
    batch_size = int(b3["batch_size"])
    save_every = int(b3["save_every"])
    eval_episodes = int(b3["eval_episodes"])

    paths = make_b3_paths(str(artifacts_root))
    register_b3_env()

    # NOTE: minus_curriculum removed because curriculum is not implemented (would be identical to full).
    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("full", dict(disable_rewriter=False, stratified_replay=True,  disable_dedup=False, disable_early_exit=False)),
        ("minus_rewriter", dict(disable_rewriter=True, stratified_replay=True,  disable_dedup=False, disable_early_exit=False)),
        ("minus_stratified_replay", dict(disable_rewriter=False, stratified_replay=False, disable_dedup=False, disable_early_exit=False)),
        ("minus_earlyexit_dedup", dict(disable_rewriter=False, stratified_replay=True,  disable_dedup=True,  disable_early_exit=True)),
    ]

    rows: List[Dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Local imports here keep B3 isolated but still reuse existing builders
    from src.config import load_app_config
    from src.store.run_store import RunStore
    from src.data.loader import build_all, jbb_category_counts, export_jbb_harmful_to_seeds_ood

    for vname, flags in variants:
        for sd in seeds:
            run_root = artifacts_root / "runs" / vname / f"seed_{sd}"
            run_root.mkdir(parents=True, exist_ok=True)

            stack_override_path = run_root / "stack_b3.yaml"
            stack_path_for_run = write_stack_override(base_stack, stack_override_path, run_root)

            runs_model = f"b3_{vname}_{target_model.replace('/','_').replace(':','_')}_seed{sd}"

            cfg_stack = load_app_config(stack_path_for_run)
            rs = RunStore(runs_root=Path(cfg_stack["paths"]["runs"]), model_name=runs_model, redact_in_csv=False)

            if not (rs.root / "seeds.jsonl").exists() or not (rs.root / "operators.json").exists():
                data_root = Path(cfg_stack["paths"]["data"])
                build_all(data_root=data_root, run_dir=rs.root)
                rs.write_meta(data_root=data_root, jbb_stats=jbb_category_counts(data_root / "jbb"))

            ood_path = rs.root / "seeds_ood.jbb.jsonl"
            if not ood_path.exists():
                data_root = Path(cfg_stack["paths"]["data"])
                export_jbb_harmful_to_seeds_ood(data_root / "jbb", ood_path)

            train_out_dir = rs.root / "b3_train"
            env_kwargs = {
                "config_path": stack_path_for_run,
                "runs_model": runs_model,
                "seed": int(sd),
                "budget_k": budget_k,
                "train_mode": True,
                "seeds_path_override": None,
                "dataset_tag": "ID-SORRY",
                "use_target_echo": True,
                "success_threshold": success_th,
                "target_model_override": target_model,
                "disable_curriculum": False,
                **flags,
            }

            sac_cfg = SACCfg(
                total_steps=train_steps,
                learning_starts=learning_starts,
                batch_size=batch_size,
                seed=int(sd),
                save_every=save_every,
            )

            train_sac_b3(
                env_id="PromptHybridB3-v0",
                env_kwargs=env_kwargs,
                cfg=sac_cfg,
                out_dir=str(train_out_dir),
                stratified=bool(flags["stratified_replay"]),
            )

            eval_kwargs = dict(env_kwargs)
            eval_kwargs.update({
                "train_mode": False,
                "seeds_path_override": str(ood_path),
                "dataset_tag": "OOD-JBB",
            })

            env = gym.make("PromptHybridB3-v0", **eval_kwargs)
            obs, _ = env.reset(seed=int(sd))

            obs_dim = int(np.prod(env.observation_space.shape))
            num_ops = int(env.action_space["op"].n)
            cont_dim = int(env.action_space["cont"].shape[0])

            actor_ckpt = train_out_dir / "checkpoints" / "actor_latest.pt"
            actor = load_actor_from_checkpoint(actor_ckpt, obs_dim, num_ops, cont_dim, device)

            for _ in range(eval_episodes):
                obs, _ = env.reset()
                done = False
                while not done:
                    a = policy_action(actor, obs, device)
                    obs, _r, term, trunc, _info = env.step(a)
                    done = bool(term or trunc)

            env.close()

            metrics_out = compute_b3_metrics(rs.jsonl_path, dataset_tag="OOD-JBB", success_threshold=success_th)
            write_json(rs.root / "b3_metrics_ood.json", metrics_out)

            nm = normalize_metrics_for_table(metrics_out)
            counts = metrics_out.get("counts", {}) if isinstance(metrics_out.get("counts", {}), dict) else {}

            metrics_json = {
                "variant": vname,
                "seed": sd,
                "eval": {
                    "asr": nm["asr"],
                    "qs": nm["qs"],
                    "stealth": nm["stealth"],
                    "impact": nm["impact"],
                    "dist1": nm["dist1"],
                    "dist2": nm["dist2"],
                    "dist3": nm["dist3"],
                },
                "counts": counts,
                "run_dir": rs.root.as_posix(),
            }

            # single tree: write metrics.json inside run_root
            write_json(run_root / "metrics.json", metrics_json)

            rows.append({
                "Variant": vname,
                "Seed": sd,
                "ASR": nm["asr"],
                "Q/S": nm["qs"],
                "Stealth": nm["stealth"],
                "Impact": nm["impact"],
                "Dist-1": nm["dist1"],
                "Dist-2": nm["dist2"],
                "Dist-3": nm["dist3"],
                "seeds": counts.get("seeds"),
                "successes": counts.get("successes"),
                "total_target_calls": counts.get("total_target_calls"),
                "run_dir": rs.root.as_posix(),
            })

    out_csv = paths.tables_root / "table_b3_component_ablation.csv"
    out_json = paths.tables_root / "table_b3_component_ablation.json"

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_json(out_json, {"rows": rows})

    print(f"[B3] wrote: {out_csv}")
    print(f"[B3] wrote: {out_json}")
    print(f"[B3] per-run metrics written under: {artifacts_root}/runs/<variant>/seed_<n>/metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", default="config/ablations/b3_component.yaml")
    args = ap.parse_args()
    main(args.config_path)
