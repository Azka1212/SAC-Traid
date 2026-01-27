from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import B2Config


def _sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _dataset_tag_from_name(name: str) -> str:
    if not name:
        return ""

    name = name.lower()
    if "sorry" in name:
        return "SORRY-BENCH"
    if "jbb" in name or "jailbreakbench" in name:
        return "JBB-OOD"

    return name.replace("_", "-").upper()


def _find_latest_sac_run_dir(model_id: str) -> Optional[Path]:
    """
    Find the most recent SAC run directory for a given model_id under:

      artifacts/runs/<sanitized_model_id>/<run_dir>/rl_sac
    """
    runs_root = Path("artifacts/runs")
    model_dir = runs_root / _sanitize_model_id(model_id)

    if not model_dir.is_dir():
        print(f"[B2] WARN: runs root not found for model: {model_dir}")
        return None

    run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        print(f"[B2] WARN: no run dirs under {model_dir}")
        return None

    # pick most recently modified run
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_run = run_dirs[0]
    sac_dir = latest_run / "rl_sac"

    if not sac_dir.is_dir():
        print(f"[B2] WARN: rl_sac dir not found in {latest_run}")
        return None

    print(f"[B2] Using SAC run dir: {sac_dir}")
    return sac_dir


def _mirror_sac_run_into_b2(sac_dir: Path, run_root: Path) -> None:
    """
    Make B2 fully self-contained:

      - Copy the entire rl_sac directory into the B2 seed folder.
      - Optionally copy key metrics files to the seed root.
      - Delete the original run directory under artifacts/runs.

    After this, ALL artifacts for this B2 run live only under:

      artifacts/ablations/b2/<variant>/<target_short>/seed_<seed>/
    """
    run_root.mkdir(parents=True, exist_ok=True)

    # 1) Copy full rl_sac -> run_root/rl_sac
    dest_sac_dir = run_root / "rl_sac"
    if dest_sac_dir.exists():
        shutil.rmtree(dest_sac_dir)

    print(f"[B2] Copying SAC run tree -> {dest_sac_dir}")
    shutil.copytree(sac_dir, dest_sac_dir)

    # 2) Try to surface the main metrics at the seed root for convenience
    #    (aggregate.py can also find them via rglob, but this is nice UX)
    for fname in ["train_metrics.json", "train_events.jsonl",
                  "eval_metrics.json", "eval_episodes.jsonl"]:
        # look for file at top of rl_sac first
        src = dest_sac_dir / fname
        if not src.is_file():
            # fallback: search anywhere under rl_sac
            candidates = list(dest_sac_dir.rglob(fname))
            if candidates:
                src = candidates[0]
            else:
                continue

        dst = run_root / fname
        shutil.copy2(src, dst)
        print(f"[B2] Surfaced {fname} at {dst}")

    # 3) Remove the original run directory under artifacts/runs
    try:
        run_dir = sac_dir.parent  # .../<run_stamp>/rl_sac -> .../<run_stamp>
        print(f"[B2] Removing original SAC run dir: {run_dir}")
        shutil.rmtree(run_dir)
    except Exception as e:
        print(f"[B2] WARN: failed to remove original SAC run dir: {e}")


def _run_train_sac(
    cfg: B2Config,
    seed: int,
    config_path: Optional[Path],
    total_steps_override: Optional[int] = None,
) -> None:
    """
    Launch the existing training CLI:

        python -m src.cli train-sac ...

    We let it write under artifacts/runs/* TEMPORARILY, then
    _mirror_sac_run_into_b2() pulls everything into artifacts/ablations/b2.
    """
    target_model_id = cfg.target.get("model_id")
    if not target_model_id:
        raise ValueError("[B2] target.model_id missing in config.")

    sac_cfg = cfg.sac or {}
    total_env_steps_yaml = int(sac_cfg.get("total_env_steps", 0))
    total_env_steps = int(total_steps_override or total_env_steps_yaml)
    if total_env_steps <= 0:
        raise ValueError(
            "[B2] total_env_steps must be > 0 "
            "(either in YAML or via --total-steps)."
        )

    train_dataset_name = cfg.data.get("train_dataset", "")
    train_tag = _dataset_tag_from_name(train_dataset_name)

    print(f"[B2][TRAIN] model_id        : {target_model_id}")
    print(f"[B2][TRAIN] seed            : {seed}")
    print(
        f"[B2][TRAIN] total_env_steps : {total_env_steps} "
        f"(yaml={total_env_steps_yaml})"
    )
    print(
        f"[B2][TRAIN] train_dataset   : {train_dataset_name} "
        f"(tag={train_tag})"
    )

    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "train-sac",
        "--config-path",
        "config/stack.yaml",
        "--runs-model",
        target_model_id,
        "--target-model",
        target_model_id,
        "--total-steps",
        str(total_env_steps),
        "--seed",
        str(seed),
        "--dataset-tag",
        train_tag,
    ]

    print(f"[B2][TRAIN] Launching:\n       {' '.join(cmd)}")
    env = os.environ.copy()
    env["B2_EXPERIMENT_NAME"] = cfg.name
    env["B2_REWARD_MODE"] = cfg.reward_mode
    if config_path is not None:
        env["B2_REWARD_CONFIG"] = str(config_path.resolve())

    subprocess.run(cmd, check=True, env=env)
    print("[B2][TRAIN] train-sac finished.")


def _run_eval_sac_if_possible(
    cfg: B2Config,
    max_episodes_override: Optional[int] = None,
) -> None:
    """
    Run eval-sac on the latest checkpoint for this model.

    The eval artifacts are produced under artifacts/runs/* and later
    mirrored into the B2 folder.
    """
    target_model_id = cfg.target.get("model_id")
    if not target_model_id:
        print("[B2][EVAL] No target.model_id; skipping eval-sac.")
        return

    sac_dir = _find_latest_sac_run_dir(target_model_id)
    if sac_dir is None:
        print("[B2][EVAL] Could not find SAC run dir; skipping eval-sac.")
        return

    ckpt_dir = sac_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"[B2][EVAL] No checkpoints dir at {ckpt_dir}; skipping eval-sac.")
        return

    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        print(f"[B2][EVAL] No step_*.pt checkpoints in {ckpt_dir}; skipping eval-sac.")
        return

    last_ckpt = ckpts[-1]

    eval_cfg = cfg.eval or {}
    episodes_yaml = int(eval_cfg.get("max_episodes", 256))
    episodes = int(max_episodes_override or episodes_yaml)

    eval_dataset_name = eval_cfg.get("dataset") or cfg.data.get("eval_dataset", "")
    eval_tag = _dataset_tag_from_name(eval_dataset_name)

    print(f"[B2][EVAL] Using checkpoint  : {last_ckpt}")
    print(f"[B2][EVAL] eval_dataset      : {eval_dataset_name} (tag={eval_tag})")
    print(f"[B2][EVAL] episodes          : {episodes} (yaml={episodes_yaml})")

    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "eval-sac",
        "--config-path",
        "config/stack.yaml",
        "--runs-model",
        target_model_id,
        "--target-model",
        target_model_id,
        "--dataset-tag",
        eval_tag,
        "--episodes",
        str(episodes),
        "--checkpoint",
        str(last_ckpt),
        "--progress-every",
        "20",
    ]

    print(f"[B2][EVAL] Launching:\n       {' '.join(cmd)}")
    env = os.environ.copy()
    env["B2_EXPERIMENT_NAME"] = cfg.name
    env["B2_REWARD_MODE"] = cfg.reward_mode

    try:
        subprocess.run(cmd, check=True, env=env)
        print("[B2][EVAL] eval-sac finished.")
    except subprocess.CalledProcessError as e:
        print(
            f"[B2][EVAL] WARNING: eval-sac failed with return code {e.returncode} "
            f"(continuing; metrics copy will use whatever exists)."
        )


def run_train_and_eval(
    cfg: B2Config,
    seed: int,
    run_root: Path,
    config_path: Optional[Path] = None,
    total_steps_override: Optional[int] = None,
    max_episodes_override: Optional[int] = None,
) -> None:
    """
    High-level entrypoint used by b2_runner.py:

      1) Run train-sac for the given seed (using main stack).
      2) Run eval-sac on the latest checkpoint (if possible).
      3) MIRROR the resulting rl_sac run tree into:

           artifacts/ablations/b2/<variant>/<target_short>/seed_<seed>/

         and delete the original run dir under artifacts/runs.
    """
    # 1) Train
    _run_train_sac(cfg, seed, config_path, total_steps_override)

    # 2) Eval (best-effort; safe to fail)
    _run_eval_sac_if_possible(cfg, max_episodes_override)

    # 3) Mirror SAC run into structured B2 folder and clean up main runs dir
    target_model_id = cfg.target.get("model_id", "")
    sac_dir = _find_latest_sac_run_dir(target_model_id)
    if sac_dir is None:
        print("[B2] ERROR: Could not locate SAC run dir after training.")
        print("      No artifacts will be mirrored for this seed.")
        return

    _mirror_sac_run_into_b2(sac_dir, run_root)

    print(f"[B2] Finished train+eval for seed={seed}. Results in: {run_root}")
