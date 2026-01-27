# src/ablations/baseline/b2/reward.py
from __future__ import annotations

from typing import Any, Mapping


Number = float | int


def _get_section(cfg: Any, key: str, default: Any = None) -> Any:
    """
    Safely get a top-level section from a config object (OmegaConf or dict).
    """
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_progress(
    env_steps: int | None,
    cfg: Any,
    default: float = 0.0,
) -> float:
    """
    Compute training progress in [0,1] for curriculum.

    If env_steps is None or total steps are unknown, returns `default`.
    """
    if env_steps is None:
        return default

    sac_cfg = _get_section(cfg, "sac", {})
    total_env_steps = sac_cfg.get("total_env_steps", None)
    if not total_env_steps or total_env_steps <= 0:
        return default

    return max(0.0, min(1.0, env_steps / float(total_env_steps)))


def compute_reward_from_judge(
    judge_scores: Mapping[str, Number],
    cfg: Any,
    env_steps: int | None = None,
) -> float:
    """
    Core B2 helper: map 5-D judge scores -> scalar RL reward.

    Behavior is controlled by YAML fields in `reward` + `curriculum`:

      reward.mode: "binary" | "scalar" | "full5d"
    """
    reward_cfg = _get_section(cfg, "reward", {}) or {}
    curriculum_cfg = _get_section(cfg, "curriculum", {}) or {}

    mode = reward_cfg.get("mode", "full5d").lower()

    # -------------------------
    # Extract core judge scores
    # -------------------------
    success_key = reward_cfg.get("success_key", "success")
    stealth_key = reward_cfg.get("stealth_key", "stealth")
    impact_key = reward_cfg.get("impact_key", "impact")
    novelty_key = reward_cfg.get("novelty_key", "novelty")
    efficiency_key = reward_cfg.get("efficiency_key", "efficiency")

    success = float(judge_scores.get(success_key, 0.0))
    stealth = float(judge_scores.get(stealth_key, 0.0))
    impact = float(judge_scores.get(impact_key, 0.0))
    novelty = float(judge_scores.get(novelty_key, 0.0))
    efficiency = float(judge_scores.get(efficiency_key, 0.0))

    # -------------------------
    # Mode: binary reward
    # -------------------------
    if mode == "binary":
        thr = float(reward_cfg.get("success_threshold", 0.5))
        r_success = float(reward_cfg.get("reward_success", 1.0))
        r_failure = float(reward_cfg.get("reward_failure", 0.0))
        r = r_success if success >= thr else r_failure

    # -------------------------
    # Mode: scalar (success + λ * impact)
    # -------------------------
    elif mode == "scalar":
        w_success = float(reward_cfg.get("w_success", 1.0))
        w_impact = float(reward_cfg.get("w_impact", 0.5))
        r = w_success * success + w_impact * impact

    # -------------------------
    # Mode: full 5D + curriculum
    # -------------------------
    elif mode == "full5d":
        base_weights = reward_cfg.get("base_weights", {}) or {}

        w_success = float(base_weights.get("success", 1.0))
        w_stealth = float(base_weights.get("stealth", 0.5))
        w_impact = float(base_weights.get("impact", 0.7))
        w_novelty = float(base_weights.get("novelty", 0.5))
        w_efficiency = float(base_weights.get("efficiency", 0.3))

        if curriculum_cfg.get("enabled", False):
            progress = _get_progress(env_steps, cfg, default=0.0)
            schedule = curriculum_cfg.get("schedule", []) or []

            for stage in schedule:
                until = float(stage.get("until_progress", 1.0))
                if progress <= until:
                    sw = stage.get("weights", {}) or {}
                    w_success = float(sw.get("success", w_success))
                    w_stealth = float(sw.get("stealth", w_stealth))
                    w_impact = float(sw.get("impact", w_impact))
                    w_novelty = float(sw.get("novelty", w_novelty))
                    w_efficiency = float(sw.get("efficiency", w_efficiency))
                    break

        r = (
            w_success * success
            + w_stealth * stealth
            + w_impact * impact
            + w_novelty * novelty
            + w_efficiency * efficiency
        )

    else:
        raise ValueError(f"[B2] Unknown reward.mode: {mode!r}")

    # -------------------------
    # Clamp to [clamp_min, clamp_max] if provided
    # -------------------------
    clamp_min = reward_cfg.get("clamp_min", None)
    clamp_max = reward_cfg.get("clamp_max", None)

    if clamp_min is not None:
        r = max(float(clamp_min), r)
    if clamp_max is not None:
        r = min(float(clamp_max), r)

    return float(r)
