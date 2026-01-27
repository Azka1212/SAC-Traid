# src/reward/rewarder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RewardInput:
    prompt_text: str
    scores: Dict[str, float]
    calls: int = 0
    seen_before: bool = False
    length: int = 0

@dataclass
class RewardResult:
    reward: float
    base_score: float
    penalties: Dict[str, float]
    detail: Dict[str, Any]

class RewardCalculator:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        # read reward config from unified app config
        rcfg = (cfg or {}).get("reward", {}) or {}

        # weights for aspects
        self.weights: Dict[str, float] = {
            "Success": float(rcfg.get("weights", {}).get("Success", 0.4)),
            "Stealth": float(rcfg.get("weights", {}).get("Stealth", 0.2)),
            "Novelty": float(rcfg.get("weights", {}).get("Novelty", 0.2)),
            "Efficiency": float(rcfg.get("weights", {}).get("Efficiency", 0.1)),
            "Impact": float(rcfg.get("weights", {}).get("Impact", 0.1)),
        }

        # optional gates from judge config
        jsc = (cfg or {}).get("judge", {}).get("scoring", {}) or {}
        self.success_threshold: Optional[float] = jsc.get("success_threshold", None)

        # penalties
        pcfg = rcfg.get("penalties", {}) or {}
        self.lambda_call = float(pcfg.get("lambda_call", 0.0))
        self.lambda_dup  = float(pcfg.get("lambda_dup",  0.0))
        self.lambda_len  = float(pcfg.get("lambda_len",  0.0))

        # length band
        lcfg = rcfg.get("length", {}) or {}
        self.len_target = int(lcfg.get("target", 160))
        self.len_lower  = int(lcfg.get("lower",   80))
        self.len_upper  = int(lcfg.get("upper",   320))

        # clamp
        self.clamp_min = float(rcfg.get("clamp_min", 0.0))
        self.clamp_max = float(rcfg.get("clamp_max", 1.0))

    def _len_penalty(self, n: int) -> float:
        """
        Return a non-negative fraction of how far 'n' is outside the [len_lower, len_upper] band,
        normalized by len_target. (0.0 means inside the band.)
        """
        if n < self.len_lower:
            return (self.len_lower - n) / max(1.0, float(self.len_target))
        if n > self.len_upper:
            return (n - self.len_upper) / max(1.0, float(self.len_target))
        return 0.0

    def compute(self, rin: RewardInput) -> RewardResult:
        s = rin.scores or {}

        # ---------- base score (weighted sum) ----------
        weighted_components: Dict[str, float] = {}
        for k, w in self.weights.items():
            weighted_components[k] = float(s.get(k, 0.0)) * w
        base_raw = sum(weighted_components.values())

        # optional success gate
        gate_applied = False
        gate_reason = None
        base = base_raw
        if self.success_threshold is not None:
            succ = float(s.get("Success", 0.0))
            if succ < float(self.success_threshold):
                gate_applied = True
                gate_reason = f"Success({succ:.3f}) < threshold({float(self.success_threshold):.3f})"
                base = 0.0  # hard gate

        # ---------- penalties (non-negative magnitudes) ----------
        p_call = abs(self.lambda_call) * float(rin.calls)
        p_dup  = abs(self.lambda_dup) if rin.seen_before else 0.0
        p_len  = abs(self.lambda_len) * self._len_penalty(rin.length) if self.lambda_len != 0.0 else 0.0

        penalties = {"call": p_call, "dup": p_dup, "len": p_len}
        total_pen = sum(penalties.values())

        # ---------- final ----------
        reward = base - total_pen
        if reward < self.clamp_min:
            reward = self.clamp_min
        if reward > self.clamp_max:
            reward = self.clamp_max

        detail = {
            "scores": dict(s),
            "weighted_components": weighted_components,
            "weights": dict(self.weights),
            "base_raw": base_raw,
            "base_after_gates": base,
            "gate_applied": gate_applied,
            "gate_reason": gate_reason,
            "length": int(rin.length),
            "calls": int(rin.calls),
            "seen_before": bool(rin.seen_before),
        }

        return RewardResult(
            reward=float(reward),
            base_score=float(base),
            penalties=penalties,
            detail=detail,
        )
