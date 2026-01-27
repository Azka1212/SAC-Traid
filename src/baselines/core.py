from __future__ import annotations
"""
src/baselines/core.py

Core utilities shared by all baseline methods (GCG, AutoDAN, RLBreaker, ...).

Responsibilities:
- Represent a single dataset example (seed) and a single attack result.
- Load seeds from JSONL files.
- Compute run-level metrics (ASR, Q/S, Distinct-n, 5D averages, runtime).

NOTE:
- This module should NOT write artifacts to disk.
- Folder layout + logging are handled by each baseline runner/attack module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json

from .gcg.config import DatasetSpec, MetricsConfig


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AttackSample:
    """A single harmful seed to attack (e.g., one AdvBench/JBB item)."""
    sample_id: str
    dataset_name: str
    prompt_text: str
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    """
    Final result for one sample under a given baseline method, target model,
    and query budget.
    """
    sample_id: str
    dataset_name: str

    target_id: str
    target_short_name: str
    method: str

    query_budget: int
    queries_used: int

    seed_prompt: str
    adversarial_suffix: str
    final_prompt: str
    model_response: str

    scores: Dict[str, float]
    reward: Optional[float] = None

    success: bool = False
    runtime_sec: float = 0.0

    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _extract_sample_id(obj: Dict[str, Any], idx: int) -> str:
    for key in ("id", "sample_id", "pair_id", "qid", "index"):
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return str(idx)


def _extract_prompt_text(obj: Dict[str, Any]) -> str:
    for key in ("prompt", "question", "instruction", "text", "input"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return json.dumps(obj, ensure_ascii=False)


def load_seeds_from_jsonl(spec: DatasetSpec) -> List[AttackSample]:
    if spec.seeds_path is None:
        raise ValueError(f"No seeds_path defined for dataset '{spec.name}'")

    path = Path(spec.seeds_path)
    if not path.exists():
        raise FileNotFoundError(f"Seeds file not found: {path}")

    samples: List[AttackSample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if spec.max_pairs is not None and idx >= spec.max_pairs:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            sample_id = _extract_sample_id(obj, idx)
            prompt = _extract_prompt_text(obj)
            category = obj.get("category") or obj.get("topic")

            samples.append(
                AttackSample(
                    sample_id=sample_id,
                    dataset_name=spec.name,
                    prompt_text=prompt,
                    category=category,
                    metadata=obj,
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_distinct_n(texts: Iterable[str], n: int) -> float:
    all_ngrams = []
    for t in texts:
        tokens = t.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i : i + n]))

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / float(len(all_ngrams))


def compute_run_metrics(
    results: List[AttackResult],
    metrics_cfg: MetricsConfig,
    success_threshold: float,
) -> Dict[str, Any]:
    if not results:
        return {
            "n_samples": 0,
            "asr": 0.0,
            "q_per_success": None,
            "avg_scores": {},
            "distinct": {},
            "runtime_total": 0.0,
            "runtime_avg": 0.0,
        }

    n = len(results)
    successes = [r for r in results if r.scores.get("Success", 0.0) >= success_threshold]
    num_success = len(successes)
    asr = num_success / float(n)

    total_queries = sum(max(0, r.queries_used) for r in results)
    q_per_success = (total_queries / float(num_success)) if num_success > 0 else None

    aspects = ["Success", "Stealth", "Novelty", "Efficiency", "Impact"]
    avg_scores: Dict[str, float] = {}
    if metrics_cfg.compute_5d:
        for a in aspects:
            vals = [r.scores.get(a) for r in results if r.scores.get(a) is not None]
            if vals:
                avg_scores[a] = sum(vals) / float(len(vals))

    distinct: Dict[str, float] = {}
    if metrics_cfg.compute_distinct_n:
        prompts = [r.final_prompt for r in results]
        for n_val in metrics_cfg.compute_distinct_n:
            distinct[f"distinct_{n_val}"] = _compute_distinct_n(prompts, n_val)

    runtime_total = sum(max(0.0, r.runtime_sec) for r in results)
    runtime_avg = runtime_total / float(n)

    out: Dict[str, Any] = {
        "n_samples": n,
        "asr": asr,
        "q_per_success": q_per_success,
        "avg_scores": avg_scores,
        "distinct": distinct,
        "runtime_total": runtime_total if metrics_cfg.compute_runtime else None,
        "runtime_avg": runtime_avg if metrics_cfg.compute_runtime else None,
        "total_queries": total_queries if metrics_cfg.compute_qps else None,
    }

    if metrics_cfg.compute_query_efficiency and q_per_success and q_per_success > 0:
        out["query_efficiency"] = asr / q_per_success
    else:
        out["query_efficiency"] = None

    return out
