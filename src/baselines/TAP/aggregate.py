# src/baselines/tap/aggregate.py
"""
Aggregate TAP runs (Part A tables).

Folder layout assumed (canonical, mirroring GCG / AutoDAN):
  <artifacts_root>/models/<target_short>/<budget>/run_*/metrics.json

Outputs (default):
  <artifacts_root>/tables/tap/
    - table_A1_tap.csv
    - table_A2_tap.csv
    - manifest_metrics.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from .config import TapExperimentConfig


METHOD = "TAP"


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().lower()
        if s in ("inf", "infinity"):
            return float("inf")
        return float(s)
    except Exception:
        return default


def _finite(values: List[float]) -> List[float]:
    return [v for v in values if isinstance(v, float) and math.isfinite(v)]


def _mean_safe(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _std_safe(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Load + filter metrics
# ---------------------------------------------------------------------------

def _load_metrics_files(root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in root.rglob("metrics.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            if "target_short_name" not in m or "query_budget" not in m:
                continue
            m["_path"] = str(path)
            out.append(m)
        except Exception:
            continue
    return out


def _allowed_targets(cfg: TapExperimentConfig) -> List[str]:
    rm_targets = list(cfg.run_matrix.target_short_names or [])
    if rm_targets:
        return rm_targets
    return [t.short_name for t in cfg.targets if bool(t.enabled)]


def _allowed_budgets(cfg: TapExperimentConfig) -> List[int]:
    return [int(b) for b in (cfg.run_matrix.query_budgets or []) if int(b) > 0]


def _filter_metrics(
    metrics_list: List[Dict[str, Any]],
    *,
    allowed_targets: List[str],
    allowed_budgets: List[int],
    allow_extra_budgets: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in metrics_list:
        t = str(m.get("target_short_name", ""))
        b = int(_to_float(m.get("query_budget", -1), default=-1))

        if not t or b < 0:
            continue
        if t not in allowed_targets:
            continue
        if (not allow_extra_budgets) and allowed_budgets and (b not in allowed_budgets):
            continue

        out.append(m)
    return out


def _group_by_target_and_budget(
    metrics_list: List[Dict[str, Any]]
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for m in metrics_list:
        t = str(m.get("target_short_name", "unknown"))
        b = int(_to_float(m.get("query_budget", -1), default=-1))
        if b < 0:
            continue
        groups.setdefault((t, b), []).append(m)
    return groups


def _write_manifest(out_root: Path, metrics_list: List[Dict[str, Any]]) -> Path:
    manifest_path = out_root / "manifest_metrics.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for m in metrics_list:
            rec = {
                "path": m.get("_path"),
                "target_short_name": m.get("target_short_name"),
                "query_budget": m.get("query_budget"),
                "repeat_idx": m.get("repeat_idx"),
                "asr": m.get("asr"),
                "qps": m.get("qps"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return manifest_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate_tap_partA(
    config_path: str,
    out_dir: Optional[str] = None,
    *,
    allow_extra_budgets: bool = False,
) -> None:
    cfg = TapExperimentConfig.from_yaml(config_path)

    artifacts_root = Path(cfg.paths.artifacts_root).resolve()
    models_root = (artifacts_root / "models").resolve()
    if not models_root.exists():
        raise RuntimeError(
            f"models_root does not exist: {models_root}\n"
            f"Expected: {models_root}/<target>/<budget>/run_*/metrics.json\n"
            f"Run the TAP runner first."
        )

    out_root = Path(out_dir).resolve() if out_dir else (artifacts_root / "tables" / "tap").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_metrics = _load_metrics_files(models_root)
    if not all_metrics:
        raise RuntimeError(f"No metrics.json found under {models_root}.")

    allowed_targets = _allowed_targets(cfg)
    allowed_budgets = _allowed_budgets(cfg)

    metrics_list = _filter_metrics(
        all_metrics,
        allowed_targets=allowed_targets,
        allowed_budgets=allowed_budgets,
        allow_extra_budgets=allow_extra_budgets,
    )
    if not metrics_list:
        raise RuntimeError(
            "No metrics matched filters.\n"
            f"- allowed_targets={allowed_targets}\n"
            f"- allowed_budgets={allowed_budgets} (allow_extra_budgets={allow_extra_budgets})\n"
            f"- models_root={models_root}"
        )

    manifest_path = _write_manifest(out_root, metrics_list)

    grouped = _group_by_target_and_budget(metrics_list)
    budgets_present = sorted(set(b for (_, b) in grouped.keys()))
    full_budget = max(budgets_present)

    # -------------------- Table A2 --------------------
    tableA2_path = out_root / "table_A2_tap.csv"
    with tableA2_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method", "target_short", "query_budget", "num_runs",
                "mean_asr", "std_asr",
                "mean_qps_finite", "std_qps_finite", "num_qps_finite",
            ]
        )

        for (t, b) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
            runs = grouped[(t, b)]
            asrs = [_to_float(m.get("asr", 0.0)) for m in runs]
            qps_vals = [_to_float(m.get("qps", float("inf")), default=float("inf")) for m in runs]
            qps_fin = _finite(qps_vals)

            w.writerow(
                [
                    METHOD,
                    t,
                    b,
                    len(runs),
                    _mean_safe(asrs),
                    _std_safe(asrs),
                    _mean_safe(qps_fin),
                    _std_safe(qps_fin),
                    len(qps_fin),
                ]
            )

    # -------------------- Table A1 --------------------
    tableA1_path = out_root / "table_A1_tap.csv"
    with tableA1_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method", "target_short", "query_budget", "num_runs",
                "mean_asr", "std_asr",
                "mean_qps_finite", "std_qps_finite", "num_qps_finite",
                "mean_distinct_2",
                "mean_stealth", "mean_impact", "mean_novelty", "mean_efficiency",
            ]
        )

        targets = sorted(set(t for (t, _) in grouped.keys()))
        for t in targets:
            key = (t, full_budget)
            if key not in grouped:
                continue
            runs = grouped[key]

            asrs = [_to_float(m.get("asr", 0.0)) for m in runs]
            qps_vals = [_to_float(m.get("qps", float("inf")), default=float("inf")) for m in runs]
            qps_fin = _finite(qps_vals)

            distinct2 = [_to_float(m.get("distinct_2", 0.0)) for m in runs]
            stealth = [_to_float(m.get("avg_stealth", 0.0)) for m in runs]
            impact = [_to_float(m.get("avg_impact", 0.0)) for m in runs]
            novelty = [_to_float(m.get("avg_novelty", 0.0)) for m in runs]
            efficiency = [_to_float(m.get("avg_efficiency", 0.0)) for m in runs]

            w.writerow(
                [
                    METHOD,
                    t,
                    full_budget,
                    len(runs),
                    _mean_safe(asrs),
                    _std_safe(asrs),
                    _mean_safe(qps_fin),
                    _std_safe(qps_fin),
                    len(qps_fin),
                    _mean_safe(distinct2),
                    _mean_safe(stealth),
                    _mean_safe(impact),
                    _mean_safe(novelty),
                    _mean_safe(efficiency),
                ]
            )

    print(f"[aggregate/tap] models_root: {models_root}")
    print(f"[aggregate/tap] loaded metrics (all): {len(all_metrics)}")
    print(f"[aggregate/tap] included after filters: {len(metrics_list)}")
    print(f"[aggregate/tap] allowed_targets: {allowed_targets}")
    print(f"[aggregate/tap] allowed_budgets: {allowed_budgets} (allow_extra_budgets={allow_extra_budgets})")
    print(f"[aggregate/tap] full_budget (A1): {full_budget}")
    print(f"[aggregate/tap] wrote: {tableA2_path}")
    print(f"[aggregate/tap] wrote: {tableA1_path}")
    print(f"[aggregate/tap] wrote: {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate TAP Part A runs into tables.")
    p.add_argument("--config-path", type=str, default="config/baselines/tap.yaml")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--allow-extra-budgets",
        action="store_true",
        help="If set, aggregate budgets found on disk even if not listed in run_matrix.query_budgets.",
    )
    args = p.parse_args()

    aggregate_tap_partA(
        config_path=args.config_path,
        out_dir=args.out_dir,
        allow_extra_budgets=args.allow_extra_budgets,
    )


if __name__ == "__main__":
    main()
