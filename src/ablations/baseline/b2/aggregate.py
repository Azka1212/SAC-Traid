# src/ablations/baseline/b2/aggregate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

# Table B2 columns we care about
# Make sure eval_metrics.json uses these exact keys, or adjust here.
METRIC_KEYS = [
    "asr",
    "q_per_success",
    "stealth",
    "impact",
    "distinct_1",
    "distinct_2",
    "distinct_3",
]

def _collect_eval_metrics(artifacts_root: Path) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Scan artifacts_root for eval_metrics.json under:

        artifacts_root/<variant>/<target_short>/seed_*/eval_metrics.json

    Returns:
        {
          (variant, target_short): [metrics_dict_per_seed, ...],
          ...
        }
    """
    results: Dict[Tuple[str, str], List[Dict]] = {}

    if not artifacts_root.exists():
        print(f"[B2][WARN] artifacts_root does not exist: {artifacts_root}")
        return results

    # variants: binary / scalar / full5d / etc.
    for variant_dir in artifacts_root.iterdir():
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name

        # target_short: e.g. deepseek_r1_7b
        for target_dir in variant_dir.iterdir():
            if not target_dir.is_dir():
                continue
            target_short = target_dir.name
            key = (variant, target_short)

            # seeds: seed_1, seed_2, ...
            for seed_dir in target_dir.glob("seed_*"):
                if not seed_dir.is_dir():
                    continue

                # By design, B2 eval should write eval_metrics.json at run_root
                metrics_path = seed_dir / "eval_metrics.json"
                if not metrics_path.is_file():
                    # fallback: search recursively in case you nest it later
                    candidates = list(seed_dir.rglob("eval_metrics.json"))
                    if candidates:
                        metrics_path = candidates[0]
                    else:
                        print(f"[B2][WARN] No eval_metrics.json found under {seed_dir}")
                        continue

                try:
                    with metrics_path.open("r", encoding="utf-8") as f:
                        metrics = json.load(f)
                except Exception as e:
                    print(f"[B2][WARN] Failed to read {metrics_path}: {e}")
                    continue

                results.setdefault(key, []).append(metrics)

    return results


def _aggregate_metrics(per_seed: List[Dict]) -> Dict[str, float]:
    """
    Average metrics across seeds for a (variant, target_short) pair.
    """
    agg: Dict[str, float] = {}
    for k in METRIC_KEYS:
        vals = []
        for m in per_seed:
            if k in m:
                try:
                    vals.append(float(m[k]))
                except (TypeError, ValueError):
                    print(f"[B2][WARN] Non-numeric value for key {k}: {m[k]!r}")
        if not vals:
            continue
        agg[k] = mean(vals)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate B2 reward ablation metrics into a CSV table.\n\n"
            "Expected layout:\n"
            "  artifacts/ablations/b2/<variant>/<target_short>/seed_*/eval_metrics.json\n\n"
            "Outputs a CSV (default: artifacts/ablations/b2/table_B2_reward.csv) with:\n"
            "  variant, target, num_seeds, asr, q_per_success, stealth, distinct_2, impact\n"
        )
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="artifacts/ablations/b2",
        help="Root folder for B2 artifacts (default: artifacts/ablations/b2).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/ablations/b2/table_B2_reward.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    results = _collect_eval_metrics(artifacts_root)

    if not results:
        print(f"[B2] No eval_metrics.json found under {artifacts_root}")
        return

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["variant", "target", "num_seeds"] + METRIC_KEYS
    lines: List[str] = []
    lines.append(",".join(header))

    # Sort so you get a stable order in the CSV: binary, full5d, scalar, etc.
    for (variant, target_short), per_seed in sorted(results.items()):
        agg = _aggregate_metrics(per_seed)
        row: List[str] = [
            variant,
            target_short,
            str(len(per_seed)),
        ]
        for k in METRIC_KEYS:
            if k in agg:
                row.append(f"{agg[k]:.4f}")
            else:
                row.append("")
        lines.append(",".join(row))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[B2] Wrote table: {out_path}")


if __name__ == "__main__":
    main()
