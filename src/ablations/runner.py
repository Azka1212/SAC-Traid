# src/ablations/runner.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_algorithm_ablation_config
from .algorithms import train_and_eval_algorithm_variant
from .metrics import collect_algorithm_results, aggregate_algorithm_results


def run_b1_algorithm_ablation(config_path: str) -> None:
    cfg = load_algorithm_ablation_config(config_path)
    cfg.artifacts_root.mkdir(parents=True, exist_ok=True)

    print(f"[B1] Running Algorithm Ablation: {cfg.ablation_name}")
    print(f"[B1] Artifacts root: {cfg.artifacts_root}")

    # 1. Run all variants × seeds 
    for variant in cfg.variants:
        for seed in cfg.seeds:
            print(f"[B1] Variant={variant.name} Seed={seed}")
            run_dir = train_and_eval_algorithm_variant(cfg, variant, seed)
            print(f"[B1]   -> run_dir={run_dir}")

    # 2. Collect metrics
    df = collect_algorithm_results(cfg)
    csv_all = cfg.artifacts_root / "b1_algorithm_raw.csv"
    df.to_csv(csv_all, index=False)
    print(f"[B1] Saved raw metrics to {csv_all}")

    # 3. Aggregate table for paper (mean/std over seeds)
    agg = aggregate_algorithm_results(df)
    csv_table = cfg.artifacts_root / "b1_algorithm_table.csv"
    agg.to_csv(csv_table, index=False)
    print(f"[B1] Saved table metrics to {csv_table}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to B1 algorithm ablation YAML",
    )
    args = parser.parse_args()
    run_b1_algorithm_ablation(args.config_path)


if __name__ == "__main__":
    main()
