# src/ablations/metrics.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd

from .config import AlgorithmAblationConfig, AlgorithmVariantConfig


METRIC_KEYS = [
    "asr",
    "q_per_success",
    "stealth",
    "impact",
    "distinct_1",
    "distinct_2",
    "distinct_3",
]


def collect_algorithm_results(
    cfg: AlgorithmAblationConfig,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for variant in cfg.variants:
        for seed in cfg.seeds:
            run_dir = (
                cfg.artifacts_root
                / variant.name
                / f"seed_{seed:03d}"
            )
            metrics_path = run_dir / cfg.metrics_filename
            if not metrics_path.exists():
                print(f"[WARN] metrics file missing: {metrics_path}")
                continue

            with metrics_path.open("r") as f:
                m = json.load(f)

            row = {
                "variant": variant.name,
                "label": variant.label,
                "seed": seed,
            }
            for k in METRIC_KEYS:
                row[k] = m.get(k, None)

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def aggregate_algorithm_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over seeds: mean and std for each metric.
    """
    metric_cols = METRIC_KEYS

    agg_funcs = {
        col: ["mean", "std"]
        for col in metric_cols
    }

    grouped = df.groupby(["variant", "label"]).agg(agg_funcs)

    # flatten MultiIndex columns: (asr, mean) -> asr_mean
    grouped.columns = [
        f"{m}_{stat}" for (m, stat) in grouped.columns
    ]
    grouped = grouped.reset_index()
    return grouped
