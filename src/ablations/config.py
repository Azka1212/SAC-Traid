# src/ablations/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class AlgorithmVariantConfig:
    name: str              # internal id, e.g. "sac_triad"
    label: str             # pretty label for tables, e.g. "SAC-Triad"
    algorithm: str         # "sac" | "ppo" | "bandit"
    config_overrides: Dict[str, Any]


@dataclass
class AlgorithmAblationConfig:
    ablation_name: str
    artifacts_root: Path
    stack_config_path: Path

    train_dataset_tag: str
    eval_dataset_tag: str

    target_model: str
    seeds: List[int]
    train_steps: int
    eval_episodes: int
    eval_budget: int

    metrics_filename: str
    variants: List[AlgorithmVariantConfig]


def load_algorithm_ablation_config(path: str | Path) -> AlgorithmAblationConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    variants = [
        AlgorithmVariantConfig(
            name=v["name"],
            label=v["label"],
            algorithm=v["algorithm"],
            config_overrides=v.get("config_overrides", {}),
        )
        for v in raw["variants"]
    ]

    return AlgorithmAblationConfig(
        ablation_name=raw["ablation_name"],
        artifacts_root=Path(raw["artifacts_root"]),
        stack_config_path=Path(raw["stack_config_path"]),
        train_dataset_tag=raw["train_dataset_tag"],
        eval_dataset_tag=raw["eval_dataset_tag"],
        target_model=raw["target_model"],
        seeds=list(raw["seeds"]),
        train_steps=int(raw["train_steps"]),
        eval_episodes=int(raw["eval_episodes"]),
        eval_budget=int(raw["eval_budget"]),
        metrics_filename=raw.get("metrics_filename", "metrics.json"),
        variants=variants,
    )
