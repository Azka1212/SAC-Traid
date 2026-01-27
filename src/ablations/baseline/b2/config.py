from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class B2Config:
    """
    Thin wrapper around the B2 YAML config (config/ablations/b2/*.yaml).

    We keep it simple:
    - Each top-level section is stored as a dict: experiment, env, sac, reward, etc.
    - Seeds are normalized to a plain list[int] from `seeds.values`.
    """

    raw: Dict[str, Any]

    experiment: Dict[str, Any]
    seeds: List[int]
    data: Dict[str, Any]
    target: Dict[str, Any]
    judge: Dict[str, Any]
    rewriter: Dict[str, Any]
    env: Dict[str, Any]
    sac: Dict[str, Any]
    reward: Dict[str, Any]
    curriculum: Dict[str, Any]
    train_logging: Dict[str, Any]
    eval: Dict[str, Any]
    paths: Dict[str, Any]

    # ----------------------------
    # Convenience properties
    # ----------------------------

    @property
    def name(self) -> str:
        return self.experiment.get("name", "")

    @property
    def group(self) -> str:
        return self.experiment.get("group", "")

    @property
    def variant(self) -> str:
        # 'binary', 'scalar', or 'full5d'
        return self.experiment.get("variant", "")

    @property
    def artifacts_root(self) -> Path:
        root = self.experiment.get("artifacts_root", "artifacts/ablations/b2")
        return Path(root)

    @property
    def seed_values(self) -> List[int]:
        return list(self.seeds)

    @property
    def reward_mode(self) -> str:
        return self.reward.get("mode", "")

    @property
    def target_short_name(self) -> str:
        return self.target.get("short_name", "target")

    @property
    def target_model_id(self) -> str:
        return self.target.get("model_id", "")

    @property
    def train_dataset_tag(self) -> str:
        # e.g. "sorry_bench" -> "SORRY-BENCH"
        ds = self.data.get("train_dataset", "sorry_bench")
        return ds.replace("_", "-").upper()

    def make_run_root(self, seed: int) -> Path:
        template = self.paths.get(
            "run_root",
            "{artifacts_root}/{variant}/{target_short}/seed_{seed}",
        )
        run_root_str = template.format(
            artifacts_root=str(self.artifacts_root),
            variant=self.variant,
            target_short=self.target_short_name,
            seed=seed,
        )
        return Path(run_root_str)

    def iter_seed_run_roots(self):
        for s in self.seed_values:
            yield s, self.make_run_root(s)

    # ----------------------------
    # Construction from YAML
    # ----------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "B2Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        experiment = raw.get("experiment", {})
        seeds_section = raw.get("seeds", {})
        data = raw.get("data", {})
        target = raw.get("target", {})
        judge = raw.get("judge", {})
        rewriter = raw.get("rewriter", {})
        env = raw.get("env", {})
        sac = raw.get("sac", {})
        reward = raw.get("reward", {})
        curriculum = raw.get("curriculum", {})
        train_logging = raw.get("train_logging", {})
        eval_section = raw.get("eval", {})
        paths = raw.get("paths", {})

        seeds_values = seeds_section.get("values", [])
        if isinstance(seeds_values, int):
            seeds_list = [seeds_values]
        else:
            seeds_list = list(seeds_values)

        return cls(
            raw=raw,
            experiment=experiment,
            seeds=seeds_list,
            data=data,
            target=target,
            judge=judge,
            rewriter=rewriter,
            env=env,
            sac=sac,
            reward=reward,
            curriculum=curriculum,
            train_logging=train_logging,
            eval=eval_section,
            paths=paths,
        )
