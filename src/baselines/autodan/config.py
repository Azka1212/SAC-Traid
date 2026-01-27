from __future__ import annotations

"""
Config loader for the AutoDAN baseline (Part A).

- Mirrors config/baselines/autodan.yaml
- Provides typed dataclasses so runner / attack / aggregate don't touch raw dicts.

IMPORTANT:
This file must NOT contain TAP config classes. Keep AutoDAN isolated.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Base configs (paths, datasets, targets, judge, metrics)
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    data_root: Path
    artifacts_root: Path


@dataclass
class DatasetSpec:
    name: str
    seeds_path: Optional[Path] = None
    operators_path: Optional[Path] = None
    pacing_path: Optional[Path] = None
    max_pairs: Optional[int] = None


@dataclass
class DatasetsConfig:
    train: DatasetSpec
    test_jbb_ood: DatasetSpec


@dataclass
class TargetSpec:
    id: str
    short_name: str
    family: str
    enabled: bool = True


@dataclass
class GuardSpec:
    id: str
    short_name: str
    enabled: bool = False


@dataclass
class JudgeDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class JudgeScoringConfig:
    aspects: List[str] = field(default_factory=lambda: ["Success", "Stealth", "Novelty", "Efficiency", "Impact"])
    success_threshold: float = 0.65
    confidence_weighting: bool = True


@dataclass
class JudgeConfig:
    model_id: str = "ollama/llama3:instruct"
    backend: str = "router"
    use_chat_api: bool = True
    decoding: JudgeDecodingConfig = field(default_factory=JudgeDecodingConfig)
    scoring: JudgeScoringConfig = field(default_factory=JudgeScoringConfig)


@dataclass
class MetricsConfig:
    compute_5d: bool = True
    compute_asr: bool = True
    compute_qps: bool = True
    compute_distinct_n: List[int] = field(default_factory=list)
    compute_runtime: bool = True
    compute_query_efficiency: bool = True


# ---------------------------------------------------------------------------
# AutoDAN-specific configs
# ---------------------------------------------------------------------------

@dataclass
class AutoDANDecodingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256


@dataclass
class AutoDANObjectiveConfig:
    weight_success: float = 1.0
    weight_impact: float = 0.3
    weight_stealth: float = 0.2
    weight_length_penalty: float = 0.0


@dataclass
class AutoDANHyperparams:
    random_seed: int = 42
    num_generations: int = 10
    population_size: int = 10
    elite_fraction: float = 0.2
    crossover_rate: float = 0.5
    max_suffix_tokens: int = 64
    decoding: AutoDANDecodingConfig = field(default_factory=AutoDANDecodingConfig)
    objective: AutoDANObjectiveConfig = field(default_factory=AutoDANObjectiveConfig)


# ---------------------------------------------------------------------------
# Run matrix + top-level logging
# ---------------------------------------------------------------------------

@dataclass
class RunMatrixConfig:
    experiment_name: str = "partA_autodan_jbb_ood"
    dataset: str = "test_jbb_ood"
    query_budgets: List[int] = field(default_factory=lambda: [5, 10, 20])
    repeats: int = 1
    target_short_names: List[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    save_events_jsonl: bool = True
    save_events_csv: bool = True
    save_metrics_json: bool = True
    save_config_snapshot: bool = True
    save_git_commit: bool = True
    max_prompt_chars: int = 8000
    redact_prompts_in_csv: bool = False


@dataclass
class AutoDANExperimentConfig:
    paths: PathsConfig
    datasets: DatasetsConfig
    targets: List[TargetSpec]
    guards: List[GuardSpec]
    judge: JudgeConfig
    metrics: MetricsConfig
    autodan: AutoDANHyperparams
    run_matrix: RunMatrixConfig
    logging: LoggingConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AutoDANExperimentConfig":
        return load_autodan_config(config_path)


# ---------------------------------------------------------------------------
# Loader utils
# ---------------------------------------------------------------------------

def _resolve_path(base_dir: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _infer_repo_root(config_path: Path) -> Path:
    cfg = config_path.resolve()
    for parent in [cfg.parent, *cfg.parents]:
        try:
            if (parent / "config").exists() and (parent / "src").exists():
                return parent
        except OSError:
            continue
    return config_path.parent.parent.parent.resolve()


def _require_path(p: Optional[Path], field_name: str) -> Path:
    if p is None:
        raise ValueError(f"Missing required path for {field_name}. Check your YAML.")
    return p


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def load_autodan_config(config_path: str | Path) -> AutoDANExperimentConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    base_dir = _infer_repo_root(config_path)

    # ----- paths -----
    paths_raw = raw.get("paths", {}) or {}
    data_root = _resolve_path(base_dir, _get(paths_raw, "data_root", "./data"))
    artifacts_root = _resolve_path(base_dir, _get(paths_raw, "artifacts_root", "./artifacts/baselines/autodan"))
    paths = PathsConfig(
        data_root=_require_path(data_root, "paths.data_root"),
        artifacts_root=_require_path(artifacts_root, "paths.artifacts_root"),
    )

    # ----- datasets -----
    ds_raw = raw.get("datasets", {}) or {}
    train_raw = ds_raw.get("train", {}) or {}
    jbb_raw = ds_raw.get("test_jbb_ood", {}) or {}
    if not jbb_raw:
        raise ValueError("datasets.test_jbb_ood is required. Define it in config/baselines/autodan.yaml.")

    train_ds = DatasetSpec(
        name=_get(train_raw, "name", "train"),
        seeds_path=_resolve_path(base_dir, train_raw.get("seeds_path")),
        operators_path=_resolve_path(base_dir, train_raw.get("operators_path")),
        pacing_path=_resolve_path(base_dir, train_raw.get("pacing_path")),
        max_pairs=train_raw.get("max_pairs"),
    )
    jbb_ds = DatasetSpec(
        name=_get(jbb_raw, "name", "test_jbb_ood"),
        seeds_path=_resolve_path(base_dir, jbb_raw.get("seeds_path")),
        operators_path=_resolve_path(base_dir, jbb_raw.get("operators_path")),
        pacing_path=_resolve_path(base_dir, jbb_raw.get("pacing_path")),
        max_pairs=jbb_raw.get("max_pairs"),
    )
    datasets = DatasetsConfig(train=train_ds, test_jbb_ood=jbb_ds)

    # ----- targets & guards -----
    targets = [
        TargetSpec(
            id=t["id"],
            short_name=t["short_name"],
            family=t.get("family", t["short_name"]),
            enabled=bool(t.get("enabled", True)),
        )
        for t in (raw.get("targets", []) or [])
    ]
    guards = [
        GuardSpec(
            id=g["id"],
            short_name=g["short_name"],
            enabled=bool(g.get("enabled", False)),
        )
        for g in (raw.get("guards", []) or [])
    ]

    # ----- judge -----
    judge_raw = raw.get("judge", {}) or {}
    judge_dec_raw = judge_raw.get("decoding", {}) or {}
    judge_scoring_raw = judge_raw.get("scoring", {}) or {}
    judge = JudgeConfig(
        model_id=str(_get(judge_raw, "model_id", "ollama/llama3:instruct")),
        backend=str(_get(judge_raw, "backend", "router")),
        use_chat_api=bool(_get(judge_raw, "use_chat_api", True)),
        decoding=JudgeDecodingConfig(
            temperature=float(_get(judge_dec_raw, "temperature", 0.0)),
            top_p=float(_get(judge_dec_raw, "top_p", 1.0)),
            max_tokens=int(_get(judge_dec_raw, "max_tokens", 256)),
        ),
        scoring=JudgeScoringConfig(
            aspects=list(_get(judge_scoring_raw, "aspects", ["Success", "Stealth", "Novelty", "Efficiency", "Impact"])),
            success_threshold=float(_get(judge_scoring_raw, "success_threshold", 0.65)),
            confidence_weighting=bool(_get(judge_scoring_raw, "confidence_weighting", True)),
        ),
    )

    # ----- metrics -----
    metrics_raw = raw.get("metrics", {}) or {}
    metrics = MetricsConfig(
        compute_5d=bool(_get(metrics_raw, "compute_5d", True)),
        compute_asr=bool(_get(metrics_raw, "compute_asr", True)),
        compute_qps=bool(_get(metrics_raw, "compute_qps", True)),
        compute_distinct_n=list(_get(metrics_raw, "compute_distinct_n", [])),
        compute_runtime=bool(_get(metrics_raw, "compute_runtime", True)),
        compute_query_efficiency=bool(_get(metrics_raw, "compute_query_efficiency", True)),
    )

    # ----- autodan hyperparams -----
    ad_raw = raw.get("autodan", {}) or {}
    dec_raw = ad_raw.get("decoding", {}) or {}
    obj_raw = ad_raw.get("objective", {}) or {}

    autodan = AutoDANHyperparams(
        random_seed=int(_get(ad_raw, "random_seed", 42)),
        num_generations=int(_get(ad_raw, "num_generations", 10)),
        population_size=int(_get(ad_raw, "population_size", 10)),
        elite_fraction=float(_get(ad_raw, "elite_fraction", 0.2)),
        crossover_rate=float(_get(ad_raw, "crossover_rate", 0.5)),
        max_suffix_tokens=int(_get(ad_raw, "max_suffix_tokens", 64)),
        decoding=AutoDANDecodingConfig(
            temperature=float(_get(dec_raw, "temperature", 0.7)),
            top_p=float(_get(dec_raw, "top_p", 0.95)),
            max_tokens=int(_get(dec_raw, "max_tokens", 256)),
        ),
        objective=AutoDANObjectiveConfig(
            weight_success=float(_get(obj_raw, "weight_success", 1.0)),
            weight_impact=float(_get(obj_raw, "weight_impact", 0.3)),
            weight_stealth=float(_get(obj_raw, "weight_stealth", 0.2)),
            weight_length_penalty=float(_get(obj_raw, "weight_length_penalty", 0.0)),
        ),
    )

    # ----- run matrix -----
    rm_raw = raw.get("run_matrix", {}) or {}
    run_matrix = RunMatrixConfig(
        experiment_name=str(_get(rm_raw, "experiment_name", "partA_autodan_jbb_ood")),
        dataset=str(_get(rm_raw, "dataset", "test_jbb_ood")),
        query_budgets=list(_get(rm_raw, "query_budgets", [5, 10, 20])),
        repeats=int(_get(rm_raw, "repeats", 1)),
        target_short_names=list(_get(rm_raw, "target_short_names", [])),
    )

    # ----- top-level logging -----
    log_raw = raw.get("logging", {}) or {}
    logging_cfg = LoggingConfig(
        save_events_jsonl=bool(_get(log_raw, "save_events_jsonl", True)),
        save_events_csv=bool(_get(log_raw, "save_events_csv", True)),
        save_metrics_json=bool(_get(log_raw, "save_metrics_json", True)),
        save_config_snapshot=bool(_get(log_raw, "save_config_snapshot", True)),
        save_git_commit=bool(_get(log_raw, "save_git_commit", True)),
        max_prompt_chars=int(_get(log_raw, "max_prompt_chars", 8000)),
        redact_prompts_in_csv=bool(_get(log_raw, "redact_prompts_in_csv", False)),
    )

    return AutoDANExperimentConfig(
        paths=paths,
        datasets=datasets,
        targets=targets,
        guards=guards,
        judge=judge,
        metrics=metrics,
        autodan=autodan,
        run_matrix=run_matrix,
        logging=logging_cfg,
        raw=raw,
    )
