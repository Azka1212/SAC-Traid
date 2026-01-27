# src/baselines/pair/config.py
from __future__ import annotations

"""
Config loader for PAIR baseline (Part A).

- Mirrors config/baselines/pair.yaml
- Provides typed dataclasses for runner / attack / aggregate.

PAIR here = iterative prompt refinement with an attacker LLM.
This module stays isolated to PAIR only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Base configs
# ---------------------------------------------------------------------------

@dataclass
class PathsConfig:
    data_root: Path
    artifacts_root: Path


@dataclass
class DatasetSpec:
    name: str
    seeds_path: Optional[Path] = None
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
class DecodingConfig:
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
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
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
# PAIR specific
# ---------------------------------------------------------------------------

@dataclass
class AttackerConfig:
    model_id: str = "ollama/yi:9b"
    backend: str = "router"
    use_chat_api: bool = True
    decoding: DecodingConfig = field(default_factory=lambda: DecodingConfig(temperature=0.7, top_p=0.9, max_tokens=256))


@dataclass
class RefinementConfig:
    include_target_response: bool = True
    include_judge_feedback: bool = True
    include_refusal_signal: bool = True
    keep_history: int = 2
    prompt_style: str = "concise"  # concise | verbose


@dataclass
class ObjectiveConfig:
    weight_success: float = 1.0
    weight_impact: float = 0.3
    weight_stealth: float = 0.2
    weight_novelty: float = 0.2
    weight_efficiency: float = 0.1
    weight_length_penalty: float = 0.0


@dataclass
class PAIRHyperparams:
    seed_source: str = "train"
    train_n_seeds: int = 128

    max_rounds: int = 25
    early_stop_on_success: bool = True

    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Run matrix + logging
# ---------------------------------------------------------------------------

@dataclass
class RunMatrixConfig:
    experiment_name: str = "partA_pair_jbb_ood"
    dataset: str = "test_jbb_ood"
    query_budgets: List[int] = field(default_factory=lambda: [25])
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
class PAIRExperimentConfig:
    paths: PathsConfig
    datasets: DatasetsConfig
    targets: List[TargetSpec]
    guards: List[GuardSpec]
    judge: JudgeConfig
    metrics: MetricsConfig
    pair: PAIRHyperparams
    run_matrix: RunMatrixConfig
    logging: LoggingConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PAIRExperimentConfig":
        return load_pair_config(config_path)


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


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_pair_config(config_path: str | Path) -> PAIRExperimentConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    base_dir = _infer_repo_root(config_path)

    # paths
    paths_raw = raw.get("paths", {}) or {}
    data_root = _resolve_path(base_dir, _get(paths_raw, "data_root", "./data"))
    artifacts_root = _resolve_path(base_dir, _get(paths_raw, "artifacts_root", "./artifacts/baselines/pair"))
    paths = PathsConfig(
        data_root=_require_path(data_root, "paths.data_root"),
        artifacts_root=_require_path(artifacts_root, "paths.artifacts_root"),
    )

    # datasets
    ds_raw = raw.get("datasets", {}) or {}
    train_raw = ds_raw.get("train", {}) or {}
    test_raw = ds_raw.get("test_jbb_ood", {}) or {}
    if not test_raw:
        raise ValueError("datasets.test_jbb_ood is required in pair.yaml.")

    train_ds = DatasetSpec(
        name=_get(train_raw, "name", "train"),
        seeds_path=_resolve_path(base_dir, train_raw.get("seeds_path")),
        max_pairs=train_raw.get("max_pairs"),
    )
    test_ds = DatasetSpec(
        name=_get(test_raw, "name", "test_jbb_ood"),
        seeds_path=_resolve_path(base_dir, test_raw.get("seeds_path")),
        max_pairs=test_raw.get("max_pairs"),
    )
    datasets = DatasetsConfig(train=train_ds, test_jbb_ood=test_ds)

    # targets & guards
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

    # judge
    judge_raw = raw.get("judge", {}) or {}
    jdec = judge_raw.get("decoding", {}) or {}
    jsc = judge_raw.get("scoring", {}) or {}
    judge = JudgeConfig(
        model_id=str(_get(judge_raw, "model_id", "ollama/llama3:instruct")),
        backend=str(_get(judge_raw, "backend", "router")),
        use_chat_api=bool(_get(judge_raw, "use_chat_api", True)),
        decoding=DecodingConfig(
            temperature=float(_get(jdec, "temperature", 0.0)),
            top_p=float(_get(jdec, "top_p", 1.0)),
            max_tokens=int(_get(jdec, "max_tokens", 256)),
        ),
        scoring=JudgeScoringConfig(
            aspects=list(_get(jsc, "aspects", ["Success", "Stealth", "Novelty", "Efficiency", "Impact"])),
            success_threshold=float(_get(jsc, "success_threshold", 0.65)),
            confidence_weighting=bool(_get(jsc, "confidence_weighting", True)),
        ),
    )

    # metrics
    metrics_raw = raw.get("metrics", {}) or {}
    metrics = MetricsConfig(
        compute_5d=bool(_get(metrics_raw, "compute_5d", True)),
        compute_asr=bool(_get(metrics_raw, "compute_asr", True)),
        compute_qps=bool(_get(metrics_raw, "compute_qps", True)),
        compute_distinct_n=list(_get(metrics_raw, "compute_distinct_n", [])),
        compute_runtime=bool(_get(metrics_raw, "compute_runtime", True)),
        compute_query_efficiency=bool(_get(metrics_raw, "compute_query_efficiency", True)),
    )

    # pair
    pair_raw = raw.get("pair", {}) or {}

    atk_raw = pair_raw.get("attacker", {}) or {}
    atk_dec_raw = atk_raw.get("decoding", {}) or {}
    attacker = AttackerConfig(
        model_id=str(_get(atk_raw, "model_id", "ollama/yi:9b")),
        backend=str(_get(atk_raw, "backend", "router")),
        use_chat_api=bool(_get(atk_raw, "use_chat_api", True)),
        decoding=DecodingConfig(
            temperature=float(_get(atk_dec_raw, "temperature", 0.7)),
            top_p=float(_get(atk_dec_raw, "top_p", 0.9)),
            max_tokens=int(_get(atk_dec_raw, "max_tokens", 256)),
        ),
    )

    ref_raw = pair_raw.get("refinement", {}) or {}
    refinement = RefinementConfig(
        include_target_response=bool(_get(ref_raw, "include_target_response", True)),
        include_judge_feedback=bool(_get(ref_raw, "include_judge_feedback", True)),
        include_refusal_signal=bool(_get(ref_raw, "include_refusal_signal", True)),
        keep_history=int(_get(ref_raw, "keep_history", 2)),
        prompt_style=str(_get(ref_raw, "prompt_style", "concise")),
    )

    obj_raw = pair_raw.get("objective", {}) or {}
    objective = ObjectiveConfig(
        weight_success=float(_get(obj_raw, "weight_success", 1.0)),
        weight_impact=float(_get(obj_raw, "weight_impact", 0.3)),
        weight_stealth=float(_get(obj_raw, "weight_stealth", 0.2)),
        weight_novelty=float(_get(obj_raw, "weight_novelty", 0.2)),
        weight_efficiency=float(_get(obj_raw, "weight_efficiency", 0.1)),
        weight_length_penalty=float(_get(obj_raw, "weight_length_penalty", 0.0)),
    )

    pdec_raw = pair_raw.get("decoding", {}) or {}
    decoding = DecodingConfig(
        temperature=float(_get(pdec_raw, "temperature", 0.0)),
        top_p=float(_get(pdec_raw, "top_p", 1.0)),
        max_tokens=int(_get(pdec_raw, "max_tokens", 256)),
    )

    pair = PAIRHyperparams(
        seed_source=str(_get(pair_raw, "seed_source", "train")),
        train_n_seeds=int(_get(pair_raw, "train_n_seeds", 128)),
        max_rounds=int(_get(pair_raw, "max_rounds", 25)),
        early_stop_on_success=bool(_get(pair_raw, "early_stop_on_success", True)),
        attacker=attacker,
        refinement=refinement,
        objective=objective,
        decoding=decoding,
        random_seed=int(_get(pair_raw, "random_seed", 42)),
    )

    # run matrix
    rm_raw = raw.get("run_matrix", {}) or {}
    run_matrix = RunMatrixConfig(
        experiment_name=str(_get(rm_raw, "experiment_name", "partA_pair_jbb_ood")),
        dataset=str(_get(rm_raw, "dataset", "test_jbb_ood")),
        query_budgets=list(_get(rm_raw, "query_budgets", [25])),
        repeats=int(_get(rm_raw, "repeats", 1)),
        target_short_names=list(_get(rm_raw, "target_short_names", [])),
    )

    # logging
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

    return PAIRExperimentConfig(
        paths=paths,
        datasets=datasets,
        targets=targets,
        guards=guards,
        judge=judge,
        metrics=metrics,
        pair=pair,
        run_matrix=run_matrix,
        logging=logging_cfg,
        raw=raw,
    )
