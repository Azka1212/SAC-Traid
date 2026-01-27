# src/baselines/xjailbreak/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


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
    train: Optional[DatasetSpec] = None
    by_key: Dict[str, DatasetSpec] = field(default_factory=dict)


@dataclass
class TargetSpec:
    id: str
    short_name: str
    family: str
    enabled: bool = True


@dataclass
class JudgeDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class JudgeScoringConfig:
    aspects: List[str] = field(default_factory=lambda: ["Success", "Stealth", "Novelty", "Efficiency", "Impact"])
    success_threshold: float = 0.65
    confidence_weighting: bool = False  # YAML has this


@dataclass
class JudgeConfig:
    model_id: str = "ollama/llama3:instruct"
    backend: str = "router"
    use_chat_api: bool = True
    decoding: JudgeDecodingConfig = field(default_factory=JudgeDecodingConfig)
    scoring: JudgeScoringConfig = field(default_factory=JudgeScoringConfig)


@dataclass
class MetricsConfig:
    compute_3d: bool = False
    compute_5d: bool = True
    compute_asr: bool = True
    compute_qps: bool = True
    compute_distinct_n: List[int] = field(default_factory=list)
    compute_runtime: bool = True
    compute_query_efficiency: bool = True


@dataclass
class AttackerDecodingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


@dataclass
class AttackerConfig:
    model_id: str = "ollama/yi:9b"
    backend: str = "router"
    use_chat_api: bool = True
    decoding: AttackerDecodingConfig = field(default_factory=AttackerDecodingConfig)


@dataclass
class AnnealConfig:
    enabled: bool = False
    start: float = 0.50
    end: float = 0.20


@dataclass
class RepresentationConfig:
    dim: int = 512
    proximity_weight: float = 0.35
    use_token_hash_embedding: bool = True
    normalize: bool = True
    anneal: AnnealConfig = field(default_factory=AnnealConfig)


@dataclass
class RefinementConfig:
    keep_history: int = 2
    prompt_style: str = "concise"  # concise | verbose
    include_target_response: bool = True
    include_judge_feedback: bool = True
    include_refusal_signal: bool = False  # YAML has this


@dataclass
class ObjectiveConfig:
    weight_success: float = 1.0
    weight_impact: float = 0.3
    weight_stealth: float = 0.2
    weight_novelty: float = 0.2
    weight_efficiency: float = 0.1
    weight_length_penalty: float = 0.0


@dataclass
class TargetDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class XJailbreakHyperparams:
    seed_source: str = "train"
    train_n_seeds: int = 128
    max_rounds: int = 25
    early_stop_on_success: bool = True

    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    decoding: TargetDecodingConfig = field(default_factory=TargetDecodingConfig)

    random_seed: int = 42


@dataclass
class RunMatrixConfig:
    experiment_name: str = "xjailbreak_eval"
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
class XJailbreakExperimentConfig:
    paths: PathsConfig
    datasets: DatasetsConfig
    targets: List[TargetSpec]
    judge: JudgeConfig
    metrics: MetricsConfig
    xjailbreak: XJailbreakHyperparams
    run_matrix: RunMatrixConfig
    logging: LoggingConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "XJailbreakExperimentConfig":
        return load_xjailbreak_config(config_path)


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


def _parse_dataset_spec(base_dir: Path, key: str, raw_ds: Dict[str, Any]) -> DatasetSpec:
    name = str(_get(raw_ds, "name", key))
    seeds_path = _resolve_path(base_dir, raw_ds.get("seeds_path"))
    max_pairs = raw_ds.get("max_pairs")
    return DatasetSpec(name=name, seeds_path=seeds_path, max_pairs=max_pairs)


def load_xjailbreak_config(config_path: str | Path) -> XJailbreakExperimentConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    base_dir = _infer_repo_root(config_path)

    paths_raw = raw.get("paths", {}) or {}
    data_root = _resolve_path(base_dir, _get(paths_raw, "data_root", "./data"))
    artifacts_root = _resolve_path(base_dir, _get(paths_raw, "artifacts_root", "./artifacts/baselines/xjailbreak"))
    paths = PathsConfig(
        data_root=_require_path(data_root, "paths.data_root"),
        artifacts_root=_require_path(artifacts_root, "paths.artifacts_root"),
    )

    rm_raw = raw.get("run_matrix", {}) or {}
    dataset_key = str(_get(rm_raw, "dataset", "test_jbb_ood"))
    run_matrix = RunMatrixConfig(
        experiment_name=str(_get(rm_raw, "experiment_name", "xjailbreak_eval")),
        dataset=dataset_key,
        query_budgets=list(_get(rm_raw, "query_budgets", [25])),
        repeats=int(_get(rm_raw, "repeats", 1)),
        target_short_names=list(_get(rm_raw, "target_short_names", [])),
    )

    ds_raw = raw.get("datasets", {}) or {}
    by_key: Dict[str, DatasetSpec] = {}
    for k, v in ds_raw.items():
        if isinstance(v, dict):
            by_key[k] = _parse_dataset_spec(base_dir, k, v)
    train_ds = by_key.get("train", None)

    # dataset alias fallback (keeps your old behavior)
    if dataset_key not in by_key:
        aliases = ["test", "test_benign", "test_jbb_ood"]
        alias_found = next((a for a in aliases if a in by_key), None)
        if alias_found is None:
            raise ValueError(f"datasets.{dataset_key} is required.")
        run_matrix.dataset = alias_found

    datasets = DatasetsConfig(train=train_ds, by_key=by_key)

    targets = [
        TargetSpec(
            id=t["id"],
            short_name=t["short_name"],
            family=t.get("family", t["short_name"]),
            enabled=bool(t.get("enabled", True)),
        )
        for t in (raw.get("targets", []) or [])
    ]

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
            confidence_weighting=bool(_get(judge_scoring_raw, "confidence_weighting", False)),
        ),
    )

    metrics_raw = raw.get("metrics", {}) or {}
    metrics = MetricsConfig(
        compute_3d=bool(_get(metrics_raw, "compute_3d", False)),
        compute_5d=bool(_get(metrics_raw, "compute_5d", True)),
        compute_asr=bool(_get(metrics_raw, "compute_asr", True)),
        compute_qps=bool(_get(metrics_raw, "compute_qps", True)),
        compute_distinct_n=list(_get(metrics_raw, "compute_distinct_n", [])),
        compute_runtime=bool(_get(metrics_raw, "compute_runtime", True)),
        compute_query_efficiency=bool(_get(metrics_raw, "compute_query_efficiency", True)),
    )

    x_raw = raw.get("xjailbreak", {}) or {}

    rep_raw = x_raw.get("representation", {}) or {}
    anneal_raw = rep_raw.get("anneal", {}) or {}
    representation = RepresentationConfig(
        dim=int(_get(rep_raw, "dim", 512)),
        proximity_weight=float(_get(rep_raw, "proximity_weight", 0.35)),
        use_token_hash_embedding=bool(_get(rep_raw, "use_token_hash_embedding", True)),
        normalize=bool(_get(rep_raw, "normalize", True)),
        anneal=AnnealConfig(
            enabled=bool(_get(anneal_raw, "enabled", False)),
            start=float(_get(anneal_raw, "start", 0.50)),
            end=float(_get(anneal_raw, "end", 0.20)),
        ),
    )

    attacker_raw = x_raw.get("attacker", {}) or {}
    attacker_dec = attacker_raw.get("decoding", {}) or {}
    attacker = AttackerConfig(
        model_id=str(_get(attacker_raw, "model_id", "ollama/yi:9b")),
        backend=str(_get(attacker_raw, "backend", "router")),
        use_chat_api=bool(_get(attacker_raw, "use_chat_api", True)),
        decoding=AttackerDecodingConfig(
            temperature=float(_get(attacker_dec, "temperature", 0.7)),
            top_p=float(_get(attacker_dec, "top_p", 0.9)),
            max_tokens=int(_get(attacker_dec, "max_tokens", 256)),
        ),
    )

    ref_raw = x_raw.get("refinement", {}) or {}
    refinement = RefinementConfig(
        keep_history=int(_get(ref_raw, "keep_history", 2)),
        prompt_style=str(_get(ref_raw, "prompt_style", "concise")),
        include_target_response=bool(_get(ref_raw, "include_target_response", True)),
        include_judge_feedback=bool(_get(ref_raw, "include_judge_feedback", True)),
        include_refusal_signal=bool(_get(ref_raw, "include_refusal_signal", False)),
    )

    obj_raw = x_raw.get("objective", {}) or {}
    objective = ObjectiveConfig(
        weight_success=float(_get(obj_raw, "weight_success", 1.0)),
        weight_impact=float(_get(obj_raw, "weight_impact", 0.3)),
        weight_stealth=float(_get(obj_raw, "weight_stealth", 0.2)),
        weight_novelty=float(_get(obj_raw, "weight_novelty", 0.2)),
        weight_efficiency=float(_get(obj_raw, "weight_efficiency", 0.1)),
        weight_length_penalty=float(_get(obj_raw, "weight_length_penalty", 0.0)),
    )

    dec_raw = x_raw.get("decoding", {}) or {}
    decoding = TargetDecodingConfig(
        temperature=float(_get(dec_raw, "temperature", 0.0)),
        top_p=float(_get(dec_raw, "top_p", 1.0)),
        max_tokens=int(_get(dec_raw, "max_tokens", 256)),
    )

    xjailbreak = XJailbreakHyperparams(
        seed_source=str(_get(x_raw, "seed_source", "train")),
        train_n_seeds=int(_get(x_raw, "train_n_seeds", 128)),
        max_rounds=int(_get(x_raw, "max_rounds", 25)),
        early_stop_on_success=bool(_get(x_raw, "early_stop_on_success", True)),
        representation=representation,
        attacker=attacker,
        refinement=refinement,
        objective=objective,
        decoding=decoding,
        random_seed=int(_get(x_raw, "random_seed", 42)),
    )

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

    return XJailbreakExperimentConfig(
        paths=paths,
        datasets=datasets,
        targets=targets,
        judge=judge,
        metrics=metrics,
        xjailbreak=xjailbreak,
        run_matrix=run_matrix,
        logging=logging_cfg,
        raw=raw,
    )
