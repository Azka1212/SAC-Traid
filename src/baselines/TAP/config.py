# src/baselines/tap/config.py
from __future__ import annotations

"""
Config loader for the TAP baseline (Part A).

- Mirrors the structure of config/baselines/tap.yaml
- Provides typed dataclasses so the TAP code doesn't need to touch raw dicts.

This version supports TAP-style tree search hyperparameters + attacker/target decoding,
while remaining backward-compatible with older YAMLs (missing fields fall back to defaults).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclass definitions (paths, datasets, targets, judge, metrics, tap)
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
    temperature: float
    top_p: float
    max_tokens: int


@dataclass
class JudgeScoringConfig:
    aspects: List[str]
    success_threshold: float
    confidence_weighting: bool = True


@dataclass
class JudgeConfig:
    model_id: str
    backend: str
    use_chat_api: bool
    decoding: JudgeDecodingConfig
    scoring: JudgeScoringConfig


@dataclass
class MetricsConfig:
    compute_5d: bool = True
    compute_asr: bool = True
    compute_qps: bool = True
    compute_distinct_n: List[int] = field(default_factory=list)
    compute_runtime: bool = True
    compute_query_efficiency: bool = True


# ------------------------- TAP hyperparams (expanded) -------------------------

@dataclass
class TapObjectiveConfig:
    weight_success: float = 1.0
    weight_impact: float = 0.0
    weight_stealth: float = 0.0
    weight_novelty: float = 0.0
    weight_efficiency: float = 0.0
    weight_length_penalty: float = 0.0


@dataclass
class TapDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class TapTreeConfig:
    branching_factor: int = 4
    width: int = 10
    depth: int = 5
    early_stop_on_success: bool = True


@dataclass
class TapAttackerConfig:
    model_id: str
    decoding: TapDecodingConfig = field(default_factory=lambda: TapDecodingConfig(max_tokens=512, temperature=0.7, top_p=0.9))


@dataclass
class TapTargetDecodingConfig:
    max_tokens: int = 256


@dataclass
class TapHyperparams:
    random_seed: int = 42
    tree: TapTreeConfig = field(default_factory=TapTreeConfig)
    objective: TapObjectiveConfig = field(default_factory=TapObjectiveConfig)
    attacker: Optional[TapAttackerConfig] = None
    target_decoding: TapTargetDecodingConfig = field(default_factory=TapTargetDecodingConfig)


# ------------------------- Run matrix + logging -------------------------

@dataclass
class RunMatrixConfig:
    experiment_name: str
    dataset: str
    query_budgets: List[int]
    repeats: int
    target_short_names: List[str]


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
class TapExperimentConfig:
    """
    Top-level TAP config object used by runner / attack / aggregate.
    """
    paths: PathsConfig
    datasets: DatasetsConfig
    targets: List[TargetSpec]
    guards: List[GuardSpec]
    judge: JudgeConfig
    metrics: MetricsConfig
    tap: TapHyperparams
    run_matrix: RunMatrixConfig
    logging: LoggingConfig

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TapExperimentConfig":
        return load_tap_config(config_path)


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------

def _resolve_path(base_dir: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _infer_repo_root(config_path: Path) -> Path:
    """
    Infer repo root robustly: parent directory that contains both config/ and src/.
    """
    cfg = config_path.resolve()
    for parent in [cfg.parent, *cfg.parents]:
        try:
            if (parent / "config").exists() and (parent / "src").exists():
                return parent
        except OSError:
            continue
    # fallback
    return config_path.parent.parent.parent.resolve()


def _require_path(p: Optional[Path], field_name: str) -> Path:
    if p is None:
        raise ValueError(f"Missing required path for {field_name}. Check your YAML.")
    return p


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def load_tap_config(config_path: str | Path) -> TapExperimentConfig:
    """
    Load and parse config/baselines/tap.yaml into a typed TapExperimentConfig.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw_loaded = yaml.safe_load(f)
    raw: Dict[str, Any] = raw_loaded if isinstance(raw_loaded, dict) else {}

    # ----- paths -----
    paths_raw = _as_dict(raw.get("paths", {}))
    base_dir = _infer_repo_root(config_path)

    data_root = _resolve_path(base_dir, paths_raw.get("data_root", "./data"))
    artifacts_root = _resolve_path(
        base_dir, paths_raw.get("artifacts_root", "./artifacts/baselines/tap")
    )

    paths = PathsConfig(
        data_root=_require_path(data_root, "paths.data_root"),
        artifacts_root=_require_path(artifacts_root, "paths.artifacts_root"),
    )

    # ----- datasets -----
    ds_raw = _as_dict(raw.get("datasets", {}))
    train_raw = _as_dict(ds_raw.get("train", {}))
    jbb_raw = _as_dict(ds_raw.get("test_jbb_ood", {}))

    if not jbb_raw:
        raise ValueError(
            "datasets.test_jbb_ood is required for TAP Part A runs. "
            "Please define it in config/baselines/tap.yaml."
        )

    train_ds = DatasetSpec(
        name=str(train_raw.get("name", "train")),
        seeds_path=_resolve_path(base_dir, train_raw.get("seeds_path")),
        operators_path=_resolve_path(base_dir, train_raw.get("operators_path")),
        pacing_path=_resolve_path(base_dir, train_raw.get("pacing_path")),
        max_pairs=train_raw.get("max_pairs"),
    )

    jbb_ds = DatasetSpec(
        name=str(jbb_raw.get("name", "test_jbb_ood")),
        seeds_path=_resolve_path(base_dir, jbb_raw.get("seeds_path")),
        operators_path=_resolve_path(base_dir, jbb_raw.get("operators_path")),
        pacing_path=_resolve_path(base_dir, jbb_raw.get("pacing_path")),
        max_pairs=jbb_raw.get("max_pairs"),
    )

    datasets = DatasetsConfig(train=train_ds, test_jbb_ood=jbb_ds)

    # ----- targets & guards -----
    targets_raw = raw.get("targets", [])
    targets: List[TargetSpec] = []
    if isinstance(targets_raw, list):
        for t in targets_raw:
            if not isinstance(t, dict):
                continue
            targets.append(
                TargetSpec(
                    id=str(t["id"]),
                    short_name=str(t["short_name"]),
                    family=str(t.get("family", t["short_name"])),
                    enabled=bool(t.get("enabled", True)),
                )
            )

    guards_raw = raw.get("guards", [])
    guards: List[GuardSpec] = []
    if isinstance(guards_raw, list):
        for g in guards_raw:
            if not isinstance(g, dict):
                continue
            guards.append(
                GuardSpec(
                    id=str(g["id"]),
                    short_name=str(g["short_name"]),
                    enabled=bool(g.get("enabled", False)),
                )
            )

    # ----- judge -----
    judge_raw = _as_dict(raw.get("judge", {}))
    dec_raw = _as_dict(judge_raw.get("decoding", {}))
    scoring_raw = _as_dict(judge_raw.get("scoring", {}))

    # IMPORTANT: default to canonical 5 aspects if missing
    default_aspects = ["Success", "Stealth", "Novelty", "Efficiency", "Impact"]
    aspects = scoring_raw.get("aspects", default_aspects)
    if not isinstance(aspects, list) or not aspects:
        aspects = default_aspects

    judge = JudgeConfig(
        model_id=str(judge_raw["model_id"]),
        backend=str(judge_raw["backend"]),
        use_chat_api=bool(judge_raw.get("use_chat_api", True)),
        decoding=JudgeDecodingConfig(
            temperature=float(dec_raw.get("temperature", 0.0)),
            top_p=float(dec_raw.get("top_p", 1.0)),
            max_tokens=int(dec_raw.get("max_tokens", 256)),
        ),
        scoring=JudgeScoringConfig(
            aspects=[str(a) for a in aspects],
            success_threshold=float(scoring_raw.get("success_threshold", 0.65)),
            confidence_weighting=bool(scoring_raw.get("confidence_weighting", True)),
        ),
    )

    # ----- metrics -----
    metrics_raw = _as_dict(raw.get("metrics", {}))
    metrics = MetricsConfig(
        compute_5d=bool(metrics_raw.get("compute_5d", True)),
        compute_asr=bool(metrics_raw.get("compute_asr", True)),
        compute_qps=bool(metrics_raw.get("compute_qps", True)),
        compute_distinct_n=list(metrics_raw.get("compute_distinct_n", [])) if isinstance(metrics_raw.get("compute_distinct_n", []), list) else [],
        compute_runtime=bool(metrics_raw.get("compute_runtime", True)),
        compute_query_efficiency=bool(metrics_raw.get("compute_query_efficiency", True)),
    )

    # ----- TAP hyperparams (expanded) -----
    tap_raw = _as_dict(raw.get("tap", {}))

    # objective
    obj_raw = _as_dict(tap_raw.get("objective", {}))
    objective = TapObjectiveConfig(
        weight_success=float(obj_raw.get("weight_success", 1.0)),
        weight_impact=float(obj_raw.get("weight_impact", 0.0)),
        weight_stealth=float(obj_raw.get("weight_stealth", 0.0)),
        weight_novelty=float(obj_raw.get("weight_novelty", 0.0)),
        weight_efficiency=float(obj_raw.get("weight_efficiency", 0.0)),
        weight_length_penalty=float(obj_raw.get("weight_length_penalty", 0.0)),
    )

    # tree
    tree_raw = _as_dict(tap_raw.get("tree", {}))
    tree = TapTreeConfig(
        branching_factor=int(tree_raw.get("branching_factor", 4)),
        width=int(tree_raw.get("width", 10)),
        depth=int(tree_raw.get("depth", 5)),
        early_stop_on_success=bool(tree_raw.get("early_stop_on_success", True)),
    )

    # attacker (optional)
    attacker_raw = _as_dict(tap_raw.get("attacker", {}))
    attacker_dec_raw = _as_dict(attacker_raw.get("decoding", {}))
    attacker: Optional[TapAttackerConfig] = None
    attacker_model_id = attacker_raw.get("model_id", None)
    if isinstance(attacker_model_id, str) and attacker_model_id.strip():
        attacker = TapAttackerConfig(
            model_id=attacker_model_id.strip(),
            decoding=TapDecodingConfig(
                temperature=float(attacker_dec_raw.get("temperature", 0.7)),
                top_p=float(attacker_dec_raw.get("top_p", 0.9)),
                max_tokens=int(attacker_dec_raw.get("max_tokens", 512)),
            ),
        )

    # target decoding
    tgt_dec_raw = _as_dict(tap_raw.get("target_decoding", {}))
    target_decoding = TapTargetDecodingConfig(
        max_tokens=int(tgt_dec_raw.get("max_tokens", 256))
    )

    tap = TapHyperparams(
        random_seed=int(tap_raw.get("random_seed", 42)),
        tree=tree,
        objective=objective,
        attacker=attacker,
        target_decoding=target_decoding,
    )

    # ----- run matrix -----
    rm_raw = _as_dict(raw.get("run_matrix", {}))
    run_matrix = RunMatrixConfig(
        experiment_name=str(rm_raw.get("experiment_name", "partA_tap_jbb_ood")),
        dataset=str(rm_raw.get("dataset", "test_jbb_ood")),
        query_budgets=list(rm_raw.get("query_budgets", [25])) if isinstance(rm_raw.get("query_budgets", [25]), list) else [25],
        repeats=int(rm_raw.get("repeats", 1)),
        target_short_names=list(rm_raw.get("target_short_names", [])) if isinstance(rm_raw.get("target_short_names", []), list) else [],
    )

    # ----- logging -----
    log_raw = _as_dict(raw.get("logging", {}))
    logging_cfg = LoggingConfig(
        save_events_jsonl=bool(log_raw.get("save_events_jsonl", True)),
        save_events_csv=bool(log_raw.get("save_events_csv", True)),
        save_metrics_json=bool(log_raw.get("save_metrics_json", True)),
        save_config_snapshot=bool(log_raw.get("save_config_snapshot", True)),
        save_git_commit=bool(log_raw.get("save_git_commit", True)),
        max_prompt_chars=int(log_raw.get("max_prompt_chars", 8000)),
        redact_prompts_in_csv=bool(log_raw.get("redact_prompts_in_csv", False)),
    )

    return TapExperimentConfig(
        paths=paths,
        datasets=datasets,
        targets=targets,
        guards=guards,
        judge=judge,
        metrics=metrics,
        tap=tap,
        run_matrix=run_matrix,
        logging=logging_cfg,
        raw=raw,
    )
