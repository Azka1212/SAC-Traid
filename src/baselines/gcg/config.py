# Stores GCG-specific hyperparameters and default configs (steps, suffix length, candidates, budgets).
from __future__ import annotations

"""
Config loader for the GCG baseline (Part A).

- Mirrors the structure of config/baselines/gcg.yaml
- Provides typed dataclasses so the rest of the baseline code
  doesn't need to touch raw dicts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclass definitions mirroring gcg.yaml
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
    # JBB-OOD dataset used by your GCG Part A run
    test_jbb_ood: DatasetSpec | None = None




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


@dataclass
class GCGObjectiveConfig:
    weight_success: float = 1.0
    weight_impact: float = 0.0
    weight_stealth: float = 0.0
    weight_length_penalty: float = 0.0


@dataclass
class GCGDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class GCGHyperparams:
    anchor_target_id: str
    train_n_seeds: int

    num_steps: int
    candidates_per_step: int
    max_suffix_tokens: int
    step_size: float
    top_k_tokens: int
    reuse_best_suffix: bool
    early_stop_improvement: int

    objective: GCGObjectiveConfig
    decoding: GCGDecodingConfig

    random_seed: int = 42


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
class GCGExperimentConfig:
    """
    Top-level GCG config object used by runner/core/attack.

    This is what `load_gcg_config()` returns.
    """
    paths: PathsConfig
    datasets: DatasetsConfig
    targets: List[TargetSpec]
    guards: List[GuardSpec]
    judge: JudgeConfig
    metrics: MetricsConfig
    gcg: GCGHyperparams
    run_matrix: RunMatrixConfig
    logging: LoggingConfig

    # Keep a copy of the raw dict for debugging / snapshotting if needed.
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "GCGExperimentConfig":
        """
        Convenience wrapper so other modules can do:
            cfg = GCGExperimentConfig.from_yaml("config/baselines/gcg.yaml")
        """
        return load_gcg_config(config_path)



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
    Infer repo root robustly.

    We prefer a parent directory that contains both:
      - config/
      - src/
    This avoids brittle assumptions like parent.parent.parent.
    """
    cfg = config_path.resolve()
    for parent in [cfg.parent, *cfg.parents]:
        try:
            if (parent / "config").exists() and (parent / "src").exists():
                return parent
        except OSError:
            # Some environments can throw on exists() for permission edge cases
            continue

    # Fallback to the previous heuristic (config/baselines/... -> repo root)
    # config_path.parent = config/baselines
    # config_path.parent.parent = config
    # config_path.parent.parent.parent = repo root
    return config_path.parent.parent.parent.resolve()


def _require_path(p: Optional[Path], field_name: str) -> Path:
    if p is None:
        raise ValueError(f"Missing required path for {field_name}. Check your YAML.")
    return p


def load_gcg_config(config_path: str | Path) -> GCGExperimentConfig:
    """
    Load and parse config/baselines/gcg.yaml into a typed GCGExperimentConfig.

    Args:
        config_path: path to the YAML file (relative or absolute).

    Returns:
        GCGExperimentConfig instance.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # ----- paths -----
    paths_raw = raw.get("paths", {})
    base_dir = _infer_repo_root(config_path)  # repo root
    data_root = _resolve_path(base_dir, paths_raw.get("data_root", "./data"))
    artifacts_root = _resolve_path(
        base_dir, paths_raw.get("artifacts_root", "./artifacts/baselines/gcg")
    )
    paths = PathsConfig(
        data_root=_require_path(data_root, "paths.data_root"),
        artifacts_root=_require_path(artifacts_root, "paths.artifacts_root"),
    )

    # ----- datasets -----
    ds_raw = raw.get("datasets", {})
    train_raw = ds_raw.get("train", {})
    jbb_raw = ds_raw.get("test_jbb_ood", {})

    # Fail-fast: AdvBench is not used in this repo right now.
    if "test_advbench" in ds_raw:
        raise ValueError(
            "datasets.test_advbench is not used in this repo right now. "
            "Remove it from config/baselines/gcg.yaml."
        )

    train_ds = DatasetSpec(
        name=train_raw.get("name", "train"),
        seeds_path=_resolve_path(base_dir, train_raw.get("seeds_path")),
        operators_path=_resolve_path(base_dir, train_raw.get("operators_path")),
        pacing_path=_resolve_path(base_dir, train_raw.get("pacing_path")),
        max_pairs=train_raw.get("max_pairs"),
    )

    jbb_ds = None
    if jbb_raw:
        jbb_ds = DatasetSpec(
            name=jbb_raw.get("name", "test_jbb_ood"),
            seeds_path=_resolve_path(base_dir, jbb_raw.get("seeds_path")),
            operators_path=_resolve_path(base_dir, jbb_raw.get("operators_path")),
            pacing_path=_resolve_path(base_dir, jbb_raw.get("pacing_path")),
            max_pairs=jbb_raw.get("max_pairs"),
        )

    datasets = DatasetsConfig(
        train=train_ds,
        test_jbb_ood=jbb_ds,
    )


    # ----- targets & guards -----
    targets = [
        TargetSpec(
            id=t["id"],
            short_name=t["short_name"],
            family=t.get("family", t["short_name"]),
            enabled=bool(t.get("enabled", True)),
        )
        for t in raw.get("targets", [])
    ]

    guards = [
        GuardSpec(
            id=g["id"],
            short_name=g["short_name"],
            enabled=bool(g.get("enabled", False)),
        )
        for g in raw.get("guards", [])
    ]

    # ----- judge -----
    judge_raw = raw.get("judge", {})
    dec_raw = judge_raw.get("decoding", {})
    scoring_raw = judge_raw.get("scoring", {})

    judge = JudgeConfig(
        model_id=judge_raw["model_id"],
        backend=judge_raw["backend"],
        use_chat_api=bool(judge_raw.get("use_chat_api", True)),
        decoding=JudgeDecodingConfig(
            temperature=float(dec_raw.get("temperature", 0.0)),
            top_p=float(dec_raw.get("top_p", 1.0)),
            max_tokens=int(dec_raw.get("max_tokens", 256)),
        ),
        scoring=JudgeScoringConfig(
            aspects=list(scoring_raw.get("aspects", [])),
            success_threshold=float(scoring_raw.get("success_threshold", 0.65)),
            confidence_weighting=bool(scoring_raw.get("confidence_weighting", True)),
        ),
    )

    # ----- metrics -----
    metrics_raw = raw.get("metrics", {})
    metrics = MetricsConfig(
        compute_5d=bool(metrics_raw.get("compute_5d", True)),
        compute_asr=bool(metrics_raw.get("compute_asr", True)),
        compute_qps=bool(metrics_raw.get("compute_qps", True)),
        compute_distinct_n=list(metrics_raw.get("compute_distinct_n", [])),
        compute_runtime=bool(metrics_raw.get("compute_runtime", True)),
        compute_query_efficiency=bool(metrics_raw.get("compute_query_efficiency", True)),
    )

    # ----- GCG hyperparams -----
    gcg_raw = raw.get("gcg", {})
    obj_raw = gcg_raw.get("objective", {})
    dec_gcg_raw = gcg_raw.get("decoding", {})

    gcg = GCGHyperparams(
        anchor_target_id=gcg_raw["anchor_target_id"],
        train_n_seeds=int(gcg_raw.get("train_n_seeds", 128)),
        num_steps=int(gcg_raw.get("num_steps", 200)),
        candidates_per_step=int(gcg_raw.get("candidates_per_step", 8)),
        max_suffix_tokens=int(gcg_raw.get("max_suffix_tokens", 32)),
        step_size=float(gcg_raw.get("step_size", 0.5)),
        top_k_tokens=int(gcg_raw.get("top_k_tokens", 64)),
        reuse_best_suffix=bool(gcg_raw.get("reuse_best_suffix", True)),
        early_stop_improvement=int(gcg_raw.get("early_stop_improvement", 3)),
        objective=GCGObjectiveConfig(
            weight_success=float(obj_raw.get("weight_success", 1.0)),
            weight_impact=float(obj_raw.get("weight_impact", 0.0)),
            weight_stealth=float(obj_raw.get("weight_stealth", 0.0)),
            weight_length_penalty=float(obj_raw.get("weight_length_penalty", 0.0)),
        ),
        decoding=GCGDecodingConfig(
            temperature=float(dec_gcg_raw.get("temperature", 0.0)),
            top_p=float(dec_gcg_raw.get("top_p", 1.0)),
            max_tokens=int(dec_gcg_raw.get("max_tokens", 256)),
        ),
        random_seed=int(gcg_raw.get("random_seed", 42)),
    )

    # ----- run matrix -----
    rm_raw = raw.get("run_matrix", {})
    run_matrix = RunMatrixConfig(
        experiment_name=rm_raw.get("experiment_name", "partA_gcg_jbb_ood"),
        dataset=rm_raw.get("dataset", "test_jbb_ood"),
        query_budgets=list(rm_raw.get("query_budgets", [5, 10, 20])),
        repeats=int(rm_raw.get("repeats", 3)),
        target_short_names=list(rm_raw.get("target_short_names", [])),
    )

    # ----- logging -----
    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        save_events_jsonl=bool(log_raw.get("save_events_jsonl", True)),
        save_events_csv=bool(log_raw.get("save_events_csv", True)),
        save_metrics_json=bool(log_raw.get("save_metrics_json", True)),
        save_config_snapshot=bool(log_raw.get("save_config_snapshot", True)),
        save_git_commit=bool(log_raw.get("save_git_commit", True)),
        max_prompt_chars=int(log_raw.get("max_prompt_chars", 8000)),
        redact_prompts_in_csv=bool(log_raw.get("redact_prompts_in_csv", False)),
    )

    # Basic sanity checks
    anchor_id = gcg.anchor_target_id
    if anchor_id not in [t.id for t in targets]:
        raise ValueError(
            f"GCG anchor_target_id={anchor_id!r} is not in targets list. "
            "Make sure gcg.anchor_target_id matches one of the target ids."
        )

    return GCGExperimentConfig(
        paths=paths,
        datasets=datasets,
        targets=targets,
        guards=guards,
        judge=judge,
        metrics=metrics,
        gcg=gcg,
        run_matrix=run_matrix,
        logging=logging_cfg,
        raw=raw,
    )
