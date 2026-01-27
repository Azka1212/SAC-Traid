# src/baselines/gptfuzzer/config.py
from __future__ import annotations

"""
Config loader for the GPTFuzzer baseline (Part A).

- Mirrors config/baselines/gptfuzzer.yaml
- Provides typed dataclasses so runner / attack / aggregate don't touch raw dicts.

Keep this isolated to GPTFuzzer only (no TAP / no SAC configs).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

AnyPath = Union[str, Path]


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
    aspects: List[str] = field(
        default_factory=lambda: ["Success", "Stealth", "Novelty", "Efficiency", "Impact"]
    )
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
# GPTFuzzer-specific configs
# ---------------------------------------------------------------------------

@dataclass
class MutatorDecodingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


@dataclass
class MutatorLLMConfig:
    model_id: str = "ollama/yi:9b"
    backend: str = "router"
    use_chat_api: bool = True
    decoding: MutatorDecodingConfig = field(default_factory=MutatorDecodingConfig)


@dataclass
class PopulationConfig:
    seed_pool_size: int = 8
    max_population: int = 32
    keep_top_k: int = 10
    deduplicate: bool = True
    dedup_key: str = "normalized_prompt"  # normalized_prompt | hash


@dataclass
class SelectionConfig:
    strategy: str = "epsilon_greedy"  # topk | roulette | epsilon_greedy
    epsilon: float = 0.2


@dataclass
class MutationOperatorConfig:
    name: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MutationConfig:
    operators: List[MutationOperatorConfig] = field(default_factory=list)


@dataclass
class ObjectiveConfig:
    weight_success: float = 1.0
    weight_impact: float = 0.3
    weight_stealth: float = 0.2
    weight_novelty: float = 0.2
    weight_efficiency: float = 0.1
    weight_length_penalty: float = 0.0
    early_stop_on_success: bool = True


@dataclass
class TargetDecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class GPTFuzzerHyperparams:
    # seed source / sampling
    seed_source: str = "train"
    train_n_seeds: int = 128

    # fuzz loop
    max_rounds: int = 200
    candidates_per_round: int = 4

    population: PopulationConfig = field(default_factory=PopulationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)

    # optional mutator llm (for llm_rewrite/paraphrase ops)
    mutator_llm: Optional[MutatorLLMConfig] = None

    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    decoding: TargetDecodingConfig = field(default_factory=TargetDecodingConfig)

    random_seed: int = 42


# ---------------------------------------------------------------------------
# Run matrix + top-level logging
# ---------------------------------------------------------------------------

@dataclass
class RunMatrixConfig:
    experiment_name: str = "partA_gptfuzzer_jbb_ood"
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
class GPTFuzzerExperimentConfig:
    paths: PathsConfig
    datasets: DatasetsConfig
    targets: List[TargetSpec]
    guards: List[GuardSpec]
    judge: JudgeConfig
    metrics: MetricsConfig
    gptfuzzer: GPTFuzzerHyperparams
    run_matrix: RunMatrixConfig
    logging: LoggingConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: AnyPath) -> "GPTFuzzerExperimentConfig":
        return load_gptfuzzer_config(config_path)


# ---------------------------------------------------------------------------
# Loader utils
# ---------------------------------------------------------------------------

def _resolve_path(base_dir: Path, maybe_path: Optional[Union[str, Path]]) -> Optional[Path]:
    if maybe_path is None:
        return None
    p = maybe_path if isinstance(maybe_path, Path) else Path(str(maybe_path))
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
    # fallback: best effort
    return config_path.parent.parent.parent.resolve()


def _require_path(p: Optional[Path], field_name: str) -> Path:
    if p is None:
        raise ValueError(f"Missing required path for {field_name}. Check your YAML.")
    return p


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _validate_loaded(cfg: "GPTFuzzerExperimentConfig") -> None:
    # Must have at least one enabled target
    enabled_targets = [t for t in cfg.targets if t.enabled]
    if not enabled_targets:
        raise ValueError("No enabled targets found in YAML (targets: enabled: true).")

    # Test dataset must have seeds_path
    test_ds = cfg.datasets.test_jbb_ood
    if test_ds.seeds_path is None:
        raise ValueError("datasets.test_jbb_ood.seeds_path is required.")

    # Sanity check some enums
    if cfg.gptfuzzer.population.dedup_key not in ("normalized_prompt", "hash"):
        raise ValueError(
            f"gptfuzzer.population.dedup_key must be one of "
            f"['normalized_prompt','hash'], got: {cfg.gptfuzzer.population.dedup_key}"
        )

    if cfg.gptfuzzer.selection.strategy not in ("topk", "roulette", "epsilon_greedy"):
        raise ValueError(
            f"gptfuzzer.selection.strategy must be one of "
            f"['topk','roulette','epsilon_greedy'], got: {cfg.gptfuzzer.selection.strategy}"
        )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_gptfuzzer_config(config_path: AnyPath) -> GPTFuzzerExperimentConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    base_dir = _infer_repo_root(config_path)

    # ----- paths -----
    paths_raw = raw.get("paths", {}) or {}
    data_root = _resolve_path(base_dir, _get(paths_raw, "data_root", "./data"))
    artifacts_root = _resolve_path(base_dir, _get(paths_raw, "artifacts_root", "./artifacts/baselines/gptfuzzer"))
    paths = PathsConfig(
        data_root=_require_path(data_root, "paths.data_root"),
        artifacts_root=_require_path(artifacts_root, "paths.artifacts_root"),
    )

    # ----- datasets -----
    ds_raw = raw.get("datasets", {}) or {}

    train_raw = ds_raw.get("train", {}) or {}
    test_raw = ds_raw.get("test_jbb_ood", {}) or {}
    if not test_raw:
        raise ValueError("datasets.test_jbb_ood is required. Define it in config/baselines/gptfuzzer.yaml.")

    train_ds = DatasetSpec(
        name=_get(train_raw, "name", "train"),
        seeds_path=_resolve_path(base_dir, train_raw.get("seeds_path")),
        operators_path=_resolve_path(base_dir, train_raw.get("operators_path")),
        pacing_path=_resolve_path(base_dir, train_raw.get("pacing_path")),
        max_pairs=train_raw.get("max_pairs"),
    )
    test_ds = DatasetSpec(
        name=_get(test_raw, "name", "test_jbb_ood"),
        seeds_path=_resolve_path(base_dir, test_raw.get("seeds_path")),
        operators_path=_resolve_path(base_dir, test_raw.get("operators_path")),
        pacing_path=_resolve_path(base_dir, test_raw.get("pacing_path")),
        max_pairs=test_raw.get("max_pairs"),
    )
    datasets = DatasetsConfig(train=train_ds, test_jbb_ood=test_ds)

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

    # ----- gptfuzzer hyperparams -----
    gf_raw = raw.get("gptfuzzer", {}) or {}

    # population
    pop_raw = gf_raw.get("population", {}) or {}
    population = PopulationConfig(
        seed_pool_size=int(_get(pop_raw, "seed_pool_size", 8)),
        max_population=int(_get(pop_raw, "max_population", 32)),
        keep_top_k=int(_get(pop_raw, "keep_top_k", 10)),
        deduplicate=bool(_get(pop_raw, "deduplicate", True)),
        dedup_key=str(_get(pop_raw, "dedup_key", "normalized_prompt")),
    )

    # selection
    sel_raw = gf_raw.get("selection", {}) or {}
    selection = SelectionConfig(
        strategy=str(_get(sel_raw, "strategy", "epsilon_greedy")),
        epsilon=float(_get(sel_raw, "epsilon", 0.2)),
    )

    # mutation
    mut_raw = gf_raw.get("mutation", {}) or {}
    ops_raw = mut_raw.get("operators", []) or []
    operators: List[MutationOperatorConfig] = []
    for o in ops_raw:
        if not isinstance(o, dict) or "name" not in o:
            continue
        operators.append(
            MutationOperatorConfig(
                name=str(o["name"]),
                weight=float(o.get("weight", 1.0)),
                params=dict(o.get("params", {}) or {}),
            )
        )
    mutation = MutationConfig(operators=operators)

    # mutator llm (optional)
    mut_llm_cfg: Optional[MutatorLLMConfig] = None
    mut_llm_raw = gf_raw.get("mutator_llm", None)
    if isinstance(mut_llm_raw, dict):
        mdec = mut_llm_raw.get("decoding", {}) or {}
        mut_llm_cfg = MutatorLLMConfig(
            model_id=str(_get(mut_llm_raw, "model_id", "ollama/yi:9b")),
            backend=str(_get(mut_llm_raw, "backend", "router")),
            use_chat_api=bool(_get(mut_llm_raw, "use_chat_api", True)),
            decoding=MutatorDecodingConfig(
                temperature=float(_get(mdec, "temperature", 0.7)),
                top_p=float(_get(mdec, "top_p", 0.9)),
                max_tokens=int(_get(mdec, "max_tokens", 256)),
            ),
        )

    # objective
    obj_raw = gf_raw.get("objective", {}) or {}
    objective = ObjectiveConfig(
        weight_success=float(_get(obj_raw, "weight_success", 1.0)),
        weight_impact=float(_get(obj_raw, "weight_impact", 0.3)),
        weight_stealth=float(_get(obj_raw, "weight_stealth", 0.2)),
        weight_novelty=float(_get(obj_raw, "weight_novelty", 0.2)),
        weight_efficiency=float(_get(obj_raw, "weight_efficiency", 0.1)),
        weight_length_penalty=float(_get(obj_raw, "weight_length_penalty", 0.0)),
        early_stop_on_success=bool(_get(obj_raw, "early_stop_on_success", True)),
    )

    # target decoding
    dec_raw = gf_raw.get("decoding", {}) or {}
    decoding = TargetDecodingConfig(
        temperature=float(_get(dec_raw, "temperature", 0.0)),
        top_p=float(_get(dec_raw, "top_p", 1.0)),
        max_tokens=int(_get(dec_raw, "max_tokens", 256)),
    )

    gptfuzzer = GPTFuzzerHyperparams(
        seed_source=str(_get(gf_raw, "seed_source", "train")),
        train_n_seeds=int(_get(gf_raw, "train_n_seeds", 128)),
        max_rounds=int(_get(gf_raw, "max_rounds", 200)),
        candidates_per_round=int(_get(gf_raw, "candidates_per_round", 4)),
        population=population,
        selection=selection,
        mutation=mutation,
        mutator_llm=mut_llm_cfg,
        objective=objective,
        decoding=decoding,
        random_seed=int(_get(gf_raw, "random_seed", 42)),
    )

    # ----- run matrix -----
    rm_raw = raw.get("run_matrix", {}) or {}
    run_matrix = RunMatrixConfig(
        experiment_name=str(_get(rm_raw, "experiment_name", "partA_gptfuzzer_jbb_ood")),
        dataset=str(_get(rm_raw, "dataset", "test_jbb_ood")),
        query_budgets=list(_get(rm_raw, "query_budgets", [25])),
        repeats=int(_get(rm_raw, "repeats", 1)),
        target_short_names=list(_get(rm_raw, "target_short_names", [])),
    )

    # ----- logging -----
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

    cfg = GPTFuzzerExperimentConfig(
        paths=paths,
        datasets=datasets,
        targets=targets,
        guards=guards,
        judge=judge,
        metrics=metrics,
        gptfuzzer=gptfuzzer,
        run_matrix=run_matrix,
        logging=logging_cfg,
        raw=raw,
    )

    _validate_loaded(cfg)
    return cfg
