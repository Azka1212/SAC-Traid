# src/baselines/pair/runner.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .config import PAIRExperimentConfig
from .attack import run_pair_attack

_METHOD = "pair"


def _build_target_lookup(cfg: PAIRExperimentConfig) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for t in cfg.targets:
        if bool(t.enabled):
            lookup[t.short_name] = t.id
    return lookup


def _resolve_run_plan(
    *,
    cfg: PAIRExperimentConfig,
    target_lookup: Dict[str, str],
    targets_override: Optional[List[str]] = None,
    budgets_override: Optional[List[int]] = None,
    repeats_override: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    rm = cfg.run_matrix

    target_shorts = targets_override if targets_override is not None else rm.target_short_names
    query_budgets = budgets_override if budgets_override is not None else rm.query_budgets
    repeats = repeats_override if repeats_override is not None else rm.repeats

    if not target_shorts:
        raise ValueError("No targets specified (empty target list). Check run_matrix.target_short_names.")

    unknown = [t for t in target_shorts if t not in target_lookup]
    if unknown:
        raise ValueError(
            f"Unknown/disabled target_short_name(s): {unknown}. "
            f"Enabled targets: {sorted(target_lookup.keys())}"
        )

    if not query_budgets:
        raise ValueError("No query budgets specified. Check run_matrix.query_budgets.")
    for b in query_budgets:
        if int(b) <= 0:
            raise ValueError(f"Invalid query budget: {b}. Must be positive integer.")

    if int(repeats) <= 0:
        raise ValueError(f"Invalid repeats: {repeats}. Must be >= 1.")

    plan: List[Tuple[str, int, int]] = []
    for ts in target_shorts:
        for qb in query_budgets:
            for r in range(int(repeats)):
                plan.append((ts, int(qb), int(r)))
    return plan


def _baseline_run_dir(*, models_root: Path, target_short: str, budget: int, repeat_idx: int) -> Path:
    return models_root / target_short / str(budget) / f"run_{repeat_idx:03d}"


def _write_latest_pointer(models_root: Path, run_dir: Path) -> None:
    try:
        rel = run_dir.relative_to(models_root)
    except Exception:
        rel = run_dir
    (models_root / "LATEST.txt").write_text(str(rel) + "\n", encoding="utf-8")


def _append_index(models_root: Path, record: Dict) -> None:
    idx_path = models_root / "INDEX.jsonl"
    with idx_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pair_partA_from_config(
    *,
    config_path: str,
    targets: Optional[List[str]] = None,
    query_budgets: Optional[List[int]] = None,
    repeats: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    cfg = PAIRExperimentConfig.from_yaml(config_path)

    artifacts_root = Path(cfg.paths.artifacts_root).resolve()
    models_root = (artifacts_root / "models").resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    target_lookup = _build_target_lookup(cfg)

    plan = _resolve_run_plan(
        cfg=cfg,
        target_lookup=target_lookup,
        targets_override=targets,
        budgets_override=query_budgets,
        repeats_override=repeats,
    )

    print(f"[{_METHOD.upper()}] Loaded config from: {config_path}")
    print(f"[{_METHOD.upper()}] artifacts_root: {artifacts_root}")
    print(f"[{_METHOD.upper()}] models_root: {models_root}")
    print(f"[{_METHOD.upper()}] planned runs: {len(plan)}")
    print(f"[{_METHOD.upper()}] layout: models/<target_short>/<budget>/run_XXX/")

    if dry_run:
        for i, (tshort, b, r) in enumerate(plan, start=1):
            print(f"[DRY {i:03d}] {tshort} budget={b} repeat={r} -> {_baseline_run_dir(models_root=models_root, target_short=tshort, budget=b, repeat_idx=r)}")
        return

    for idx, (target_short, budget, repeat_idx) in enumerate(plan, start=1):
        target_id = target_lookup[target_short]
        run_dir = _baseline_run_dir(models_root=models_root, target_short=target_short, budget=budget, repeat_idx=repeat_idx)
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{idx}/{len(plan)}] Running {_METHOD.upper()} | "
            f"target={target_short} ({target_id}) | budget={budget} | repeat={repeat_idx} | run_dir={run_dir}"
        )

        t0 = time.time()
        metrics = run_pair_attack(
            cfg=cfg,
            target_id=target_id,
            target_short_name=target_short,
            query_budget=budget,
            repeat_idx=repeat_idx,
            run_dir=run_dir,
        )
        t1 = time.time()

        _write_latest_pointer(models_root, run_dir)

        rec = {
            "ts": int(time.time()),
            "method": _METHOD,
            "config_path": str(Path(config_path).resolve()),
            "target_short": target_short,
            "target_id": target_id,
            "budget": budget,
            "repeat_idx": repeat_idx,
            "run_dir": str(run_dir),
            "runtime_sec": round(t1 - t0, 4),
            "asr": float(metrics.get("asr", 0.0)),
            "qps": metrics.get("qps", None),
        }
        _append_index(models_root, rec)

        asr = float(metrics.get("asr", 0.0))
        qps = metrics.get("qps", None)
        qps_str = "inf" if (qps is None or qps == float("inf")) else f"{float(qps):.2f}"
        print(f"    -> Completed: ASR={asr:.3f}, Q/S={qps_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PAIR baseline (Part A) from a YAML config.")
    parser.add_argument("--config-path", type=str, default="config/baselines/pair.yaml")
    parser.add_argument("--targets", type=str, nargs="*", default=None)
    parser.add_argument("--query-budgets", type=int, nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_pair_partA_from_config(
        config_path=args.config_path,
        targets=args.targets,
        query_budgets=args.query_budgets,
        repeats=args.repeats,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
