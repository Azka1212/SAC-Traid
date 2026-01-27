# src/baselines/rlbreaker/runner.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import RlbreakerExperimentConfig
from .attack import run_rlbreaker_attack

_METHOD = "rlbreaker"


def _build_target_lookup(cfg: RlbreakerExperimentConfig) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for t in cfg.targets:
        if bool(t.enabled):
            lookup[str(t.short_name)] = str(t.id)
    return lookup


def _resolve_run_plan(
    *,
    cfg: RlbreakerExperimentConfig,
    target_lookup: Dict[str, str],
    targets_override: Optional[List[str]] = None,
    budgets_override: Optional[List[int]] = None,
    repeats_override: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    rm = cfg.run_matrix
    target_shorts = targets_override if targets_override is not None else rm.target_short_names
    budgets = budgets_override if budgets_override is not None else rm.query_budgets
    repeats = repeats_override if repeats_override is not None else rm.repeats

    if not target_shorts:
        raise ValueError("No targets specified. Set run_matrix.target_short_names or pass --targets.")

    unknown = [t for t in target_shorts if t not in target_lookup]
    if unknown:
        raise ValueError(f"Unknown/disabled targets: {unknown}. Enabled: {sorted(target_lookup.keys())}")

    if not budgets:
        raise ValueError("No query budgets specified.")
    for b in budgets:
        if int(b) <= 0:
            raise ValueError(f"Invalid query budget: {b}")

    if repeats is None or int(repeats) <= 0:
        raise ValueError(f"Invalid repeats: {repeats}")

    plan: List[Tuple[str, int, int]] = []
    for ts in target_shorts:
        for b in budgets:
            for r in range(int(repeats)):
                plan.append((ts, int(b), int(r)))
    return plan


def _baseline_run_dir(models_root: Path, target_short: str, budget: int, repeat_idx: int) -> Path:
    return models_root / target_short / str(budget) / f"run_{repeat_idx:03d}"


def _write_latest_pointer(models_root: Path, run_dir: Path) -> None:
    try:
        rel = run_dir.relative_to(models_root)
    except Exception:
        rel = run_dir
    (models_root / "LATEST.txt").write_text(str(rel) + "\n", encoding="utf-8")


def _append_index(models_root: Path, record: Dict[str, Any]) -> None:
    idx_path = models_root / "INDEX.jsonl"
    with idx_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_rlbreaker_partA_from_config(
    *,
    config_path: str,
    targets: Optional[List[str]] = None,
    query_budgets: Optional[List[int]] = None,
    repeats: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    cfg = RlbreakerExperimentConfig.from_yaml(config_path)

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
    print(f"[{_METHOD.upper()}] dataset: {cfg.run_matrix.dataset}")
    print(f"[{_METHOD.upper()}] planned runs: {len(plan)}")
    print(f"[{_METHOD.upper()}] layout: models/<target_short>/<budget>/run_XXX/")

    if dry_run:
        for i, (tshort, b, r) in enumerate(plan, start=1):
            print(f"[DRY {i:03d}] {tshort} budget={b} repeat={r} -> {_baseline_run_dir(models_root, tshort, b, r)}")
        return

    for idx, (target_short, budget, repeat_idx) in enumerate(plan, start=1):
        target_id = target_lookup[target_short]
        run_dir = _baseline_run_dir(models_root, target_short, budget, repeat_idx)
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{idx}/{len(plan)}] Running {_METHOD.upper()} | "
            f"target={target_short} ({target_id}) | budget={budget} | repeat={repeat_idx} | run_dir={run_dir}"
        )

        t0 = time.time()
        ok = True
        err: Optional[str] = None
        metrics: Dict[str, Any] = {}

        try:
            metrics = run_rlbreaker_attack(
                cfg=cfg,
                target_id=target_id,
                target_short_name=target_short,
                query_budget=budget,
                repeat_idx=repeat_idx,
                run_dir=run_dir,
            )
        except Exception as e:
            ok = False
            err = repr(e)
            print(f"    !! Run failed: {err}")

        t1 = time.time()

        _write_latest_pointer(models_root, run_dir)
        rec: Dict[str, Any] = {
            "ts": int(time.time()),
            "method": _METHOD,
            "config_path": str(Path(config_path).resolve()),
            "dataset": cfg.run_matrix.dataset,
            "target_short": target_short,
            "target_id": target_id,
            "budget": budget,
            "repeat_idx": repeat_idx,
            "run_dir": str(run_dir),
            "runtime_sec": round(t1 - t0, 4),
            "ok": bool(ok),
            "error": err,
            "asr": float(metrics.get("asr", 0.0)) if metrics else 0.0,
            "qps": metrics.get("qps", None) if metrics else None,
        }
        _append_index(models_root, rec)

        if ok:
            asr = float(metrics.get("asr", 0.0))
            qps = metrics.get("qps", None)
            qps_str = "inf" if (qps is None or qps == float("inf")) else f"{float(qps):.2f}"
            print(f"    -> Completed: ASR={asr:.3f}, Q/S={qps_str}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run RLBreaker baseline (Part A harness).")
    p.add_argument("--config-path", type=str, default="config/baselines/rlbreaker.yaml")
    p.add_argument("--targets", type=str, nargs="*", default=None)
    p.add_argument("--query-budgets", type=int, nargs="*", default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_rlbreaker_partA_from_config(
        config_path=args.config_path,
        targets=args.targets,
        query_budgets=args.query_budgets,
        repeats=args.repeats,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
