from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.eval.metrics import compute_metrics


@dataclass
class ModelCfg:
    target_id: str
    target_short_name: str
    run_dir: str
    checkpoint_path: str
    seeds_path_override: str


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _pick_aspect_mean(judge_means: Dict[str, Any], name: str) -> float:
    # map keys case-insensitively
    target = name.lower()
    for k, v in judge_means.items():
        if str(k).lower() == target:
            try:
                return float(v)
            except Exception:
                return 0.0
    return 0.0


def _baseline_style_metrics(
    *,
    minimal: Dict[str, Any],
    target_id: str,
    target_short_name: str,
    repeat_idx: int,
    query_budget: int,
    total_runtime_sec: float,
) -> Dict[str, Any]:
    counts = minimal.get("counts", {}) or {}
    metrics = minimal.get("metrics", {}) or {}
    judge_means = minimal.get("judge_means", {}) or {}

    num_pairs = int(counts.get("pairs", 0))
    num_success = int(counts.get("successes", 0))
    total_queries = int(counts.get("total_target_calls", 0))

    asr = float(metrics.get("ASR", 0.0))
    distinct_1 = float(metrics.get("Distinct1", 0.0))
    distinct_2 = float(metrics.get("Distinct2", 0.0))
    distinct_3 = float(metrics.get("Distinct3", 0.0))

    avg_queries_per_pair = (total_queries / num_pairs) if num_pairs > 0 else 0.0
    qps = (total_queries / num_success) if num_success > 0 else float("inf")

    # This matches your example: query_efficiency = ASR / QPS
    # (e.g., 0.39 / 2.564 = 0.152)
    query_efficiency = (asr / qps) if (qps != float("inf") and qps > 0) else 0.0

    runtime_per_pair = (total_runtime_sec / num_pairs) if num_pairs > 0 else 0.0

    out = {
        "target_id": target_id,
        "target_short_name": target_short_name,
        "repeat_idx": repeat_idx,
        "query_budget": query_budget,
        "num_pairs": num_pairs,
        "num_success": num_success,
        "total_queries": total_queries,
        "total_runtime_sec": total_runtime_sec,
        "avg_queries_per_pair": avg_queries_per_pair,
        "asr": asr,
        "qps": qps,
        "avg_success": _pick_aspect_mean(judge_means, "Success"),
        "avg_stealth": _pick_aspect_mean(judge_means, "Stealth"),
        "avg_novelty": _pick_aspect_mean(judge_means, "Novelty"),
        "avg_efficiency": _pick_aspect_mean(judge_means, "Efficiency"),
        "avg_impact": _pick_aspect_mean(judge_means, "Impact"),
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
        "query_efficiency": query_efficiency,
        "runtime_per_pair": runtime_per_pair,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, help="Path to sac_triad eval YAML")
    args = ap.parse_args()

    cfg_path = Path(args.config_path)
    cfg = _load_yaml(cfg_path)

    exp_name = str(cfg["experiment_name"])
    artifacts_root = Path(str(cfg["artifacts_root"]))
    stack_config_path = str(cfg["config_path"])

    dataset_tag = str(cfg["dataset_tag"])
    episodes = int(cfg["episodes"])
    query_budget = int(cfg["query_budget"])
    repeat_idx = int(cfg.get("repeat_idx", 0))
    success_threshold = float(cfg.get("success_threshold", 0.65))
    use_target_echo = bool(cfg.get("use_target_echo", True))

    m = cfg["model"]
    model = ModelCfg(
        target_id=str(m["target_id"]),
        target_short_name=str(m["target_short_name"]),
        run_dir=str(m["run_dir"]),
        checkpoint_path=str(m["checkpoint_path"]),
        seeds_path_override=str(m["seeds_path_override"]),
    )

    run_dir = Path(model.run_dir)
    ckpt = Path(model.checkpoint_path)
    seeds = Path(model.seeds_path_override)

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not seeds.exists():
        raise FileNotFoundError(
            f"seeds_path_override not found: {seeds}\n"
            f"Tip: generate it once via export-ood for this runs_model."
        )

    # -----------------------------
    # Output folder (baseline-like)
    # artifacts/eval/sac_triad/<short>/qb_<budget>/run_<repeat>/
    # -----------------------------
    out_dir = artifacts_root / exp_name / model.target_short_name / f"qb_{query_budget}" / f"run_{repeat_idx}"
    _safe_mkdir(out_dir)

    # snapshot config used
    if cfg.get("save_config_snapshot", True):
        _write_json(out_dir / "config.snapshot.json", cfg)

    # -----------------------------
    # 1) Run eval-sac (writes events into run_dir)
    # -----------------------------
    cmd = [
        "python",
        "-m",
        "src.cli",
        "eval-sac",
        "--checkpoint",
        str(ckpt),
        "--runs-model",
        model.target_id,
        "--target-model",
        model.target_id,
        "--seeds-path-override",
        str(seeds),
        "--dataset-tag",
        dataset_tag,
        "--episodes",
        str(episodes),
        "--config-path",
        stack_config_path,
    ]
    if use_target_echo:
        cmd.append("--use-target-echo")

    t0 = time.time()
    import subprocess

    subprocess.run(cmd, check=True)
    total_runtime_sec = time.time() - t0

    # -----------------------------
    # 2) Compute metrics from run_dir/events.jsonl
    # -----------------------------
    events_jsonl = run_dir / "events.jsonl"
    if not events_jsonl.exists():
        raise FileNotFoundError(f"Expected events.jsonl in run_dir, not found: {events_jsonl}")

    minimal = compute_metrics(
        events_jsonl,
        dataset_tag=dataset_tag,
        success_threshold=success_threshold,
    )

    if "error" in minimal:
        raise RuntimeError(f"Metrics compute failed: {minimal['error']}")

    full = _baseline_style_metrics(
        minimal=minimal,
        target_id=model.target_id,
        target_short_name=model.target_short_name,
        repeat_idx=repeat_idx,
        query_budget=query_budget,
        total_runtime_sec=float(total_runtime_sec),
    )

    # -----------------------------
    # 3) Save bundle like baselines
    # -----------------------------
    _write_json(out_dir / "metrics.minimal.json", minimal)
    _write_json(out_dir / "metrics.json", full)

    # copy logs from run_dir into bundle
    if cfg.get("save_events_jsonl", True):
        _copy_if_exists(run_dir / "events.jsonl", out_dir / "events.jsonl")
    if cfg.get("save_events_csv", True):
        _copy_if_exists(run_dir / "events.csv", out_dir / "events.csv")

    # simple index + latest pointer
    idx_line = {
        "experiment": exp_name,
        "target_id": model.target_id,
        "target_short_name": model.target_short_name,
        "query_budget": query_budget,
        "repeat_idx": repeat_idx,
        "out_dir": str(out_dir),
    }
    with (out_dir / "INDEX.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(idx_line) + "\n")

    (out_dir / "LATEST.txt").write_text(str(out_dir) + "\n", encoding="utf-8")

    print(f"[DONE] Saved baseline-style bundle at: {out_dir}")
    print(f"[DONE] metrics.json -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
