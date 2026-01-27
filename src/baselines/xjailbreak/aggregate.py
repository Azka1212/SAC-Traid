# src/baselines/xjailbreak/aggregate.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


METHOD = "xjailbreak"


def _find_metrics_files(artifacts_root: Path) -> List[Path]:
    models_root = artifacts_root / "models"
    if not models_root.exists():
        return []
    return sorted(models_root.glob("*/*/run_*/metrics.json"))


def _row_sort_key(r: Dict[str, Any]) -> Tuple[str, int, int]:
    t = str(r.get("target_short_name", ""))
    qb = r.get("query_budget", 0)
    ri = r.get("repeat_idx", 0)
    try:
        qb_i = int(qb)
    except Exception:
        qb_i = 0
    try:
        ri_i = int(ri)
    except Exception:
        ri_i = 0
    return (t, qb_i, ri_i)


def aggregate_xjailbreak(artifacts_root: Path, out_dir: Optional[Path] = None) -> Path:
    artifacts_root = artifacts_root.resolve()
    out_dir = (out_dir or (artifacts_root / "tables" / METHOD)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = _find_metrics_files(artifacts_root)

    fieldnames = [
        "target_short_name",
        "repeat_idx",
        "query_budget",
        "num_pairs",
        "num_success",
        "total_queries",
        "total_runtime_sec",
        "avg_queries_per_pair",
        "asr",
        "qps",
        "avg_success",
        "avg_stealth",
        "avg_novelty",
        "avg_efficiency",
        "avg_impact",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "query_efficiency",
        "runtime_per_pair",
    ]

    rows: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []

    manifest_path = out_dir / "manifest_metrics.jsonl"
    bad_path = out_dir / "bad_metrics.jsonl"

    required_core = {"target_short_name", "repeat_idx", "query_budget", "num_pairs", "num_success", "asr", "qps"}

    with manifest_path.open("w", encoding="utf-8") as mf:
        for p in metrics_files:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(d, dict):
                    raise ValueError("metrics.json is not a dict")

                d["_metrics_path"] = str(p)
                mf.write(json.dumps({"metrics_path": str(p)}, ensure_ascii=False) + "\n")

                missing = sorted([k for k in required_core if k not in d])
                if missing:
                    bad.append(
                        {
                            "metrics_path": str(p),
                            "missing_keys": missing,
                        }
                    )
                    # still include row; but you'll see it in bad_metrics.jsonl

                rows.append(d)
            except Exception as e:
                bad.append({"metrics_path": str(p), "error": str(e)})

    # write bad list if any
    if bad:
        with bad_path.open("w", encoding="utf-8") as bf:
            for rec in bad:
                bf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # deterministic order
    rows.sort(key=_row_sort_key)

    out_csv = out_dir / "table_A_xjailbreak.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})

    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", type=str, default="./artifacts/baselines/xjailbreak")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    out = aggregate_xjailbreak(
        artifacts_root=Path(args.artifacts_root),
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(f"[{METHOD.upper()}] wrote: {out}")


if __name__ == "__main__":
    main()
