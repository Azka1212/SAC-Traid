# src/baselines/pair/aggregate.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_METHOD = "pair"


def _find_metrics_files(artifacts_root: Path) -> List[Path]:
    models_root = artifacts_root / "models"
    return sorted(models_root.glob("*/" "*"/ "run_*" / "metrics.json"))


def aggregate_pair(artifacts_root: Path, out_dir: Optional[Path] = None) -> None:
    artifacts_root = artifacts_root.resolve()
    out_dir = (artifacts_root / "tables" / _METHOD) if out_dir is None else out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = _find_metrics_files(artifacts_root)

    rows: List[Dict[str, Any]] = []
    manifest_path = out_dir / "manifest_metrics.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for p in metrics_files:
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            m["_metrics_path"] = str(p)
            rows.append(m)
            mf.write(json.dumps({"metrics_path": str(p)}, ensure_ascii=False) + "\n")

    # A1: per target summary
    a1_path = out_dir / "table_A1_pair.csv"
    a1_fields = [
        "target_short_name",
        "query_budget",
        "repeat_idx",
        "num_pairs",
        "num_success",
        "asr",
        "qps",
        "avg_success",
        "avg_stealth",
        "avg_novelty",
        "avg_efficiency",
        "avg_impact",
        "total_runtime_sec",
        "runtime_per_pair",
        "avg_queries_per_pair",
    ]
    with a1_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=a1_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in a1_fields})

    # A2: just a lighter view
    a2_path = out_dir / "table_A2_pair.csv"
    a2_fields = ["target_short_name", "query_budget", "asr", "qps", "query_efficiency"]
    with a2_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=a2_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in a2_fields})

    print(f"[{_METHOD.upper()}] Wrote {a1_path}")
    print(f"[{_METHOD.upper()}] Wrote {a2_path}")
    print(f"[{_METHOD.upper()}] Wrote {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate PAIR runs into Part A CSV tables.")
    ap.add_argument("--artifacts-root", type=str, default="artifacts/baselines/pair")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    aggregate_pair(Path(args.artifacts_root), Path(args.out_dir) if args.out_dir else None)


if __name__ == "__main__":
    main()
