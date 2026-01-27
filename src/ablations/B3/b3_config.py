from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
import time

@dataclass
class B3Paths:
    root: Path
    runs_root: Path
    tables_root: Path

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    return cfg or {}

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_b3_paths(artifacts_root: str) -> B3Paths:
    root = Path(artifacts_root)
    runs_root = root / "runs"
    tables_root = root / "tables"
    ensure_dir(root); ensure_dir(runs_root); ensure_dir(tables_root)
    return B3Paths(root=root, runs_root=runs_root, tables_root=tables_root)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
