# src/rl/utils/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable
import csv
import json
import os
import time

import torch


def nowstr() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class CSVLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fieldnames = list(fieldnames)

        self._f = path.open("a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)

        # If new file, write header
        if path.stat().st_size == 0:
            self._w.writeheader()
            self._f.flush()

    def log(self, row: Dict[str, Any]) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    # Optional convenience: allows `with CSVLogger(...) as lg: ...`
    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")
