# src/ablations/baseline/b2/b2_run_store.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import csv
import json

# Reuse helpers from the main RunStore, but we do NOT reuse its directory logic.
from src.store.run_store import env_fingerprint, _jsonify, CSV_FIELDS


class B2RunStore:
    """
    Minimal, self-contained run store for B2 ablations.

    - No A0 framework/bucket routing.
    - No automatic run-id discovery.
    - Caller passes an explicit run_root directory.
    - Always writes inside that run_root:

        run_root/
          events.jsonl
          events.csv
          run_meta.json
          schema.json

    The CSV schema is identical to the main RunStore's CSV_FIELDS so that
    downstream tooling (e.g., A0, metrics) can consume it if needed.
    """

    def __init__(self, run_root: Path, *, redact_in_csv: bool = False) -> None:
        self.root = Path(run_root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.redact_in_csv = bool(redact_in_csv)

        # Paths
        self.jsonl_path = self.root / "events.jsonl"
        self.csv_path = self.root / "events.csv"
        self.schema_path = self.root / "schema.json"
        self.meta_path = self.root / "run_meta.json"

        # Write a simple schema file (once)
        if not self.schema_path.exists():
            schema = {"csv_fields": CSV_FIELDS, "notes": "Flat schema for B2; nested fields JSON-encoded."}
            self.schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        # Open CSV writer (append mode)
        file_exists = self.csv_path.exists()
        self._csv_f = self.csv_path.open("a", newline="", encoding="utf-8")
        self._csv_w = csv.DictWriter(
            self._csv_f,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        if not file_exists:
            self._csv_w.writeheader()
            self._csv_f.flush()

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    def write_meta(self, *, model_name: str, **extra: Any) -> None:
        """
        Write a small run_meta.json with environment fingerprint and any extra
        fields passed in (e.g., reward_mode, seed, target_model, etc.).
        """
        current: Dict[str, Any] = {}

        if self.meta_path.exists():
            try:
                current = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}

        current.setdefault("run_dir", self.root.as_posix())
        current.setdefault("model_name", model_name)
        current.setdefault("environment", env_fingerprint())

        for k, v in extra.items():
            # convert Paths to strings for JSON
            if isinstance(v, Path):
                current[k] = v.as_posix()
            else:
                current[k] = v

        self.meta_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------
    def append_event(self, event: Dict[str, Any]) -> None:
        """
        Append a single event to events.jsonl and a flattened row to events.csv.

        Expected keys (not all required):
          phase, seed_id, pair_id, category, operator,
          sliders, scores, reward, meta, prompt_text,
          target_id, response_meta, dataset, dataset_tag, step, turn, query_count
        """
        event = dict(event)  # shallow copy so we can modify safely
        event.setdefault("time", datetime.now().isoformat())

        # --- JSONL ---
        with self.jsonl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(event, ensure_ascii=False) + "\n")

        # --- CSV (flattened) ---
        meta = event.get("meta") or {}

        row: Dict[str, Any] = {
            "time": event.get("time"),
            "phase": event.get("phase"),
            "seed_id": event.get("seed_id"),
            "pair_id": event.get("pair_id") or meta.get("pair_id"),
            "category": event.get("category"),
            "operator": event.get("operator"),
            "step": event.get("step") or meta.get("step"),
            "turn": event.get("turn") or meta.get("turn"),
            "query_count": event.get("query_count") or meta.get("query_count"),
            "sliders": _jsonify(event.get("sliders")),
            "scores": _jsonify(event.get("scores") or meta.get("scores")),
            "reward": event.get("reward"),
            "meta": _jsonify(meta),
            # optionally redact prompt text for privacy
            "prompt_text": (None if self.redact_in_csv else event.get("prompt_text")),
            "target_id": event.get("target_id"),
            "response_meta": _jsonify(event.get("response_meta")),
            "dataset": event.get("dataset"),
            "dataset_tag": event.get("dataset_tag"),
        }

        self._csv_w.writerow(row)
        self._csv_f.flush()

    # Convenience wrapper (optional)
    def append_rewrite_event(
        self,
        *,
        seed_id: Any,
        category: Any,
        operator: str,
        sliders: Dict[str, Any],
        meta: Dict[str, Any],
        prompt_text: Optional[str] = None,
        pair_id: Optional[str] = None,
        dataset: Optional[str] = None,
        dataset_tag: Optional[str] = None,
        step: Optional[int] = None,
        turn: Optional[int] = None,
        query_count: Optional[int] = None,
    ) -> None:
        event: Dict[str, Any] = {
            "phase": "rewrite",
            "seed_id": seed_id,
            "category": category,
            "operator": operator,
            "sliders": sliders,
            "meta": meta,
        }
        if prompt_text is not None:
            event["prompt_text"] = prompt_text
        if pair_id is not None:
            event["pair_id"] = pair_id
        if dataset is not None:
            event["dataset"] = dataset
        if dataset_tag is not None:
            event["dataset_tag"] = dataset_tag
        if step is not None:
            event["step"] = step
        if turn is not None:
            event["turn"] = turn
        if query_count is not None:
            event["query_count"] = query_count

        self.append_event(event)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._csv_f.close()
        except Exception:
            pass
