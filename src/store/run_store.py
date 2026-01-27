# src/store/run_store.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import csv, json, platform, re, time
import hashlib
import os


# ----------------------------
# Helpers
# ----------------------------
def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", str(text).strip().lower()).strip("_")


def _model_hash(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def env_fingerprint() -> Dict[str, Any]:
    try:
        import torch  # optional
        torch_v = torch.__version__
        cuda_av = torch.cuda.is_available()
        cuda_dev = torch.cuda.get_device_name(0) if cuda_av else None
    except ImportError:
        torch_v, cuda_av, cuda_dev = None, None, None
    except Exception:
        torch_v, cuda_av, cuda_dev = None, None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_v,
        "cuda_available": cuda_av,
        "cuda_device": cuda_dev,
        "time": datetime.now().isoformat(),
    }


def _jsonify(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


# ----------------------------
# CSV Schema (A0-ready)
# ----------------------------
CSV_FIELDS = [
    "time",
    "phase",
    "seed_id",
    "pair_id",
    "category",
    "operator",
    "step",
    "turn",
    "query_count",
    "sliders",
    "scores",
    "reward",
    "meta",
    "prompt_text",
    "target_id",
    "response_meta",
    "dataset",
    "dataset_tag",
]


class RunStore:
    """
    A0 goal structure:

      artifacts/runs/<framework>/<model_slug>/<bucket>/<run_id>/

    - framework comes from env: SAC_TRIAD_FRAMEWORK
    - bucket comes from env: SAC_TRIAD_BUCKET   (train|test)
    - run_id comes from env: SAC_TRIAD_RUN_ID  (already in your system)

    If no env vars are set, it falls back to legacy:

      artifacts/runs/<model_slug>/<run_id-or-timestamp>/
    """

    def __init__(
        self,
        runs_root: Path,
        model_name: str,
        reuse_latest: bool = True,
        *,
        run_id: Optional[str] = None,
        redact_in_csv: bool = False,
    ):
        self.model_name = model_name
        self.model_slug = _slugify(model_name.replace("/", "_").replace(":", "_"))
        self.runs_root = Path(runs_root)
        self.redact_in_csv = redact_in_csv

        # --- NEW (A0 routing) ---
        env_framework = os.environ.get("SAC_TRIAD_FRAMEWORK") or os.environ.get("SAC_TRIAD_RUN_FRAMEWORK")
        env_bucket = os.environ.get("SAC_TRIAD_BUCKET") or os.environ.get("SAC_TRIAD_RUN_BUCKET")

        fw_slug = _slugify(env_framework) if env_framework else ""
        bucket_slug = _slugify(env_bucket) if env_bucket else ""

        # Base model root
        # If framework is set => runs_root/framework/model_slug/(bucket)
        # Else => runs_root/model_slug (legacy)
        if fw_slug:
            base = self.runs_root / fw_slug / self.model_slug
            if bucket_slug:
                base = base / bucket_slug
            self.model_root = base
        else:
            self.model_root = self.runs_root / self.model_slug

        self.model_root.mkdir(parents=True, exist_ok=True)

        # run_id priority: explicit arg > env var > None
        env_run_id = os.environ.get("SAC_TRIAD_RUN_ID")
        self.run_id = run_id or env_run_id

        # Directory selection
        if self.run_id:
            self.root = self.model_root / _slugify(self.run_id)
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            # Legacy timestamp folders
            if reuse_latest:
                subdirs = [p for p in self.model_root.iterdir() if p.is_dir()]
                if subdirs:
                    self.root = max(subdirs, key=lambda p: p.stat().st_mtime)
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    self.root = self.model_root / ts
                    self.root.mkdir(parents=True, exist_ok=True)
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                self.root = self.model_root / ts
                self.root.mkdir(parents=True, exist_ok=True)

        # keep both for compatibility
        self.run_path: Path = self.root

        # Files
        self.jsonl_path = self.root / "events.jsonl"
        self.csv_path = self.root / "events.csv"
        self.schema_path = self.root / "schema.json"

        schema = {"csv_fields": CSV_FIELDS, "notes": "Flat schema; nested fields JSON-encoded."}
        self.schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        file_exists = self.csv_path.exists()
        self._csv_f = self.csv_path.open("a", newline="", encoding="utf-8")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            self._csv_w.writeheader()
            self._csv_f.flush()

    def write_meta(self, *args, **kwargs):
        meta_path = self.root / "run_meta.json"
        current: Dict[str, Any] = {}

        if meta_path.exists():
            try:
                current = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}

        current.setdefault("run_ts", self.root.name)
        current.setdefault("run_id", self.root.name)
        current.setdefault("model_name", self.model_name)
        current.setdefault("model_id_hash", _model_hash(self.model_name))
        current.setdefault("environment", env_fingerprint())

        # NEW: record structured routing (helps debugging later)
        current.setdefault("framework", os.environ.get("SAC_TRIAD_FRAMEWORK"))
        current.setdefault("bucket", os.environ.get("SAC_TRIAD_BUCKET"))

        for k, v in kwargs.items():
            current[k] = str(v) if isinstance(v, Path) else v

        meta_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_event(self, event: Dict[str, Any]):
        event.setdefault("time", datetime.now().isoformat())

        with self.jsonl_path.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(event, ensure_ascii=False) + "\n")

        meta = event.get("meta") or {}
        flat = {
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
            "prompt_text": (None if self.redact_in_csv else event.get("prompt_text")),
            "target_id": event.get("target_id"),
            "response_meta": _jsonify(event.get("response_meta")),
            "dataset": event.get("dataset"),
            "dataset_tag": event.get("dataset_tag"),
        }

        self._csv_w.writerow(flat)
        self._csv_f.flush()

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
    ):
        row: Dict[str, Any] = {
            "phase": "rewrite",
            "seed_id": seed_id,
            "category": category,
            "operator": operator,
            "sliders": sliders,
            "meta": meta,
        }
        if prompt_text is not None:
            row["prompt_text"] = prompt_text
        if pair_id is not None:
            row["pair_id"] = pair_id
        if dataset is not None:
            row["dataset"] = dataset
        if dataset_tag is not None:
            row["dataset_tag"] = dataset_tag
        if step is not None:
            row["step"] = step
        if turn is not None:
            row["turn"] = turn
        if query_count is not None:
            row["query_count"] = query_count

        self.append_event(row)

    def close(self):
        try:
            self._csv_f.close()
        except Exception:
            pass
