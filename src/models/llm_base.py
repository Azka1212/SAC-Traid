# src/models/llm_base.py
from __future__ import annotations

import time
from typing import Any, Dict, List

Message = Dict[str, str]  # {"role": "user"|"system"|"assistant", "content": str}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sanitize_decoding(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept common decoding knobs and clamp them to safe ranges; drop unknown keys.

    Supported keys:
      - temperature: float in [0.0, 1.5]
      - top_p: float in [0.0, 1.0]
      - max_tokens: int in [1, 4096]
      - stop: str | list[str] | None
    """
    allowed_defaults: Dict[str, Any] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 256,
        "stop": None,
    }

    out: Dict[str, Any] = {}
    for k, default in allowed_defaults.items():
        out[k] = kwargs.get(k) if (k in kwargs and kwargs[k] is not None) else default

    out["temperature"] = float(clamp(float(out["temperature"]), 0.0, 1.5))
    out["top_p"] = float(clamp(float(out["top_p"]), 0.0, 1.0))
    out["max_tokens"] = int(max(1, min(int(out["max_tokens"]), 4096)))

    # Keep payload clean: omit stop if unset
    if out.get("stop") is None:
        out.pop("stop", None)

    return out


# -----------------------------------------------------------------------------
# Base interface
# -----------------------------------------------------------------------------


class BaseLLM:
    """
    Normalized interface for any backend.

    Contract:
      generate(messages, **decoding) -> {
          "text": str,
          "usage": dict,           # may be partial depending on backend
          "latency_ms": int,
          "model": str,
          "raw": Any               # raw provider JSON
      }
    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate(self, messages: List[Message], **decoding) -> Dict[str, Any]:
        raise NotImplementedError

    # small helper for latency timing
    def _stamp(self) -> float:
        return time.time()
