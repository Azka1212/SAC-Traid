# src/models/ollama_llm.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import ConnectionError, ReadTimeout

from .llm_base import BaseLLM, Message, sanitize_decoding


def _strip_ollama_prefix(model_id: str) -> str:
    """Allow either 'ollama/llama3:instruct' or 'llama3:instruct'."""
    return model_id.split("ollama/", 1)[-1] if model_id.startswith("ollama/") else model_id


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


class OllamaLLM(BaseLLM):
    """
    Talks directly to Ollama's REST API (no router).
    Uses /api/generate and collapses chat messages into a single prompt.
    """

    def __init__(
        self,
        model_id: str,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        timeout: kept for backward compatibility, but defaults can be overridden via env:

          OLLAMA_CONNECT_TIMEOUT (seconds)  default: 10
          OLLAMA_READ_TIMEOUT    (seconds)  default: max(timeout, 600)
          OLLAMA_HTTP_RETRIES                 default: 2
          OLLAMA_HTTP_BACKOFF                 default: 2.0

        Why:
        - connect timeout should be short (server up/down)
        - read timeout must be long enough for slow generations / cold starts
        """
        super().__init__(model_id)
        self.short_id = _strip_ollama_prefix(model_id)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE", "http://localhost:11434")).rstrip(
            "/"
        )

        # Backward compatible: if caller passes timeout=120 it still works,
        # but we increase read timeout by default to reduce spurious failures.
        connect_timeout = _env_float("OLLAMA_CONNECT_TIMEOUT", 10.0)

        # If user didn’t override anything, prefer a safer default read timeout (10 min)
        default_read = max(float(timeout), 600.0)
        read_timeout = _env_float("OLLAMA_READ_TIMEOUT", default_read)

        self.timeout: Tuple[float, float] = (connect_timeout, read_timeout)

        self.retries = max(0, _env_int("OLLAMA_HTTP_RETRIES", 2))
        self.backoff = max(0.0, _env_float("OLLAMA_HTTP_BACKOFF", 2.0))

    @staticmethod
    def _messages_to_prompt(messages: List[Message]) -> str:
        """Simple, deterministic join. Good enough for adapters layer."""
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate(self, messages: List[Message], **decoding) -> Dict[str, Any]:
        d = sanitize_decoding(decoding)

        payload: Dict[str, Any] = {
            "model": self.short_id,
            "prompt": self._messages_to_prompt(messages),
            "stream": False,
            "options": {
                "temperature": d["temperature"],
                "top_p": d["top_p"],
                "num_predict": d["max_tokens"],
            },
        }

        # Ollama supports stop as either a string or list; we pass through.
        if "stop" in d:
            payload["stop"] = d["stop"]

        url = f"{self.base_url}/api/generate"
        t0 = time.time()

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()

                data = r.json()
                text = data.get("response", "")

                # Ollama may not return token counts; keep fields consistent.
                usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

                return {
                    "text": text,
                    "usage": usage,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "model": self.model_id,
                    "raw": data,
                }

            except (ReadTimeout, ConnectionError) as e:
                last_err = e
                if attempt < self.retries:
                    # exponential backoff: 2,4,8... seconds (configurable base)
                    sleep_s = self.backoff * (2**attempt)
                    time.sleep(sleep_s)
                    continue
                raise

            except Exception as e:
                # Don't retry unknown errors by default
                raise

        # should be unreachable
        if last_err is not None:
            raise last_err
        raise RuntimeError("OllamaLLM.generate failed unexpectedly")
