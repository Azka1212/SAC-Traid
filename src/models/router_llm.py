# src/models/router_llm.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests

from .llm_base import BaseLLM, Message, sanitize_decoding


def _normalize_base_url(base_url: Optional[str]) -> str:
    """
    Accept:
      - http://localhost:4000
      - http://localhost:4000/v1
    Return a base that includes /v1
    """
    url = (base_url or os.environ.get("ROUTER_BASE_URL") or "http://localhost:4000").rstrip("/")
    if url.endswith("/v1"):
        return url
    return url + "/v1"


class RouterLLM(BaseLLM):
    """
    OpenAI-compatible router client.

    Expects an OpenAI-style endpoint:
      POST {base_url}/chat/completions

    Returns:
      {
        "text": str,
        "usage": dict,
        "latency_ms": int,
        "model": str,
        "raw": Any
      }
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
    ):
        super().__init__(model_id=model_id)
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key if api_key is not None else os.environ.get("ROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.timeout_s = float(timeout_s)

    def generate(self, messages: List[Message], **decoding) -> Dict[str, Any]:
        t0 = time.time()

        dec = sanitize_decoding(dict(decoding or {}))
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": dec.get("temperature"),
            "top_p": dec.get("top_p"),
            "max_tokens": dec.get("max_tokens"),
        }
        # optional stop
        if "stop" in dec:
            payload["stop"] = dec["stop"]

        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()
        raw = resp.json()

        # Parse output text
        text = ""
        try:
            # OpenAI chat format
            text = raw["choices"][0]["message"]["content"]
        except Exception:
            # fallback formats
            try:
                text = raw["choices"][0].get("text", "")
            except Exception:
                text = ""

        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}

        t1 = time.time()
        return {
            "text": str(text),
            "usage": usage if isinstance(usage, dict) else {},
            "latency_ms": int(round((t1 - t0) * 1000.0)),
            "model": self.model_id,
            "raw": raw,
        }
