# src/models/router.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests


class RouterClient:
    """
    Minimal convenience client for an OpenAI-compatible router.

    Notes:
    - Keeps response keys consistent with BaseLLM.generate(): text/usage/latency_ms/model/raw
    - Does NOT implement retries here (RouterLLM already does). This is a thin helper.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or os.getenv("ROUTER_BASE") or "http://localhost:4000").rstrip("/")
        self.api_key = api_key or os.getenv("ROUTER_API_KEY") or "dummy"
        self.timeout = float(timeout)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}/v1/models",
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        payload.update(kwargs)

        t0 = time.time()
        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        latency_ms = int((time.time() - t0) * 1000)

        r.raise_for_status()
        data = r.json()

        # Keep this robust to missing/odd response shapes.
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        return {
            "text": text,
            "latency_ms": latency_ms,
            "usage": data.get("usage", {}),
            "model": model,
            "raw": data,
        }
