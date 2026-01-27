# src/models/utils.py
from __future__ import annotations
from typing import Any


def extract_text_from_gen(out: Any) -> str:
    if not isinstance(out, dict):
        return (str(out) if out is not None else "")
    for k in ("text", "content", "response"):
        v = out.get(k)
        if isinstance(v, str) and v.strip():
            return v
    choices = out.get("choices")
    if isinstance(choices, list) and choices:
        ch0 = choices[0]
        if isinstance(ch0, dict):
            msg = ch0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(ch0.get("text"), str):
                return ch0["text"]
    return ""
