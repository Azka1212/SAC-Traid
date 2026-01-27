# src/config.py
from __future__ import annotations
from pathlib import Path
import os
import re
from typing import Any, Dict

import yaml

# Matches ${ENV} and ${ENV:-default}
_ENV_RE = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")

def _expand_env_str(s: str) -> str:
    """Expand ${VAR} or ${VAR:-default} inside a string.

    Semantics match POSIX parameter expansion for ':-':
    - if VAR is unset OR empty -> use default
    - otherwise                -> use value
    """
    def repl(m: re.Match):
        key, default = m.group(1), (m.group(2) or "")
        val = os.environ.get(key)
        return default if (val is None or val == "") else val
    return _ENV_RE.sub(repl, s)

def _expand(obj: Any) -> Any:
    """Recursively expand env vars in all strings within a nested structure."""
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_str(obj)
    return obj

def load_app_config(path: str | Path = "config/stack.yaml") -> Dict[str, Any]:
    """
    Load the application config with environment expansion.

    Supports both:
      - Unified stack.yaml (top-level has an 'app' key)
      - Legacy app.yaml schema (no 'app' key; the whole file is the app config)

    Returns the *app* section as a dict with keys like:
      paths, router, targets, rewriter, judge, logging, etc.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")

    # Safe YAML load, then expand env vars recursively
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = _expand(raw)

    # If this is the unified stack, return only the 'app' subsection
    if isinstance(raw, dict) and "app" in raw and isinstance(raw["app"], dict):
        return raw["app"]

    # Otherwise treat the whole file as the app config (legacy layout)
    if isinstance(raw, dict):
        return raw

    raise ValueError(f"Unsupported config structure in {p}; expected mapping at top level.")
