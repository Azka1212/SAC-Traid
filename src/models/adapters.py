# src/models/adapters.py
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from .llm_base import BaseLLM, Message
from .ollama_llm import OllamaLLM
from .router_llm import RouterLLM


# -----------------------------------------------------------------------------
# LLM factory
# -----------------------------------------------------------------------------


def _strip_ollama_prefix(model_id: str) -> str:
    """
    Accept either:
      - "ollama/llama3:instruct"  -> "llama3:instruct"
      - "llama3:instruct"         -> "llama3:instruct"
    """
    return model_id.split("/", 1)[1] if model_id.startswith("ollama/") else model_id


def make_llm(
    model_id: str,
    *,
    router_base: Optional[str] = None,
    backend: Optional[str] = None,
) -> BaseLLM:
    """
    Factory:
      - default -> RouterLLM (OpenAI-compatible router)
      - "ollama/*" or backend="ollama_direct" -> OllamaLLM

    NOTE: This routing behavior is part of the pipeline contract.
    """
    # Auto-route Ollama models by prefix
    if model_id.startswith("ollama/"):
        mid = _strip_ollama_prefix(model_id)
        print(
            f"[adapters.make_llm] Using OllamaLLM for {model_id} -> {mid}",
            file=sys.stderr,
        )
        return OllamaLLM(mid)

    # Explicit override to talk to Ollama directly
    if (backend or "").lower() == "ollama_direct":
        mid = _strip_ollama_prefix(model_id)
        print(
            f"[adapters.make_llm] Using OllamaLLM (backend override) {model_id} -> {mid}",
            file=sys.stderr,
        )
        return OllamaLLM(mid)

    # Default: router
    print(
        f"[adapters.make_llm] Using RouterLLM for {model_id} base={router_base}",
        file=sys.stderr,
    )
    return RouterLLM(model_id, base_url=router_base)


# -----------------------------------------------------------------------------
# Adapters (pipeline-facing wrappers)
# -----------------------------------------------------------------------------


class TargetAdapter:
    """Thin wrapper with conservative defaults for target models."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, prompt_text: str, **decoding) -> Dict[str, Any]:
        messages: List[Message] = [{"role": "user", "content": prompt_text}]
        defaults = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 512}
        defaults.update(decoding)
        return self.llm.generate(messages, **defaults)


class RewriterAdapter:
    """
    Adapter used by PromptRewriter.

    Accepts either:
      - an already-built llm, OR
      - kwargs (model_id/backend/router_base/use_chat_api) like PromptRewriter passes.

    This adapter supports both:
      - generate(messages, **decoding)  where messages is OpenAI-style list[Message]
      - generate(prompt_text, **decoding) where prompt_text is str/Any
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        *,
        model_id: Optional[str] = None,
        backend: Optional[str] = None,
        router_base: Optional[str] = None,
        use_chat_api: bool = True,
    ):
        if llm is not None:
            self.llm = llm
        else:
            if not model_id:
                raise ValueError("RewriterAdapter requires either llm or model_id.")
            # Let make_llm decide Router vs Ollama. If model_id starts with 'ollama/',
            # it will use OllamaLLM and NOT hit your router at :4000.
            self.llm = make_llm(model_id=model_id, router_base=router_base, backend=backend)

        # NOTE: stored for compatibility / future switching behavior (do not remove).
        self.use_chat_api = use_chat_api

    def generate(self, payload: Any, **decoding) -> Dict[str, Any]:
        # The rewriter prefers .generate(messages, **decoding).
        # Support both chat payloads (list of messages) and plain prompt strings.
        if isinstance(payload, list):
            messages: List[Message] = payload  # already OpenAI-style messages
        else:
            messages = [{"role": "user", "content": str(payload)}]

        defaults = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256}
        defaults.update(decoding)
        return self.llm.generate(messages, **defaults)

    # Back-compat if caller tries .chat(...) or .complete(...)
    def chat(self, messages: List[Message], **decoding) -> Dict[str, Any]:
        return self.generate(messages, **decoding)

    def complete(self, prompt_text: str, **decoding) -> Dict[str, Any]:
        return self.generate([{"role": "user", "content": prompt_text}], **decoding)


class JudgeAdapter:
    """Thin wrapper for judge models; mirrors TargetAdapter style."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def judge(self, prompt_and_response: str, **decoding) -> Dict[str, Any]:
        messages: List[Message] = [{"role": "user", "content": prompt_and_response}]
        defaults = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256}
        defaults.update(decoding)
        return self.llm.generate(messages, **defaults)
