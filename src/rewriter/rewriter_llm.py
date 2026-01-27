# src/rewriter/rewriter_llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.models.adapters import RewriterAdapter  # must have .generate(...)


@dataclass
class RewriterInput:
    seed_id: str
    category: str
    seed_text: str
    operator_name: str
    sliders: Dict[str, float]


@dataclass
class RewriterResult:
    text: str
    meta: Dict[str, Any]


class PromptRewriter:
    def __init__(self, cfg: Dict[str, Any], run_store):
        self.cfg = cfg
        self.rs = run_store

        rw = cfg["rewriter"]
        self.client = RewriterAdapter(
            model_id=rw["model_id"],
            backend=rw.get("backend", "router"),
            router_base=cfg.get("router", {}).get("base_url"),
            use_chat_api=rw.get("use_chat_api", True),
        )

        # IMPORTANT: disable any router fallback – we only want the adapter
        self._fallback = None

    def _map_sliders(self, sliders: Dict[str, float]):
        dec = self.cfg["rewriter"]["decoding"]

        tmin, tmax = float(dec["temperature_min"]), float(dec["temperature_max"])
        temp = tmin + (tmax - tmin) * float(sliders["temperature"])

        mn, mx = dec["max_new_tokens"]
        max_new_tokens = int(float(mn) + (float(mx) - float(mn)) * float(sliders["length"]))

        p = float(sliders["persona"])
        persona_key = "neutral" if p < 0.33 else ("formal" if p < 0.66 else "role")
        persona_tag = self.cfg["rewriter"]["persona_tags"][persona_key]

        return temp, max_new_tokens, persona_key, persona_tag

    def _messages(self, seed_text: str, operator_name: str, persona_tag: str) -> List[Dict[str, str]]:
        system = (
            "You rewrite text for surface form only (wording, phrasing, formatting). "
            "Preserve the original meaning and intent. Do not add or remove instructions."
        )
        user = (
            "Apply the following transformation strictly as a style operation:\n"
            f"OPERATOR: {operator_name}\n"
            f"Persona: {persona_tag}\n\n"
            f"Seed text:\n{seed_text}\n\n"
            "Return only the rewritten text."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _single_prompt(self, seed_text: str, operator_name: str, persona_tag: str) -> str:
        return (
            "Rewrite the text for surface form only (wording, phrasing, formatting). "
            "Preserve the original meaning and intent. Do not add or remove instructions.\n\n"
            f"OPERATOR: {operator_name}\n"
            f"Persona: {persona_tag}\n\n"
            f"Seed text:\n{seed_text}\n\n"
            "Return only the rewritten text."
        )

    def _invoke(self, payload: Any, *, temperature: float, max_tokens: int, top_p: float) -> Any:
        # ALWAYS prefer the adapter (must expose .generate)
        if hasattr(self.client, "generate"):
            return self.client.generate(payload, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

        # If your adapter exposes chat/complete instead, keep these as backups:
        if hasattr(self.client, "chat"):
            msgs = payload if isinstance(payload, list) else [{"role": "user", "content": str(payload)}]
            return self.client.chat(msgs, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

        if hasattr(self.client, "complete"):
            prompt = payload if isinstance(payload, str) else str(payload[-1]["content"])
            return self.client.complete(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

        # No adapter method found → explicit error (do NOT fall back to router)
        raise AttributeError("RewriterAdapter does not expose generate/chat/complete")

    def _extract_text(self, out: Any) -> str:
        """Be forgiving about adapter return shapes."""
        if isinstance(out, str):
            return out.strip()

        if isinstance(out, dict):
            txt = out.get("text") or out.get("content") or out.get("response")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()

            choices = out.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0]
                if isinstance(ch0, dict):
                    msg = ch0.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"].strip()
                    if isinstance(ch0.get("text"), str):
                        return ch0["text"].strip()

        # Fallback
        return (str(out) if out is not None else "").strip()

    def rewrite(self, rin: RewriterInput) -> RewriterResult:
        temp, max_new_tokens, persona_key, persona_tag = self._map_sliders(rin.sliders)

        top_p = self.cfg["rewriter"]["decoding"]["top_p"]
        use_chat = bool(self.cfg["rewriter"].get("use_chat_api", True))

        payload = (
            self._messages(rin.seed_text, rin.operator_name, persona_tag)
            if use_chat
            else self._single_prompt(rin.seed_text, rin.operator_name, persona_tag)
        )

        out = self._invoke(
            payload,
            temperature=float(temp),
            max_tokens=int(max_new_tokens),
            top_p=float(top_p),
        )

        text = self._extract_text(out)

        max_chars = self.cfg.get("logging", {}).get("max_prompt_chars")
        if max_chars:
            text = text[: int(max_chars)]

        usage = out.get("usage", {}) if isinstance(out, dict) else {}
        meta = {
            "len_tokens": usage.get("completion_tokens"),
            "style_signature": f"{rin.operator_name}|{persona_key}",
            "persona_key": persona_key,
        }

        return RewriterResult(text=text, meta=meta)
