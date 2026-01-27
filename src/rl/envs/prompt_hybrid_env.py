# src/rl/envs/prompt_hybrid_env.py
from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.data.loader import (
    load_operators_for_m5,
    load_seeds_for_m5,
    validate_m5_ready,
    _load_seeds_jsonl,  # keep only if this file actually calls it; otherwise remove this name
)
from src.config import load_app_config
from src.store.run_store import RunStore

from src.models.utils import extract_text_from_gen
from src.rewriter.rewriter_llm import PromptRewriter, RewriterInput
from src.judge.judge_llm import Judge, JudgeInput
from src.reward.rewarder import RewardCalculator, RewardInput


def _hash01(s: str) -> float:
    h = hashlib.sha1((s or "").encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 10_000) / 10_000.0


def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


class PromptHybridEnv(gym.Env):
    """
    One-step episode environment that wraps your LLM pipeline.

    Train = ID/SORRY (default in-run seeds/operators)
    Test  = OOD (via seeds_path_override), operators remain SORRY.

    All events include `dataset_tag` for filtering/metrics.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str = "config/stack.yaml",
        runs_model: Optional[str] = None,
        persist_raw: Optional[bool] = None,
        redact_in_csv: Optional[bool] = None,
        seed: int = 0,
        **kwargs,
    ):
        # runtime overrides
        rewriter_model: Optional[str] = kwargs.pop("rewriter_model", None)
        judge_model: Optional[str] = kwargs.pop("judge_model", None)
        target_model: Optional[str] = kwargs.pop("target_model", None)
        use_target_echo: bool = bool(kwargs.pop("use_target_echo", False))
        seeds_path_override: Optional[str] = kwargs.pop("seeds_path_override", None)
        categories_path_override: Optional[str] = kwargs.pop("categories_path_override", None)  # reserved; not used yet
        dataset_tag: Optional[str] = kwargs.pop("dataset_tag", None)
        kwargs.pop("render_mode", None)  # ignore gym kw

        super().__init__()
        self.rng = random.Random(seed)
        self.cfg = load_app_config(config_path)

        runs_root = self.cfg["paths"]["runs"]
        if runs_model is None:
            tgts = self.cfg.get("targets", [])
            if tgts:
                t0 = tgts[0]
                runs_model = (t0.get("id") if isinstance(t0, dict) else str(t0))
            else:
                runs_model = "unspecified"

        if persist_raw is None:
            persist_raw = bool(self.cfg.get("logging", {}).get("persist_raw_prompts", True))
        if redact_in_csv is None:
            redact_in_csv = bool(self.cfg.get("logging", {}).get("redact_in_csv", False))

        self.persist_raw = persist_raw
        self.rs = RunStore(runs_root=runs_root, model_name=runs_model, redact_in_csv=redact_in_csv)

        # ---------- Seeds & Operators ----------
        self._init_seeds_and_ops(
            seeds_path_override=seeds_path_override,
            dataset_tag=dataset_tag,
        )

        # Local (mutable) copy of config to apply overrides
        cfg_local = deepcopy(self.cfg)
        if rewriter_model:
            cfg_local.setdefault("rewriter", {})["model_id"] = rewriter_model
        if judge_model:
            cfg_local.setdefault("judge", {})["model_id"] = judge_model

        # Components
        self.rewriter = PromptRewriter(cfg_local, self.rs)
        self.judge = Judge(cfg_local, self.rs)
        self.r_calc = RewardCalculator(self.cfg)

        # Optional target echo
        self.use_target_echo = bool(use_target_echo)
        self.target = None
        self.target_model_id: Optional[str] = None
        self._init_target_echo(target_model=target_model)

        # Observation and Action spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Dict(
            {
                "op": spaces.Discrete(self.num_ops),
                "cont": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )

        self._ctx: Optional[Dict[str, Any]] = None
        self._seen_prompt_hashes: set[str] = set()

    # -------------------------
    # Init helpers (structure only)
    # -------------------------
    def _init_seeds_and_ops(self, *, seeds_path_override: Optional[str], dataset_tag: Optional[str]) -> None:
        if seeds_path_override:
            # OOD: keep SORRY operators from the run; load seeds externally
            seeds_path = Path(seeds_path_override)
            if not seeds_path.exists():
                raise FileNotFoundError(f"[env] seeds_path_override not found: {seeds_path}")

            self.ops = load_operators_for_m5(self.rs)
            self.op_names = [o["name"] for o in self.ops]
            self.num_ops = len(self.op_names)

            self.seeds = _load_seeds_jsonl(seeds_path)
            self.dataset_tag = dataset_tag or f"OOD::{seeds_path.stem}"
            return

        # ID train: use SORRY snapshot seeds/operators
        validate_m5_ready(self.rs)
        self.seeds = load_seeds_for_m5(self.rs)
        self.ops = load_operators_for_m5(self.rs)
        self.op_names = [o["name"] for o in self.ops]
        self.num_ops = len(self.op_names)
        self.dataset_tag = dataset_tag or "ID-SORRY"

    def _init_target_echo(self, *, target_model: Optional[str]) -> None:
        if not self.use_target_echo:
            return

        if not target_model:
            tgts = self.cfg.get("targets", [])
            if tgts:
                t0 = tgts[0]
                target_model = (t0.get("id") if isinstance(t0, dict) else str(t0))

        if target_model:
            from src.models.adapters import make_llm, TargetAdapter
            router_base = self.cfg.get("router", {}).get("base_url")
            llm = make_llm(model_id=target_model, router_base=router_base)
            self.target = TargetAdapter(llm)
            self.target_model_id = target_model

    # -------------------------
    # Small utilities
    # -------------------------
    def _pair_id(self, seed_id: Any, op_name: str) -> str:
        return f"{seed_id}::{op_name}"

    def _obs(self, seed_ex: Dict[str, Any], op_name_hint: str) -> np.ndarray:
        cat = str(seed_ex.get("category", "Uncategorized") or "Uncategorized")
        text = str(seed_ex.get("text", "") or "")
        cat_h = _hash01(cat)
        op_h = _hash01(op_name_hint)
        max_chars = int(self.cfg.get("logging", {}).get("max_prompt_chars", 8000))
        seed_len_norm = min(len(text) / max_chars, 1.0)
        return np.array([cat_h, op_h, seed_len_norm, 1.0], dtype=np.float32)

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng.seed(seed)

        seed_ex = self.rng.choice(self.seeds)
        op_hint = self.rng.choice(self.op_names)  # used only as a feature hint
        obs = self._obs(seed_ex, op_hint)

        self._ctx = {"seed": seed_ex}
        info = {
            "seed_id": seed_ex.get("question_id"),
            "category": seed_ex.get("category"),
            "dataset_tag": self.dataset_tag,
        }
        return obs, info

    def step(self, action: Dict[str, Any]):
        assert self._ctx is not None, "Call reset() first."

        op_idx = int(action["op"])
        op_idx = max(0, min(self.num_ops - 1, op_idx))
        op_name = self.op_names[op_idx]

        cont = np.asarray(action["cont"], dtype=np.float32)
        cont = np.clip(cont, 0.0, 1.0)
        sliders = {"length": float(cont[0]), "temperature": float(cont[1]), "persona": float(cont[2])}

        seed_ex = self._ctx["seed"]

        # 1) Rewriter
        try:
            rin = RewriterInput(
                seed_id=seed_ex.get("question_id"),
                category=seed_ex.get("category"),
                seed_text=seed_ex.get("text", ""),
                operator_name=op_name,
                sliders=sliders,
            )
            r = self.rewriter.rewrite(rin)

            ev = {
                "phase": "rewrite",
                "seed_id": rin.seed_id,
                "category": rin.category,
                "operator": op_name,
                "sliders": sliders,
                "meta": r.meta,
                "dataset_tag": self.dataset_tag,
                "pair_id": self._pair_id(rin.seed_id, op_name),
            }
            if self.persist_raw:
                ev["prompt_text"] = r.text
            self.rs.append_event(ev)

            prompt_text = r.text or ""
        except Exception as e:
            info = {"error_stage": "rewrite", "error": str(e), "dataset_tag": self.dataset_tag}
            return self._obs(seed_ex, op_name), 0.0, True, False, info

        # 2) Optional target echo
        target_calls = 0
        target_text = ""
        target_model_used = self.target_model_id
        if self.use_target_echo and self.target is not None:
            try:
                out = self.target.generate(prompt_text, temperature=0.2, max_tokens=128)
                target_calls += 1
                target_text = extract_text_from_gen(out)
                target_model_used = out.get("model") or target_model_used

                ev = {
                    "phase": "rewrite_target_echo",
                    "target_id": out.get("model"),
                    "seed_id": rin.seed_id,
                    "operator": op_name,
                    "response_meta": {"len_tokens": out.get("usage", {}).get("completion_tokens")},
                    "dataset_tag": self.dataset_tag,
                    "pair_id": self._pair_id(rin.seed_id, op_name),
                }
                if self.persist_raw:
                    ev["target_text"] = target_text
                self.rs.append_event(ev)
            except Exception as e:
                self.rs.append_event(
                    {
                        "phase": "rewrite_target_error",
                        "seed_id": rin.seed_id,
                        "operator": op_name,
                        "response_meta": {"error": str(e)},
                        "dataset_tag": self.dataset_tag,
                    }
                )

        # 3) Judge
        j_latency = None
        j_parsed_ok = None
        try:
            jin = JudgeInput(
                prompt_text=prompt_text,
                category=rin.category,
                seed_id=rin.seed_id,
                operator=op_name,
                target_response=target_text if target_text else None,
            )
            jres = self.judge.score(jin)
            j_latency = jres.latency_ms
            j_parsed_ok = bool(jres.parsed_ok)

            # NOTE: keeping this line because it's in your code (even if unused)
            judge_alias = {k.lower(): float(v) for k, v in (jres.scores or {}).items()}

            self.rs.append_event(
                {
                    "phase": "judge",
                    "seed_id": rin.seed_id,
                    "category": rin.category,
                    "operator": op_name,
                    "scores": jres.scores,
                    "response_meta": {
                        "latency_ms": jres.latency_ms,
                        "model": jres.model_id,
                        "parsed_ok": jres.parsed_ok,
                    },
                    "dataset_tag": self.dataset_tag,
                    "pair_id": self._pair_id(rin.seed_id, op_name),
                }
            )
        except Exception as e:
            info = {
                "error_stage": "judge",
                "error": str(e),
                "prompt_text": prompt_text,
                "target_text": target_text,
                "target_model": target_model_used,
                "target_calls": target_calls,
                "dataset_tag": self.dataset_tag,
            }
            return self._obs(seed_ex, op_name), 0.0, True, False, info

        # 4) Reward
        try:
            p_hash = _sha1_text(prompt_text)
            seen_before = p_hash in self._seen_prompt_hashes
            self._seen_prompt_hashes.add(p_hash)

            rr = self.r_calc.compute(
                RewardInput(
                    prompt_text=prompt_text,
                    scores=jres.scores,
                    calls=target_calls,
                    seen_before=seen_before,
                    length=len(prompt_text or ""),
                )
            )
            self.rs.append_event(
                {
                    "phase": "reward",
                    "seed_id": rin.seed_id,
                    "category": rin.category,
                    "operator": op_name,
                    "reward": rr.reward,
                    "base_score": rr.base_score,
                    "penalties": rr.penalties,
                    "detail": rr.detail,
                    "dataset_tag": self.dataset_tag,
                    "pair_id": self._pair_id(rin.seed_id, op_name),
                }
            )
            reward = float(rr.reward)
        except Exception as e:
            info = {
                "error_stage": "reward",
                "error": str(e),
                "prompt_text": prompt_text,
                "target_text": target_text,
                "target_model": target_model_used,
                "target_calls": target_calls,
                "dataset_tag": self.dataset_tag,
            }
            return self._obs(seed_ex, op_name), 0.0, True, False, info

        obs = self._obs(seed_ex, op_name)
        info = {
            "scores": jres.scores,
            "seed_id": rin.seed_id,
            "category": rin.category,
            "operator": op_name,
            "sliders": sliders,
            "prompt_text": prompt_text,
            "target_text": target_text,
            "target_model": target_model_used,
            "judge_parsed_ok": j_parsed_ok,
            "judge_latency_ms": j_latency,
            "target_calls": target_calls,
            "seen_before": seen_before,
            "reward": reward,
            "dataset_tag": self.dataset_tag,
        }
        return obs, reward, True, False, info

    def render(self):
        return None

    def close(self):
        try:
            self.rs.close()
        except Exception:
            pass
