from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import load_app_config
from src.data.loader import _load_seeds_jsonl, build_all
from src.models.utils import extract_text_from_gen
from src.rewriter.rewriter_llm import PromptRewriter, RewriterInput
from src.judge.judge_llm import Judge, JudgeInput

from .b2_run_store import B2RunStore
from .reward import compute_reward_from_judge  # <-- B2-specific reward


def _hash01(s: str) -> float:
    h = hashlib.sha1((s or "").encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 10_000) / 10_000.0


def _sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


class B2PromptEnv(gym.Env):
    """
    One-step episode environment for B2 reward ablation.

    Differences from PromptHybridEnv:
      - All logging goes through B2RunStore inside run_root.
      - Seeds/operators are snapshotted directly into run_root (no artifacts/runs).
      - Reward is computed via src.ablations.baseline.b2.reward.compute_reward_from_judge
        using B2 YAML fields: reward.mode, base_weights, curriculum, etc.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str = "config/stack.yaml",
        run_root: str = "artifacts/ablations/b2/tmp",
        *,
        seed: int = 0,
        # optional overrides
        rewriter_model: Optional[str] = None,
        judge_model: Optional[str] = None,
        target_model: Optional[str] = None,
        use_target_echo: bool = False,
        seeds_path_override: Optional[str] = None,
        dataset_tag: Optional[str] = None,
        reward_mode: Optional[str] = None,
        # NEW: B2 reward config + curriculum + total steps
        reward_cfg: Optional[Dict[str, Any]] = None,
        curriculum_cfg: Optional[Dict[str, Any]] = None,
        total_env_steps: Optional[int] = None,
        persist_raw: Optional[bool] = None,
        redact_in_csv: Optional[bool] = None,
        **kwargs,
    ):
        # consume any extra Gym kwargs like render_mode
        kwargs.pop("render_mode", None)

        super().__init__()

        self.rng = random.Random(seed)
        self.cfg = load_app_config(config_path)

        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)

        if persist_raw is None:
            persist_raw = bool(self.cfg.get("logging", {}).get("persist_raw_prompts", True))
        if redact_in_csv is None:
            redact_in_csv = bool(self.cfg.get("logging", {}).get("redact_in_csv", False))

        self.persist_raw = persist_raw
        self.rs = B2RunStore(self.run_root, redact_in_csv=redact_in_csv)

        # ---------- Seeds & Operators ----------
        self._init_seeds_and_ops(
            seeds_path_override=seeds_path_override,
            dataset_tag=dataset_tag,
            config_path=config_path,
        )

        # Local mutable copy of config so we can override models
        cfg_local = deepcopy(self.cfg)

        if rewriter_model:
            cfg_local.setdefault("rewriter", {})["model_id"] = rewriter_model
        if judge_model:
            cfg_local.setdefault("judge", {})["model_id"] = judge_model

        # Reward config is now B2-specific; we keep a small dict for compute_reward_from_judge.
        # If reward_cfg / curriculum_cfg are not passed, we fall back to stack config.
        self._reward_cfg = reward_cfg or cfg_local.get("reward", {}) or {}
        self._curriculum_cfg = curriculum_cfg or cfg_local.get("curriculum", {}) or {}

        # Allow explicit reward_mode override (e.g., binary / scalar / full5d)
        if reward_mode is not None:
            self._reward_cfg = dict(self._reward_cfg)  # shallow copy
            self._reward_cfg["mode"] = reward_mode

        # This is the total training steps for curriculum progress
        self._total_env_steps_cfg = int(total_env_steps or 0)

        # Small config object passed into compute_reward_from_judge
        self._reward_cfg_for_fn: Dict[str, Any] = {
            "reward": self._reward_cfg,
            "curriculum": self._curriculum_cfg,
            "sac": {"total_env_steps": self._total_env_steps_cfg},
        }

        # Components
        self.rewriter = PromptRewriter(cfg_local, self.rs)
        self.judge = Judge(cfg_local, self.rs)

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

        # Track total env steps for curriculum progress in B2
        self._env_steps: int = 0

    # -------------------------
    # Seed/operator snapshot
    # -------------------------
    def _ensure_snapshot(self, config_path: str) -> None:
        """
        Ensure that run_root contains SORRY snapshot:
          - seeds.jsonl
          - operators.json
          - categories.json, pacing.json (not strictly required here).

        Uses the same build_all() helper as M2, but writes directly into run_root.
        """
        seeds_path = self.run_root / "seeds.jsonl"
        ops_path = self.run_root / "operators.json"

        if seeds_path.exists() and ops_path.exists():
            return

        data_root = Path(self.cfg["paths"]["data"])
        build_all(data_root=data_root, run_dir=self.run_root)

    def _init_seeds_and_ops(
        self,
        *,
        seeds_path_override: Optional[str],
        dataset_tag: Optional[str],
        config_path: str,
    ) -> None:
        """
        Initialize seeds and operators for B2.

        Handles both:
          - ID training (SORRY snapshot under run_root)
          - OOD eval (explicit seeds_path_override)

        Also robust to different operators.json formats:
          - {"operators": [ {...}, {...} ]}
          - {"ops": [ {...}, {...} ]}
          - [ {...}, {...} ]
          - {"name1": {...}, "name2": {...}, ...}
        """

        def _load_ops(path: Path) -> List[Dict[str, Any]]:
            raw = json.loads(path.read_text(encoding="utf-8"))

            # Case 1: dict with "operators" or "ops" key
            if isinstance(raw, dict):
                if "operators" in raw and isinstance(raw["operators"], list):
                    ops_list = raw["operators"]
                elif "ops" in raw and isinstance(raw["ops"], list):
                    ops_list = raw["ops"]
                else:
                    # Fallback: dict mapping names -> config
                    ops_list = []
                    for k, v in raw.items():
                        if isinstance(v, dict):
                            op_cfg = dict(v)
                            op_cfg.setdefault("name", k)
                            ops_list.append(op_cfg)
                        else:
                            # value is not a dict; just store name
                            ops_list.append({"name": str(k)})
            elif isinstance(raw, list):
                ops_list = []
                for item in raw:
                    if isinstance(item, dict):
                        if "name" not in item:
                            # try to infer a name from other fields or index
                            # (we just leave it unnamed and cast later)
                            ops_list.append(dict(item))
                        else:
                            ops_list.append(dict(item))
                    else:
                        # plain string or something else
                        ops_list.append({"name": str(item)})
            else:
                raise TypeError(
                    f"[B2Env] Unsupported operators.json format: {type(raw)}"
                )

            return ops_list

        if seeds_path_override:
            # OOD: seeds are external, operators still from SORRY snapshot in run_root
            seeds_path = Path(seeds_path_override)
            if not seeds_path.exists():
                raise FileNotFoundError(f"[B2Env] seeds_path_override not found: {seeds_path}")

            # Ensure we have operators snapshot
            self._ensure_snapshot(config_path)
            ops_path = self.run_root / "operators.json"
            if not ops_path.exists():
                raise FileNotFoundError(f"[B2Env] operators.json missing at: {ops_path}")

            self.ops = _load_ops(ops_path)
            self.op_names = [
                (o["name"] if isinstance(o, dict) and "name" in o else str(o))
                for o in self.ops
            ]
            self.num_ops = len(self.op_names)

            self.seeds = _load_seeds_jsonl(seeds_path)
            self.dataset_tag = dataset_tag or f"B2-OOD::{seeds_path.stem}"
            return

        # ID train: snapshot directly in run_root (no artifacts/runs)
        self._ensure_snapshot(config_path)
        seeds_path = self.run_root / "seeds.jsonl"
        ops_path = self.run_root / "operators.json"

        if not seeds_path.exists():
            raise FileNotFoundError(f"[B2Env] seeds.jsonl missing at: {seeds_path}")
        if not ops_path.exists():
            raise FileNotFoundError(f"[B2Env] operators.json missing at: {ops_path}")

        self.seeds = _load_seeds_jsonl(seeds_path)
        self.ops = _load_ops(ops_path)
        self.op_names = [
            (o["name"] if isinstance(o, dict) and "name" in o else str(o))
            for o in self.ops
        ]
        self.num_ops = len(self.op_names)
        self.dataset_tag = dataset_tag or "B2-SORRY"


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

        # Count global env steps for curriculum
        self._env_steps += 1

        op_idx = int(action["op"])
        op_idx = max(0, min(self.num_ops - 1, op_idx))
        op_name = self.op_names[op_idx]

        cont = np.asarray(action["cont"], dtype=np.float32)
        cont = np.clip(cont, 0.0, 1.0)
        sliders = {
            "length": float(cont[0]),
            "temperature": float(cont[1]),
            "persona": float(cont[2]),
        }

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

        # 4) Reward via B2-specific helper
        try:
            p_hash = _sha1_text(prompt_text)
            seen_before = p_hash in self._seen_prompt_hashes
            self._seen_prompt_hashes.add(p_hash)

            # Compute scalar reward from 5D judge scores
            reward_val = compute_reward_from_judge(
                judge_scores=jres.scores,
                cfg=self._reward_cfg_for_fn,
                env_steps=self._env_steps,
            )

            self.rs.append_event(
                {
                    "phase": "reward",
                    "seed_id": rin.seed_id,
                    "category": rin.category,
                    "operator": op_name,
                    "reward": reward_val,
                    # meta fields so you can inspect reward behaviour later if needed
                    "detail": {
                        "env_steps": self._env_steps,
                        "total_env_steps_cfg": self._total_env_steps_cfg,
                        "reward_mode": self._reward_cfg.get("mode"),
                    },
                    "dataset_tag": self.dataset_tag,
                    "pair_id": self._pair_id(rin.seed_id, op_name),
                }
            )
            reward = float(reward_val)
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
