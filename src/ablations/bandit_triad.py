# src/ablations/bandit_triad.py
from __future__ import annotations

from typing import Any
import random

import torch

from ..rl.envs import PromptHybridEnv


def _build_env(
    stack_config_path: str,
    runs_model: str,
    target_model: str,
    dataset_tag: str,
    seeds_path_override: str | None,
) -> Any:
    env = PromptHybridEnv(
        config_path=stack_config_path,
        runs_model=runs_model,
        rewriter_model="ollama/yi:9b",
        judge_model="ollama/llama3:instruct",
        target_model=target_model,
        dataset_tag=dataset_tag,
        seeds_path_override=seeds_path_override,
        use_target_echo=False,
    )
    return env


# -------------------------------------------------------------------
# Contextual bandit evaluation
# -------------------------------------------------------------------


def eval_bandit_triad(
    stack_config_path: str,
    runs_model: str,
    target_model: str,
    seeds_path_override: str,
    dataset_tag: str,
    episodes: int,
    epsilon: float = 0.1,
    default_temp: float = 0.7,
) -> None:
    """
    Bandit-Triad baseline (no training step):

      - build Triad env on JBB-OOD seeds
      - epsilon-greedy over operators, temp ~ jitter around default_temp
      - env is responsible for logging events with dataset_tag
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = _build_env(
        stack_config_path=stack_config_path,
        runs_model=runs_model,
        target_model=target_model,
        dataset_tag=dataset_tag,
        seeds_path_override=seeds_path_override,
    )

    num_ops = env.action_space["op"].n

    # simple running-average bandit stats
    op_value = [0.0 for _ in range(num_ops)]
    op_counts = [1e-6 for _ in range(num_ops)]  # avoid div by zero

    for ep in range(episodes):
        obs, info = env.reset()
        done = False

        while not done:
            # epsilon-greedy operator choice
            if random.random() < epsilon:
                op = random.randint(0, num_ops - 1)
            else:
                avg_values = [op_value[i] / op_counts[i] for i in range(num_ops)]
                op = max(range(num_ops), key=lambda i: avg_values[i])

            # temperature with small jitter, stay in [0, 1]
            temp = min(
                max(default_temp + 0.1 * (random.random() - 0.5), 0.0),
                1.0,
            )

            action = {
                "op": op,
                "cont": [0.5, float(temp), 0.5],  # length, temperature, persona
            }
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # update stats with immediate reward (whatever env returns)
            op_value[op] += float(reward)
            op_counts[op] += 1.0

            obs = next_obs

    # env has written events to RunStore/events.jsonl with dataset_tag
