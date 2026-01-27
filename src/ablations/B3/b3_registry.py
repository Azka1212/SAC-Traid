from __future__ import annotations
import gymnasium as gym

def register_b3_env() -> None:
    # Safe to call multiple times.
    try:
        gym.spec("PromptHybridB3-v0")
        return
    except Exception:
        pass

    gym.register(
        id="PromptHybridB3-v0",
        entry_point="src.ablations.B3.b3_env:PromptHybridB3Env",
    )
