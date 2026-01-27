# src/rl/registry.py
from gymnasium.envs.registration import register

_REG_ID = "PromptHybrid-v0"
_DONE = False

def register_prompt_hybrid_env():
    global _DONE
    if _DONE:
        return _REG_ID
    try:
        register(
            id=_REG_ID,
            entry_point="src.rl.envs.prompt_hybrid_env:PromptHybridEnv",
        )
    except Exception:
        # If already registered by another import path, we still want to proceed safely.
        pass
    _DONE = True
    return _REG_ID
