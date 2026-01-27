# Package initializer for baseline jailbreak methods and registry (GCG, later AutoDAN/RLBreaker/etc.).
# src/baselines/rlbreaker/__init__.py
"""
RLBREAKER baseline (SAFE scaffold).

This module provides a *policy-compliant* RL-style prompt optimizer that:
- runs end-to-end in the same harness style as your other baselines
- logs events + metrics
- aggregates tables

It does NOT implement jailbreak / policy bypass behavior.
"""
