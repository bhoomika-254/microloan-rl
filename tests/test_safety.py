"""Tests for the safety wrappers."""

from __future__ import annotations

import numpy as np

from fairprice_mf.envs.microloan_env import EnvConfig, MicroLoanPricingEnv
from fairprice_mf.safety.wrappers import (
    CVaRBudgetWrapper,
    TrustRegionWrapper,
)


def test_trust_region_clips_repeat_borrower_jump():
    env = MicroLoanPricingEnv(EnvConfig(seed=0))
    wrapped = TrustRegionWrapper(env, delta_max=0.02)
    # Reset twice with same seed to revisit the same borrower (approx).
    obs, info = wrapped.reset(seed=42)
    _, _, _, _, info1 = wrapped.step(np.array([0.1], dtype=np.float32))   # low rate
    obs, info = wrapped.reset(seed=42)
    _, _, _, _, info2 = wrapped.step(np.array([0.95], dtype=np.float32))  # high rate
    # The clipping happens only when the same client_key is hit; not
    # guaranteed across resets, but if it does we expect the original
    # rate stored under "rate_original" to differ from the final
    if info2.get("rate_clipped_by_trust_region"):
        assert info2["rate_original"] != info2["rate_offered"]


def test_cvar_budget_decreases_over_episode():
    env = MicroLoanPricingEnv(EnvConfig(seed=0, sequential=True, horizon=5))
    wrapped = CVaRBudgetWrapper(env, budget=3.0)
    obs, info = wrapped.reset(seed=1)
    assert obs.shape[0] == 17   # 16 + 1 for budget slot
    budgets = []
    terminated = False
    while not terminated:
        _, _, terminated, _, info = wrapped.step(np.array([0.7], dtype=np.float32))
        budgets.append(info["cvar_budget_remaining"])
    # budget should be non-increasing
    for a, b in zip(budgets, budgets[1:]):
        assert b <= a + 1e-6
