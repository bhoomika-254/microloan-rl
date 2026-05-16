"""Tests for the MicroLoanPricing environment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import fairprice_mf  # noqa: F401 — registers env ids
from fairprice_mf.envs.microloan_env import EnvConfig, MicroLoanPricingEnv, OBS_DIM


def test_env_passes_gymnasium_checker_continuous():
    env = MicroLoanPricingEnv(EnvConfig(seed=0))
    # env_checker requires reset/step to be well-behaved
    check_env(env.unwrapped, skip_render_check=True)


def test_env_passes_gymnasium_checker_discrete():
    env = MicroLoanPricingEnv(EnvConfig(seed=0, discrete_actions=11))
    check_env(env.unwrapped, skip_render_check=True)


def test_reset_returns_correct_shape():
    env = MicroLoanPricingEnv(EnvConfig(seed=42))
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "protected_group" in info
    assert info["protected_group"] in (0, 1)


def test_step_returns_5_tuple():
    env = MicroLoanPricingEnv(EnvConfig(seed=1))
    env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "rate_offered" in info


def test_regulatory_cap_is_hard():
    env = MicroLoanPricingEnv(EnvConfig(seed=1, regulatory_cap=0.25))
    env.reset(seed=1)
    # Action 1.0 maps to r_min + 1.0*(cap - r_min) = cap = 0.25
    obs, reward, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["rate_offered"] <= 0.25 + 1e-6


def test_seed_reproducibility():
    env1 = MicroLoanPricingEnv(EnvConfig(seed=123))
    env2 = MicroLoanPricingEnv(EnvConfig(seed=123))
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    np.testing.assert_array_almost_equal(obs1, obs2)


def test_discrete_actions_map_to_grid():
    env = MicroLoanPricingEnv(EnvConfig(seed=1, discrete_actions=11))
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    assert abs(info["rate_offered"] - env.config.r_min) < 1e-6


def test_sequential_horizon():
    env = MicroLoanPricingEnv(EnvConfig(seed=1, sequential=True, horizon=5))
    env.reset(seed=1)
    terminated = False
    steps = 0
    while not terminated and steps < 10:
        _, _, terminated, _, _ = env.step(np.array([0.3], dtype=np.float32))
        steps += 1
    assert steps == 5


def test_acceptance_correlates_with_rate():
    """High rates should reduce acceptance, low rates should increase it."""
    accepts_low = 0
    accepts_high = 0
    n = 200
    env = MicroLoanPricingEnv(EnvConfig(seed=7))
    for i in range(n):
        env.reset(seed=i)
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))  # ~ r_min
        accepts_low += int(info["accepted"])
        env.reset(seed=i)
        _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))  # ~ r_cap
        accepts_high += int(info["accepted"])
    assert accepts_low > accepts_high
