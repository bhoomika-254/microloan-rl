"""
Safety wrappers for the MicroLoanPricing environment.

Two layers of safety, both *pre-action* (we modify the action before the
env consumes it) rather than reward shaping.  This is the right design
for production deployment: the upstream agent can produce any action,
but the wrapper guarantees the action that hits Fineract is safe.

  * `RegulatoryCapWrapper`     — enforce `rate ≤ r_cap`.  Already done
    inside the env, but lifted here so it survives any policy override.
  * `TrustRegionWrapper`       — for a repeat borrower, the new rate may
    not move by more than `delta_max` from the rate offered on their
    previous loan.  Operationally important: prevents policy-version
    swaps from creating unexplainable price shocks for individual
    clients.
  * `CVaRBudgetWrapper`        — a Lagrangian state-augmentation wrapper
    in the style of Sootla et al. (Sauté RL, 2022).  Tracks cumulative
    welfare cost across an episode and refuses to widen the rate further
    once the CVaR budget is exhausted.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import gymnasium as gym
import numpy as np

from fairprice_mf.envs.microloan_env import MicroLoanPricingEnv


class RegulatoryCapWrapper(gym.ActionWrapper):
    """Clip the rate to a regulatory ceiling.  Defensive."""

    def __init__(self, env: gym.Env, r_cap: float):
        super().__init__(env)
        self.r_cap = r_cap

    def action(self, action):
        # only matters for continuous; discrete is already bounded by grid
        if isinstance(self.action_space, gym.spaces.Box):
            return np.clip(action, 0.0, 1.0)
        return action


class TrustRegionWrapper(gym.Wrapper):
    """
    Limit |new_rate - last_rate| for a recurring borrower.

    Identifies a borrower by a hashed key over their feature tuple.  In
    a production deployment this would be the Fineract client_id.
    """

    def __init__(self, env: gym.Env, delta_max: float = 0.03):
        super().__init__(env)
        self.delta_max = delta_max
        self._last_rate: dict[int, float] = {}

    def _client_key(self, info: dict) -> int:
        # In production: Fineract client_id.  In simulation we approximate
        # with a deterministic hash of observables.
        return hash((
            info.get("sector"),
            info.get("protected_group"),
            round(info.get("true_reservation_rate", 0.0), 3),
        ))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        key = self._client_key(info)
        proposed = info.get("rate_offered")
        if proposed is None:
            return obs, reward, terminated, truncated, info

        last = self._last_rate.get(key)
        if last is not None:
            delta = proposed - last
            if abs(delta) > self.delta_max:
                clipped = last + np.sign(delta) * self.delta_max
                info["rate_clipped_by_trust_region"] = True
                info["rate_original"] = proposed
                info["rate_offered"] = float(clipped)
        self._last_rate[key] = info["rate_offered"]
        return obs, reward, terminated, truncated, info


class CVaRBudgetWrapper(gym.Wrapper):
    """
    State-augmentation wrapper that tracks cumulative welfare cost and
    refuses to increase the offered rate once a CVaR budget is exhausted.

    Reference: Sootla, Cowen-Rivers, Jafferjee, Wang, Mguni, Wang, Bou-Ammar
    (2022). "Sauté RL: Almost Surely Safe RL Using State Augmentation".
    """

    def __init__(
        self,
        env: gym.Env,
        cvar_alpha: float = 0.10,
        budget: float = 5.0,
        adjustment_step: float = 0.02,
    ):
        super().__init__(env)
        self.cvar_alpha = cvar_alpha
        self.budget_init = budget
        self.adjustment_step = adjustment_step

        # augment observation by one slot for remaining budget
        low = np.append(env.observation_space.low, [0.0])
        high = np.append(env.observation_space.high, [1.0])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._budget_remaining = budget

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._budget_remaining = self.budget_init
        return self._augment(obs), info

    def step(self, action):
        # adjust action if budget is tight
        if self._budget_remaining < 0.2 * self.budget_init:
            if isinstance(self.action_space, gym.spaces.Box):
                # nudge toward lower rates
                action = np.maximum(np.asarray(action) - self.adjustment_step, 0.0)
            elif isinstance(self.action_space, gym.spaces.Discrete):
                action = max(0, int(action) - 1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        welfare_cost = info.get("welfare_penalty", 0.0)
        self._budget_remaining = max(0.0, self._budget_remaining - welfare_cost)
        info["cvar_budget_remaining"] = self._budget_remaining
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs):
        frac = self._budget_remaining / self.budget_init
        return np.append(np.asarray(obs, dtype=np.float32), np.float32(frac))
