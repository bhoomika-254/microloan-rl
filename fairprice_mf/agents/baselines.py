"""
Baseline pricing policies for the MicroLoanPricing environment.

Three reference agents:

  * `RuleBasedPolicy`           — status quo: a small lookup table from
                                   risk-bucket to rate.  This is what most
                                   MFIs deploy today.
  * `ProfitMaxLogisticPolicy`   — supervised learning baseline: fit an
                                   acceptance and default model from logged
                                   (rate, outcome) tuples, then pick the
                                   rate that maximizes expected profit
                                   given those models.  This is the
                                   Phillips-2015 / Ban-Keskin family.
  * `RandomPolicy`              — uniform random in [r_min, r_cap].
                                   Useful for data-collection sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fairprice_mf.envs.microloan_env import EnvConfig, OBS_INDEX


@dataclass
class RuleBasedPolicy:
    """
    Standard 3-bucket microfinance pricing:

      * good payer + low shock      -> 0.14 (14%)
      * standard borrower           -> 0.18
      * risky borrower / high shock -> 0.24

    Bucketing is on (prior_repayment_rate, region_shock_idx).
    """

    rate_good: float = 0.14
    rate_standard: float = 0.18
    rate_risky: float = 0.24

    def __call__(self, obs: np.ndarray) -> float:
        prr = float(obs[OBS_INDEX["prior_repayment_rate"]])
        shock = float(obs[OBS_INDEX["region_shock_idx"]])
        if prr > 0.85 and shock < 0.0:
            return self.rate_good
        if prr < 0.50 or shock > 0.3:
            return self.rate_risky
        return self.rate_standard


class ProfitMaxLogisticPolicy:
    """
    Fit two logistic models from logged (obs, rate, accepted, defaulted)
    tuples:  P(accept | obs, rate)  and  P(default | obs, rate).
    Then offer the rate that maximizes expected profit:

        argmax_r  P(accept) * (1 - P(default)) * principal_expected * r
                 - P(accept) * P(default) * principal_expected * loss_given_default

    For simplicity we use closed-form sklearn on a grid of candidate rates.
    """

    def __init__(self, rate_grid: Optional[np.ndarray] = None):
        from sklearn.linear_model import LogisticRegression

        self.rate_grid = (
            rate_grid if rate_grid is not None else np.linspace(0.06, 0.40, 18)
        )
        self.accept_model = LogisticRegression(max_iter=1000)
        self.default_model = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(
        self,
        observations: np.ndarray,
        rates: np.ndarray,
        accepted: np.ndarray,
        defaulted: np.ndarray,
    ):
        # design matrix: features + rate
        X = np.column_stack([observations, rates.reshape(-1, 1)])
        self.accept_model.fit(X, accepted.astype(int))
        # Default model only trained on accepted loans
        mask = accepted.astype(bool)
        if mask.sum() > 10:
            self.default_model.fit(X[mask], defaulted[mask].astype(int))
        else:
            # not enough data; fall back to constant 0.1 default
            self.default_model = _ConstantModel(0.1)
        self._fitted = True

    def __call__(self, obs: np.ndarray) -> float:
        if not self._fitted:
            return 0.18  # safe fallback
        candidates = np.column_stack([
            np.tile(obs, (len(self.rate_grid), 1)),
            self.rate_grid.reshape(-1, 1),
        ])
        p_accept = self.accept_model.predict_proba(candidates)[:, 1]
        if hasattr(self.default_model, "predict_proba"):
            p_default = self.default_model.predict_proba(candidates)[:, 1]
        else:
            p_default = np.full(len(self.rate_grid), self.default_model.value)
        loss_given_default = 0.6
        expected_profit = (
            p_accept * ((1 - p_default) * self.rate_grid - p_default * loss_given_default)
        )
        return float(self.rate_grid[int(np.argmax(expected_profit))])


@dataclass
class _ConstantModel:
    value: float

    def predict_proba(self, X):  # noqa: N802 — sklearn interface
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.value), np.full(n, self.value)])


class RandomPolicy:
    def __init__(self, low: float = 0.06, high: float = 0.40, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.low = low
        self.high = high

    def __call__(self, obs: np.ndarray) -> float:
        return float(self.rng.uniform(self.low, self.high))


def policy_to_normalized_action(rate: float, env_config: EnvConfig) -> np.ndarray:
    """Map a raw rate to the env's normalized [0, 1] action."""
    a = (rate - env_config.r_min) / max(
        1e-6, env_config.regulatory_cap - env_config.r_min
    )
    return np.array([float(np.clip(a, 0.0, 1.0))], dtype=np.float32)
