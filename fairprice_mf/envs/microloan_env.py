"""
MicroLoanPricingEnv — the core Gymnasium environment.

Two modes are supported:

  * `sequential=False` (default): single-step *contextual bandit* style.
        At each `reset()` the env samples a new borrower.  The agent emits
        a rate and `step()` returns the realized profit / welfare reward.
        Episode length 1.  This matches how loan-pricing decisions actually
        happen in production at most MFIs and gives the cleanest evaluation.

  * `sequential=True`: a multi-step *portfolio* episode in which the agent
        prices N borrowers in a row, with a slowly drifting macro state
        between borrowers.  This is what we use for stress-test rollouts.

Observation space
-----------------
A flat float32 Box.  The first `n_features` entries are the borrower's
*non-protected* observable features.  The macro-economic features (e.g.
shock indicator, regional PAR-30) come next.  The protected attribute is
*not* part of the observation passed to the agent — the env keeps it in
`info["protected_group"]` for the fairness auditor.

Action space
------------
`Box(low=r_min, high=r_max, shape=(1,))` for continuous-action algorithms.
For DQN comparability, instantiate with `discrete_actions=11` to get
`Discrete(11)` over rates evenly spaced in `[r_min, r_max]`.

Reward
------
`R = profit - lambda_welfare * welfare_penalty`

where the welfare penalty captures rate-as-fraction-of-cap and an
over-indebtedness flag.  The constraint costs for fairness and CVaR are
emitted separately via `info["constraint_costs"]` and consumed by the
constrained-RL wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fairprice_mf.envs.borrower import (
    BorrowerCohort,
    BorrowerFeatures,
    BorrowerLatents,
    BusinessSector,
    CohortParams,
)
from fairprice_mf.envs.shocks import (
    BaselineRegime,
    EconomicShock,
    ShockSampler,
)


@dataclass
class EnvConfig:
    """All knobs for the environment in one place."""

    # rate bounds (annual, decimal)
    r_min: float = 0.06
    r_max: float = 0.45
    regulatory_cap: float = 0.40            # hard ceiling, action masked above this
    baseline_rate: float = 0.18

    # loan structure
    principal_min: float = 50.0
    principal_max: float = 1000.0
    n_installments: int = 12
    servicing_cost_per_loan: float = 5.0

    # reward shaping
    welfare_weight: float = 0.30            # λ_w in the reward equation
    over_indebtedness_threshold: float = 0.40   # installment/income ratio cap
    over_indebtedness_penalty: float = 10.0

    # constraint thresholds (consumed by wrappers, not by the env itself)
    fairness_slack_eps: float = 0.02        # |E[rate|g=0] - E[rate|g=1]| ≤ eps
    cvar_alpha: float = 0.10                # worst-10% tail
    cvar_welfare_floor: float = -2.0        # min acceptable CVaR_α(welfare)

    # discrete-action variant: None for continuous, int K for K rate buckets
    discrete_actions: Optional[int] = None

    # mode
    sequential: bool = False
    horizon: int = 12

    # rng
    seed: Optional[int] = None

    # cohort overrides
    cohort_params: Optional[CohortParams] = None
    # if provided, fixed shock used every reset; otherwise sample one
    fixed_shock: Optional[str] = None


# Indexing helper for the observation vector.  Keep this in sync with
# `_build_observation` below.
OBS_INDEX = {
    "income_proxy": 0,
    "prior_loans_norm": 1,
    "prior_repayment_rate": 2,
    "age_band_norm": 3,
    "region_shock_idx": 4,
    "group_member": 5,
    "group_par30": 6,
    # sector is one-hot encoded over 5 entries: indices 7..11
    "sector_onehot_start": 7,
    # macroeconomic features
    "macro_shock_baseline": 12,
    "macro_shock_inflation": 13,
    "macro_shock_drought": 14,
    "macro_shock_pandemic": 15,
}
OBS_DIM = 16


class MicroLoanPricingEnv(gym.Env):
    """Gymnasium environment for fairness-aware micro-loan rate pricing."""

    metadata = {"render_modes": ["human"], "name": "MicroLoanPricing-v0"}

    def __init__(self, config: Optional[Union[EnvConfig, dict]] = None, **kwargs):
        super().__init__()
        # accept dict for clean integration with libraries like Hydra
        if isinstance(config, dict):
            config = EnvConfig(**config)
        if config is None:
            config = EnvConfig(**kwargs)
        self.config = config

        cohort_params = config.cohort_params or CohortParams(seed=config.seed)
        self.cohort = BorrowerCohort(cohort_params)
        self.shock_sampler = ShockSampler(seed=config.seed)

        # spaces
        if config.discrete_actions is not None:
            self.action_space = spaces.Discrete(config.discrete_actions)
            self._rate_grid = np.linspace(
                config.r_min, min(config.r_max, config.regulatory_cap),
                config.discrete_actions,
            )
        else:
            # continuous; we use [0, 1] internally and map to [r_min, r_cap]
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            )
            self._rate_grid = None

        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(OBS_DIM,), dtype=np.float32,
        )

        # episode state
        self._current_x: Optional[BorrowerFeatures] = None
        self._current_z: Optional[BorrowerLatents] = None
        self._current_shock: EconomicShock = BaselineRegime()
        self._t: int = 0

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.cohort.reseed(seed)
            self.shock_sampler = ShockSampler(seed=seed)

        if self.config.fixed_shock:
            self._current_shock = self.shock_sampler.get_by_name(self.config.fixed_shock)
        else:
            self._current_shock = self.shock_sampler.sample()

        self._current_x, self._current_z = self.cohort.sample()
        # apply shock once at reset (and again per step in sequential mode)
        self._current_z = self._current_shock.apply(
            self._current_x, self._current_z, t_in_loan=0, rng=self.cohort.rng,
        )
        self._t = 0
        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(self, action):
        rate = self._action_to_rate(action)

        # safety: regulatory cap is a hard ceiling
        rate = float(min(rate, self.config.regulatory_cap))

        principal = float(self.cohort.rng.uniform(
            self.config.principal_min, self.config.principal_max,
        ))

        accepted = self.cohort.acceptance(self._current_x, self._current_z, rate)

        if accepted:
            outcome = self.cohort.simulate_repayment(
                self._current_x,
                self._current_z,
                rate,
                self.config.n_installments,
                principal,
            )
            profit = (
                outcome["total_received"]
                - principal
                - self.config.servicing_cost_per_loan
            )
            welfare_penalty = self._welfare_penalty(rate, principal, outcome)
            reward = profit - self.config.welfare_weight * welfare_penalty
            default_realized = outcome["defaulted"]
        else:
            outcome = None
            profit = 0.0
            welfare_penalty = 0.0
            reward = 0.0  # no loan happened; neutral outcome
            default_realized = False

        # constraint costs go out via info, the env itself stays unconstrained
        constraint_costs = {
            "fairness": self._fairness_constraint_signal(rate),
            "cvar_welfare": welfare_penalty,    # auditor estimates CVaR_α over many episodes
        }

        info = self._build_info()
        info.update({
            "accepted": accepted,
            "rate_offered": rate,
            "principal": principal,
            "profit": float(profit),
            "welfare_penalty": float(welfare_penalty),
            "default_realized": bool(default_realized),
            "outcome": outcome,
            "constraint_costs": constraint_costs,
            "shock": self._current_shock.name,
        })

        if self.config.sequential:
            self._t += 1
            terminated = self._t >= self.config.horizon
            if not terminated:
                # advance: new borrower, same shock regime, slight drift
                self._current_x, self._current_z = self.cohort.sample()
                self._current_z = self._current_shock.apply(
                    self._current_x, self._current_z, t_in_loan=self._t,
                    rng=self.cohort.rng,
                )
            obs = self._build_observation()
        else:
            terminated = True
            obs = self._build_observation()

        return obs, float(reward), bool(terminated), False, info

    # -- internal helpers ---------------------------------------------------

    def _action_to_rate(self, action) -> float:
        if isinstance(self.action_space, spaces.Discrete):
            return float(self._rate_grid[int(action)])
        a = np.asarray(action, dtype=np.float32).reshape(-1)[0]
        a = float(np.clip(a, 0.0, 1.0))
        return self.config.r_min + a * (self.config.regulatory_cap - self.config.r_min)

    def _build_observation(self) -> np.ndarray:
        if self._current_x is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        x = self._current_x
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[OBS_INDEX["income_proxy"]] = x.income_proxy
        # prior_loans normalized
        obs[OBS_INDEX["prior_loans_norm"]] = min(x.prior_loans, 10) / 10.0
        obs[OBS_INDEX["prior_repayment_rate"]] = x.prior_repayment_rate
        obs[OBS_INDEX["age_band_norm"]] = x.age_band / 4.0
        obs[OBS_INDEX["region_shock_idx"]] = x.region_shock_idx
        obs[OBS_INDEX["group_member"]] = float(x.group_member)
        obs[OBS_INDEX["group_par30"]] = x.group_par30
        # sector one-hot
        sector_idx = int(x.sector)
        obs[OBS_INDEX["sector_onehot_start"] + sector_idx] = 1.0
        # macro shock one-hot
        shock_map = {"baseline": 12, "inflation": 13, "drought": 14, "pandemic": 15}
        obs[shock_map[self._current_shock.name]] = 1.0
        return obs

    def _build_info(self) -> dict:
        """Auditor consumes this; *not* policy input."""
        if self._current_x is None:
            return {}
        return {
            "protected_group": int(self._current_x.protected_group),
            "sector": int(self._current_x.sector),
            "true_default_prob_baseline": float(self._current_z.default_prob_base),
            "true_reservation_rate": float(self._current_z.reservation_rate),
            "shock": self._current_shock.name,
            "t": self._t,
        }

    def _welfare_penalty(self, rate: float, principal: float, outcome: dict) -> float:
        # rate-as-fraction-of-cap, scaled
        rate_pen = (rate - self.config.r_min) / max(
            1e-6, self.config.regulatory_cap - self.config.r_min
        )
        # over-indebtedness: monthly installment relative to income proxy
        installment = principal * (1 + rate) / self.config.n_installments
        # rescale income_proxy [0, 1] into a monthly income figure
        income_monthly = 50.0 + 450.0 * self._current_x.income_proxy
        ratio = installment / max(1.0, income_monthly)
        over_indebt = max(0.0, ratio - self.config.over_indebtedness_threshold)
        oi_pen = self.config.over_indebtedness_penalty * over_indebt
        # default suffering for the borrower
        default_pen = 8.0 if outcome["defaulted"] else 0.0
        return float(rate_pen + oi_pen + default_pen)

    def _fairness_constraint_signal(self, rate: float) -> float:
        """
        Per-episode contribution to the demographic-parity constraint.

        The constraint is `|E[rate | g=0] - E[rate | g=1]| ≤ ε`.
        Per-episode we just emit the signed contribution:
            +rate if g=1, -rate if g=0.
        The auditor / wrapper averages this and takes the absolute value.
        """
        return rate if self._current_x.protected_group == 1 else -rate

    def render(self):
        x = self._current_x
        z = self._current_z
        if x is None:
            print("[env not reset]")
            return
        print(
            f"shock={self._current_shock.name:>9s} "
            f"g={x.protected_group} sector={x.sector.name:>12s} "
            f"income={x.income_proxy:.2f} cycle={x.prior_loans} "
            f"reservation~={z.reservation_rate:.3f} "
            f"p_default~={z.default_prob_base:.3f}"
        )
