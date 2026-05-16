"""
Borrower simulator for microfinance loan pricing.

Design notes
------------
This module is intentionally a *calibrated* simulator, not a learned generative
model.  In the absence of accessible Mifos historical data, we anchor the
simulator's behaviour against:

  * Kiva publicly-released loan data (acceptance, sector, geography mix).
  * Published microfinance elasticity literature
        - Karlan & Zinman (2008) on rate elasticity of demand in South Africa
        - Dehejia, Montgomery & Morduch (2012) on Bangladesh microcredit
  * Reasonable economic priors documented in `docs/calibration.md`.

The borrower is modelled as a structural causal model with three latent
variables:

  * `reservation_rate`  ~ Beta(α(x, e), β(x, e))
        the maximum interest rate the borrower would accept.
  * `default_prob_base` = σ(w · features + b)
        the *baseline* default probability at the median rate.
  * `repayment_serial_corr` = ρ
        first-order autocorrelation in installment outcomes.

Acceptance is a noisy threshold:  accept = 1 iff rate <= reservation_rate + ε
where ε is Gaussian.  Default probability scales upward with offered rate
according to the elasticity calibration.

Crucially we *separate* the protected attribute (e.g. gender, caste) from
the predictive features.  The simulator can be configured with or without
*structural* dependence of repayment on the protected attribute, allowing
us to test how the policy behaves under each ground truth.

This file has zero ML dependencies; it is pure numpy and is fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class BusinessSector(IntEnum):
    AGRICULTURE = 0
    RETAIL = 1
    SERVICES = 2
    MANUFACTURING = 3
    OTHER = 4


@dataclass
class BorrowerFeatures:
    """Observable features of a single borrower."""

    # numeric features
    income_proxy: float           # standardized monthly income proxy in [0, 1]
    prior_loans: int              # number of past loans (loan cycle)
    prior_repayment_rate: float   # fraction of past installments paid on time, in [0, 1]
    age_band: int                 # 0..4 for [<25, 25-34, 35-44, 45-54, 55+]
    region_shock_idx: float       # local economic shock indicator in [-1, 1]
    group_member: int             # 1 if part of a joint-liability group
    group_par30: float            # group's portfolio-at-risk 30-day, in [0, 1]
    sector: BusinessSector        # categorical, encoded as int

    # protected attribute, observable but not given to the policy network
    protected_group: int          # 0 or 1


@dataclass
class BorrowerLatents:
    """Hidden state of the borrower, drawn at episode reset."""

    reservation_rate: float
    default_prob_base: float
    repayment_corr: float


@dataclass
class CohortParams:
    """
    Population-level parameters that govern how borrowers are drawn.

    Calibrated values live in fairprice_mf/calibration/kiva_priors.yaml.
    Anything here is a default that allows the env to instantiate without
    a config file (useful for tests).
    """

    n_features: int = 8
    protected_group_prevalence: float = 0.45
    # elasticity of acceptance to rate, per 1 percentage-point increase in rate
    rate_acceptance_elasticity: float = -0.12
    # elasticity of default to rate (same units)
    rate_default_elasticity: float = 0.08
    # if True, the protected attribute has a (small) *structural* effect on
    # repayment ability through income heterogeneity.  We test policies under
    # both regimes.
    protected_affects_repayment: bool = False
    # noise on the acceptance threshold
    acceptance_noise_std: float = 0.005
    # cohort-level base default prob at median rate
    base_default_prob: float = 0.08
    # cohort-level mean reservation rate (annual, decimal)
    base_reservation_rate: float = 0.28
    seed: Optional[int] = None


class BorrowerCohort:
    """
    A sampler for microfinance borrowers.

    Usage
    -----
    >>> cohort = BorrowerCohort(CohortParams(seed=0))
    >>> features, latents = cohort.sample()
    >>> features.income_proxy
    """

    def __init__(self, params: CohortParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

    # -- internal helpers ---------------------------------------------------

    def _draw_features(self) -> BorrowerFeatures:
        p = self.params
        rng = self.rng

        income_proxy = float(np.clip(rng.beta(2.0, 5.0), 0.0, 1.0))
        prior_loans = int(rng.geometric(p=0.45) - 1)  # mostly first-cycle
        prior_repayment_rate = float(
            np.clip(rng.beta(8.0, 1.5) if prior_loans > 0 else 0.0, 0.0, 1.0)
        )
        age_band = int(rng.choice([0, 1, 2, 3, 4], p=[0.18, 0.34, 0.28, 0.15, 0.05]))
        region_shock_idx = float(rng.normal(0.0, 0.25))
        region_shock_idx = float(np.clip(region_shock_idx, -1.0, 1.0))
        group_member = int(rng.binomial(1, 0.65))
        group_par30 = float(np.clip(rng.beta(2.0, 30.0), 0.0, 1.0)) if group_member else 0.0
        sector = BusinessSector(int(rng.choice(list(BusinessSector), p=[0.32, 0.28, 0.22, 0.13, 0.05])))
        protected_group = int(rng.binomial(1, p.protected_group_prevalence))

        return BorrowerFeatures(
            income_proxy=income_proxy,
            prior_loans=prior_loans,
            prior_repayment_rate=prior_repayment_rate,
            age_band=age_band,
            region_shock_idx=region_shock_idx,
            group_member=group_member,
            group_par30=group_par30,
            sector=sector,
            protected_group=protected_group,
        )

    def _draw_latents(self, x: BorrowerFeatures) -> BorrowerLatents:
        """Latents are conditional on features and on cohort params."""
        p = self.params

        # Reservation rate: higher for higher-income, lower for prior good repayers
        # (good repayers are stickier and accept lower rates).  Mean is shifted
        # by the population base reservation rate, with realistic dispersion.
        mean_res = (
            p.base_reservation_rate
            + 0.04 * x.income_proxy
            - 0.06 * x.prior_repayment_rate
            + 0.03 * x.region_shock_idx  # local shock pushes reservation up
        )
        # Beta parameters chosen so that variance is ~0.04
        alpha, beta = self._beta_params_from_mean(mean_res, var=0.0025)
        reservation_rate = float(np.clip(self.rng.beta(alpha, beta), 0.05, 0.60))

        # Default prob baseline (at median rate, no shock)
        logit = (
            -2.0                              # cohort prior
            - 1.2 * x.prior_repayment_rate    # good repayers default less
            - 0.8 * x.income_proxy            # higher income, less default
            + 1.5 * x.region_shock_idx        # shock raises default
            + 2.0 * x.group_par30             # group contagion
        )
        # Optionally inject a tiny structural protected-group effect through
        # heterogeneity in unmodelled income.  This is the "hard case" for
        # fairness.
        if p.protected_affects_repayment and x.protected_group == 1:
            logit += 0.15
        default_prob_base = float(1.0 / (1.0 + np.exp(-logit)))
        default_prob_base = float(np.clip(default_prob_base, 0.005, 0.55))

        # Repayment serial correlation
        repayment_corr = float(np.clip(0.35 + 0.2 * x.region_shock_idx, 0.0, 0.85))

        return BorrowerLatents(
            reservation_rate=reservation_rate,
            default_prob_base=default_prob_base,
            repayment_corr=repayment_corr,
        )

    @staticmethod
    def _beta_params_from_mean(mean: float, var: float) -> tuple[float, float]:
        """Convert (mean, variance) into Beta(α, β) parameters."""
        mean = float(np.clip(mean, 1e-3, 1 - 1e-3))
        max_var = mean * (1 - mean) - 1e-6
        var = float(np.clip(var, 1e-6, max_var))
        common = mean * (1 - mean) / var - 1
        return mean * common, (1 - mean) * common

    # -- public API ---------------------------------------------------------

    def sample(self) -> tuple[BorrowerFeatures, BorrowerLatents]:
        x = self._draw_features()
        z = self._draw_latents(x)
        return x, z

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    # -- response model -----------------------------------------------------

    def acceptance(self, x: BorrowerFeatures, z: BorrowerLatents, rate: float) -> bool:
        """
        Decide whether the borrower accepts the offered rate.

        Threshold model:  accept iff rate <= reservation_rate + ε
        where ε ~ N(0, σ_acc).
        """
        eps = self.rng.normal(0.0, self.params.acceptance_noise_std)
        return bool(rate <= z.reservation_rate + eps)

    def default_probability(
        self,
        x: BorrowerFeatures,
        z: BorrowerLatents,
        rate: float,
        baseline_rate: float = 0.18,
    ) -> float:
        """
        Probability of *eventual* default on a loan offered at `rate`.

        Calibrated so that a +1 percentage-point increase in rate raises
        the default odds in a manner consistent with `rate_default_elasticity`.
        """
        logit_base = np.log(z.default_prob_base / (1 - z.default_prob_base))
        # marginal effect of (rate - baseline) on log-odds
        rate_effect = self.params.rate_default_elasticity * 100.0 * (rate - baseline_rate)
        logit = logit_base + rate_effect
        return float(1.0 / (1.0 + np.exp(-logit)))

    def simulate_repayment(
        self,
        x: BorrowerFeatures,
        z: BorrowerLatents,
        rate: float,
        n_installments: int,
        principal: float,
    ) -> dict:
        """
        Simulate a full repayment schedule given a borrower and offered rate.

        Returns a dict with realized cashflows and a default flag.
        """
        p_default_total = self.default_probability(x, z, rate)
        # Per-installment miss probability that integrates to p_default_total
        # over `n_installments` correlated draws.  We use a simple Markov chain.
        miss_prob = 1 - (1 - p_default_total) ** (1.0 / n_installments)

        installment_amount = principal * (1 + rate) / n_installments
        cashflows = np.zeros(n_installments, dtype=float)
        missed = False
        last_miss = 0
        rng = self.rng
        for t in range(n_installments):
            # Markov-correlated miss
            adj_miss = miss_prob + z.repayment_corr * (last_miss - miss_prob)
            adj_miss = float(np.clip(adj_miss, 0.0, 0.95))
            miss = int(rng.random() < adj_miss)
            if miss:
                missed = True
                cashflows[t] = 0.0
            else:
                cashflows[t] = installment_amount
            last_miss = miss

        defaulted = bool(missed and float(cashflows.sum()) < 0.7 * principal * (1 + rate))
        return {
            "cashflows": cashflows,
            "total_received": float(cashflows.sum()),
            "expected_total": float(principal * (1 + rate)),
            "defaulted": defaulted,
            "installments_missed": int((cashflows == 0).sum()),
            "p_default_at_offer": p_default_total,
        }
