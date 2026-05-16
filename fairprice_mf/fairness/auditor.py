"""
Fairness auditor for the MicroLoanPricing environment.

Three families of metric:

  1. *Demographic parity of rates*  — `|E[a | g=0] − E[a | g=1]| ≤ ε`.
     Pricing analogue of the classical approval-rate parity.
  2. *Disparate impact ratio*       — `min_g P(accept | g) / max_g P(accept | g)`.
     The 80-percent rule, applied to acceptance.
  3. *Counterfactual fairness*       — re-sample the same borrower with the
     protected attribute flipped (all other latents held constant), score
     under the policy, and report the absolute rate-shift distribution.
     This is the genuinely novel piece of our auditor.

All three are computed from rollout logs.  Counterfactual fairness depends
on having access to the simulator (it cannot be computed from logs alone),
so it lives in this module rather than as a generic library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from fairprice_mf.envs.borrower import BorrowerFeatures, BusinessSector
from fairprice_mf.envs.microloan_env import MicroLoanPricingEnv, OBS_DIM, OBS_INDEX


@dataclass
class FairnessReport:
    n_episodes: int
    mean_rate_by_group: dict[int, float]
    rate_parity_gap: float
    acceptance_rate_by_group: dict[int, float]
    disparate_impact_ratio: float
    counterfactual_rate_shift_mean: Optional[float] = None
    counterfactual_rate_shift_p95: Optional[float] = None
    per_group_default_rate: dict[int, float] = field(default_factory=dict)
    per_group_mean_welfare_penalty: dict[int, float] = field(default_factory=dict)

    def passes(self, eps: float = 0.02, di_floor: float = 0.80) -> dict[str, bool]:
        return {
            "rate_parity": self.rate_parity_gap <= eps,
            "disparate_impact": self.disparate_impact_ratio >= di_floor,
        }

    def pretty(self) -> str:
        lines = [
            f"FairnessReport over {self.n_episodes} episodes",
            f"  rate parity gap            : {self.rate_parity_gap:.4f}",
            f"  disparate impact ratio     : {self.disparate_impact_ratio:.4f}",
            f"  mean rate  | g=0           : {self.mean_rate_by_group.get(0, float('nan')):.4f}",
            f"  mean rate  | g=1           : {self.mean_rate_by_group.get(1, float('nan')):.4f}",
            f"  acceptance | g=0           : {self.acceptance_rate_by_group.get(0, float('nan')):.4f}",
            f"  acceptance | g=1           : {self.acceptance_rate_by_group.get(1, float('nan')):.4f}",
        ]
        if self.counterfactual_rate_shift_mean is not None:
            lines.append(
                f"  counterfactual shift (mean): {self.counterfactual_rate_shift_mean:.4f}"
            )
            lines.append(
                f"  counterfactual shift (p95) : {self.counterfactual_rate_shift_p95:.4f}"
            )
        return "\n".join(lines)


class FairnessAuditor:
    """Accumulates per-episode statistics and produces a FairnessReport."""

    def __init__(self):
        self._records: list[dict] = []

    def record(self, info: dict) -> None:
        # we only care about completed loan-pricing decisions
        if "rate_offered" not in info:
            return
        self._records.append({
            "g": info["protected_group"],
            "rate": info["rate_offered"],
            "accepted": info["accepted"],
            "default": info.get("default_realized", False),
            "welfare": info.get("welfare_penalty", 0.0),
        })

    def report(
        self,
        counterfactual_shifts: Optional[np.ndarray] = None,
    ) -> FairnessReport:
        if not self._records:
            return FairnessReport(
                n_episodes=0,
                mean_rate_by_group={},
                rate_parity_gap=float("nan"),
                acceptance_rate_by_group={},
                disparate_impact_ratio=float("nan"),
            )

        rates = np.array([r["rate"] for r in self._records])
        groups = np.array([r["g"] for r in self._records])
        accepts = np.array([r["accepted"] for r in self._records], dtype=bool)
        defaults = np.array([r["default"] for r in self._records], dtype=bool)
        welfare = np.array([r["welfare"] for r in self._records])

        mean_rate = {}
        accept_rate = {}
        default_rate = {}
        welfare_mean = {}
        for g in [0, 1]:
            mask = groups == g
            if not mask.any():
                continue
            mean_rate[g] = float(rates[mask].mean())
            accept_rate[g] = float(accepts[mask].mean())
            # default rate conditional on accepted loans only
            accepted_mask = mask & accepts
            default_rate[g] = (
                float(defaults[accepted_mask].mean())
                if accepted_mask.any() else 0.0
            )
            welfare_mean[g] = float(welfare[mask].mean())

        if 0 in mean_rate and 1 in mean_rate:
            parity = abs(mean_rate[0] - mean_rate[1])
        else:
            parity = float("nan")

        if 0 in accept_rate and 1 in accept_rate:
            ar0, ar1 = accept_rate[0], accept_rate[1]
            denom = max(ar0, ar1)
            di = (min(ar0, ar1) / denom) if denom > 0 else 1.0
        else:
            di = float("nan")

        cf_mean = cf_p95 = None
        if counterfactual_shifts is not None and len(counterfactual_shifts) > 0:
            cf_mean = float(np.mean(np.abs(counterfactual_shifts)))
            cf_p95 = float(np.quantile(np.abs(counterfactual_shifts), 0.95))

        return FairnessReport(
            n_episodes=len(self._records),
            mean_rate_by_group=mean_rate,
            rate_parity_gap=parity,
            acceptance_rate_by_group=accept_rate,
            disparate_impact_ratio=di,
            per_group_default_rate=default_rate,
            per_group_mean_welfare_penalty=welfare_mean,
            counterfactual_rate_shift_mean=cf_mean,
            counterfactual_rate_shift_p95=cf_p95,
        )

    def reset(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Counterfactual fairness auditing — needs simulator access
# ---------------------------------------------------------------------------

def counterfactual_audit(
    policy_fn: Callable[[np.ndarray], float],
    env: MicroLoanPricingEnv,
    n_samples: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """
    For each of `n_samples` borrowers, build the counterfactual observation
    that *would* have been generated had the protected attribute been flipped
    (in our env this only matters when CohortParams.protected_affects_repayment
    is True, but we still measure the policy's sensitivity).

    Returns
    -------
    np.ndarray of shape (n_samples,) — signed rate shift `rate(g=1) - rate(g=0)`
    for the same underlying borrower features.

    Notes
    -----
    Because the observation passed to the policy does not contain the protected
    attribute directly, the *direct* counterfactual rate shift will be zero
    in expectation.  Non-zero shift implies the policy has picked up a proxy
    correlation through other features — exactly what we want to detect.
    """
    rng = np.random.default_rng(seed)
    shifts = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        # build counterfactual: flip protected attribute and re-build obs
        x = env._current_x  # noqa: SLF001 — auditor needs internals
        z = env._current_z

        # rate under original (call policy on whatever obs the env shows)
        obs_orig = env._build_observation()
        rate_orig = float(policy_fn(obs_orig))

        # construct flipped borrower with same observable features
        x_flipped = BorrowerFeatures(
            income_proxy=x.income_proxy,
            prior_loans=x.prior_loans,
            prior_repayment_rate=x.prior_repayment_rate,
            age_band=x.age_band,
            region_shock_idx=x.region_shock_idx,
            group_member=x.group_member,
            group_par30=x.group_par30,
            sector=x.sector,
            protected_group=1 - x.protected_group,
        )
        env._current_x = x_flipped  # noqa: SLF001
        obs_flipped = env._build_observation()
        rate_flipped = float(policy_fn(obs_flipped))

        # restore
        env._current_x = x
        shifts[i] = rate_flipped - rate_orig
    return shifts
