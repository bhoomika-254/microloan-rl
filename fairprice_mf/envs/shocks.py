"""
Economic shock generator for robustness testing.

This module produces structured perturbations to the simulator that mimic
realistic macroeconomic stressors faced by microfinance institutions:

  * `InflationShock`         — broad upward shift in reservation rates and
                               default probabilities.
  * `DroughtShock`           — sector-conditional, hits agriculture hardest.
  * `PandemicShock`          — temporary collapse in retail and services
                               cashflows with a recovery curve.
  * `BaselineRegime`         — no shock; the calm-water baseline.

Domain randomization training samples a shock from the menu at each episode
to encourage the learned policy to be robust across regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from fairprice_mf.envs.borrower import BorrowerFeatures, BorrowerLatents, BusinessSector


class EconomicShock(Protocol):
    """Interface for a shock that mutates borrower latents in-place."""

    name: str

    def apply(
        self,
        x: BorrowerFeatures,
        z: BorrowerLatents,
        t_in_loan: int,
        rng: np.random.Generator,
    ) -> BorrowerLatents:
        ...


@dataclass
class BaselineRegime:
    name: str = "baseline"

    def apply(self, x, z, t_in_loan, rng):
        return z


@dataclass
class InflationShock:
    """Broad upward shift; affects all borrowers but income-sensitive ones more."""
    name: str = "inflation"
    severity: float = 0.05  # 5 percentage points equivalent

    def apply(self, x, z, t_in_loan, rng):
        bump = self.severity * (1 + 0.5 * (1 - x.income_proxy))
        new_default = float(np.clip(z.default_prob_base + bump * 0.7, 0.0, 0.95))
        new_res = float(np.clip(z.reservation_rate + bump * 0.3, 0.05, 0.65))
        return BorrowerLatents(
            reservation_rate=new_res,
            default_prob_base=new_default,
            repayment_corr=z.repayment_corr,
        )


@dataclass
class DroughtShock:
    """Hits agriculture hard, retail mildly, services minimally."""
    name: str = "drought"
    severity: float = 0.12

    SECTOR_MULTIPLIERS = {
        BusinessSector.AGRICULTURE: 1.0,
        BusinessSector.RETAIL: 0.35,
        BusinessSector.SERVICES: 0.10,
        BusinessSector.MANUFACTURING: 0.15,
        BusinessSector.OTHER: 0.20,
    }

    def apply(self, x, z, t_in_loan, rng):
        mult = self.SECTOR_MULTIPLIERS[x.sector]
        bump = self.severity * mult
        new_default = float(np.clip(z.default_prob_base + bump, 0.0, 0.95))
        return BorrowerLatents(
            reservation_rate=z.reservation_rate,
            default_prob_base=new_default,
            repayment_corr=min(0.9, z.repayment_corr + 0.1 * mult),
        )


@dataclass
class PandemicShock:
    """Big retail/services hit with a recovery curve over `recovery_periods`."""
    name: str = "pandemic"
    initial_severity: float = 0.20
    recovery_periods: int = 6

    SECTOR_MULTIPLIERS = {
        BusinessSector.AGRICULTURE: 0.10,
        BusinessSector.RETAIL: 1.00,
        BusinessSector.SERVICES: 1.20,
        BusinessSector.MANUFACTURING: 0.60,
        BusinessSector.OTHER: 0.50,
    }

    def apply(self, x, z, t_in_loan, rng):
        recovery_frac = max(0.0, 1.0 - t_in_loan / self.recovery_periods)
        mult = self.SECTOR_MULTIPLIERS[x.sector]
        bump = self.initial_severity * mult * recovery_frac
        new_default = float(np.clip(z.default_prob_base + bump, 0.0, 0.95))
        return BorrowerLatents(
            reservation_rate=z.reservation_rate,
            default_prob_base=new_default,
            repayment_corr=z.repayment_corr,
        )


class ShockSampler:
    """
    Sample a shock at episode reset for domain-randomization training.

    Args
    ----
    weights: probability of each shock type.  Order:
        (baseline, inflation, drought, pandemic).  Defaults to
        (0.55, 0.20, 0.15, 0.10) which is the recommended training mix.
    """

    def __init__(
        self,
        weights: Optional[tuple[float, float, float, float]] = None,
        seed: Optional[int] = None,
    ):
        if weights is None:
            weights = (0.55, 0.20, 0.15, 0.10)
        assert abs(sum(weights) - 1.0) < 1e-6, "weights must sum to 1"
        self.weights = weights
        self.rng = np.random.default_rng(seed)
        self.shocks: list[EconomicShock] = [
            BaselineRegime(),
            InflationShock(),
            DroughtShock(),
            PandemicShock(),
        ]

    def sample(self) -> EconomicShock:
        idx = int(self.rng.choice(len(self.shocks), p=self.weights))
        return self.shocks[idx]

    def get_by_name(self, name: str) -> EconomicShock:
        for s in self.shocks:
            if s.name == name:
                return s
        raise KeyError(f"Unknown shock: {name}")
