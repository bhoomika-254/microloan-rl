"""Tests for the borrower simulator."""

from __future__ import annotations

import numpy as np

from fairprice_mf.envs.borrower import BorrowerCohort, CohortParams


def test_sample_produces_in_range_features():
    cohort = BorrowerCohort(CohortParams(seed=0))
    for _ in range(100):
        x, z = cohort.sample()
        assert 0.0 <= x.income_proxy <= 1.0
        assert 0.0 <= x.prior_repayment_rate <= 1.0
        assert -1.0 <= x.region_shock_idx <= 1.0
        assert 0 <= x.age_band <= 4
        assert x.protected_group in (0, 1)
        assert 0.05 <= z.reservation_rate <= 0.60
        assert 0.005 <= z.default_prob_base <= 0.55


def test_default_probability_monotonically_increases_with_rate():
    cohort = BorrowerCohort(CohortParams(seed=0))
    x, z = cohort.sample()
    p_low = cohort.default_probability(x, z, rate=0.10)
    p_mid = cohort.default_probability(x, z, rate=0.20)
    p_high = cohort.default_probability(x, z, rate=0.35)
    assert p_low < p_mid < p_high


def test_simulate_repayment_returns_dict():
    cohort = BorrowerCohort(CohortParams(seed=0))
    x, z = cohort.sample()
    out = cohort.simulate_repayment(x, z, rate=0.2, n_installments=12, principal=500.0)
    assert "cashflows" in out
    assert out["cashflows"].shape == (12,)
    assert "defaulted" in out
    assert "p_default_at_offer" in out


def test_protected_attribute_does_not_affect_repayment_by_default():
    """The default config has protected_affects_repayment=False.
    Marginal default rates should be statistically indistinguishable
    across the protected groups in expectation."""
    cohort = BorrowerCohort(CohortParams(
        seed=0, protected_affects_repayment=False,
    ))
    defaults_g0 = []
    defaults_g1 = []
    for _ in range(2000):
        x, z = cohort.sample()
        if x.protected_group == 0:
            defaults_g0.append(z.default_prob_base)
        else:
            defaults_g1.append(z.default_prob_base)
    mean0 = float(np.mean(defaults_g0))
    mean1 = float(np.mean(defaults_g1))
    assert abs(mean0 - mean1) < 0.01
