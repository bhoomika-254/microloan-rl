"""Tests for the fairness auditor."""

from __future__ import annotations

import numpy as np

from fairprice_mf.agents.baselines import RuleBasedPolicy, policy_to_normalized_action
from fairprice_mf.envs.microloan_env import EnvConfig, MicroLoanPricingEnv
from fairprice_mf.fairness.auditor import FairnessAuditor, counterfactual_audit


def _run_policy(policy, env, n_episodes):
    auditor = FairnessAuditor()
    for i in range(n_episodes):
        obs, info = env.reset(seed=i)
        rate = policy(obs)
        norm_a = policy_to_normalized_action(rate, env.config)
        _, _, _, _, step_info = env.step(norm_a)
        auditor.record(step_info)
    return auditor.report()


def test_rule_based_policy_is_demographically_neutral():
    env = MicroLoanPricingEnv(EnvConfig(seed=0))
    report = _run_policy(RuleBasedPolicy(), env, n_episodes=500)
    # Rule-based policy doesn't see protected attribute; should be close to parity
    assert report.rate_parity_gap < 0.02


def test_auditor_reports_disparate_impact():
    env = MicroLoanPricingEnv(EnvConfig(seed=0))
    report = _run_policy(RuleBasedPolicy(), env, n_episodes=300)
    # DI ratio is in (0, 1]
    assert 0.0 < report.disparate_impact_ratio <= 1.0


def test_counterfactual_audit_runs():
    env = MicroLoanPricingEnv(EnvConfig(seed=0))
    rule = RuleBasedPolicy()
    shifts = counterfactual_audit(rule, env, n_samples=50, seed=0)
    assert shifts.shape == (50,)
    # rule-based policy is invariant to protected attribute (which it can't see)
    # but other observable correlates may cause tiny shifts; bound them
    assert np.mean(np.abs(shifts)) < 0.02


def test_fairness_report_pretty_prints():
    env = MicroLoanPricingEnv(EnvConfig(seed=0))
    report = _run_policy(RuleBasedPolicy(), env, n_episodes=30)
    s = report.pretty()
    assert "rate parity gap" in s
