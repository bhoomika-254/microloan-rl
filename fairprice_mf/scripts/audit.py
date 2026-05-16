"""``fairprice-audit`` — run a fairness audit over a policy/environment combo.

Example::

    fairprice-audit --policy rule_based --episodes 5000 --seed 0
    fairprice-audit --policy profit_max --episodes 5000 --report-json audit.json

The audit produces:
  - demographic-parity gap on offered rates
  - disparate-impact ratio (80% rule)
  - per-group default rate
  - counterfactual rate-shift on a flipped-protected-attribute replay

It does NOT require a trained model — it works on any callable mapping
``obs -> rate``. To audit a trained Fair-CQL checkpoint, use ``--policy fair_cql
--checkpoint path/to/model.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from fairprice_mf.agents.baselines import (
    ProfitMaxLogisticPolicy,
    RandomPolicy,
    RuleBasedPolicy,
)
from fairprice_mf.envs.microloan_env import EnvConfig, MicroLoanPricingEnv
from fairprice_mf.fairness.auditor import FairnessAuditor, counterfactual_audit


def _build_policy(name: str, env: MicroLoanPricingEnv, checkpoint: str | None) -> Callable:
    if name == "rule_based":
        return RuleBasedPolicy()
    if name == "random":
        return RandomPolicy(low=env.config.r_min, high=env.config.regulatory_cap, seed=0)
    if name == "profit_max":
        pol = ProfitMaxLogisticPolicy()
        rule = RuleBasedPolicy()
        obs_list, rate_list, acc_list, def_list = [], [], [], []
        obs, _info = env.reset()
        for _ in range(2000):
            rate = rule(obs)
            action = _rate_to_action(rate, env)
            nxt, _, terminated, _t, step = env.step(action)
            obs_list.append(obs)
            rate_list.append(rate)
            acc_list.append(int(step["accepted"]))
            def_list.append(int(step.get("default_realized", 0)))
            obs = nxt if not terminated else env.reset()[0]
        pol.fit(
            np.array(obs_list), np.array(rate_list),
            np.array(acc_list), np.array(def_list),
        )
        return pol
    if name == "fair_cql":
        if checkpoint is None:
            raise SystemExit("--checkpoint required for --policy fair_cql")
        return _load_cql_policy(checkpoint, env)
    raise SystemExit(f"unknown policy '{name}'")


def _rate_to_action(rate: float, env: MicroLoanPricingEnv):
    """Convert a raw rate to the action format the env wants."""
    if env.config.discrete_actions is not None:
        # nearest discrete bucket
        grid = np.asarray(env.config.discrete_actions, dtype=np.float32)
        return int(np.argmin(np.abs(grid - rate)))
    # continuous: normalized in [-1, 1]
    lo, hi = env.config.r_min, env.config.regulatory_cap
    norm = 2.0 * (rate - lo) / (hi - lo) - 1.0
    return np.array([np.clip(norm, -1.0, 1.0)], dtype=np.float32)


def _load_cql_policy(checkpoint: str, env: MicroLoanPricingEnv):
    import torch  # local import — torch is heavy
    from fairprice_mf.agents.fair_cql import CQLConfig, FairCQLTrainer

    cfg = CQLConfig(obs_dim=env.observation_space.shape[0])
    trainer = FairCQLTrainer(cfg)
    state = torch.load(checkpoint, map_location="cpu")
    trainer.q.load_state_dict(state["q_state_dict"])

    def policy(obs: np.ndarray) -> float:
        idx = trainer.act(obs)
        grid = (env.config.discrete_actions
                or np.linspace(env.config.r_min, env.config.regulatory_cap, cfg.action_dim).tolist())
        return float(np.asarray(grid)[idx])

    return policy


def run_audit(
    policy_name: str,
    episodes: int,
    seed: int,
    checkpoint: str | None,
) -> dict:
    env = MicroLoanPricingEnv(EnvConfig(seed=seed))
    policy = _build_policy(policy_name, env, checkpoint)
    auditor = FairnessAuditor()

    obs, info = env.reset(seed=seed)
    for _ in range(episodes):
        rate = float(policy(obs))
        action = _rate_to_action(rate, env)
        next_obs, reward, terminated, _truncated, step = env.step(action)
        # merge reset-info (protected) with step-info (rate, accepted, default)
        record = {**info, **step, "reward": float(reward)}
        auditor.record(record)
        if terminated:
            obs, info = env.reset()
        else:
            obs = next_obs

    report = auditor.report()

    # Counterfactual sweep
    cf_shifts = counterfactual_audit(policy, env, n_samples=min(1000, episodes), seed=seed)
    cf_mean = float(np.mean(cf_shifts)) if len(cf_shifts) else None
    cf_p95 = float(np.percentile(np.abs(cf_shifts), 95)) if len(cf_shifts) else None

    return {
        "policy": policy_name,
        "episodes": episodes,
        "seed": seed,
        "parity_gap": float(report.rate_parity_gap),
        "disparate_impact": float(report.disparate_impact_ratio),
        "default_rate_g0": float(report.per_group_default_rate.get(0, float("nan"))),
        "default_rate_g1": float(report.per_group_default_rate.get(1, float("nan"))),
        "mean_rate_g0": float(report.mean_rate_by_group.get(0, float("nan"))),
        "mean_rate_g1": float(report.mean_rate_by_group.get(1, float("nan"))),
        "acceptance_rate_g0": float(report.acceptance_rate_by_group.get(0, float("nan"))),
        "acceptance_rate_g1": float(report.acceptance_rate_by_group.get(1, float("nan"))),
        "passes": report.passes(),
        "counterfactual_mean_shift": cf_mean,
        "counterfactual_p95_abs_shift": cf_p95,
        "pretty": report.pretty(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fairprice-audit", description=__doc__)
    parser.add_argument(
        "--policy",
        default="rule_based",
        choices=["rule_based", "random", "profit_max", "fair_cql"],
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a trained Fair-CQL .pt checkpoint")
    parser.add_argument("--report-json", type=Path, default=None,
                        help="If set, write the audit dict here as JSON")
    args = parser.parse_args(argv)

    result = run_audit(args.policy, args.episodes, args.seed, args.checkpoint)
    print(result["pretty"])
    cf_val = result["counterfactual_mean_shift"]
    if cf_val is not None:
        print(f"\n  Counterfactual mean rate shift: {cf_val:+.4f}")
    print(f"  Parity gap passes (ε=0.02):    {result['passes']['rate_parity']}")
    print(f"  Disparate impact ≥ 0.80:       {result['passes']['disparate_impact']}")

    if args.report_json is not None:
        out = {k: v for k, v in result.items() if k != "pretty"}
        args.report_json.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nWrote JSON report → {args.report_json}")

    # exit non-zero if fairness fails — useful for CI gating
    if not all(result["passes"].values()):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
