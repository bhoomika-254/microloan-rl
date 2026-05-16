"""End-to-end benchmark: rule-based vs profit-max vs Fair-CQL.

Reports:
  - mean reward per episode
  - fairness metrics (parity gap, disparate impact)
  - per-group default rates
  - acceptance rate

Used both as a sanity check (``--smoke``) in CI and as the main benchmarking
entry point. A full run with ``--episodes 5000 --train-steps 10000`` takes
a couple of minutes on CPU.

Example::

    python scripts/run_benchmark.py --episodes 5000 --train-steps 10000
    python scripts/run_benchmark.py --smoke   # 200 episodes / 100 train steps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from fairprice_mf.agents.baselines import (
    ProfitMaxLogisticPolicy,
    RuleBasedPolicy,
)
from fairprice_mf.agents.data_gen import (
    dataset_to_tensors,
    discrete_idx_to_normalized_action,
    generate_offline_dataset,
    rate_to_discrete_idx,
)
from fairprice_mf.agents.fair_cql import CQLConfig, FairCQLTrainer, TransitionBatch
from fairprice_mf.envs.microloan_env import EnvConfig, MicroLoanPricingEnv
from fairprice_mf.fairness.auditor import FairnessAuditor


def evaluate_policy(policy_fn, env: MicroLoanPricingEnv, episodes: int, seed: int) -> dict:
    """Run a policy in the env for `episodes` episodes and collect metrics."""
    auditor = FairnessAuditor()
    rewards = []
    accepted_count = 0
    obs, info = env.reset(seed=seed)
    for _ in range(episodes):
        rate = float(policy_fn(obs))
        # convert rate -> env action
        if env.config.discrete_actions is not None:
            idx = rate_to_discrete_idx(rate, env)
            action = idx
        else:
            idx = rate_to_discrete_idx(rate, env)
            action = discrete_idx_to_normalized_action(idx, env)
        next_obs, reward, terminated, _truncated, step = env.step(action)
        record = {**info, **step, "reward": float(reward)}
        auditor.record(record)
        rewards.append(float(reward))
        accepted_count += int(step["accepted"])
        if terminated:
            obs, info = env.reset()
        else:
            obs = next_obs

    report = auditor.report()
    passes = report.passes()
    return {
        "mean_reward": float(np.mean(rewards)),
        "acceptance_rate": accepted_count / episodes,
        "parity_gap": float(report.rate_parity_gap),
        "disparate_impact": float(report.disparate_impact_ratio),
        "default_rate_g0": float(report.per_group_default_rate.get(0, float("nan"))),
        "default_rate_g1": float(report.per_group_default_rate.get(1, float("nan"))),
        "mean_rate_g0": float(report.mean_rate_by_group.get(0, float("nan"))),
        "mean_rate_g1": float(report.mean_rate_by_group.get(1, float("nan"))),
        "passes_parity": passes["rate_parity"],
        "passes_di": passes["disparate_impact"],
    }


def train_fair_cql(
    n_dataset_episodes: int,
    train_steps: int,
    seed: int,
    fairness_weight_init: float = 1.0,
    fairness_lr: float = 5e-3,
) -> tuple[FairCQLTrainer, dict]:
    """Generate offline data, train Fair-CQL, return trainer + final metrics."""
    import torch

    print(f"  ↳ generating offline dataset ({n_dataset_episodes} eps)...")
    dataset = generate_offline_dataset(
        n_episodes=n_dataset_episodes, seed=seed,
    )
    obs, actions, rewards, next_obs, dones, protected, rate_offered = dataset_to_tensors(dataset)

    cfg = CQLConfig(
        obs_dim=obs.shape[1],
        action_dim=11,
        train_steps=train_steps,
        batch_size=min(256, len(dataset)),
        fairness_weight_init=fairness_weight_init,
        fairness_lr=fairness_lr,
    )
    trainer = FairCQLTrainer(cfg)

    print(f"  ↳ training Fair-CQL for {train_steps} steps...")
    last_metrics: dict = {}
    rng = np.random.default_rng(seed)
    for step in range(train_steps):
        idx = rng.integers(0, len(dataset), size=cfg.batch_size)
        batch = TransitionBatch(
            obs=obs[idx],
            actions=actions[idx],
            rewards=rewards[idx],
            next_obs=next_obs[idx],
            dones=dones[idx],
            protected=protected[idx],
            rate_offered=rate_offered[idx],
        )
        last_metrics = trainer.update(batch)
        if step % max(1, train_steps // 10) == 0:
            print(
                f"    step {step:>6}: "
                f"td={last_metrics.get('td_loss', 0):.3f} "
                f"cql={last_metrics.get('cql_loss', 0):.3f} "
                f"fair={last_metrics.get('fair_loss', 0):.4f} "
                f"λ={last_metrics.get('lambda_fair', 0):.3f} "
                f"gap={last_metrics.get('rate_gap', 0):+.4f}"
            )
    return trainer, last_metrics


def make_fair_cql_policy(trainer: FairCQLTrainer, env: MicroLoanPricingEnv):
    grid = np.asarray(
        env.config.discrete_actions
        if env.config.discrete_actions is not None
        else np.linspace(env.config.r_min, env.config.regulatory_cap, trainer.cfg.action_dim).tolist(),
        dtype=np.float32,
    )

    def policy(obs: np.ndarray) -> float:
        idx = trainer.act(obs.astype(np.float32))
        return float(grid[idx])

    return policy


def make_profit_max_policy(env: MicroLoanPricingEnv, seed: int):
    rule = RuleBasedPolicy()
    pol = ProfitMaxLogisticPolicy()
    obs_list, rate_list, acc_list, def_list = [], [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(2000):
        rate = rule(obs)
        if env.config.discrete_actions is not None:
            action = rate_to_discrete_idx(rate, env)
        else:
            action = discrete_idx_to_normalized_action(rate_to_discrete_idx(rate, env), env)
        nxt, _r, terminated, _t, step = env.step(action)
        obs_list.append(obs)
        rate_list.append(rate)
        acc_list.append(int(step["accepted"]))
        def_list.append(int(step.get("default_realized", 0)))
        obs = nxt if not terminated else env.reset()[0]
    pol.fit(
        np.array(obs_list),
        np.array(rate_list),
        np.array(acc_list),
        np.array(def_list),
    )
    return pol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Evaluation episodes per policy.")
    parser.add_argument("--train-steps", type=int, default=10_000,
                        help="Number of Fair-CQL gradient steps.")
    parser.add_argument("--dataset-episodes", type=int, default=None,
                        help="Episodes for offline data collection (default: --episodes).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: tiny budgets, skip Fair-CQL eval slowdowns.")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--skip-cql", action="store_true",
                        help="Skip the (slow) Fair-CQL training step.")
    args = parser.parse_args(argv)

    if args.smoke:
        args.episodes = min(args.episodes, 200)
        args.train_steps = min(args.train_steps, 100)
    dataset_eps = args.dataset_episodes or args.episodes

    env = MicroLoanPricingEnv(EnvConfig(seed=args.seed))

    results: dict[str, dict] = {}

    print("=" * 64)
    print("Baseline 1: rule-based 3-bucket pricing")
    print("=" * 64)
    t0 = time.time()
    results["rule_based"] = evaluate_policy(RuleBasedPolicy(), env, args.episodes, args.seed)
    print(f"  done in {time.time() - t0:.1f}s — reward {results['rule_based']['mean_reward']:+.3f}, "
          f"parity {results['rule_based']['parity_gap']:+.4f}")

    print()
    print("=" * 64)
    print("Baseline 2: profit-max logistic (Phillips/Ban style)")
    print("=" * 64)
    env_pm = MicroLoanPricingEnv(EnvConfig(seed=args.seed + 1))
    t0 = time.time()
    pm = make_profit_max_policy(env_pm, args.seed + 1)
    results["profit_max"] = evaluate_policy(pm, env, args.episodes, args.seed)
    print(f"  done in {time.time() - t0:.1f}s — reward {results['profit_max']['mean_reward']:+.3f}, "
          f"parity {results['profit_max']['parity_gap']:+.4f}")

    if not args.skip_cql:
        print()
        print("=" * 64)
        print("Method: FairPrice-MF (Fair-CQL with λ-Lagrangian fairness)")
        print("=" * 64)
        t0 = time.time()
        trainer, last_metrics = train_fair_cql(
            n_dataset_episodes=dataset_eps,
            train_steps=args.train_steps,
            seed=args.seed,
        )
        policy = make_fair_cql_policy(trainer, env)
        results["fair_cql"] = evaluate_policy(policy, env, args.episodes, args.seed)
        results["fair_cql"]["train_steps"] = args.train_steps
        results["fair_cql"]["final_lambda_fair"] = last_metrics.get("lambda_fair", 0)
        print(f"  done in {time.time() - t0:.1f}s — reward {results['fair_cql']['mean_reward']:+.3f}, "
              f"parity {results['fair_cql']['parity_gap']:+.4f}, "
              f"λ_fair={results['fair_cql']['final_lambda_fair']:.3f}")

    # Pretty summary
    print()
    print("=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)
    header = f"{'policy':<14} {'reward':>10} {'accept':>8} {'parity':>10} {'DI':>8} {'g0/g1 def':>14}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        def_str = f"{r['default_rate_g0']:.2f}/{r['default_rate_g1']:.2f}"
        print(
            f"{name:<14} "
            f"{r['mean_reward']:>+10.3f} "
            f"{r['acceptance_rate']:>8.2%} "
            f"{r['parity_gap']:>+10.4f} "
            f"{r['disparate_impact']:>8.3f} "
            f"{def_str:>14}"
        )

    if args.report_json:
        args.report_json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nWrote → {args.report_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
