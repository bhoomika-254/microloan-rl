"""Train a Fair-CQL policy offline and save a reloadable checkpoint.

Example::

    python scripts/train_offline.py \\
        --dataset-episodes 50000 \\
        --train-steps 30000 \\
        --fairness-weight-init 1.0 \\
        --fairness-lr 5e-3 \\
        --out runs/fair_cql_v0.pt

The output ``.pt`` can be loaded by ``fairprice-serve --policy fair_cql --checkpoint ...``
or by ``fairprice-audit --policy fair_cql --checkpoint ...``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fairprice_mf.agents.data_gen import dataset_to_tensors, generate_offline_dataset
from fairprice_mf.agents.fair_cql import CQLConfig, FairCQLTrainer, TransitionBatch
from fairprice_mf.envs.microloan_env import EnvConfig, MicroLoanPricingEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-episodes", type=int, default=50_000)
    parser.add_argument("--train-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cql-weight", type=float, default=1.0)
    parser.add_argument("--fairness-weight-init", type=float, default=1.0)
    parser.add_argument("--fairness-lr", type=float, default=5e-3)
    parser.add_argument("--fairness-eps", type=float, default=0.02)
    parser.add_argument("--exploration-rate", type=float, default=0.15,
                        help="ε for the data-collection mixture (rule-based + ε·random).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/fair_cql.pt"))
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = MicroLoanPricingEnv(EnvConfig(seed=args.seed))

    print(f"[{time.strftime('%H:%M:%S')}] Generating {args.dataset_episodes} offline transitions...")
    t0 = time.time()
    dataset = generate_offline_dataset(
        n_episodes=args.dataset_episodes,
        exploration_rate=args.exploration_rate,
        seed=args.seed,
    )
    obs, actions, rewards, next_obs, dones, protected, rate_offered = dataset_to_tensors(dataset)
    print(f"  obs shape: {tuple(obs.shape)}, "
          f"mean reward: {rewards.mean():+.3f}, "
          f"protected balance: {protected.float().mean():.2f}, "
          f"took {time.time() - t0:.1f}s")

    cfg = CQLConfig(
        obs_dim=obs.shape[1],
        action_dim=11,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        cql_weight=args.cql_weight,
        fairness_weight_init=args.fairness_weight_init,
        fairness_lr=args.fairness_lr,
        fairness_eps=args.fairness_eps,
    )
    trainer = FairCQLTrainer(cfg)

    print(f"[{time.strftime('%H:%M:%S')}] Training Fair-CQL for {args.train_steps} steps...")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    metrics_log: list[dict] = []
    for step in range(args.train_steps):
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
        m = trainer.update(batch)
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  step {step:>6} ({elapsed:>5.1f}s): "
                f"td={m.get('td_loss', 0):.3f} "
                f"cql={m.get('cql_loss', 0):.3f} "
                f"fair={m.get('fair_loss', 0):.4f} "
                f"λ={m.get('lambda_fair', 0):.3f} "
                f"gap={m.get('rate_gap', 0):+.4f}"
            )
            metrics_log.append({"step": step, **{k: float(v) for k, v in m.items()}})

    # Save checkpoint
    action_grid = (
        np.asarray(env.config.discrete_actions)
        if env.config.discrete_actions is not None
        else np.linspace(env.config.r_min, env.config.regulatory_cap, cfg.action_dim)
    )
    torch.save(
        {
            "q_state_dict": trainer.q.state_dict(),
            "q_target_state_dict": trainer.q_target.state_dict(),
            "config": cfg.__dict__,
            "action_grid": action_grid.tolist(),
            "obs_dim": int(obs.shape[1]),
            "metrics_log": metrics_log,
        },
        args.out,
    )
    print(f"\n[{time.strftime('%H:%M:%S')}] Saved checkpoint → {args.out}")
    # Also dump training-log JSON for plots
    log_path = args.out.with_suffix(".log.json")
    log_path.write_text(json.dumps(metrics_log, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] Saved training log → {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
