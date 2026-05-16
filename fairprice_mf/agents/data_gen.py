"""
Generate an offline dataset of (obs, action, reward, next_obs, info)
transitions using a stochastic mixture of baseline policies.

This mimics how an MFI would build a logged dataset: most decisions made
by the rule-based system, occasional ε-greedy exploration injected to
ensure coverage of the rate space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fairprice_mf.agents.baselines import RandomPolicy, RuleBasedPolicy
from fairprice_mf.envs.microloan_env import EnvConfig, MicroLoanPricingEnv


@dataclass
class LoggedTransition:
    obs: np.ndarray
    action_idx: int      # discretized for CQL
    rate_offered: float
    reward: float
    next_obs: np.ndarray
    done: bool
    protected_group: int
    sector: int
    accepted: bool
    default_realized: bool
    shock: str


def rate_to_discrete_idx(rate: float, env: MicroLoanPricingEnv) -> int:
    cfg = env.config
    grid = np.linspace(cfg.r_min, min(cfg.r_max, cfg.regulatory_cap), 11)
    return int(np.argmin(np.abs(grid - rate)))


def discrete_idx_to_normalized_action(idx: int, env: MicroLoanPricingEnv) -> np.ndarray:
    cfg = env.config
    grid = np.linspace(cfg.r_min, min(cfg.r_max, cfg.regulatory_cap), 11)
    rate = float(grid[idx])
    a = (rate - cfg.r_min) / max(1e-6, cfg.regulatory_cap - cfg.r_min)
    return np.array([float(np.clip(a, 0.0, 1.0))], dtype=np.float32)


def generate_offline_dataset(
    n_episodes: int = 50_000,
    exploration_rate: float = 0.15,
    seed: int = 0,
    env_config: Optional[EnvConfig] = None,
) -> list[LoggedTransition]:
    rng = np.random.default_rng(seed)
    env = MicroLoanPricingEnv(env_config or EnvConfig(seed=seed))
    rule = RuleBasedPolicy()
    random_pi = RandomPolicy(low=env.config.r_min, high=env.config.regulatory_cap, seed=seed + 1)
    is_discrete = env.config.discrete_actions is not None

    transitions = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        # decide which policy logs this decision
        use_random = rng.random() < exploration_rate
        rate = random_pi(obs) if use_random else rule(obs)
        action_idx = rate_to_discrete_idx(rate, env)
        # action for env: integer if discrete, normalized vector if continuous
        if is_discrete:
            env_action = action_idx
        else:
            env_action = discrete_idx_to_normalized_action(action_idx, env)
        next_obs, reward, terminated, truncated, step_info = env.step(env_action)
        transitions.append(LoggedTransition(
            obs=obs.copy(),
            action_idx=action_idx,
            rate_offered=step_info["rate_offered"],
            reward=float(reward),
            next_obs=next_obs.copy(),
            done=bool(terminated),
            protected_group=int(info["protected_group"]),
            sector=int(info["sector"]),
            accepted=bool(step_info["accepted"]),
            default_realized=bool(step_info["default_realized"]),
            shock=step_info["shock"],
        ))
    return transitions


def dataset_to_tensors(dataset: list[LoggedTransition]):
    """Pack a list of transitions into tensors for FairCQLTrainer.update()."""
    import torch

    obs = torch.tensor(np.stack([t.obs for t in dataset]), dtype=torch.float32)
    actions = torch.tensor([t.action_idx for t in dataset], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in dataset], dtype=torch.float32)
    next_obs = torch.tensor(np.stack([t.next_obs for t in dataset]), dtype=torch.float32)
    dones = torch.tensor([float(t.done) for t in dataset], dtype=torch.float32)
    protected = torch.tensor([t.protected_group for t in dataset], dtype=torch.long)
    rate_offered = torch.tensor([t.rate_offered for t in dataset], dtype=torch.float32)
    return obs, actions, rewards, next_obs, dones, protected, rate_offered
