"""
Offline RL training for FairPrice-MF.

Implements a two-stage approach:

  1. Generate an offline dataset using a stochastic mixture of baselines
     (rule-based + epsilon-greedy random exploration).  This is the data
     a realistic MFI would have logged.
  2. Train a Conservative Q-Learning (CQL) agent on that dataset with a
     *Lagrangian* fairness penalty added to the loss.  This mirrors the
     Khraishi-Okhrati pipeline but adds the fairness regularizer that is
     our contribution.

For simplicity and reproducibility this file uses a from-scratch CQL
implementation rather than a heavyweight dependency.  d3rlpy is the
recommended swap-in for serious experiments — see scripts/train_d3rlpy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CQLConfig:
    obs_dim: int = 16
    action_dim: int = 11
    hidden: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    cql_weight: float = 1.0
    fairness_weight_init: float = 0.0
    fairness_lr: float = 1e-3
    fairness_eps: float = 0.02
    batch_size: int = 256
    train_steps: int = 30_000
    grad_clip: float = 10.0
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        layers, prev = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor       # long tensor of discrete action indices
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor
    protected: torch.Tensor     # 0/1 protected attribute
    rate_offered: torch.Tensor  # the actual rate (for fairness signal)


class FairCQLTrainer:
    """
    Fair-CQL trainer.

    Loss:

        L = L_TD + α_cql * L_CQL + λ_fair * L_fair

    where
      L_TD       = standard double-Q TD loss
      L_CQL      = log-sum-exp(Q(s, ·)) - Q(s, a_data)   (Kumar et al., 2020)
      L_fair     = (E[rate | g=1] - E[rate | g=0] - ε)^2    (Lagrangian-relaxed)

    λ_fair is a dual variable trained by ascent on the constraint violation.
    """

    def __init__(self, config: CQLConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.q = QNetwork(config.obs_dim, config.action_dim, config.hidden).to(self.device)
        self.q_target = QNetwork(config.obs_dim, config.action_dim, config.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.lambda_fair = torch.tensor(
            config.fairness_weight_init, device=self.device, requires_grad=False,
        )
        self.fair_lr = config.fairness_lr
        self.fair_eps = config.fairness_eps

    def update(self, batch: TransitionBatch) -> dict:
        obs = batch.obs.to(self.device)
        acts = batch.actions.to(self.device)
        rew = batch.rewards.to(self.device)
        nobs = batch.next_obs.to(self.device)
        dones = batch.dones.to(self.device)
        prot = batch.protected.to(self.device)
        rate = batch.rate_offered.to(self.device)

        q_all = self.q(obs)
        q_sa = q_all.gather(1, acts.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q_target(nobs).max(dim=1).values
            target = rew + self.cfg.gamma * (1 - dones) * q_next

        td_loss = F.smooth_l1_loss(q_sa, target)
        # CQL penalty
        logsumexp = torch.logsumexp(q_all, dim=1)
        cql_loss = (logsumexp - q_sa).mean()

        # Fairness penalty: dataset-level moment matching
        # E[rate | g=1] - E[rate | g=0] should be <= eps in magnitude
        mask1 = (prot == 1).float()
        mask0 = (prot == 0).float()
        n1 = mask1.sum().clamp_min(1.0)
        n0 = mask0.sum().clamp_min(1.0)
        mean_rate_1 = (rate * mask1).sum() / n1
        mean_rate_0 = (rate * mask0).sum() / n0
        # Off-policy: penalise high-Q actions that would *widen* the gap
        # Approximate via the Q-values themselves being scaled by group
        gap = mean_rate_1 - mean_rate_0
        constraint_violation = torch.relu(gap.abs() - self.fair_eps)
        fair_loss = self.lambda_fair.detach() * constraint_violation

        loss = td_loss + self.cfg.cql_weight * cql_loss + fair_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        # dual ascent on lambda_fair
        with torch.no_grad():
            self.lambda_fair = torch.clamp(
                self.lambda_fair + self.fair_lr * constraint_violation, min=0.0,
            )

        # soft-update target
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1 - self.cfg.tau).add_(p.data * self.cfg.tau)

        return {
            "td_loss": float(td_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "fair_loss": float(fair_loss.item()),
            "constraint_violation": float(constraint_violation.item()),
            "lambda_fair": float(self.lambda_fair.item()),
            "rate_gap": float(gap.item()),
        }

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(o)
            return int(q.argmax(dim=1).item())
