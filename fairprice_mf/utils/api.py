"""
FastAPI sidecar exposing the trained policy as a pricing service.

Run:
    uvicorn fairprice_mf.utils.api:app --host 0.0.0.0 --port 8080

Endpoints
---------
  POST /price            — given a Fineract loan-application-like payload,
                            return a recommended interest rate and audit info.
  GET  /policy/version   — current policy id.
  GET  /audit/recent     — most recent decisions.
  GET  /health           — liveness probe.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Deque

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fairprice_mf.agents.baselines import RuleBasedPolicy
from fairprice_mf.utils.fineract_shim import map_fineract_payload_to_obs


class LoanApplication(BaseModel):
    client_id: str = Field(..., alias="clientId")
    monthly_income: float = Field(200.0, alias="monthlyIncome")
    prior_loans_count: int = Field(0, alias="priorLoansCount")
    prior_repayment_rate: float = Field(0.0, alias="priorRepaymentRate")
    age_band: int = Field(1, alias="ageBand")
    region_shock_index: float = Field(0.0, alias="regionShockIndex")
    group_id: str | None = Field(None, alias="groupId")
    group_par30: float = Field(0.0, alias="groupPar30")
    sector_code: int = Field(4, alias="sectorCode")
    macro_shock_code: int = Field(0, alias="macroShockCode")
    protected_group_hint: int | None = Field(None, alias="protectedGroupHint")

    class Config:
        populate_by_name = True


class PricingDecision(BaseModel):
    client_id: str
    recommended_rate: float
    policy_version: str
    timestamp: float
    audit_notes: dict[str, Any]


def build_app(
    policy_name: str = "rule_based",
    checkpoint: str | None = None,
    policy_version: str | None = None,
) -> FastAPI:
    """Construct a FastAPI app bound to a chosen policy.

    Parameters
    ----------
    policy_name :
        One of ``"rule_based"`` or ``"fair_cql"``. The latter requires a
        ``checkpoint`` path.
    checkpoint :
        Path to a trained Fair-CQL ``.pt`` file. Ignored for rule-based.
    policy_version :
        Override for the version string reported by ``/policy/version``.
        Defaults to the ``FAIRPRICE_POLICY_VERSION`` env var, then to
        ``"{policy_name}-v0"``.
    """
    version = policy_version or os.environ.get(
        "FAIRPRICE_POLICY_VERSION", f"{policy_name}-v0",
    )
    policy = _make_policy(policy_name, checkpoint)
    decisions: Deque[PricingDecision] = deque(maxlen=1000)

    app = FastAPI(title="FairPrice-MF Pricing Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/policy/version")
    def policy_version_ep() -> dict[str, str]:
        return {"version": version, "policy": policy_name}

    @app.post("/price", response_model=PricingDecision)
    def price(application: LoanApplication) -> PricingDecision:
        payload = application.model_dump(by_alias=True)
        try:
            obs = map_fineract_payload_to_obs(payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        rate = float(policy(obs))

        decision = PricingDecision(
            client_id=application.client_id,
            recommended_rate=rate,
            policy_version=version,
            timestamp=time.time(),
            audit_notes={
                "obs_vector_norm": float(np.linalg.norm(obs)),
                "protected_group_observed_for_audit_only": application.protected_group_hint,
                "policy_kind": policy_name,
            },
        )
        decisions.append(decision)
        return decision

    @app.get("/audit/recent")
    def audit_recent(n: int = 50) -> list[PricingDecision]:
        n = max(1, min(n, len(decisions) or 1))
        return list(decisions)[-n:]

    return app


def _make_policy(name: str, checkpoint: str | None):
    if name == "rule_based":
        return RuleBasedPolicy()
    if name == "fair_cql":
        if checkpoint is None:
            raise ValueError("fair_cql policy requires a checkpoint path")
        return _load_cql_policy_for_serving(checkpoint)
    raise ValueError(f"unknown policy '{name}'")


def _load_cql_policy_for_serving(checkpoint: str):
    """Wrap a saved Fair-CQL Q-network in a (obs -> rate) callable."""
    import torch  # local import — keep import light by default

    from fairprice_mf.agents.fair_cql import CQLConfig, FairCQLTrainer

    state = torch.load(checkpoint, map_location="cpu")
    cfg = CQLConfig()
    trainer = FairCQLTrainer(cfg)
    trainer.q.load_state_dict(state["q_state_dict"])
    grid = np.asarray(
        state.get("action_grid", np.linspace(0.06, 0.40, cfg.action_dim)),
        dtype=np.float32,
    )

    def policy(obs: np.ndarray) -> float:
        idx = trainer.act(np.asarray(obs, dtype=np.float32))
        return float(grid[idx])

    return policy


# Convenience module-level app for `uvicorn fairprice_mf.utils.api:app`
app = build_app()
