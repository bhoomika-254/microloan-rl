"""
Fineract integration shim.

A thin FastAPI service that exposes:

  POST /price            — given a Fineract loan-application payload,
                            return a recommended interestRateDifferential
                            and a per-decision audit record.
  GET  /policy/version   — currently-loaded policy identifier.
  GET  /audit/recent     — last N decisions, suitable for compliance review.

The service does *not* call the Fineract API itself; instead, a Fineract
hook or external orchestrator calls this service before creating a loan,
takes the returned rate, and sets `interestRateDifferential` on the loan
product via the standard Fineract endpoint
`PUT /api/v1/loanproducts/{id}` or
`POST /api/v1/loans` with a per-loan override.

This separation keeps Fineract pristine (no in-tree ML dependencies) and
makes our module trivial to deploy as a sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FineractClient:
    """Minimal Fineract API client used only for audit/round-trip examples."""
    base_url: str
    tenant: str = "default"
    user: str = "mifos"
    password: str = "password"

    def get_loan_product(self, product_id: int) -> dict:
        import requests
        r = requests.get(
            f"{self.base_url}/api/v1/loanproducts/{product_id}",
            headers={"Fineract-Platform-TenantId": self.tenant},
            auth=(self.user, self.password),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def set_floating_rate_differential(
        self, product_id: int, differential: float,
    ) -> dict:
        import requests
        r = requests.put(
            f"{self.base_url}/api/v1/loanproducts/{product_id}",
            headers={"Fineract-Platform-TenantId": self.tenant},
            auth=(self.user, self.password),
            json={"interestRateDifferential": float(differential)},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()


def map_fineract_payload_to_obs(payload: dict) -> np.ndarray:
    """
    Build a 16-d observation vector from a Fineract loan-application payload.

    This is a *deliberately conservative* mapping.  Fields that aren't
    available are defaulted; the auditor logs which fields fell back.

    Expected payload keys (best-effort):
      clientId, monthlyIncome, priorLoansCount, priorRepaymentRate,
      ageBand, regionShockIndex, groupId, groupPar30, sectorCode,
      protectedGroupHint  (used only for audit, NOT for inference)
    """
    obs = np.zeros(16, dtype=np.float32)
    # 0: income proxy in [0, 1] — scale by 1000 USD/month upper bound
    obs[0] = float(np.clip(payload.get("monthlyIncome", 200.0) / 1000.0, 0.0, 1.0))
    obs[1] = float(np.clip(payload.get("priorLoansCount", 0) / 10.0, 0.0, 1.0))
    obs[2] = float(np.clip(payload.get("priorRepaymentRate", 0.0), 0.0, 1.0))
    obs[3] = float(np.clip(payload.get("ageBand", 1) / 4.0, 0.0, 1.0))
    obs[4] = float(np.clip(payload.get("regionShockIndex", 0.0), -1.0, 1.0))
    obs[5] = float(1.0 if payload.get("groupId") else 0.0)
    obs[6] = float(np.clip(payload.get("groupPar30", 0.0), 0.0, 1.0))
    # 7..11: sector one-hot
    sector = int(payload.get("sectorCode", 4))   # default OTHER
    sector = max(0, min(4, sector))
    obs[7 + sector] = 1.0
    # 12..15: shock one-hot, default baseline
    shock_code = int(payload.get("macroShockCode", 0))
    shock_code = max(0, min(3, shock_code))
    obs[12 + shock_code] = 1.0
    return obs
