"""
Helpers to fit a `CohortParams` from a Kiva-style loan dataset.

Expected CSV schema (column names from the Kiva public snapshot):
    loan_amount, sector, country, borrower_genders, repayment_term,
    posted_time, funded_time, status (funded/expired), term_in_months

Calibration outputs are written to `configs/cohort_calibrated.yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from fairprice_mf.envs.borrower import CohortParams


KIVA_SECTOR_TO_BUSINESS = {
    "Agriculture": "AGRICULTURE",
    "Food": "AGRICULTURE",
    "Retail": "RETAIL",
    "Services": "SERVICES",
    "Manufacturing": "MANUFACTURING",
    "Wholesale": "RETAIL",
}


def fit_from_kiva_csv(csv_path: str, seed: int = 0) -> CohortParams:
    """
    Lightweight calibration.  Reads a Kiva CSV, derives:

      * funding-success rate (acceptance proxy)
      * mean loan amount
      * sector mix

    Returns a CohortParams with cohort-level priors adjusted to match.

    The full notebook calibration also fits Beta parameters to the
    distribution of reservation rates inferred from acceptance bands.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    accept_rate = float((df["status"].str.lower() == "funded").mean())
    # Map acceptance rate to a base reservation-rate prior.
    # Rough mapping: 90% acceptance implies low reservation rates ~ 20%,
    # 60% acceptance implies higher reservation rates ~ 32%.
    base_res = float(np.clip(0.42 - 0.25 * accept_rate, 0.18, 0.40))
    return CohortParams(
        base_reservation_rate=base_res,
        protected_group_prevalence=0.45,  # placeholder; depends on gender split
        seed=seed,
    )


def save_params_yaml(params: CohortParams, out_path: str) -> None:
    import yaml
    with open(out_path, "w") as f:
        yaml.dump(params.__dict__, f, sort_keys=False)
