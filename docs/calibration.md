# Calibration Protocol

The `MicroLoanPricing-v0` environment is a parameterised simulator. This document explains how its parameters are chosen so that simulation results are not pure fiction.

> **TL;DR.** We anchor the simulator's three most consequential distributions — reservation rate, default probability, and macro-shock effect sizes — against the public Kiva loan-snapshot dataset and the published World Bank microfinance literature. Where data are missing or ethically off-limits, we document the prior we use and treat it as a hyperparameter to sweep, not a fact.

---

## 1. What we calibrate against

We use two public, freely-redistributable sources:

| Source | What we extract | Where it feeds in |
|---|---|---|
| [Kiva snapshot](https://build.kiva.org/docs/data/snapshots) | Sector distribution, mean loan size, funding-success rate, geographic shock indices | `BorrowerCohort` priors; `fairprice_mf.calibration.kiva.fit_from_kiva_csv` |
| World Bank / CGAP microfinance briefs | Approximate ranges for portfolio-at-risk-30 (PAR-30) and average interest yields in MFIs | Bounds for `default_prob_base` and `reservation_rate_alpha/beta` |

We deliberately do **not** use lender-side Kiva data (the bias literature on Kiva — Sarkar & Alvari 2020 — establishes that *lenders* discriminate, but the funding-success rate after platform allocation is closer to a borrower-side signal).

---

## 2. The parameters and what they mean

`BorrowerCohort` takes a `CohortParams` dataclass. The key fields:

| Field | Default | Calibration source |
|---|---|---|
| `reservation_rate_alpha`, `reservation_rate_beta` | (2.0, 8.0) | Beta-fit to Kiva funding-success vs. offered yield. Mean ≈ 0.20. |
| `default_prob_base` | 0.08 | CGAP MFI annualised default rates, mid-decile. |
| `default_prob_rate_coef` | 1.4 | Logistic coefficient — empirically chosen so the marginal default rate rises ~3pp per 10pp rate hike, consistent with Karlan & Zinman (2019). |
| `default_prob_repay_coef` | -1.8 | Repayment history weight. Negative — past repayment reduces default. |
| `welfare_penalty_coef` | 0.6 | Cents-on-the-dollar consumption loss per interest dollar (proxy). |
| `protected_affects_repayment` | `False` | When `True`, attributes a small (~1pp) repayment differential to the protected group — used to stress-test the fairness Lagrangian. |

These are not point estimates from a regression — they are priors. The `notebooks/01_kiva_calibration.ipynb` notebook walks through the actual fitting; results are saved as a YAML override.

---

## 3. The procedure

### 3.1 Download

Kiva snapshots are huge (~GB). For the calibration notebook we work with the loans-only CSV slice:

```bash
mkdir -p data/raw
# Replace URL with the current Kiva snapshot location.
# See https://build.kiva.org/docs/data/snapshots for the latest.
curl -L -o data/raw/kiva_loans.csv "<snapshot_url>"
```

### 3.2 Filter

We keep only:

- `status in {paid, defaulted, refunded}`  (resolved loans only)
- `loan_amount in [25, 2000]` USD (within the microfinance window)
- `country_code` in the Mifos partner set (default: KE, IN, BD, PH, GT, MX)
- Non-null sector

### 3.3 Fit

Three regressions are run:

1. **Reservation rate proxy.** Bin funding-success rate by offered yield (Kiva auctions a rate); fit Beta CDF. Output: `reservation_rate_alpha`, `reservation_rate_beta`.
2. **Default base rate.** Per-sector default frequency. Take mid-decile, optionally per-sector vector.
3. **Sector × shock interaction.** Within rough geographic bins, regress default rate on a coded macro-event dummy (e.g. droughts in 2011 East Africa). Output: per-sector shock multipliers.

Outputs are written to `configs/calibration_kiva.yaml`. The simulator picks these up if the env is constructed via `EnvConfig.from_yaml(path)`.

### 3.4 Validate

The simulator's own behaviour is then sanity-checked against held-out Kiva loans:

- The simulated acceptance rate at the mean Kiva yield should be within ±5pp of the empirical funding-success rate.
- The simulated mean default rate should be within ±3pp of the empirical filtered default rate.
- The sector hierarchy of default rates (Agriculture > Services > Retail, etc.) should match in ordering.

Failing any check ⇒ adjust priors and re-run. We log all calibration runs in `runs/calibration/`.

---

## 4. What we explicitly cannot calibrate

- **Welfare penalty.** There is no public dataset that gives per-loan borrower welfare. We use a coefficient based on welfare-economics literature (e.g. Banerjee et al. 2015) and treat it as a sensitivity-analysis hyperparameter.
- **Protected-attribute–outcome links.** Sarkar & Alvari (2020) document lender-side bias on Kiva but not the *causal* link between gender and repayment, which is a contested research question. We default to `protected_affects_repayment=False` and document deviations explicitly.
- **Group-liability dynamics.** Kiva loans are individual; group-lending defaults look very different (cross-guarantee, peer monitoring). We use a one-feature proxy (`group_par30`) and flag this as a major v1.0 limitation.

---

## 5. Reproducing the calibration

```bash
# Stub notebook scaffolds the workflow; cells need a downloaded snapshot.
jupyter lab notebooks/01_kiva_calibration.ipynb
```

A successful run produces `configs/calibration_kiva.yaml` plus a one-page `calibration_report.md` summarising deviations from defaults.

---

## 6. If you have real portfolio data

If your institution has an anonymised loan portfolio it can share **with consent**, the calibration helper `fit_from_loan_panel(path)` in `fairprice_mf.calibration` is the entry point — same outputs, applied to your data. Please open an issue first; we will help you set up a re-identification-safe pipeline (k-anonymity ≥ 5, no client IDs, no district below a population threshold).
