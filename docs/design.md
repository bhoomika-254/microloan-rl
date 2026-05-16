# FairPrice-MF — Design Notes

This document explains the **why** behind the code. For the **how-to-use** see the [README](../README.md); for the full research argument see [PROPOSAL.md](../PROPOSAL.md).

---

## 1. Goals & non-goals

**Goals**

- Produce a reproducible, simulation-backed reinforcement-learning pricing policy that an MFI running Mifos / Apache Fineract could plausibly adopt.
- Make fairness and risk constraints **first-class** rather than bolt-on. They enter the training objective, the safety wrappers, and the audit.
- Stay deployable. The whole system has to coexist with Fineract's existing floating-interest-rates API and not require schema migrations.

**Non-goals**

- We do **not** try to learn from a real MFI portfolio without consent (we have none). All numbers in this repo come from a parameterized simulator that is calibrated against the public Kiva snapshot.
- We do **not** claim sample efficiency. CQL is conservative by design; the value here is *safety*, not data-efficiency.
- We do **not** attempt online RL in production. The deployment story is offline-trained policy → shadow-mode evaluation → human-in-the-loop rollout.

---

## 2. The CMDP

We model micro-loan pricing as a Constrained Markov Decision Process:

| Symbol | Meaning |
|---|---|
| $s_t$ | borrower features ($x_t$) + sector one-hot + macro shock one-hot |
| $a_t$ | offered interest rate (11-bucket discrete, or continuous in $[r_{\min}, r_{\mathrm{cap}}]$) |
| $r_t$ | $\text{profit}_t - \beta \cdot \text{welfare\_penalty}_t$ |
| $g_t$ | protected attribute (held out of $s_t$; used only by auditor) |
| $c_t^{\mathrm{fair}}$ | $\mathbb{E}[a \mid g=1] - \mathbb{E}[a \mid g=0]$ |
| $c_t^{\mathrm{risk}}$ | $-\mathrm{CVaR}_\alpha(\text{welfare}_t)$ |

Objective:
$$
\max_\pi \; \mathbb{E}\big[\sum_t \gamma^t r_t\big] \quad \text{s.t.} \quad |c^{\mathrm{fair}}| \le \varepsilon, \quad c^{\mathrm{risk}} \le \kappa
$$

The protected attribute is deliberately excluded from $s_t$. The policy is therefore *blind* in the input layer; the fairness Lagrangian penalises **observed** disparities, not declared use of the attribute, which catches proxy discrimination.

---

## 3. Module map

```
fairprice_mf/
├── envs/
│   ├── borrower.py        # Structural causal model of borrower decisions
│   ├── shocks.py          # Plug-in economic shock regimes
│   └── microloan_env.py   # Gymnasium Env wrapping the two above
├── agents/
│   ├── baselines.py       # Rule-based, profit-max logistic, random
│   ├── data_gen.py        # Logs offline transitions with behaviour policy
│   └── fair_cql.py        # Fair-CQL trainer (TD + CQL + λ-fair)
├── fairness/
│   └── auditor.py         # Demographic parity, DI, counterfactual
├── safety/
│   └── wrappers.py        # Regulatory cap, trust region, CVaR budget
├── calibration/
│   └── kiva.py            # Maps public Kiva data → simulator priors
└── utils/
    ├── api.py             # FastAPI sidecar (factory + module-level app)
    └── fineract_shim.py   # Loan-application payload ↔ obs vector
```

### Why these boundaries?

- **`envs/` is the contract.** As long as a future contributor produces a `MicroLoanPricingEnv`-compatible object with the same `info` keys, every other module continues to work. This lets us swap in a real-portfolio replay environment later.
- **`agents/` does not know about fairness.** The fairness constraint is injected through the *batch* (the `protected` and `rate_offered` tensors) rather than through any class. That keeps the door open to plugging in another offline-RL trainer (IQL, BCQ, BEAR) without rewriting the auditor.
- **`safety/` is action-space middleware.** The wrappers compose: `CVaRBudgetWrapper(TrustRegionWrapper(RegulatoryCapWrapper(env)))`. The order matters — see `safety/wrappers.py` docstrings.
- **`utils/api.py` uses a factory.** `build_app(policy_name=...)` returns a fresh FastAPI app, which is what lets us run audit-mode and serve-mode side-by-side or under test.

---

## 4. The borrower simulator

`BorrowerCohort` is a structural causal model, not a sklearn classifier. The DAG is roughly:

```
income_proxy ──┐
prior_repay ───┼─► reservation_rate ──► accept/decline
sector ────────┘                          │
                                          ▼
                       offered_rate ──► default_realized
                                          │
                                          ▼
                                  per_step welfare
```

Three deliberate choices:

1. **Acceptance has a Beta-distributed reservation rate**, not a logistic. This is more realistic for low-income borrowers, who often have a hard ceiling above which they walk away rather than a smooth probability.
2. **Default has a Markov component**. A borrower who repaid last period is more likely to repay this period — this is well-attested in microfinance and is what makes the sequential mode interesting.
3. **The protected attribute *can* affect repayment** (toggleable). This is morally fraught — we leave it off by default. When on, the simulator becomes an environment in which the optimal *profit-maximising* policy is itself unfair, and the fairness Lagrangian has actual work to do.

See `docs/calibration.md` for how the parameters are anchored against the Kiva public snapshot.

---

## 5. The Fair-CQL trainer

We adapt Kumar et al. (2020) CQL with an additive Lagrangian fairness regulariser.

The loss per minibatch:

$$
\mathcal{L} = \underbrace{(Q(s, a) - y)^2}_{\text{TD}} + \alpha_{\mathrm{CQL}} \underbrace{\big[\log\textstyle\sum_{a'} e^{Q(s,a')} - Q(s, a_{\text{data}})\big]}_{\text{CQL conservatism}} + \lambda_{\mathrm{fair}} \cdot \underbrace{\big(|\bar a_{g=1} - \bar a_{g=0}| - \varepsilon\big)_+^2}_{\text{fairness}}
$$

where $\bar a_{g=k}$ is the in-batch average offered rate for protected group $k$, and $\lambda_{\mathrm{fair}}$ is updated by dual ascent:

$$
\lambda_{\mathrm{fair}} \leftarrow \max\big(0,\; \lambda_{\mathrm{fair}} + \eta \cdot (|\bar a_{g=1} - \bar a_{g=0}| - \varepsilon)\big)
$$

**Why this form?**

- Lagrangian (rather than CPO-style trust region) keeps the trainer a single optimiser; no need for second-order corrections. Trades theoretical convergence guarantees for engineering simplicity.
- Fairness is computed on the *behaviour-policy actions actually logged in the batch* (`rate_offered`), not on the network's predicted action. This stays well-defined for offline data and avoids the chicken-and-egg of "fair w.r.t. a policy we haven't learned yet".
- ε-tolerance ($\varepsilon = 0.02$ by default) admits noise in finite-batch group means. Tighter ε ⇒ tighter parity, lower reward.

**Limitations:**

- Demographic parity is one fairness criterion; equalized-odds and calibration-within-groups give different policies. We default to parity because it is what regulators ask for and what is most legally defensible under Indian RBI fair-lending guidance.
- The trainer only enforces parity *in-distribution*. A deployed policy could drift outside the training distribution and re-introduce disparity; the FastAPI sidecar's `/audit/recent` endpoint exists precisely for ongoing monitoring.

---

## 6. Safety: two layers

| Layer | Implementation | Purpose |
|---|---|---|
| Hard cap | `RegulatoryCapWrapper` | Clip any rate to `[r_min, r_cap]`. Belt-and-suspenders for the regulator. |
| Trust region | `TrustRegionWrapper` | For repeat borrowers, $|a_t - a_{t-1}| \le \delta$. Prevents jarring price changes. |
| CVaR budget | `CVaRBudgetWrapper` | Sauté-RL state augmentation. Tracks remaining welfare budget; terminates an episode if exhausted. |

The first two are *defensive*. Even if the trainer goes wild, they guarantee no proposed rate violates the regulatory ceiling or shocks an existing customer. The third is the *learned* safety layer — its CVaR is shaped by training.

---

## 7. The Fineract integration

We do **not** propose a schema change. Fineract already supports floating-rate products via `interestRateDifferential` on `POST /api/v1/loanproducts`. Our deployment plan is:

1. Train Fair-CQL offline on portfolio replay (or, in this repo, the simulator).
2. Run `fairprice-serve` as a sidecar on the same host or pod as Fineract.
3. When Fineract proposes a rate (or a human officer does), make a parallel call to `POST /price` for the same client.
4. Log both rates. Initially, **shadow mode only** — Fineract continues to bill the human/rule-based rate.
5. After 30+ days of shadow operation, route a fraction of new loans through the model rate, with a kill-switch.

The Fineract shim (`utils/fineract_shim.py`) handles the payload-to-obs mapping and is deliberately conservative: missing fields fall back to safe defaults rather than failing.

---

## 8. What's not in v0

- An online learning loop. (Outside scope of a 3-month DMP, and ethically the wrong default for microfinance.)
- Multi-product reasoning (only single-loan pricing).
- Group liability mechanics beyond the `group_par30` covariate. Real joint-liability dynamics are a research project of their own.
- A web UI for auditors. We expose JSON via the sidecar; a Mifos plugin would be follow-up work.

See the [PROPOSAL.md roadmap](../PROPOSAL.md#10-timeline) for what is scheduled and what is explicitly post-DMP.
