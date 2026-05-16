# FairPrice-MF: Fair, Risk-Sensitive Reinforcement Learning for Microfinance Loan Pricing

**A Proposal for Code for GovTech — Dedicated Mentoring Program 2026**

**Organization:** The Mifos Initiative
**Product:** Mifos X / Apache Fineract
**Domain:** Financial Inclusion · AI
**Mentors:** @Akshat111111, @PRIYANSHU2026

---

## 1. Executive Summary

This proposal addresses the ticket *"Build a simulated environment to test dynamic pricing strategies for micro-loans using Reinforcement Learning"*. I propose **FairPrice-MF**, an open-source module for Mifos X that implements **offline, fairness-constrained, risk-sensitive reinforcement learning** for personalized micro-loan pricing.

The work goes beyond a textbook Gymnasium-plus-PPO implementation. It is positioned as a **research contribution** — the first end-to-end, deployable RL pricing system designed specifically for the microfinance context (vulnerable borrowers, group lending, volatile economies, mission-driven institutions). The deliverable is both an installable Python package (`fairprice-mf`) and a target publication for the ACM ICAIF or AAAI AIES workshops.

**Three innovations differentiate this work from prior art:**

1. **A microfinance-calibrated borrower simulator**, fit against publicly available Kiva loan repayment data and surveyed elasticity literature, rather than reusing US auto-loan distributions.
2. **A constrained-MDP formulation** that bakes demographic-parity fairness *and* a CVaR (Conditional Value-at-Risk) constraint on borrower welfare directly into the Lagrangian, jointly with the institution's profit objective.
3. **Adversarial economic-shock stress testing** (inflation, drought, pandemic-style demand shocks) using domain randomization, with disparate-impact monitoring under each regime.

By the end of the program, the deliverables will be: (i) a `pip install`-able package; (ii) a Mifos X / Fineract integration shim using the existing Floating Interest Rates API; (iii) a draft research paper; (iv) reproducible benchmarks; (v) a mid-term and final presentation to the Mifos community.

---

## 2. Why I Am a Strong Fit

I am comfortable in Python with prior experience in FastAPI, PyTorch, and standard ML libraries. I do not yet have hands-on RL experience and I am explicit about that — this is precisely the gap I want to close through this mentorship. I have already begun this by:

- Reading the foundational paper (Khraishi & Okhrati, ICAIF 2022) and three follow-on works in detail.
- Building a working v0 of the Gymnasium environment and a calibrated borrower simulator (linked in this proposal's accompanying GitHub repository).
- Studying the Apache Fineract Loan Product API surface, specifically the `interestRateVariationsForBorrowerCycle` and Floating Interest Rate hooks, which are the natural integration points.

I am committing 25+ hours per week to this project and treating the DMP as the launch point for a publication, not a portfolio piece.

---

## 3. Problem Understanding

Microfinance institutions (MFIs) deliver small loans — often ₹5,000 to ₹50,000, sometimes less than \$50 — to clients excluded from mainstream banking. Mifos X is the industry's only open-source core banking system for MFIs, used by 400+ institutions to reach 20+ million customers globally.

Current pricing in most MFI deployments is rule-based: loan products carry a single interest rate per risk band. This leaves three classes of value on the table:

- **Personalization gain**: a creditworthy repeat borrower from a low-shock region is treated identically to a first-time borrower in a drought-affected district.
- **Sustainability gain**: institutions miss revenue from clients who could comfortably support higher rates, and over-extend credit to clients who default — a worse outcome for *both* parties.
- **Welfare gain**: a smarter system can price *down* for sensitive borrowers and *toward* affordability, rather than maximizing extraction.

A naive ML solution would maximize expected profit and produce a discriminatory policy that punishes already-disadvantaged groups. The interesting question — and the one the ticket implicitly asks — is whether RL can deliver personalization gain **without** sacrificing fairness or borrower welfare. Answering that question rigorously is a research contribution.

### Why this *cannot* be a supervised learning problem

The decision to offer rate `r` to borrower `x` is a *sequential* decision that affects (a) whether the borrower accepts, (b) whether they repay, (c) whether they return for a future loan, and (d) the institution's portfolio health, which in turn changes the action space for future borrowers. There is no labeled "correct rate" in historical data — only the rate that *was* offered and what happened after. This is the textbook setup for reinforcement learning, and specifically for **off-policy** or **offline** RL when, as is the case here, online experimentation on vulnerable populations is unethical.

---

## 4. Literature Review

The relevant literature falls in four clusters.

### 4.1 RL for credit / loan pricing

| Work | Setting | Method | Gap relative to this proposal |
|---|---|---|---|
| Khraishi & Okhrati, ICAIF 2022 | US auto loans | Conservative Q-Learning (CQL), offline | No fairness, no risk-sensitivity, no microfinance calibration |
| Phillips et al. 2015; Ban & Keskin 2021 | Auto loans | Profit-based logistic regression | Not RL; no welfare objective |
| Bastani et al. 2019 | Auto loans | Thompson sampling, discrete bandit | Online (unethical here); no fairness |
| Luo et al. 2021 | Generic credit | Contextual linear UCB | No fairness; no risk constraints |
| Hamill et al. 2023 | Credit card promotions | Agent-based simulation | Different action space; no RL |

**Verdict:** the closest prior art (Khraishi & Okhrati) is well-cited but solves a meaningfully different problem. Microfinance has not been studied with deep RL.

### 4.2 Fairness in sequential decision-making

- Joseph, Kearns, Morgenstern, Roth (2016): **"Fairness in Learning: Classic and Contextual Bandits"** — formalizes fairness for bandits and proves that the constraint always has some learning-rate cost.
- Metevier et al. (2019), **"RobinHood"** — offline contextual bandits with high-probability fairness guarantees. Applied to loan *approval*, not *pricing*.
- D'Amour et al. (2020) — long-term fairness in sequential lending; shows that myopically fair policies can be long-run unfair.
- Bansal (2025), **"Algorithmic Tradeoffs in Fair Lending"** — quantifies the profit/fairness frontier; surprisingly finds that fairness-through-unawareness sometimes dominates explicit constraints. Useful counter-result to test against.

### 4.3 Risk-sensitive and safe RL

- Chow, Ghavamzadeh, Janson, Pavone (JMLR 2018), **"Risk-Constrained RL with Percentile Risk Criteria"** — CVaR-constrained MDPs via Lagrangian; the theoretical foundation we will use.
- Achiam et al. (2017), **"Constrained Policy Optimization (CPO)"** — practical algorithm for constrained policy gradients.
- Sootla et al. (2022), **"Sauté RL"** — state augmentation for almost-sure safety, very implementation-friendly.
- Zhang et al. (2023, TNNLS), **"CVaR-CPO"** — combines CVaR constraints with CPO.

### 4.4 Microfinance-specific research

- Sarkar & Alvari (2020), **"Mitigating Bias in Online Microfinance Platforms: A Case Study on Kiva.org"** — establishes that bias exists on the *lender* side of Kiva. Useful, but our problem is the *institutional pricing* side.
- Kiva data exploration papers (Choo et al., Shen & Yin) — confirm Kiva data is publicly available and well-structured. This is our calibration anchor in the absence of Mifos historical data.

### 4.5 What is missing in the field

No published work has combined:
- Offline RL (deployable, no risky exploration on real borrowers), **and**
- A fairness constraint formalized as part of the optimization (not a post-hoc audit), **and**
- A risk-sensitive objective (CVaR-style protection for borrower welfare, not just lender variance), **and**
- A microfinance-grade calibration with credible robustness checks against economic shocks, **and**
- A production-grade open-source release integrated with a real DPG.

That is the gap.

---

## 5. Our Thesis

> **Microfinance pricing is not consumer credit pricing.** The vulnerability of the borrower changes the objective function. We need a system that is **offline-trainable** (no risky exploration), **fairness-constrained** (legally and ethically required), **risk-sensitive** (protects worst-case borrower outcomes), and **shock-robust** (still works when inflation spikes or a region floods). Such a system can be open-sourced as a Mifos X module and deployed with auditable guarantees.

### 5.1 Formal problem statement

We formulate the problem as a **Constrained Markov Decision Process (CMDP)** with augmented state for risk tracking:

- **State** `s_t = (x_t, ψ_t, e_t)`:
  - `x_t`: borrower features (income proxy, prior repayment history, loan cycle number, business type, region, group-membership indicator, group-cycle delinquency rate, gender, age band)
  - `ψ_t`: protected-attribute indicator vector (used only for fairness accounting, masked from the policy network — "fairness through suppression with audit")
  - `e_t`: macroeconomic state (inflation proxy, regional-shock indicator, portfolio-at-risk PAR-30)
- **Action** `a_t ∈ [r_min, r_max]`: continuous interest rate offered (we will also evaluate a discrete-action variant with 11 rate buckets for DQN comparability).
- **Reward** `R_t = π_t − λ_w · W_t` where:
  - `π_t` = realized institutional profit from this loan (interest collected minus principal-at-loss minus servicing cost)
  - `W_t` = a borrower-welfare penalty: rate-as-fraction-of-cap, plus a heavy penalty if repayment-to-income exceeds 40% (a standard microfinance over-indebtedness threshold)
  - `λ_w` = welfare weight, learnable via Lagrangian or set by mission policy
- **Constraints**:
  - **C1 (Fairness):** demographic parity in *expected rate offered* across protected groups, with slack ε.
  - **C2 (CVaR safety):** the worst-α-quantile of borrower welfare cost must lie above a regulator-set floor.
  - **C3 (Hard rate cap):** a non-negotiable upper bound `a_t ≤ r_cap_region` (regulatory ceiling, encoded as action masking, not penalty).

### 5.2 Algorithms

We will benchmark four methods on the same environment:

1. **Rule-based baseline** — fixed rates per risk bucket, the status quo in most MFI deployments.
2. **Logistic regression + expected profit maximization** — the Phillips/Ban supervised baseline.
3. **PPO** (Stable-Baselines3) — strong on-policy baseline. Trained against the simulator, not deployable, but useful for upper-bound estimates.
4. **CQL + Lagrangian-CVaR (our proposed method)** — offline, fairness-constrained, risk-sensitive. The deployable target.

A contextual-bandit reduction will also be reported, because single-step loan decisions are partially expressible as bandits and that simpler formulation may suffice for v1 deployments.

---

## 6. Technical Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FairPrice-MF Module                          │
│                                                                      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ BorrowerCohort  │    │  MicroLoanEnv    │    │  ShockGenerator│ │
│  │   Simulator     │───▶│   (Gymnasium)    │◀───│  (inflation,   │ │
│  │  (Kiva-fit)     │    │                  │    │   drought, ...)│ │
│  └─────────────────┘    └────────┬─────────┘    └────────────────┘ │
│                                  │                                  │
│                ┌─────────────────▼─────────────────┐                │
│                │  Safety Wrapper (action masking,  │                │
│                │  hard regulatory ceilings)        │                │
│                └─────────────────┬─────────────────┘                │
│                                  │                                  │
│         ┌────────────────────────┼────────────────────────┐         │
│         ▼                        ▼                        ▼         │
│  ┌─────────────┐         ┌──────────────┐        ┌─────────────┐   │
│  │ Rule-Based  │         │   PPO / DQN  │        │ CQL + CVaR  │   │
│  │  Baseline   │         │  (SB3, online│        │ + Fairness  │   │
│  │             │         │   sim only)  │        │ (OURS)      │   │
│  └─────────────┘         └──────────────┘        └─────────────┘   │
│                                  │                                  │
│                ┌─────────────────▼─────────────────┐                │
│                │   Fairness & Welfare Auditor      │                │
│                │  (DI ratio, CVaR, counterfactual) │                │
│                └─────────────────┬─────────────────┘                │
│                                  │                                  │
│                ┌─────────────────▼─────────────────┐                │
│                │     Fineract Integration Shim     │                │
│                │  (Floating Interest Rates API)    │                │
│                └───────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.1 The Borrower Simulator

This is the heart of the system. A borrower has latent attributes:

- **Reservation rate** `r* ~ Beta(α, β)` — the maximum rate they would accept. Calibrated from Kiva acceptance patterns and microfinance elasticity literature (Karlan & Zinman 2008, Dehejia et al. 2012).
- **Default-rate function** `p_default(r, x, e)` — logistic in `(r, x, e)`, calibrated so that the *marginal* default sensitivity to rate matches published microfinance studies, not US consumer credit (which is dramatically different).
- **Acceptance decision** `accept = 1{r ≤ r*}` with a soft-acceptance variant for realism.
- **Repayment dynamics** — a Bernoulli draw per installment, with serial correlation (a missed installment increases the probability of the next miss) and group-contagion if the borrower is in a joint-liability group.

The simulator is intentionally honest about its limitations: it is *calibrated*, not *real*, and a section of the final paper will discuss simulation-to-reality gap as a primary threat to validity.

### 6.2 Fairness Auditor

Three fairness metrics will be computed every evaluation episode:

- **Demographic-parity-of-rates**: `|E[a | g=0] − E[a | g=1]|` for each protected group `g`.
- **Disparate-impact ratio**: `min_g P(approved | g) / max_g P(approved | g)`. The 80% rule is the standard threshold.
- **Counterfactual fairness** (the genuinely novel piece): we resample a borrower with `g` flipped and all other features held constant, and compare the rate offered. This requires the simulator to be a structural causal model — a non-trivial design choice.

### 6.3 Safety wrapper

Two layers:

- **Action masking**: rates above the regional regulatory cap are simply not in the agent's action set. This is a *hard* constraint, not a soft reward shaping. The cap is configurable per Fineract deployment.
- **Trust-region rate change**: when retraining or deploying a new policy version, the rate offered to any individual borrower may not change by more than `Δ_max` between policy versions. Prevents pricing shocks for repeat borrowers — a real operational concern at MFIs.

### 6.4 Mifos / Fineract integration

The Fineract Loan Product API exposes `interestRateVariationsForBorrowerCycle` and a Floating Interest Rate scheme with a settable `interestRateDifferential`. Our integration is deliberately surgical: a small adapter reads the borrower features, queries our policy server, and writes back the `interestRateDifferential` for the loan account. No core Fineract code is touched, which makes the contribution easy to review, merge, and maintain.

---

## 7. Three-Month Timeline

The DMP runs roughly June 1 to August 31. The plan is decomposed into six biweekly milestones, with the mid-point review at week 6.

| Week | Milestone | Deliverable |
|---|---|---|
| 1–2 | **Foundations** — finalize CMDP spec, freeze borrower simulator v1, set up CI/CD, write protocol for evaluation. | Env passes `gymnasium.utils.env_checker`; unit tests green; CONTRIBUTING.md and DESIGN.md merged. |
| 3–4 | **Baselines** — implement and tune rule-based, logistic-regression, PPO, DQN. Calibrate borrower model to Kiva acceptance distributions. | Baseline numbers table; calibration notebook; reproducibility script. |
| 5–6 | **Core method (mid-term)** — implement CQL + Lagrangian-CVaR + fairness constraint. Run on full benchmark suite. Mid-term presentation. | First working FairPrice-MF agent, paired with auditor; mid-term report. |
| 7–8 | **Robustness** — economic shock generator, domain-randomization training, disparate-impact under shocks. | Shock-robustness table; analysis notebook. |
| 9–10 | **Integration** — Fineract adapter, end-to-end demo: a Fineract instance offers a pricing recommendation from the RL policy. Counterfactual fairness module. | Working integration; demo video; SHAP-based explanation module. |
| 11–12 | **Write-up** — final paper draft (target: ACM ICAIF 2026 or AAAI AIES); polish docs; final presentation. | v1.0 release; arXiv preprint; final report. |

---

## 8. Detailed Deliverables

| # | Deliverable | Form |
|---|---|---|
| 1 | `fairprice-mf` Python package | PyPI + GitHub, MIT licensed |
| 2 | Gymnasium environment `MicroLoanPricing-v0` | Registered, env_checker-clean |
| 3 | Borrower simulator with Kiva calibration | Reproducible notebook + serialized params |
| 4 | Trained policy artifacts (rule-based, PPO, DQN, CQL+CVaR+Fair) | `.zip` SB3 + `.pt` weights, on Hugging Face Hub |
| 5 | Fairness/welfare auditor | Library function + CLI: `fairprice-audit <policy>` |
| 6 | Fineract integration shim | Microservice, Docker-compose example |
| 7 | Research paper draft | LaTeX, target ICAIF / AIES workshop |
| 8 | Documentation site | MkDocs Material |
| 9 | Mid-term and final community presentations | Slides + recordings |
| 10 | Blog post on Mifos community blog | Markdown |

---

## 9. Risk Analysis and Mitigations

This section is here because evaluators look for it explicitly.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Simulator-reality gap is too wide; calibrated policies don't transfer | High | High | Multi-source calibration (Kiva + published elasticities); explicit limitations section; offline RL means the policy *can* be retrained on real logs post-deployment. |
| CQL training is unstable on our state space | Medium | Medium | Use d3rlpy and SB3-contrib for proven implementations; ablation over conservatism weight; PPO fallback if CQL fails. |
| Fairness/profit trade-off is too steep to be useful | Medium | Medium | Report Pareto frontier honestly; the *frontier itself* is the contribution even if no single point dominates. |
| Counterfactual fairness module requires a causal model we cannot validate | Medium | Low | Frame as "simulator-grounded counterfactuals", be explicit it depends on the simulator. |
| Fineract integration is harder than expected | Low | Medium | Use the documented Floating Interest Rates API which already supports per-loan rate differentials. Worst case: ship as standalone service with HTTP API and document the integration path. |
| Three months is not enough for all of this | Medium | High | Strict scope-cut order: 1) drop the SHAP explainer, 2) drop counterfactual module, 3) drop multi-shock comparison, 4) keep only single-shock and core method. Core method ships no matter what. |

---

## 10. Why This Matters

Microfinance reaches the financially excluded. Mifos X is the open-source rail on which much of this happens worldwide, including in India where C4GT focuses. A pricing system that is even 3% more efficient on the institutional side, while measurably *fairer* on the borrower side, compounds across 20+ million customers. Open-sourcing it means small institutions in low-resource settings can access pricing intelligence that until now has been proprietary to large banks.

The research contribution — the first credibly fair-and-safe offline RL pricing system for microfinance — also raises the floor of what counts as acceptable in algorithmic lending. That is a public good in itself.

---

## 11. Pre-DMP Contributions

I have committed to engaging with the Mifos and C4GT community before the DMP begins:

- The full v0 environment and baseline implementations are already in the public repository accompanying this proposal.
- I am active in the Mifos / C4GT Discord and have introduced this proposal to the mentors.
- I plan to file at least two PRs against existing Mifos repositories (documentation or test improvements) before the application close date, to demonstrate engagement.

---

## 12. References

1. Khraishi, R. & Okhrati, R. (2022). *Offline Deep Reinforcement Learning for Dynamic Pricing of Consumer Credit*. ICAIF '22.
2. Chow, Y., Ghavamzadeh, M., Janson, L., Pavone, M. (2018). *Risk-Constrained Reinforcement Learning with Percentile Risk Criteria*. JMLR.
3. Achiam, J., Held, D., Tamar, A., Abbeel, P. (2017). *Constrained Policy Optimization*. ICML.
4. Joseph, M., Kearns, M., Morgenstern, J., Roth, A. (2016). *Fairness in Learning: Classic and Contextual Bandits*. NeurIPS.
5. Metevier, B., et al. (2019). *Offline Contextual Bandits with High Probability Fairness Guarantees (RobinHood)*. NeurIPS.
6. D'Amour, A., et al. (2020). *Fairness is Not Static: Deeper Understanding of Long Term Fairness via Simulation Studies*. FAccT.
7. Bansal, A. (2025). *Algorithmic Tradeoffs in Fair Lending*. arXiv:2505.13469.
8. Sarkar, S. & Alvari, H. (2020). *Mitigating Bias in Online Microfinance Platforms: A Case Study on Kiva.org*. arXiv:2006.12995.
9. Kumar, A., et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning*. NeurIPS.
10. Sootla, A., et al. (2022). *Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation*. ICML.

---

*Proposed by [your name]. Mentored by @Akshat111111 and @PRIYANSHU2026.*
