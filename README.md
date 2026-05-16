# FairPrice-MF

**Fair, risk-sensitive reinforcement learning for microfinance loan pricing.**

[![tests](https://github.com/USER/fairprice-mf/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/fairprice-mf/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)]()
[![C4GT DMP 2026](https://img.shields.io/badge/C4GT-DMP%202026-orange)](https://codeforgovtech.in)

A research-grade module for **Mifos X / Apache Fineract** that learns personalized, fairness-constrained, risk-sensitive interest rate policies for micro-loans, using offline reinforcement learning.

> **Why?** Microfinance reaches the financially excluded — 20+ million customers via Mifos alone. Existing pricing is rule-based and leaves both efficiency and fairness on the table. A smarter system can price more accurately for the institution *and* more humanely for the borrower, while passing fair-lending audits.

This repository is the public artifact accompanying a [C4GT DMP 2026 proposal](./PROPOSAL.md) to The Mifos Initiative.

---

## Highlights

- 🏗 **A calibrated Gymnasium environment** (`MicroLoanPricing-v0`) modelling microfinance borrowers, group-liability dynamics, and economic shocks.
- ⚖️ **Fairness baked in**, not bolted on: demographic-parity and CVaR constraints enter the training objective as Lagrangian terms; an auditor reports disparate impact and counterfactual rate-shift.
- 🛡 **Two-layer safety**: regulatory rate caps and trust-region rate changes prevent unexplainable price shocks for repeat borrowers.
- 🌧 **Stress tested** against simulated economic regimes (inflation, drought, pandemic) via domain randomization.
- 🔌 **Fineract-ready**: integration shim maps loan applications to/from the existing Floating Interest Rates API.

---

## Install

```bash
# Editable install (recommended while iterating)
git clone https://github.com/USER/fairprice-mf.git
cd fairprice-mf
pip install -e .[dev]
```

Or via PyPI once published:

```bash
pip install fairprice-mf
```

---

## 60-second example

```python
import numpy as np
import gymnasium as gym
import fairprice_mf  # registers MicroLoanPricing-v0

env = gym.make("MicroLoanPricing-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, _, info = env.step(np.array([0.4], dtype=np.float32))
print(f"Offered rate: {info['rate_offered']:.3f}, accepted: {info['accepted']}, reward: {reward:.2f}")
```

A full baseline-vs-FairPrice comparison:

```bash
python scripts/run_benchmark.py --episodes 5000
```

---

## Repository layout

```
fairprice-mf/
├── fairprice_mf/
│   ├── envs/                 # Gymnasium env, borrower simulator, shocks
│   │   ├── microloan_env.py
│   │   ├── borrower.py
│   │   └── shocks.py
│   ├── agents/               # baselines + offline RL trainer
│   │   ├── baselines.py
│   │   ├── data_gen.py
│   │   └── fair_cql.py       # Conservative Q-Learning + fairness Lagrangian
│   ├── fairness/             # auditor with DI, parity, counterfactual
│   ├── safety/               # action wrappers: caps, trust region, CVaR budget
│   ├── calibration/          # Kiva-data calibration helpers
│   └── utils/                # Fineract shim, FastAPI sidecar
├── tests/                    # 19+ pytest tests
├── scripts/                  # CLI entry points
├── configs/                  # YAML configs (Hydra-style)
├── notebooks/                # exploratory + calibration notebooks
├── docs/                     # design notes & API docs
└── PROPOSAL.md               # the full C4GT DMP proposal
```

---

## The science

We model micro-loan pricing as a **Constrained Markov Decision Process (CMDP)**:

- **State:** borrower features (income proxy, prior repayment, group PAR-30, sector, region shock) + a macroeconomic regime indicator. The *protected* attribute (gender / caste / region) is **never** in the policy's observation — it is held aside for the auditor.
- **Action:** an interest rate, continuous or discrete (11 buckets).
- **Reward:** profit minus a borrower-welfare penalty, with a hard regulatory ceiling.
- **Constraints:** demographic-parity on offered rates, CVaR-floor on borrower welfare, hard regulatory cap.

The flagship method is a **fairness-constrained Conservative Q-Learning** trainer (`fairprice_mf.agents.fair_cql.FairCQLTrainer`) that augments the Khraishi-Okhrati 2022 CQL formulation with a Lagrangian fairness regularizer. The dual variable is trained by gradient ascent on the fairness-constraint violation. CVaR safety enters via the [Sauté RL](https://arxiv.org/abs/2202.06558) state-augmentation wrapper.

See [PROPOSAL.md §5](./PROPOSAL.md#5-our-thesis) for the formal problem statement and [docs/design.md](./docs/design.md) for the architectural details.

---

## Running the tests

```bash
pytest -v
```

The test suite covers (i) Gymnasium API conformance via `env_checker`, (ii) borrower-simulator monotonicity properties, (iii) the auditor's reporting, and (iv) the safety wrappers' guarantees.

---

## Roadmap (C4GT DMP 2026)

- [x] v0: Gymnasium env, baselines, fairness auditor, safety wrappers (this PR)
- [ ] Calibration against Kiva public snapshot
- [ ] Fair-CQL benchmarking against PPO/DQN/profit-max baselines
- [ ] Domain-randomization training + shock-regime evaluation
- [ ] FastAPI sidecar + Fineract integration demo
- [ ] Counterfactual fairness module + SHAP-based explanations
- [ ] Research paper (target: ACM ICAIF 2026 or AAAI AIES)
- [ ] v1.0 release + Mifos community blog post

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). PRs welcome — especially for additional shock models, alternative offline RL backbones (`d3rlpy`, `corl`), and Fineract integration tests.

## License

MIT. See [LICENSE](./LICENSE).

## Acknowledgments

Mentored by **@Akshat111111** and **@PRIYANSHU2026** as part of the C4GT DMP 2026 program with **The Mifos Initiative**.

Built on the shoulders of:

- Khraishi & Okhrati (2022) for the offline-RL credit pricing framework.
- Chow, Ghavamzadeh, Janson, Pavone (2018) for CVaR-constrained RL theory.
- Joseph, Kearns, Morgenstern, Roth (2016) for fair bandits.
- Sootla et al. (2022) for the Sauté RL safety wrapper.
- The Apache Fineract community for the open-source rails of financial inclusion.
