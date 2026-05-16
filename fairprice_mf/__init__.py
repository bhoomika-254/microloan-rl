"""FairPrice-MF: Fair, risk-sensitive RL for microfinance loan pricing."""

__version__ = "0.1.0"

from gymnasium.envs.registration import register

register(
    id="MicroLoanPricing-v0",
    entry_point="fairprice_mf.envs.microloan_env:MicroLoanPricingEnv",
    max_episode_steps=1,  # single-step bandit-style by default; configurable
)

register(
    id="MicroLoanPricingSequential-v0",
    entry_point="fairprice_mf.envs.microloan_env:MicroLoanPricingEnv",
    kwargs={"sequential": True, "horizon": 12},
    max_episode_steps=12,
)
