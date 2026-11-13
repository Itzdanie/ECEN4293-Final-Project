"""Tier 3: Tier 1's score/side state plus exact credits and ults ready (0-5)
per team. Monte Carlo only - the state space is too large to enumerate exactly."""

from winprob.tier3.monte_carlo import simulate_win_probability

__all__ = ["simulate_win_probability"]
