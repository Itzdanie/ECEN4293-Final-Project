"""Tier 2: Tier 1's score/side state plus buy type and loss streak, solved by
Gauss-Seidel over a sparse system and cross-checked with Monte Carlo."""

from winprob.tier2.exact import solve_tier2, win_probability_from_start
from winprob.tier2.monte_carlo import simulate_win_probability
from winprob.tier2.states import all_states, boundary_value, equation_terms

__all__ = [
    "solve_tier2",
    "win_probability_from_start",
    "simulate_win_probability",
    "all_states",
    "boundary_value",
    "equation_terms",
]
