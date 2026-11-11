"""Tier 1: score + side/overtime state, solved exactly (LU, recursion) and by Monte Carlo."""

from winprob.tier1.exact import solve_linear_system, solve_recursive
from winprob.tier1.monte_carlo import simulate_win_probability
from winprob.tier1.overtime import ot_win_probability
from winprob.tier1.root_finding import (
    required_round_win_rate_bisection,
    required_round_win_rate_secant,
)
from winprob.tier1.states import WIN_SCORE, map_winner

__all__ = [
    "WIN_SCORE",
    "map_winner",
    "ot_win_probability",
    "solve_linear_system",
    "solve_recursive",
    "simulate_win_probability",
    "required_round_win_rate_bisection",
    "required_round_win_rate_secant",
]
