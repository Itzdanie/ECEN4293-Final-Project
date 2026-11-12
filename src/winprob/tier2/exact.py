"""Tier 2 exact solver: builds the sparse system from
:mod:`winprob.tier2.states` and solves it with Gauss-Seidel.
"""

from __future__ import annotations

import numpy as np

from winprob.numerics.gauss_seidel import gauss_seidel
from winprob.tier2.economy_data import POST_PISTOL_BUY_DIST
from winprob.tier2.states import State, all_states, boundary_value, equation_terms


def solve_tier2(tol: float = 1e-8, max_iter: int = 500) -> tuple[dict[State, float], int]:
    """Exact Tier 2 win probabilities for every (score, buy type, loss
    streak) state, via Gauss-Seidel. Returns ``(table, n_iterations)``."""
    states = all_states()
    index = {state: i for i, state in enumerate(states)}
    n = len(states)

    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    constants = np.zeros(n)

    for state, i in index.items():
        for next_state, weight in equation_terms(state):
            boundary = boundary_value(next_state[0], next_state[1])
            if boundary is not None:
                constants[i] += weight * boundary
            else:
                neighbors[i].append((index[next_state], weight))

    solution, n_iter = gauss_seidel(neighbors, constants, tol=tol, max_iter=max_iter)
    table = {state: float(solution[i]) for state, i in index.items()}
    return table, n_iter


def win_probability_from_start(table: dict[State, float]) -> float:
    """P(team A wins the whole map), averaging round 2's buy type/streak
    over who wins the opening pistol (round 1 is a coin flip)."""
    total = 0.0
    for a_won_pistol in (True, False):
        b_won_pistol = not a_won_pistol
        score_a, score_b = (1, 0) if a_won_pistol else (0, 1)
        streak_a = 0 if a_won_pistol else 1
        streak_b = 0 if b_won_pistol else 1
        for buy_a, p_buy_a in POST_PISTOL_BUY_DIST[a_won_pistol].items():
            for buy_b, p_buy_b in POST_PISTOL_BUY_DIST[b_won_pistol].items():
                state = (score_a, score_b, buy_a, buy_b, streak_a, streak_b)
                total += 0.5 * p_buy_a * p_buy_b * table[state]
    return total
