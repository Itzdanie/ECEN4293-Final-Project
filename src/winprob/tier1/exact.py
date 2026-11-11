"""Tier 1 exact solvers.

Two independent implementations of the same Markov chain, kept separate on
purpose so they can cross-check each other:

- :func:`solve_linear_system` builds the dense ``A p = b`` system and solves
  it with the hand-rolled LU solver in :mod:`winprob.numerics.linalg`.
- :func:`solve_recursive` walks the same recursion top-down with memoization,
  sharing no solving logic with the linear-system path.
"""

from __future__ import annotations

import numpy as np

from winprob.numerics.linalg import lu_solve
from winprob.tier1.overtime import ot_win_probability
from winprob.tier1.states import (
    WIN_SCORE,
    Side,
    is_overtime_entry,
    map_winner,
    round_win_prob_a,
    side_for_team_a,
)

State = tuple[int, int]


def _unknown_states() -> list[State]:
    """Every (score_a, score_b) with both scores below 13, except the 12-12
    overtime entry, which is resolved by the closed form instead."""
    return [
        (a, b)
        for a in range(WIN_SCORE)
        for b in range(WIN_SCORE)
        if not is_overtime_entry(a, b)
    ]


def _boundary_value(score_a: int, score_b: int, p_atk: float, a_started: Side) -> float:
    """P(A wins) for a state that doesn't need the linear system/recursion:
    a clinched map, or the 12-12 overtime entry."""
    winner = map_winner(score_a, score_b)
    if winner is not None:
        return winner
    if is_overtime_entry(score_a, score_b):
        side_a_first_ot_round = side_for_team_a(score_a + score_b, a_started)
        _, _, p = ot_win_probability(p_atk, side_a_first_ot_round)
        return p
    raise ValueError(f"({score_a}, {score_b}) is neither terminal nor the overtime entry")


def _fill_terminal_boundary(table: dict[State, float]) -> None:
    for k in range(WIN_SCORE):
        table[(WIN_SCORE, k)] = 1.0
        table[(k, WIN_SCORE)] = 0.0


def solve_linear_system(p_atk: float, a_started: Side) -> dict[State, float]:
    """Exact Tier 1 win probabilities via a dense linear system, solved
    with the hand-rolled LU solver."""
    unknowns = _unknown_states()
    index = {state: i for i, state in enumerate(unknowns)}
    n = len(unknowns)

    a_matrix = np.zeros((n, n))
    b_vector = np.zeros(n)

    for (score_a, score_b), i in index.items():
        a_matrix[i, i] = 1.0
        side_a = side_for_team_a(score_a + score_b, a_started)
        p_round = round_win_prob_a(p_atk, side_a)

        for next_state, weight in (
            ((score_a + 1, score_b), p_round),
            ((score_a, score_b + 1), 1.0 - p_round),
        ):
            j = index.get(next_state)
            if j is not None:
                a_matrix[i, j] -= weight
            else:
                b_vector[i] += weight * _boundary_value(*next_state, p_atk, a_started)

    solution = lu_solve(a_matrix, b_vector)

    table = {state: float(solution[i]) for state, i in index.items()}
    table[(WIN_SCORE - 1, WIN_SCORE - 1)] = _boundary_value(
        WIN_SCORE - 1, WIN_SCORE - 1, p_atk, a_started
    )
    _fill_terminal_boundary(table)
    return table


def solve_recursive(p_atk: float, a_started: Side) -> dict[State, float]:
    """Exact Tier 1 win probabilities via top-down memoized recursion - an
    independent implementation from :func:`solve_linear_system`."""
    cache: dict[State, float] = {}

    def p_win(score_a: int, score_b: int) -> float:
        state = (score_a, score_b)
        if state in cache:
            return cache[state]

        winner = map_winner(score_a, score_b)
        if winner is not None:
            value = winner
        elif is_overtime_entry(score_a, score_b):
            value = _boundary_value(score_a, score_b, p_atk, a_started)
        else:
            side_a = side_for_team_a(score_a + score_b, a_started)
            p_round = round_win_prob_a(p_atk, side_a)
            value = p_round * p_win(score_a + 1, score_b) + (1.0 - p_round) * p_win(
                score_a, score_b + 1
            )

        cache[state] = value
        return value

    for a in range(WIN_SCORE):
        for b in range(WIN_SCORE):
            p_win(a, b)

    _fill_terminal_boundary(cache)
    return cache
