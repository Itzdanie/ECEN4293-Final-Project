"""Exact Tier 1 win probability for any single score, including deep overtime.

``solve_linear_system``'s table stops at the 12-12 overtime entry, but a real
map can go on (13-12, 13-13, 14-13, ...). Overtime is a win-by-two chain
where the score only matters through the lead, which is 0 or +-1 at the start
of each round, so every overtime state reduces to the closed-form pair
probability from :mod:`winprob.tier1.overtime`.
"""

from __future__ import annotations

from winprob.tier1.exact import solve_linear_system
from winprob.tier1.overtime import ot_win_probability
from winprob.tier1.states import (
    ROUNDS_PER_HALF,
    Side,
    map_winner,
    other_side,
    round_win_prob_a,
    side_for_team_a,
)


def exact_win_probability(
    score_a: int,
    score_b: int,
    p_atk: float,
    a_started: Side,
    table: dict | None = None,
) -> float:
    """P(team A wins the map) from ``(score_a, score_b)``. Pass a precomputed
    ``table`` (from :func:`solve_linear_system`) to avoid re-solving for many scores."""
    winner = map_winner(score_a, score_b)
    if winner is not None:
        return winner

    if score_a >= ROUNDS_PER_HALF and score_b >= ROUNDS_PER_HALF:
        side_first = side_for_team_a(2 * ROUNDS_PER_HALF, a_started)
        _, _, p_tied = ot_win_probability(p_atk, side_first)
        lead = score_a - score_b
        if lead == 0:
            return p_tied
        # one round left in the pair, played on the opposite side to the first
        q = round_win_prob_a(p_atk, other_side(side_first))
        return q + (1.0 - q) * p_tied if lead == 1 else q * p_tied

    if table is None:
        table = solve_linear_system(p_atk, a_started)
    return table[(score_a, score_b)]
