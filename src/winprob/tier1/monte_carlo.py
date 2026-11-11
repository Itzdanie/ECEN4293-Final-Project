"""Tier 1 Monte Carlo simulator.

Plays out the rest of the map ``n`` times, independently of the exact
solvers - including overtime, which is simulated round by round (sides
swapping every round, win-by-two) rather than shortcut through the closed
form in :mod:`winprob.tier1.overtime`. That keeps this a genuine empirical
cross-check of both regulation and overtime.
"""

from __future__ import annotations

import numpy as np

from winprob.tier1.states import ROUNDS_PER_HALF, Side, WIN_SCORE, map_winner

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def _round_win_prob_a(total_rounds_played: np.ndarray, p_half1_a: float, p_half2_a: float) -> np.ndarray:
    """Vectorized P(team A wins the next round) for each trial's total
    rounds played so far: one swap at round 13 in regulation, then a swap
    every round in overtime (``total_rounds_played >= 24``).
    """
    in_half1 = total_rounds_played < ROUNDS_PER_HALF
    in_overtime = total_rounds_played >= 2 * ROUNDS_PER_HALF
    ot_same_side_as_half2 = ((total_rounds_played - 2 * ROUNDS_PER_HALF) % 2) == 0

    return np.where(
        in_half1,
        p_half1_a,
        np.where(in_overtime, np.where(ot_same_side_as_half2, p_half2_a, p_half1_a), p_half2_a),
    )


def simulate_win_probability(
    p_atk: float,
    a_started: Side,
    score_a: int,
    score_b: int,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Estimate P(team A wins the map) from (score_a, score_b) over ``n``
    simulated playouts. Returns ``(p_hat, ci_low, ci_high)`` for a 95%
    normal-approximation confidence interval.
    """
    winner = map_winner(score_a, score_b)
    if winner is not None:
        return winner, winner, winner

    p_half1_a = p_atk if a_started == "atk" else 1.0 - p_atk
    p_half2_a = 1.0 - p_half1_a

    sa = np.full(n, score_a, dtype=np.int64)
    sb = np.full(n, score_b, dtype=np.int64)
    a_wins = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    while active.any():
        idx = np.flatnonzero(active)
        total = sa[idx] + sb[idx]
        p_round = _round_win_prob_a(total, p_half1_a, p_half2_a)

        wins_round = rng.random(idx.size) < p_round
        sa[idx] += wins_round
        sb[idx] += ~wins_round

        both_in_ot = (sa[idx] >= ROUNDS_PER_HALF) & (sb[idx] >= ROUNDS_PER_HALF)
        margin = sa[idx] - sb[idx]
        decided_ot = both_in_ot & (np.abs(margin) >= 2)
        decided_regulation = (~both_in_ot) & ((sa[idx] == WIN_SCORE) | (sb[idx] == WIN_SCORE))
        decided = decided_ot | decided_regulation

        decided_idx = idx[decided]
        a_wins[decided_idx] = np.where(decided_ot[decided], margin[decided] > 0, sa[decided_idx] == WIN_SCORE)
        active[decided_idx] = False

    p_hat = float(a_wins.mean())
    se = float(np.sqrt(p_hat * (1.0 - p_hat) / n))
    ci_low = max(0.0, p_hat - Z_95 * se)
    ci_high = min(1.0, p_hat + Z_95 * se)
    return p_hat, ci_low, ci_high
