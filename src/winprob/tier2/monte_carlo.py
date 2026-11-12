"""Tier 2 Monte Carlo simulator - plays out score, buy type, and loss
streak together, independently of the exact Gauss-Seidel solver.
"""

from __future__ import annotations

import numpy as np

from winprob.economy.buy_type import BUY_TYPES
from winprob.tier1.states import ROUNDS_PER_HALF, WIN_SCORE, map_winner
from winprob.tier2.economy_data import (
    AFTER_LOSS_BUY_DIST,
    AFTER_WIN_BUY_DIST,
    MAX_LOSS_STREAK,
    POST_PISTOL_BUY_DIST,
    ROUND_WIN_RATE,
)

Z_95 = 1.959963984540054

_BUY_INDEX = {buy_type: i for i, buy_type in enumerate(BUY_TYPES)}
_ROUND_WIN_MATRIX = np.array(
    [[ROUND_WIN_RATE[(a, b)] for b in BUY_TYPES] for a in BUY_TYPES]
)


def _sample_buy_type(rng: np.random.Generator, dist: dict[str, float], n: int) -> np.ndarray:
    probs = [dist[buy_type] for buy_type in BUY_TYPES]
    return rng.choice(len(BUY_TYPES), size=n, p=probs)


def _draw_next_buy_types(
    rng: np.random.Generator,
    won: np.ndarray,
    prior_streak: np.ndarray,
    pistol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized version of ``states._next_buy_and_streak``: returns the
    new (buy_type_index, streak) arrays for one team across all trials."""
    n = len(won)
    # a pistol round resets the streak outright (0 if won, 1 if lost),
    # ignoring whatever streak the team carried in - matches
    # states._next_buy_and_streak exactly.
    streak = np.where(
        pistol,
        np.where(won, 0, 1),
        np.where(won, 0, np.minimum(prior_streak + 1, MAX_LOSS_STREAK)),
    )
    buy_idx = np.zeros(n, dtype=np.int64)

    for is_pistol in (True, False):
        for is_win in (True, False):
            mask = (pistol == is_pistol) & (won == is_win)
            count = int(mask.sum())
            if count == 0:
                continue
            if is_pistol:
                dist = POST_PISTOL_BUY_DIST[is_win]
                buy_idx[mask] = _sample_buy_type(rng, dist, count)
            elif is_win:
                buy_idx[mask] = _sample_buy_type(rng, AFTER_WIN_BUY_DIST, count)
            else:
                # streak is already capped/updated above; group by its value
                for s in range(1, MAX_LOSS_STREAK + 1):
                    sub_mask = mask & (streak == s)
                    sub_count = int(sub_mask.sum())
                    if sub_count > 0:
                        buy_idx[sub_mask] = _sample_buy_type(rng, AFTER_LOSS_BUY_DIST[s], sub_count)

    return buy_idx, streak


def simulate_win_probability(
    score_a: int,
    score_b: int,
    buy_a: str,
    buy_b: str,
    streak_a: int,
    streak_b: int,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Estimate P(team A wins the map) from a Tier 2 state over ``n``
    simulated playouts. Returns ``(p_hat, ci_low, ci_high)``.
    """
    winner = map_winner(score_a, score_b)
    if winner is not None:
        return winner, winner, winner

    sa = np.full(n, score_a, dtype=np.int64)
    sb = np.full(n, score_b, dtype=np.int64)
    buy_a_idx = np.full(n, _BUY_INDEX[buy_a], dtype=np.int64)
    buy_b_idx = np.full(n, _BUY_INDEX[buy_b], dtype=np.int64)
    streak_a_arr = np.full(n, streak_a, dtype=np.int64)
    streak_b_arr = np.full(n, streak_b, dtype=np.int64)

    a_wins = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    while active.any():
        idx = np.flatnonzero(active)
        total = sa[idx] + sb[idx]
        pistol = (total == 0) | (total == ROUNDS_PER_HALF)

        p_a_round = np.where(pistol, 0.5, _ROUND_WIN_MATRIX[buy_a_idx[idx], buy_b_idx[idx]])
        a_wins_round = rng.random(idx.size) < p_a_round

        new_buy_a, new_streak_a = _draw_next_buy_types(rng, a_wins_round, streak_a_arr[idx], pistol)
        new_buy_b, new_streak_b = _draw_next_buy_types(rng, ~a_wins_round, streak_b_arr[idx], pistol)

        sa[idx] += a_wins_round
        sb[idx] += ~a_wins_round
        buy_a_idx[idx] = new_buy_a
        buy_b_idx[idx] = new_buy_b
        streak_a_arr[idx] = new_streak_a
        streak_b_arr[idx] = new_streak_b

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
