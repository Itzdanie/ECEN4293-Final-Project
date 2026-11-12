"""Tier 2 state and transition rules: Tier 1's score/side state, plus each
team's buy type and loss streak.

Score handling (first to 13, win-by-two overtime, the halftime swap at
round 12) is identical to Tier 1 and reused directly from
:mod:`winprob.tier1.states` - only the economy layer (buy type, loss
streak, and the round-win probability they imply) is new here.
"""

from __future__ import annotations

from winprob.economy.buy_type import BUY_TYPES
from winprob.tier1.states import ROUNDS_PER_HALF, map_winner
from winprob.tier2.economy_data import (
    AFTER_LOSS_BUY_DIST,
    AFTER_WIN_BUY_DIST,
    MAX_LOSS_STREAK,
    POST_PISTOL_BUY_DIST,
    ROUND_WIN_RATE,
)

# (score_a, score_b, buy_a, buy_b, streak_a, streak_b)
State = tuple[int, int, str, str, int, int]


def is_pistol_next(score_a: int, score_b: int) -> bool:
    """True when the upcoming round is round 13 - a pistol round, where
    buy type doesn't apply and the round is a coin flip (CLAUDE.md)."""
    return score_a + score_b == ROUNDS_PER_HALF


def _next_buy_and_streak(won: bool, prior_streak: int, pistol_next: bool) -> tuple[int, dict[str, float]]:
    """A team's loss streak after this round (deterministic) and the
    distribution its NEXT round's buy type is drawn from (empirical)."""
    if pistol_next:
        return (0 if won else 1), POST_PISTOL_BUY_DIST[won]
    if won:
        return 0, AFTER_WIN_BUY_DIST
    return min(prior_streak + 1, MAX_LOSS_STREAK), AFTER_LOSS_BUY_DIST[min(prior_streak + 1, MAX_LOSS_STREAK)]


def equation_terms(state: State) -> list[tuple[State, float]]:
    """``[(next_state, probability), ...]`` for a Tier 2 state's balance
    equation: ``value(state) = sum(probability * value(next_state))``.

    Each of the "A wins" / "A loses" branches fans out into up to 4x4 next
    states, since each team's next buy type is itself a distribution, not
    a single outcome - so a state can have up to 32 terms.
    """
    score_a, score_b, buy_a, buy_b, streak_a, streak_b = state
    pistol_next = is_pistol_next(score_a, score_b)
    p_a_wins = 0.5 if pistol_next else ROUND_WIN_RATE[(buy_a, buy_b)]

    streak_a_win, dist_a_win = _next_buy_and_streak(True, streak_a, pistol_next)
    streak_b_lose, dist_b_lose = _next_buy_and_streak(False, streak_b, pistol_next)
    streak_a_lose, dist_a_lose = _next_buy_and_streak(False, streak_a, pistol_next)
    streak_b_win, dist_b_win = _next_buy_and_streak(True, streak_b, pistol_next)

    terms: list[tuple[State, float]] = []

    for buy_a2, p_buy_a in dist_a_win.items():
        for buy_b2, p_buy_b in dist_b_lose.items():
            weight = p_a_wins * p_buy_a * p_buy_b
            if weight > 0.0:
                terms.append(((score_a + 1, score_b, buy_a2, buy_b2, streak_a_win, streak_b_lose), weight))

    for buy_a2, p_buy_a in dist_a_lose.items():
        for buy_b2, p_buy_b in dist_b_win.items():
            weight = (1.0 - p_a_wins) * p_buy_a * p_buy_b
            if weight > 0.0:
                terms.append(((score_a, score_b + 1, buy_a2, buy_b2, streak_a_lose, streak_b_win), weight))

    return terms


def boundary_value(score_a: int, score_b: int) -> float | None:
    """P(A wins), for a state that doesn't need buy type/streak at all: a
    clinched map, or the 12-12 overtime entry (both teams get fixed, equal
    credits in overtime, so it reduces to a plain coin flip). Returns
    ``None`` if (score_a, score_b) isn't a boundary - i.e. it's a genuine
    unknown needing buy type and streak.
    """
    winner = map_winner(score_a, score_b)
    if winner is not None:
        return winner
    if score_a == ROUNDS_PER_HALF and score_b == ROUNDS_PER_HALF:
        return 0.5
    return None


def all_states() -> list[State]:
    """Every Tier 2 state needing buy type and streak: both scores below
    13, excluding (0,0) (the match's starting point, derived separately
    since nothing transitions into it) and (12,12) (the overtime entry,
    a boundary value)."""
    return [
        (a, b, buy_a, buy_b, streak_a, streak_b)
        for a in range(13)
        for b in range(13)
        if (a, b) != (0, 0) and (a, b) != (ROUNDS_PER_HALF, ROUNDS_PER_HALF)
        for buy_a in BUY_TYPES
        for buy_b in BUY_TYPES
        for streak_a in range(MAX_LOSS_STREAK + 1)
        for streak_b in range(MAX_LOSS_STREAK + 1)
    ]
