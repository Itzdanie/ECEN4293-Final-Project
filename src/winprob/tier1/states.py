"""Tier 1 game-state rules: side assignment, round-win probability, and the
map-winner check (first to 13 in regulation, win-by-two once tied 12-12).
"""

from __future__ import annotations

WIN_SCORE = 13
ROUNDS_PER_HALF = 12

Side = str  # "atk" or "def"


def other_side(side: Side) -> Side:
    if side not in ("atk", "def"):
        raise ValueError(f"side must be 'atk' or 'def', got {side!r}")
    return "def" if side == "atk" else "atk"


def side_for_team_a(total_rounds_played: int, a_started: Side) -> Side:
    """Which side team A is on for the next round, given the single
    half-swap after round 12. Only valid up to the 12-12 overtime entry
    (``total_rounds_played <= 24``) - overtime swaps every round instead,
    see :mod:`winprob.tier1.overtime`.
    """
    return a_started if total_rounds_played < ROUNDS_PER_HALF else other_side(a_started)


def round_win_prob_a(p_atk: float, side_a: Side) -> float:
    """P(team A wins the next round), given the attacking side's round-win
    probability for this map/matchup and which side team A is currently on.
    """
    return p_atk if side_a == "atk" else 1.0 - p_atk


def is_overtime_entry(score_a: int, score_b: int) -> bool:
    """True exactly at the 12-12 tie where overtime's win-by-two rule takes over."""
    return score_a == ROUNDS_PER_HALF and score_b == ROUNDS_PER_HALF


def map_winner(score_a: int, score_b: int) -> float | None:
    """1.0 if team A has won the map, 0.0 if team B has, else ``None`` (still
    in progress). Handles both regulation (first to 13) and overtime
    (win by two, once both scores are at least 12).
    """
    if score_a >= ROUNDS_PER_HALF and score_b >= ROUNDS_PER_HALF:
        if score_a - score_b >= 2:
            return 1.0
        if score_b - score_a >= 2:
            return 0.0
        return None
    if score_a == WIN_SCORE:
        return 1.0
    if score_b == WIN_SCORE:
        return 0.0
    return None
