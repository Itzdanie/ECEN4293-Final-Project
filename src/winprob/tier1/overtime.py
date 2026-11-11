"""Overtime: win-by-two, sides swap every round.

a map tied at 12-12 is decided pair by pair (two
rounds per pair, one attack and one defense round for each team, since
sides swap every round). ``w`` is the chance team A wins both rounds of a
pair outright, ``l`` is the chance it loses both; a split pair (1-1) just
moves on to the next pair. Conditioning on a pair being decisive gives
``P(A wins the map) = w / (w + l)``.
"""

from __future__ import annotations

from winprob.tier1.states import Side, other_side, round_win_prob_a


def ot_win_probability(p_atk: float, side_a_first_ot_round: Side) -> tuple[float, float, float]:
    """Return ``(w, l, p)`` for a team entering overtime on
    ``side_a_first_ot_round`` for the first overtime round.
    """
    side_a_round1 = side_a_first_ot_round
    side_a_round2 = other_side(side_a_round1)

    p_a_round1 = round_win_prob_a(p_atk, side_a_round1)
    p_a_round2 = round_win_prob_a(p_atk, side_a_round2)

    w = p_a_round1 * p_a_round2
    l = (1.0 - p_a_round1) * (1.0 - p_a_round2)
    return w, l, w / (w + l)
