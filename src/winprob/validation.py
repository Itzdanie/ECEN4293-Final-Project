"""Requirement 10: reject impossible game states with an error that says
what is wrong.

Every check raises :class:`InvalidStateError` (a ``ValueError``) whose
message names the offending value and the rule it breaks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from winprob.economy.buy_type import BUY_TYPES
from winprob.economy.constants import CREDIT_CAP, MAX_LOSS_STREAK, PLAYERS_PER_TEAM, TEAM_CREDIT_CAP
from winprob.tier1.states import ROUNDS_PER_HALF, WIN_SCORE
from winprob.tier3.regression_defaults import MAP_EFFECTS

MAX_ULTS = 5
MAX_SURVIVORS = 5


class InvalidStateError(ValueError):
    """The game state asked about cannot occur in a real map."""


def _require_int(name: str, value) -> int:
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise InvalidStateError(f"{name} must be a whole number, got {value!r}")
    return int(value)


def validate_scores(score_a: int, score_b: int) -> None:
    """A map is first to 13, and from 12-12 it goes to win-by-two overtime.
    So a finished map is 13-k for k <= 11, or a two-round margin once both
    teams have at least 12; anything past that cannot happen."""
    a = _require_int("score_a", score_a)
    b = _require_int("score_b", score_b)
    if a < 0 or b < 0:
        raise InvalidStateError(f"scores can't be negative, got {a}-{b}")
    hi, lo = max(a, b), min(a, b)
    if lo >= ROUNDS_PER_HALF:
        if hi - lo > 2:
            raise InvalidStateError(
                f"score {a}-{b} is past the win condition: in overtime a team wins "
                f"as soon as it leads by 2, so a lead of {hi - lo} can't occur"
            )
    elif hi > WIN_SCORE:
        raise InvalidStateError(
            f"score {a}-{b} is past the win condition: a map ends when a team "
            f"reaches {WIN_SCORE} (unless both teams reach {ROUNDS_PER_HALF} - "
            f"then it's win by two)"
        )


def validate_side(a_started: str) -> None:
    if a_started not in ("atk", "def"):
        raise InvalidStateError(f"a_started must be 'atk' or 'def', got {a_started!r}")


def validate_round_win_probability(p_atk: float) -> None:
    if not isinstance(p_atk, (int, float)) or math.isnan(p_atk) or not 0.0 <= p_atk <= 1.0:
        raise InvalidStateError(f"p_atk is a probability and must be between 0 and 1, got {p_atk!r}")


def validate_bank(name: str, bank: float) -> None:
    if bank < 0:
        raise InvalidStateError(f"{name} can't be negative, got {bank}")
    if bank > TEAM_CREDIT_CAP:
        raise InvalidStateError(
            f"{name} of {bank} is over the credit cap: {CREDIT_CAP} per player "
            f"x {PLAYERS_PER_TEAM} players = {TEAM_CREDIT_CAP} per team"
        )


def validate_ults(name: str, ults: int) -> None:
    if not 0 <= ults <= MAX_ULTS:
        raise InvalidStateError(
            f"{name} must be between 0 and {MAX_ULTS} (a team has {PLAYERS_PER_TEAM} players), got {ults}"
        )


def validate_streak(name: str, streak: int) -> None:
    if not 0 <= streak <= MAX_LOSS_STREAK:
        raise InvalidStateError(
            f"{name} must be between 0 and {MAX_LOSS_STREAK} (the loss bonus stops growing at "
            f"{MAX_LOSS_STREAK} straight losses), got {streak}"
        )


def validate_buy_type(name: str, buy: str) -> None:
    if buy not in BUY_TYPES:
        raise InvalidStateError(f"{name} must be one of {BUY_TYPES}, got {buy!r}")


def validate_survivors(name: str, survivors: int) -> None:
    if not 0 <= survivors <= MAX_SURVIVORS:
        raise InvalidStateError(f"{name} must be between 0 and {MAX_SURVIVORS}, got {survivors}")


def validate_map(map_name: str | None) -> None:
    if map_name is not None and map_name not in MAP_EFFECTS:
        raise InvalidStateError(f"unknown map {map_name!r}; known maps: {sorted(MAP_EFFECTS)}")


def is_reset_round(score_a: int, score_b: int) -> bool:
    """True when the next round is a pistol round (round 1 or 13) or overtime,
    where credits and ult points reset - so bank and ult inputs are ignored."""
    total = score_a + score_b
    return total in (0, ROUNDS_PER_HALF) or total >= 2 * ROUNDS_PER_HALF


def validate_tier1_state(score_a: int, score_b: int, a_started: str = "atk", p_atk: float | None = None) -> None:
    validate_scores(score_a, score_b)
    validate_side(a_started)
    if p_atk is not None:
        validate_round_win_probability(p_atk)


def validate_tier2_scores(score_a: int, score_b: int) -> None:
    """A score Tier 2 covers: any regulation score up to the 12-12 overtime entry."""
    validate_scores(score_a, score_b)
    if max(score_a, score_b) > ROUNDS_PER_HALF and min(score_a, score_b) >= ROUNDS_PER_HALF:
        raise InvalidStateError(
            f"Tier 2 stops at the {ROUNDS_PER_HALF}-{ROUNDS_PER_HALF} overtime entry "
            f"(overtime credits are equal and fixed); got {score_a}-{score_b}. Use Tier 1 or Tier 3."
        )


def validate_tier2_state(
    score_a: int, score_b: int, buy_a: str, buy_b: str, streak_a: int, streak_b: int
) -> None:
    validate_tier2_scores(score_a, score_b)
    validate_buy_type("buy_a", buy_a)
    validate_buy_type("buy_b", buy_b)
    validate_streak("streak_a", streak_a)
    validate_streak("streak_b", streak_b)


def validate_tier3_state(
    score_a: int,
    score_b: int,
    bank_a: float,
    bank_b: float,
    ults_a: int,
    ults_b: int,
    streak_a: int,
    streak_b: int,
    survivors_a: int = 0,
    survivors_b: int = 0,
    a_started: str = "atk",
    map_name: str | None = None,
    ready_agents_a: Sequence[str] | None = None,
    ready_agents_b: Sequence[str] | None = None,
) -> None:
    validate_scores(score_a, score_b)
    validate_side(a_started)
    validate_bank("bank_a", bank_a)
    validate_bank("bank_b", bank_b)
    validate_ults("ults_a", ults_a)
    validate_ults("ults_b", ults_b)
    validate_streak("streak_a", streak_a)
    validate_streak("streak_b", streak_b)
    validate_survivors("survivors_a", survivors_a)
    validate_survivors("survivors_b", survivors_b)
    validate_map(map_name)
    if is_reset_round(score_a, score_b) and (ults_a or ults_b):
        total = score_a + score_b
        raise InvalidStateError(
            f"ults can't be ready at the start of round {total + 1}: pistol and overtime "
            f"rounds reset every ult, so ults_a and ults_b must be 0"
        )
