"""Tier 3 Monte Carlo: Tier 1's score/side state, plus exact credits,
survivors, and ults ready (0-5) per team. Monte Carlo only - too many
combinations to enumerate exactly (CLAUDE.md).

Each round, a team's spend is decided directly from its current bank
(classified into a real buy-type bucket), and its *effective* loadout adds
real "free" carried-over value for each surviving player from last round
(bank alone can't tell "broke with 4 survivors" apart from "broke and
wiped," but those are very different positions). That effective loadout
feeds the Phase 5 regression (plus an ult-use boost on the logit) to get a
round-win probability, oriented by which side team A is on (reusing Tier
1's side rules exactly). Credits, survivors, and effectively the economy
all reset at the real pistol rounds and in overtime, matching how the
real game resets them.

Setting ``economy_and_ults_affect_outcome=False`` makes round-win
probability always equal ``p_atk`` - credits, survivors, and ults still
evolve in the background, but don't feed back into the outcome - which is
what Requirement 7 checks: with that off, Tier 3 should reduce to Tier 1's
own model, round for round.
"""

from __future__ import annotations

import numpy as np

from winprob.economy.constants import OVERTIME_ROUND_CREDITS, PISTOL_ROUND_CREDITS
from winprob.tier1.states import ROUNDS_PER_HALF, WIN_SCORE, Side, map_winner
from winprob.tier3.economy_sim import (
    PLANT_RATE,
    SURVIVOR_VALUE,
    apply_round_income,
    decide_spend,
    draw_survivors,
)
from winprob.tier3.regression_defaults import (
    BASE_INTERCEPT,
    DELTA_LOADOUT_COEF,
    HALF2_COEF,
    MAP_EFFECTS,
    POOL_AVERAGE_MAP_EFFECT,
)
from winprob.tier3.ult_model import MAX_LOGIT_BOOST, step_ults

Z_95 = 1.959963984540054


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _map_winner_vec(score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:
    """Vectorized :func:`winprob.tier1.states.map_winner`: 1.0/0.0 if
    decided, NaN if not."""
    both_in_ot = (score_a >= ROUNDS_PER_HALF) & (score_b >= ROUNDS_PER_HALF)
    margin = score_a - score_b
    a_wins = (both_in_ot & (margin >= 2)) | (~both_in_ot & (score_a == WIN_SCORE))
    b_wins = (both_in_ot & (margin <= -2)) | (~both_in_ot & (score_b == WIN_SCORE))
    return np.where(a_wins, 1.0, np.where(b_wins, 0.0, np.nan))


def _no_future_value_in_saving(my_score: np.ndarray, opp_score: np.ndarray, total: np.ndarray) -> np.ndarray:
    """True when there's no reason to save credits this round instead of
    spending everything affordable: either this is round 12 (whatever's
    banked gets wiped by round 13's pistol reset regardless of outcome),
    losing this round ends the map outright (match point against me), or
    I'm one win from clinching but losing just ties 12-12 into overtime's
    own reset.
    """
    round_12 = total == ROUNDS_PER_HALF - 1
    my_loss_ends_it = _map_winner_vec(my_score, opp_score + 1) == 0.0
    one_win_from_clinch_but_tie_resets = (my_score == ROUNDS_PER_HALF) & (opp_score == ROUNDS_PER_HALF - 1)
    return round_12 | my_loss_ends_it | one_win_from_clinch_but_tie_resets


def _a_is_atk(total: np.ndarray, a_started_atk: bool) -> np.ndarray:
    """Vectorized version of Tier 1's side rule: one swap after round 12,
    then overtime swaps every round."""
    in_half1 = total < ROUNDS_PER_HALF
    in_ot = total >= 2 * ROUNDS_PER_HALF
    ot_parity_even = ((total - 2 * ROUNDS_PER_HALF) % 2) == 0
    half2_side_atk = not a_started_atk
    ot_side_atk = np.where(ot_parity_even, half2_side_atk, a_started_atk)
    return np.where(in_half1, a_started_atk, np.where(in_ot, ot_side_atk, half2_side_atk))


def simulate_win_probability(
    score_a: int,
    score_b: int,
    bank_a: float,
    bank_b: float,
    ults_a: int,
    ults_b: int,
    streak_a: int,
    streak_b: int,
    n: int,
    rng: np.random.Generator,
    survivors_a: int = 0,
    survivors_b: int = 0,
    a_started: Side = "atk",
    economy_and_ults_affect_outcome: bool = True,
    p_atk: float = 0.5,
    ult_boost_scale: float = 1.0,
    map_name: str | None = None,
) -> tuple[float, float, float]:
    """Estimate P(team A wins the map) from a Tier 3 state over ``n``
    simulated playouts. Returns ``(p_hat, ci_low, ci_high)``.

    ``map_name``, if given, uses that map's real fitted attack-side effect
    (one of the 12 in ``winprob.tier3.regression_defaults.MAP_EFFECTS``);
    otherwise falls back to the round-count-weighted average across all 12.
    """
    winner = map_winner(score_a, score_b)
    if winner is not None:
        return winner, winner, winner

    if map_name is None:
        map_effect = POOL_AVERAGE_MAP_EFFECT
    elif map_name in MAP_EFFECTS:
        map_effect = MAP_EFFECTS[map_name]
    else:
        raise ValueError(
            f"{map_name!r} isn't one of the maps the round-win model was fit on: "
            f"{sorted(MAP_EFFECTS)}"
        )

    a_started_atk = a_started == "atk"

    sa = np.full(n, score_a, dtype=np.int64)
    sb = np.full(n, score_b, dtype=np.int64)
    bank_a_arr = np.full(n, float(bank_a))
    bank_b_arr = np.full(n, float(bank_b))
    ults_a_arr = np.full(n, ults_a, dtype=np.int64)
    ults_b_arr = np.full(n, ults_b, dtype=np.int64)
    streak_a_arr = np.full(n, streak_a, dtype=np.int64)
    streak_b_arr = np.full(n, streak_b, dtype=np.int64)
    survivors_a_arr = np.full(n, survivors_a, dtype=np.int64)
    survivors_b_arr = np.full(n, survivors_b, dtype=np.int64)

    a_wins = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)

    while active.any():
        idx = np.flatnonzero(active)
        total = sa[idx] + sb[idx]
        pistol = (total == 0) | (total == ROUNDS_PER_HALF)
        overtime = total >= 2 * ROUNDS_PER_HALF
        reset = pistol | overtime
        a_is_atk = _a_is_atk(total, a_started_atk)

        bank_a_arr[idx] = np.where(pistol, PISTOL_ROUND_CREDITS, bank_a_arr[idx])
        bank_b_arr[idx] = np.where(pistol, PISTOL_ROUND_CREDITS, bank_b_arr[idx])
        bank_a_arr[idx] = np.where(overtime, OVERTIME_ROUND_CREDITS, bank_a_arr[idx])
        bank_b_arr[idx] = np.where(overtime, OVERTIME_ROUND_CREDITS, bank_b_arr[idx])
        # gear (and so survivor carryover) resets along with credits at a
        # pistol round or in overtime, same as the real economy.
        survivors_a_arr[idx] = np.where(reset, 0, survivors_a_arr[idx])
        survivors_b_arr[idx] = np.where(reset, 0, survivors_b_arr[idx])

        force_a = _no_future_value_in_saving(sa[idx], sb[idx], total)
        force_b = _no_future_value_in_saving(sb[idx], sa[idx], total)
        spend_a = np.where(pistol, PISTOL_ROUND_CREDITS, np.where(overtime, OVERTIME_ROUND_CREDITS, decide_spend(bank_a_arr[idx], force_a)))
        spend_b = np.where(pistol, PISTOL_ROUND_CREDITS, np.where(overtime, OVERTIME_ROUND_CREDITS, decide_spend(bank_b_arr[idx], force_b)))
        effective_loadout_a = spend_a + survivors_a_arr[idx] * SURVIVOR_VALUE
        effective_loadout_b = spend_b + survivors_b_arr[idx] * SURVIVOR_VALUE

        if economy_and_ults_affect_outcome:
            delta_loadout_atk = np.where(
                a_is_atk, effective_loadout_a - effective_loadout_b, effective_loadout_b - effective_loadout_a
            ) / 1000.0
            half2 = (~(total < ROUNDS_PER_HALF)).astype(float)
            logit = BASE_INTERCEPT + map_effect + DELTA_LOADOUT_COEF * delta_loadout_atk + HALF2_COEF * half2

            ults_used_a, ults_a_arr[idx] = step_ults(rng, ults_a_arr[idx])
            ults_used_b, ults_b_arr[idx] = step_ults(rng, ults_b_arr[idx])
            boost = ult_boost_scale * MAX_LOGIT_BOOST
            logit += np.where(a_is_atk, ults_used_a.astype(float), -ults_used_a.astype(float)) * boost
            logit += np.where(a_is_atk, -ults_used_b.astype(float), ults_used_b.astype(float)) * boost

            p_atk_wins = _sigmoid(logit)
            p_a_wins_round = np.where(a_is_atk, p_atk_wins, 1.0 - p_atk_wins)
            # pistol rounds are ~50/50 regardless of buy (CLAUDE.md) - not
            # run through the regression at all.
            p_a_wins_round = np.where(pistol, 0.5, p_a_wins_round)
        else:
            ults_used_a, ults_a_arr[idx] = step_ults(rng, ults_a_arr[idx])
            ults_used_b, ults_b_arr[idx] = step_ults(rng, ults_b_arr[idx])
            # matches Tier 1 exactly when this toggle is off (Requirement 7):
            # Tier 1 has no pistol-round special case, so neither does this.
            p_a_wins_round = np.where(a_is_atk, p_atk, 1.0 - p_atk)

        a_wins_round = rng.random(idx.size) < p_a_wins_round

        # a pistol loss hard-resets the streak to 1 (fresh economy),
        # regardless of whatever streak was carried in - same rule as Tier 2.
        streak_after_a = np.where(pistol, np.where(a_wins_round, 0, 1), np.where(a_wins_round, 0, np.minimum(streak_a_arr[idx] + 1, 3)))
        streak_after_b = np.where(pistol, np.where(~a_wins_round, 0, 1), np.where(~a_wins_round, 0, np.minimum(streak_b_arr[idx] + 1, 3)))

        planted = rng.random(idx.size) < PLANT_RATE
        bank_a_arr[idx] = apply_round_income(
            bank_a_arr[idx] - spend_a, a_wins_round, streak_after_a,
            a_is_atk, planted, rng,
        )
        bank_b_arr[idx] = apply_round_income(
            bank_b_arr[idx] - spend_b, ~a_wins_round, streak_after_b,
            ~a_is_atk, planted, rng,
        )

        streak_a_arr[idx] = streak_after_a
        streak_b_arr[idx] = streak_after_b
        survivors_a_arr[idx] = draw_survivors(rng, a_wins_round)
        survivors_b_arr[idx] = draw_survivors(rng, ~a_wins_round)

        sa[idx] += a_wins_round
        sb[idx] += ~a_wins_round

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
