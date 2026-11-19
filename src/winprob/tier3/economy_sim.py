"""Tier 3's buy policy, survivor carryover, and round income.

The buy decision is driven directly by a team's actual current bank
(classified into Eco/Semi-eco/Semi-buy/Full buy via Phase 4's real
thresholds), not a historical average of what similar teams did -
Tier 2 had to rely on the latter since it doesn't track exact credits,
but Tier 3 does, so bank itself is the more complete signal. Survivors
(players who didn't die last round) separately add real loadout value on
top of whatever's freshly spent, since a team can be flush with gear from
survivors while banking very little - bank alone can't tell those two
situations apart.
"""

from __future__ import annotations

import numpy as np

from winprob.economy.buy_type import BUY_TYPES, ECO_MAX, SEMI_BUY_MAX, SEMI_ECO_MAX
from winprob.economy.constants import (
    KILL_REWARD,
    TEAM_CREDIT_CAP,
    TEAM_LOSS_BONUS_BY_STREAK,
    TEAM_ROUND_WIN_REWARD,
)

# Mean real team loadout per buy-type bucket (credits), from
# team_round_loadout over the real Buys data - what a team spends once its
# bank puts it in that bucket.
MEAN_LOADOUT_BY_BUY_TYPE = {
    "Eco": 2228.0,
    "Semi-eco": 7232.0,
    "Semi-buy": 12756.0,
    "Full buy": 19020.0,
}
_BUCKET_SPEND = np.array([MEAN_LOADOUT_BY_BUY_TYPE[b] for b in BUY_TYPES])  # Eco, Semi-eco, Semi-buy, Full buy order
_BUCKET_THRESHOLDS = np.array([ECO_MAX, SEMI_ECO_MAX, SEMI_BUY_MAX])

# Real medians were ~4.4-4.9 (win) and ~1.8-2.4 (loss) kills per team-round
# (see winprob.economy.constants); kept as round, fixed numbers rather than
# simulated kill-by-kill, since only their average economic contribution
# matters for the round-win model here, not their variance.
MEAN_KILLS_IF_WON = 4.0
MEAN_KILLS_IF_LOST = 2.0

# team-total loss bonus, indexed by streak 0-3 (0 unused)
_LOSS_BONUS_ARRAY = np.array([0, *(TEAM_LOSS_BONUS_BY_STREAK[k] for k in (1, 2, 3))])

# Own-team survivor count (0-5) by round outcome, from the real data: a
# won round leaves a real, varied spread of survivors (mean 2.85); a lost
# round is overwhelmingly a full wipe (mean 0.17).
SURVIVOR_DIST = {
    True: np.array([0.015, 0.148, 0.240, 0.271, 0.224, 0.102]),
    False: np.array([0.886, 0.072, 0.030, 0.010, 0.002, 0.000]),
}
SURVIVOR_VALUE = 3000.0  # credits of "free" carried-over loadout per survivor (real data: ~2700-3100/survivor, close to linear)


def decide_spend(bank: np.ndarray, force: np.ndarray | None = None) -> np.ndarray:
    """What a team spends this round, driven directly by its current bank
    (classified into a buy-type bucket via the real thresholds), capped by
    that bank.

    ``force``, where True, overrides the bucket to Full buy regardless of
    bank - for rounds where saved credits have no future to be useful in
    (see ``winprob.tier3.monte_carlo._no_future_value_in_saving``), so
    there's never a reason to eco instead of spending everything you can.
    """
    bucket = np.searchsorted(_BUCKET_THRESHOLDS, bank, side="right")
    if force is not None:
        bucket = np.where(force, len(BUY_TYPES) - 1, bucket)
    return np.minimum(_BUCKET_SPEND[bucket], bank)


def draw_survivors(rng: np.random.Generator, won: np.ndarray) -> np.ndarray:
    """How many of a team's own players survive this round, by outcome."""
    n = len(won)
    survivors = np.zeros(n, dtype=np.int64)
    for outcome in (True, False):
        mask = won == outcome
        count = int(mask.sum())
        if count > 0:
            survivors[mask] = rng.choice(6, size=count, p=SURVIVOR_DIST[outcome])
    return survivors


def apply_round_income(
    bank_after_spend: np.ndarray,
    won: np.ndarray,
    streak_after: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """A team's bank after this round's win/loss bonus and kill reward,
    capped at the team credit cap. All amounts are team totals (five players).

    The spike-plant bonus is deliberately NOT included, though it is real
    income for attackers (300/player, validated). Fed through this model's
    buy buckets it gave attackers ~+1.7k of loadout per round against
    +0.13k to +0.34k in the real data, on top of the attacker edge already
    in the fitted intercept (a double count). Without it the simulated gap
    is +0.18k / +0.26k, matching the real data. See NOTES.md.
    """
    bonus = np.where(won, TEAM_ROUND_WIN_REWARD, _LOSS_BONUS_ARRAY[streak_after])
    kills_mean = np.where(won, MEAN_KILLS_IF_WON, MEAN_KILLS_IF_LOST)
    kills = rng.poisson(kills_mean)

    income = bonus + kills * KILL_REWARD
    return np.minimum(bank_after_spend + income, TEAM_CREDIT_CAP)
