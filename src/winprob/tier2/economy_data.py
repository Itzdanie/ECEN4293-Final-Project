"""Real-data tables that drive Tier 2's Markov chain: round-win rate by buy-type
pair, and how buy type transitions round to round.

Everything here is measured from the real chiro.gg/VLR data (league +
international tier), not invented. See the module docstrings below for
exactly where each number comes from and which cells had to be filled in.
"""

from __future__ import annotations

from winprob.economy.buy_type import BUY_TYPES

# --- Round-win rate by (attacker's buy type, defender's buy type) ----------
#
# From economy_buy_matchups.csv, "Gun rounds (no pistols)" scope. 11 of the
# 16 cells have a reliable real sample; chiro.gg hides cells under 200
# rounds, which hit same-bucket "mirror" matchups and the Eco/Semi-eco pair:
#
#   - Mirrors (Eco-Eco, Semi-eco-Semi-eco, Semi-buy-Semi-buy): set to 0.50,
#     matching Full buy-vs-Full buy's real 50.1% for the one mirror that
#     *does* have enough rounds.
#   - Eco-vs-Semi-eco / Semi-eco-vs-Eco: neither direction has enough
#     rounds (40 and 13 respectively). Filled from the Phase 5 logistic
#     model instead (mean loadout per bucket, map-weighted), with the two
#     directions forced to sum to 1 so the chain stays consistent. This is
#     the only place the model is used as a fallback - everywhere else,
#     real chiro.gg numbers are used directly, because a spot-check found
#     the continuous-loadout model is off by as much as 15pp against real
#     data for some buy-bucket pairs (the model's Δloadout effect isn't
#     linear enough across the full credit range to extrapolate safely).
ROUND_WIN_RATE: dict[tuple[str, str], float] = {
    ("Eco", "Eco"): 0.50,
    ("Eco", "Semi-eco"): 0.379,  # model fallback (real sample: 40 rounds)
    ("Eco", "Semi-buy"): 0.099,
    ("Eco", "Full buy"): 0.133,
    ("Semi-eco", "Eco"): 1.0 - 0.379,  # complement of the fallback above
    ("Semi-eco", "Semi-eco"): 0.50,
    ("Semi-eco", "Semi-buy"): 0.308,
    ("Semi-eco", "Full buy"): 0.209,
    ("Semi-buy", "Eco"): 0.904,
    ("Semi-buy", "Semi-eco"): 0.708,
    ("Semi-buy", "Semi-buy"): 0.50,
    ("Semi-buy", "Full buy"): 0.403,
    ("Full buy", "Eco"): 0.858,
    ("Full buy", "Semi-eco"): 0.789,
    ("Full buy", "Semi-buy"): 0.604,
    ("Full buy", "Full buy"): 0.501,
}


def _normalized(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# --- Buy-type transitions, round to round -----------------------------------
#
# Measured directly from Rounds + Buys (league/international, patch 12.00+):
# for each team-round, its buy type, whether it won, and next round's buy
# type (within the same half, excluding transitions into a pistol round).
# Conditioned on (won this round, resulting loss streak) - NOT on the
# previous buy type: a real check found teams on a loss streak still full
# buy surprisingly often (they'd often eco'd earlier specifically to bank
# credits for this round), so prior buy type adds little once streak is
# known, and dropping it keeps the model a lot simpler.

AFTER_WIN_BUY_DIST = _normalized(
    {"Full buy": 0.826, "Semi-buy": 0.160, "Semi-eco": 0.013, "Eco": 0.001}
)

# Keyed by the loss streak entering the NEXT round (1, 2, or capped at 3).
AFTER_LOSS_BUY_DIST = {
    1: _normalized({"Full buy": 0.556, "Eco": 0.281, "Semi-eco": 0.094, "Semi-buy": 0.069}),
    2: _normalized({"Full buy": 0.722, "Semi-eco": 0.131, "Semi-buy": 0.088, "Eco": 0.059}),
    3: _normalized({"Full buy": 0.606, "Semi-eco": 0.215, "Semi-buy": 0.111, "Eco": 0.068}),
}

MAX_LOSS_STREAK = 3

# --- Post-pistol buy type (rounds 2 and 14) ---------------------------------
#
# Measured the same way, specifically for the round right after a pistol
# round (round 1 or round 13).

POST_PISTOL_BUY_DIST = {
    True: _normalized({"Semi-buy": 0.936, "Semi-eco": 0.050, "Full buy": 0.014}),  # won the pistol
    False: _normalized({"Eco": 0.916, "Semi-eco": 0.081, "Semi-buy": 0.002}),  # lost the pistol
}


def _fill_missing_buy_types(dist: dict[str, float]) -> dict[str, float]:
    """Some buy types never showed up in a given real distribution (e.g.
    nobody fully eco's right after winning a pistol) - treat them as 0."""
    return {buy_type: dist.get(buy_type, 0.0) for buy_type in BUY_TYPES}


AFTER_WIN_BUY_DIST = _fill_missing_buy_types(AFTER_WIN_BUY_DIST)
AFTER_LOSS_BUY_DIST = {k: _fill_missing_buy_types(v) for k, v in AFTER_LOSS_BUY_DIST.items()}
POST_PISTOL_BUY_DIST = {k: _fill_missing_buy_types(v) for k, v in POST_PISTOL_BUY_DIST.items()}
