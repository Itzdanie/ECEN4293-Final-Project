"""Per-team loadout and VLR's four buy-type buckets: Eco / Semi-eco / Semi-buy / Full buy.

VLR/chiro.gg don't publish their exact bucket thresholds, so these were
reverse-engineered: team loadout in credits (summed price(gun) + price(armor)
over the team's five players, per round) is cut at 5000 / 10000 / 15000,
chosen because they reproduce chiro.gg's published gun-round win rates
(``economy_buy_matchups.csv``) closely:

  - Full buy vs Eco:       85.3% ours vs 85.8% chiro.gg
  - Full buy vs Full buy:  50.0% ours vs 50.1% chiro.gg
  - Semi-buy vs Eco:       90.4% ours vs 90.4% chiro.gg

The Semi-eco boundary matches less precisely - chiro.gg likely folds in
signal we don't have (e.g. per-player buys, not just the team total). Treat
Semi-eco/Semi-buy as the least certain of the three cut points.
"""

from __future__ import annotations

import pandas as pd

from winprob.economy.prices import ARMOR_PRICES, GUN_PRICES

ECO_MAX = 5000
SEMI_ECO_MAX = 10000
SEMI_BUY_MAX = 15000

BUY_TYPES = ["Eco", "Semi-eco", "Semi-buy", "Full buy"]

PISTOL_ROUNDS = {1, 13}


def is_pistol_round(round_number: int) -> bool:
    """Rounds 1 and 13: ~50/50 regardless of buy, modeled on their own
    rather than through loadout."""
    return round_number in PISTOL_ROUNDS


def classify_buy_type(team_loadout: float) -> str:
    """Bucket a team's total round loadout (credits) into one of VLR's four buy types."""
    if team_loadout < ECO_MAX:
        return "Eco"
    if team_loadout < SEMI_ECO_MAX:
        return "Semi-eco"
    if team_loadout < SEMI_BUY_MAX:
        return "Semi-buy"
    return "Full buy"


def team_round_loadout(buys: pd.DataFrame) -> pd.DataFrame:
    """Sum each team's per-player loadout for every (match_id, round), from
    a Buys dataframe (as loaded by ``winprob.data.load_buys``).

    Loadout is the *equipped* gun + armor for that round, not ``spent`` - a
    team holding guns bought in an earlier round can spend almost nothing
    and still have a full loadout.
    """
    loadout = buys["gun"].map(GUN_PRICES).fillna(0) + buys["armor"].map(ARMOR_PRICES).fillna(0)
    return (
        buys.assign(loadout=loadout)
        .groupby(["match_id", "round", "team"], as_index=False)["loadout"]
        .sum()
    )


def classify_team_rounds(buys: pd.DataFrame) -> pd.DataFrame:
    """:func:`team_round_loadout`, with a ``buy_type`` column added."""
    out = team_round_loadout(buys)
    out["buy_type"] = out["loadout"].apply(classify_buy_type)
    return out
