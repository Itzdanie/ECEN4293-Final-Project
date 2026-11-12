"""Build the training table for the round-win logistic model from loaded
Rounds and Buys dataframes.
"""

from __future__ import annotations

import pandas as pd

from winprob.economy.buy_type import PISTOL_ROUNDS, team_round_loadout

DEFAULT_TIERS = ("league", "international")
LAST_REGULATION_ROUND = 24  # overtime (its own credit rules) is excluded from this fit


def build_training_table(
    rounds: pd.DataFrame,
    buys: pd.DataFrame,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
) -> pd.DataFrame:
    """One row per gun round: whether attack won, the scaled loadout gap,
    which half it was, and the map - the inputs to the round-win model.

    Restricted to ``recorded == 1`` rounds, the given tiers, regulation
    (rounds 1-24, no overtime), and non-pistol rounds. The join against
    Buys naturally restricts this to patch 12.00+, where loadout data exists.
    """
    loadout = team_round_loadout(buys)

    r = rounds[
        (rounds["recorded"] == 1)
        & (rounds["tier"].isin(tiers))
        & (rounds["round"] <= LAST_REGULATION_ROUND)
        & (~rounds["round"].isin(PISTOL_ROUNDS))
        & rounds["winner"].notna()
    ].copy()

    r = r.merge(
        loadout.rename(columns={"team": "atk_team", "loadout": "atk_loadout"}),
        on=["match_id", "round", "atk_team"],
        how="inner",
    )
    r = r.merge(
        loadout.rename(columns={"team": "def_team", "loadout": "def_loadout"}),
        on=["match_id", "round", "def_team"],
        how="inner",
    )

    r["atk_won"] = (r["winner"] == r["atk_team"]).astype(float)
    r["delta_loadout"] = (r["atk_loadout"] - r["def_loadout"]) / 1000.0
    r["half2"] = (r["round"] >= 13).astype(float)

    return r[["match_id", "round", "map", "atk_won", "delta_loadout", "half2"]]
