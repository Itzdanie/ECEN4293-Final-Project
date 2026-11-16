"""Cleaning the per-player ult-readiness export for fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from winprob.ults.roles import check_known_agents

# rounds the export zeroes out (chiro.gg reconstruction bug, confirmed by
# their dev) - excluded from every ults-aware fit, like pistols and OT.
EXCLUDED_ROUNDS = (12, 24)


def clean_ult_rows(ult: pd.DataFrame, maps: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows safe to fit on: complete 5-player team-rounds, and
    never rounds 12/24. Adds ``team_name`` (from Maps' team_a/team_b) so
    rows can be matched to Rounds' ``atk_team``/``def_team``.

    Rounds are not otherwise filtered here (pistol/overtime rows are all
    ``ready == 0`` by the game's reset rules, which later steps rely on).
    """
    check_known_agents(ult["agent"].unique())

    size = ult.groupby(["match_id", "round", "team"])["slot"].transform("size")
    out = ult[(size == 5) & ~ult["round"].isin(EXCLUDED_ROUNDS)].copy()

    names = maps[["match_id", "team_a", "team_b"]]
    out = out.merge(names, on="match_id", how="inner")
    out["team_name"] = np.where(out["team"] == "a", out["team_a"], out["team_b"])
    return out.drop(columns=["team_a", "team_b"]).reset_index(drop=True)
