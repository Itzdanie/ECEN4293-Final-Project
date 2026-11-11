"""Data-integrity checks run on the loaded frames before anything else uses them."""

from __future__ import annotations

import pandas as pd


def filter_recorded(rounds: pd.DataFrame) -> pd.DataFrame:
    """Drop rounds VLR has no event log for (``recorded == 0``).

    Round winners are fine on the dropped rows; only economy and kill
    figures need this filter (CLAUDE.md, Rounds dataset notes).
    """
    return rounds.loc[rounds["recorded"] == 1]


def check_scoreline(maps: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """Check that counting round winners per ``match_id`` reproduces Maps' score.

    Returns one row per mismatching match, with a ``reason`` of:

    - ``"missing_from_rounds"``: the map has no round-level rows at all.
      Expected for most matches before ~2022, where VLR has no round log.
    - ``"count_mismatch"``: round rows exist but don't add up to the score
      recorded in Maps. A real anomaly worth looking at (team-name mismatch
      between the two files, a forfeit, a VLR data-entry error, ...).

    An empty frame means every match checked out exactly.
    """
    present_ids = set(rounds["match_id"].unique())
    win_counts = rounds.dropna(subset=["winner"]).groupby(["match_id", "winner"]).size()

    rows = []
    for match_id, team_a, team_b, score_a, score_b in maps[
        ["match_id", "team_a", "team_b", "score_a", "score_b"]
    ].itertuples(index=False):
        if match_id not in present_ids:
            rows.append((match_id, team_a, team_b, score_a, score_b, 0, 0, "missing_from_rounds"))
            continue
        won_a = win_counts.get((match_id, team_a), 0)
        won_b = win_counts.get((match_id, team_b), 0)
        if won_a != score_a or won_b != score_b:
            rows.append(
                (match_id, team_a, team_b, score_a, score_b, won_a, won_b, "count_mismatch")
            )

    return pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "team_a",
            "team_b",
            "score_a",
            "score_b",
            "rounds_won_a",
            "rounds_won_b",
            "reason",
        ],
    )
