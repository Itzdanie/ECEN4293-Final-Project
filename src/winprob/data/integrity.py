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


def check_ult_ready(ult: pd.DataFrame) -> dict[str, int]:
    """Counts of rule violations in the per-player ult-readiness export.

    A clean file has zero for each of these, apart from the two documented
    export artifacts reported separately:

    - ``spent_without_ready``: used an ult that wasn't charged.
    - ``ready_at_reset_round``: ready at the start of round 1, 13 or any
      overtime round (>= 25) - impossible, those rounds reset ult points.
    - ``duplicate_slots``: more than one row for the same player-round.
    - ``rows_in_rounds_12_24``: rows in the two zeroed-out rounds (an
      export bug, not game behavior - they are excluded from fitting).
    - ``incomplete_team_rounds``: team-rounds with fewer than 5 player rows.
    """
    reset_rounds = (ult["round"].isin([1, 13])) | (ult["round"] >= 25)
    sizes = ult.groupby(["match_id", "round", "team"]).size()
    return {
        "spent_without_ready": int(((ult["spent"] == 1) & (ult["ready"] == 0)).sum()),
        "ready_at_reset_round": int((reset_rounds & (ult["ready"] == 1)).sum()),
        "duplicate_slots": int(
            ult.duplicated(["match_id", "round", "team", "slot"]).sum()
        ),
        "rows_in_rounds_12_24": int(ult["round"].isin([12, 24]).sum()),
        "incomplete_team_rounds": int((sizes < 5).sum()),
    }
