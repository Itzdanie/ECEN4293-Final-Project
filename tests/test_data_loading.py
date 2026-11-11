from pathlib import Path

import pandas as pd
import pytest

from winprob.data import check_scoreline, filter_recorded, load_buys, load_maps, load_rounds

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def maps_df() -> pd.DataFrame:
    return load_maps(FIXTURES / "maps_fixture.csv")


@pytest.fixture
def rounds_df() -> pd.DataFrame:
    return load_rounds(FIXTURES / "rounds_fixture.csv")


@pytest.fixture
def buys_df() -> pd.DataFrame:
    return load_buys(FIXTURES / "buys_fixture.csv")


def test_load_maps_keeps_patch_as_string(maps_df):
    assert maps_df["patch"].dtype == "string"
    assert set(maps_df["patch"]) == {"13.10", "9.02"}


def test_load_maps_parses_date(maps_df):
    assert pd.api.types.is_datetime64_any_dtype(maps_df["date"])


def test_load_rounds_dtypes(rounds_df):
    assert rounds_df["patch"].dtype == "string"
    assert rounds_df["round"].dtype == "int64"
    assert rounds_df["recorded"].isin([0, 1]).all()


def test_filter_recorded_drops_unlogged_rounds(rounds_df):
    assert (rounds_df["recorded"] == 0).sum() == 1
    filtered = filter_recorded(rounds_df)
    assert (filtered["recorded"] == 0).sum() == 0
    assert len(filtered) == len(rounds_df) - 1


def test_load_buys_handles_missing_gun_and_armor(buys_df):
    assert buys_df["patch"].dtype == "string"
    # pistol-round purchases and un-reported buys both leave gun/armor/spent blank
    assert buys_df["gun"].isna().any()
    assert buys_df["armor"].isna().any()
    assert buys_df["spent"].isna().any()


def test_check_scoreline_passes_clean_match(maps_df, rounds_df):
    mismatches = check_scoreline(maps_df, rounds_df)
    assert "vlr-fx-1" not in set(mismatches["match_id"])


def test_check_scoreline_flags_match_missing_from_rounds(maps_df, rounds_df):
    mismatches = check_scoreline(maps_df, rounds_df)
    row = mismatches.loc[mismatches["match_id"] == "vlr-fx-2"].iloc[0]
    assert row["reason"] == "missing_from_rounds"
    assert row["rounds_won_a"] == 0 and row["rounds_won_b"] == 0


def test_check_scoreline_flags_count_mismatch(maps_df, rounds_df):
    mismatches = check_scoreline(maps_df, rounds_df)
    row = mismatches.loc[mismatches["match_id"] == "vlr-fx-3"].iloc[0]
    assert row["reason"] == "count_mismatch"
    assert row["rounds_won_a"] == 6
    assert row["rounds_won_b"] == 4


def test_check_scoreline_returns_empty_frame_for_fully_consistent_data(maps_df, rounds_df):
    clean_maps = maps_df.loc[maps_df["match_id"] == "vlr-fx-1"]
    clean_rounds = rounds_df.loc[rounds_df["match_id"] == "vlr-fx-1"]
    mismatches = check_scoreline(clean_maps, clean_rounds)
    assert mismatches.empty
