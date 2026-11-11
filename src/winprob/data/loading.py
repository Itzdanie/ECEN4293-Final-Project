"""Readers for the raw chiro.gg CSV exports

Every reader keeps ``patch`` as a string dtype (values like ``13.10`` are not
valid floats) and parses ``date`` into a real datetime column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

_MAPS_DTYPES = {
    "match_id": "string",
    "patch": "string",
    "event": "string",
    "tier": "string",
    "map": "string",
    "series_id": "string",
    "best_of": "string",
    "team_a": "string",
    "team_b": "string",
    "region_a": "string",
    "region_b": "string",
    "winner": "string",
    "comp_a": "string",
    "comp_b": "string",
    "a_started": "string",
}

_ROUNDS_DTYPES = {
    "match_id": "string",
    "patch": "string",
    "event": "string",
    "tier": "string",
    "map": "string",
    "round": "int64",
    "atk_team": "string",
    "def_team": "string",
    "winner": "string",
    "winner_side": "string",
    "outcome": "string",
    "recorded": "int64",
    "plant_site": "string",
    "first_kill_side": "string",
    "atk_kills": "int64",
    "def_kills": "int64",
    "atk_spent": "float64",
    "def_spent": "float64",
    "atk_bank": "float64",
    "def_bank": "float64",
}

_BUYS_DTYPES = {
    "match_id": "string",
    "patch": "string",
    "event": "string",
    "tier": "string",
    "map": "string",
    "round": "int64",
    "player": "string",
    "team": "string",
    "agent": "string",
    "side": "string",
    "gun": "string",
    "armor": "string",
    "spent": "float64",
    "bank": "float64",
}


def _read_csv(path: Path, dtypes: dict[str, str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtypes, parse_dates=["date"])


def load_maps(path: Path | str = RAW_DATA_DIR / "maps_combined.csv") -> pd.DataFrame:
    """Load one row per map."""
    return _read_csv(Path(path), _MAPS_DTYPES)


def load_rounds(path: Path | str = RAW_DATA_DIR / "rounds_combined.csv") -> pd.DataFrame:
    """Load one row per round.

    ``plant_time`` is left out of the dtype map on purpose: it's mm:ss text
    for planted rounds and blank otherwise, so it stays pandas' inferred
    object dtype rather than forcing a type that doesn't fit every row.
    """
    return _read_csv(Path(path), _ROUNDS_DTYPES)


def load_buys(path: Path | str = RAW_DATA_DIR / "buys_combined.csv") -> pd.DataFrame:
    """Load one row per player per round. This is the largest raw file
    (~6.4M rows, ~1 GB as of the Nov 2026 export)."""
    return _read_csv(Path(path), _BUYS_DTYPES)


def _read_aggregate_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_economy_buy_matchups(
    path: Path | str = RAW_DATA_DIR / "economy_buy_matchups.csv",
) -> pd.DataFrame:
    """Round win rate for every pair of VLR buy buckets (Eco/Semi-eco/Semi-buy/Full buy)."""
    return _read_aggregate_csv(Path(path))


def load_economy_guns(path: Path | str = RAW_DATA_DIR / "economy_guns.csv") -> pd.DataFrame:
    """Current price and win rate for every gun; seeds the patch-keyed price table."""
    return _read_aggregate_csv(Path(path))


def load_economy_archetypes(
    path: Path | str = RAW_DATA_DIR / "economy_archetypes.csv",
) -> pd.DataFrame:
    """Win rates for team weapon-mix archetypes."""
    return _read_aggregate_csv(Path(path))


def load_ult_per_round_by_agent(
    path: Path | str = RAW_DATA_DIR / "ult_per_round_by_agent.csv",
) -> pd.DataFrame:
    """Per-agent ultimate readiness/usage/lift figures."""
    return _read_aggregate_csv(Path(path))
