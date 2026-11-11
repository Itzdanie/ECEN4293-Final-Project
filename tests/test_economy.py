import pandas as pd
import pytest

from winprob.economy import (
    armor_price,
    classify_buy_type,
    classify_team_rounds,
    gun_price,
    is_pistol_round,
    team_round_loadout,
)
from winprob.economy.buy_type import ECO_MAX, SEMI_BUY_MAX, SEMI_ECO_MAX


def test_gun_and_armor_prices():
    assert gun_price("vandal") == 2900
    assert gun_price("classic") == 0
    assert armor_price("heavy") == 1000


def test_unknown_or_missing_gear_prices_as_zero():
    assert gun_price(None) == 0
    assert gun_price(float("nan")) == 0
    assert armor_price(None) == 0


@pytest.mark.parametrize(
    "loadout,expected",
    [
        (0, "Eco"),
        (ECO_MAX - 1, "Eco"),
        (ECO_MAX, "Semi-eco"),
        (SEMI_ECO_MAX - 1, "Semi-eco"),
        (SEMI_ECO_MAX, "Semi-buy"),
        (SEMI_BUY_MAX - 1, "Semi-buy"),
        (SEMI_BUY_MAX, "Full buy"),
        (19500, "Full buy"),
    ],
)
def test_classify_buy_type_boundaries(loadout, expected):
    assert classify_buy_type(loadout) == expected


def test_is_pistol_round():
    assert is_pistol_round(1)
    assert is_pistol_round(13)
    assert not is_pistol_round(2)
    assert not is_pistol_round(12)


def test_team_round_loadout_sums_per_player_and_handles_missing_gear():
    buys = pd.DataFrame(
        [
            {"match_id": "m1", "round": 2, "team": "A", "player": "p1", "gun": "vandal", "armor": "heavy"},
            {"match_id": "m1", "round": 2, "team": "A", "player": "p2", "gun": "phantom", "armor": "heavy"},
            {"match_id": "m1", "round": 2, "team": "B", "player": "p3", "gun": "classic", "armor": None},
            {"match_id": "m1", "round": 2, "team": "B", "player": "p4", "gun": None, "armor": None},
        ]
    )
    out = team_round_loadout(buys).set_index("team")["loadout"]
    assert out["A"] == 2900 + 1000 + 2900 + 1000  # vandal+heavy, phantom+heavy
    assert out["B"] == 0


def test_classify_team_rounds_adds_buy_type_column():
    # A full team of five rifles + heavy armor (19500) vs. a team with nothing (0).
    buys = pd.DataFrame(
        [{"match_id": "m1", "round": 2, "team": "A", "player": f"p{i}", "gun": "vandal", "armor": "heavy"} for i in range(5)]
        + [{"match_id": "m1", "round": 2, "team": "B", "player": f"p{i}", "gun": None, "armor": None} for i in range(5)]
    )
    out = classify_team_rounds(buys).set_index("team")["buy_type"]
    assert out["A"] == "Full buy"
    assert out["B"] == "Eco"
