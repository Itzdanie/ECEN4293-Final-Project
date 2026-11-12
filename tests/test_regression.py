import numpy as np
import pandas as pd
import pytest

from winprob.numerics.logistic import sigmoid
from winprob.regression.dataset import build_training_table
from winprob.regression.round_win_model import fit_round_win_model, predict_round_win_probability


def _buys_row(match_id, rnd, team, player, gun, armor):
    return {
        "match_id": match_id,
        "round": rnd,
        "team": team,
        "player": player,
        "gun": gun,
        "armor": armor,
    }


def _rounds_row(match_id, rnd, atk_team, def_team, winner, tier="league", recorded=1):
    return {
        "match_id": match_id,
        "round": rnd,
        "map": "ascent",
        "tier": tier,
        "atk_team": atk_team,
        "def_team": def_team,
        "winner": winner,
        "recorded": recorded,
    }


def test_build_training_table_filters_pistols_ot_and_unrecorded_rounds():
    buys = pd.DataFrame(
        [_buys_row("m1", r, t, f"{t}_p1", "vandal", "heavy") for r in [1, 2, 13, 25] for t in ["A", "B"]]
    )
    rounds = pd.DataFrame(
        [
            _rounds_row("m1", 1, "A", "B", "A"),  # pistol round 1 - excluded
            _rounds_row("m1", 2, "A", "B", "A"),  # kept
            _rounds_row("m1", 13, "B", "A", "B"),  # pistol round 13 - excluded
            _rounds_row("m1", 25, "A", "B", "A"),  # overtime - excluded
            _rounds_row("m1", 3, "A", "B", "B", recorded=0),  # unrecorded - excluded
            _rounds_row("m1", 4, "A", "B", "A", tier="other"),  # wrong tier - excluded
        ]
    )
    table = build_training_table(rounds, buys)
    assert sorted(table["round"]) == [2]


def test_build_training_table_computes_delta_loadout_and_half():
    buys = pd.DataFrame(
        [
            _buys_row("m1", 2, "A", "a1", "vandal", "heavy"),  # 3900
            _buys_row("m1", 2, "B", "b1", "classic", None),  # 0
            _buys_row("m1", 14, "A", "a1", "vandal", "heavy"),
            _buys_row("m1", 14, "B", "b1", "classic", None),
        ]
    )
    rounds = pd.DataFrame(
        [
            _rounds_row("m1", 2, "A", "B", "A"),
            _rounds_row("m1", 14, "B", "A", "B"),  # B attacking in the second half
        ]
    )
    table = build_training_table(rounds, buys).set_index("round")

    assert table.loc[2, "delta_loadout"] == pytest.approx(3.9)
    assert table.loc[2, "half2"] == 0.0
    assert table.loc[2, "atk_won"] == 1.0

    assert table.loc[14, "delta_loadout"] == pytest.approx(-3.9)  # B (atk) minus A (def)
    assert table.loc[14, "half2"] == 1.0
    assert table.loc[14, "atk_won"] == 1.0  # winner is B, and B is attacking


def _synthetic_table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    maps = rng.choice(["ascent", "bind", "haven"], size=n)
    delta_loadout = rng.normal(0, 3, n)
    half2 = rng.integers(0, 2, n).astype(float)
    map_effect = {"ascent": 0.0, "bind": 0.3, "haven": -0.2}
    z = 0.1 + 0.5 * delta_loadout + 0.2 * half2 + np.array([map_effect[m] for m in maps])
    atk_won = (rng.random(n) < sigmoid(z)).astype(float)
    return pd.DataFrame({"map": maps, "delta_loadout": delta_loadout, "half2": half2, "atk_won": atk_won})


def test_fit_round_win_model_recovers_loadout_and_map_direction():
    table = _synthetic_table()
    model = fit_round_win_model(table)

    assert model["n_iter"] <= 25
    assert model["beta"][model["feature_names"].index("delta_loadout")] > 0

    p_bind = predict_round_win_probability(model, delta_loadout=0.0, half2=False, map_name="bind")
    p_ascent = predict_round_win_probability(model, delta_loadout=0.0, half2=False, map_name="ascent")
    p_haven = predict_round_win_probability(model, delta_loadout=0.0, half2=False, map_name="haven")
    assert p_bind > p_ascent > p_haven  # matches the +0.3 / 0 / -0.2 true map effects


def test_predict_matches_manual_sigmoid():
    table = _synthetic_table()
    model = fit_round_win_model(table)

    beta = model["beta"]
    feature_names = model["feature_names"]
    x = np.zeros(len(beta))
    x[feature_names.index("intercept")] = 1.0
    x[feature_names.index("delta_loadout")] = 2.5
    x[feature_names.index("half2")] = 1.0
    # "ascent" is the (alphabetically first) reference level: no map dummy set

    expected = float(sigmoid(np.dot(beta, x)))
    actual = predict_round_win_probability(model, delta_loadout=2.5, half2=True, map_name="ascent")
    assert actual == pytest.approx(expected)


def test_a_bigger_loadout_edge_always_predicts_a_higher_win_probability():
    # Requirement 9, in miniature: a full buy beats an eco more often than a
    # full buy beats a full buy, which just needs delta_loadout's beta > 0.
    table = _synthetic_table()
    model = fit_round_win_model(table)

    p_full_vs_eco = predict_round_win_probability(model, delta_loadout=17.5, half2=False, map_name="ascent")
    p_full_vs_full = predict_round_win_probability(model, delta_loadout=0.0, half2=False, map_name="ascent")
    assert p_full_vs_eco > p_full_vs_full
