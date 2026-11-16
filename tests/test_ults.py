import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from winprob.data.integrity import check_ult_ready
from winprob.numerics.logistic import fit_newton, fit_newton_penalized, log_likelihood, sigmoid
from winprob.regression import ult_win_model
from winprob.regression.ult_win_model import (
    build_ult_training_table,
    choose_penalties,
    fit_ult_win_model,
)
from winprob.ults.habits import charge_transitions, hazard
from winprob.ults.prep import clean_ult_rows
from winprob.ults.roles import AGENT_SUBROLE, AGENTS, ROLES, SUB_ROLE_AGENTS, role_of


# --- roles -------------------------------------------------------------------


def test_every_agent_is_in_exactly_one_sub_role():
    listed = [a for agents in SUB_ROLE_AGENTS.values() for a in agents]
    assert len(listed) == len(set(listed)) == len(AGENTS)
    assert {role_of(AGENT_SUBROLE[a]) for a in AGENTS} == set(ROLES)


# --- penalized Newton ----------------------------------------------------------


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    y = (rng.random(n) < sigmoid(X @ np.array([0.2, 0.8, -0.5, 0.3]))).astype(float)
    return X, y


def test_zero_penalty_reproduces_plain_newton():
    X, y = _synthetic()
    plain, _, _ = fit_newton(X, y)
    pen, _, _ = fit_newton_penalized(X, y, np.zeros(X.shape[1]))
    assert np.allclose(plain, pen, atol=1e-8)


def test_penalized_newton_matches_scipy_and_shrinks_toward_zero():
    X, y = _synthetic()
    penalty = np.array([0.0, 50.0, 50.0, 50.0])

    def neg_objective(b):
        return -(log_likelihood(X, y, b) - 0.5 * np.sum(penalty * b * b))

    ours, n_iter, history = fit_newton_penalized(X, y, penalty)
    ref = minimize(neg_objective, np.zeros(4), method="BFGS", options={"gtol": 1e-9}).x
    assert np.allclose(ours, ref, atol=1e-4)
    assert n_iter < 15
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:]))  # monotone ascent

    plain, _, _ = fit_newton(X, y)
    assert np.all(np.abs(ours[1:]) < np.abs(plain[1:]))


# --- integrity / cleaning ------------------------------------------------------


def _ult_rows(match="m1", rounds=(1, 2), team="a", agents=("omen", "raze", "jett", "sova", "cypher"), ready=0, spent=0):
    return [
        {"match_id": match, "round": r, "team": team, "slot": s, "agent": a, "ready": ready, "spent": spent}
        for r in rounds
        for s, a in enumerate(agents)
    ]


def test_check_ult_ready_counts_each_kind_of_violation():
    rows = pd.DataFrame(_ult_rows(rounds=(1, 2, 12)))
    rows.loc[(rows["round"] == 2) & (rows["slot"] == 0), "spent"] = 1  # spent without ready
    rows.loc[(rows["round"] == 1) & (rows["slot"] == 1), "ready"] = 1  # ready on pistol round
    rows = rows.iloc[:-1]  # round 12 is now incomplete (4 rows)
    found = check_ult_ready(rows)
    assert found == {
        "spent_without_ready": 1,
        "ready_at_reset_round": 1,
        "duplicate_slots": 0,
        "rows_in_rounds_12_24": 4,
        "incomplete_team_rounds": 1,
    }


def test_clean_ult_rows_drops_incomplete_team_rounds_and_rounds_12_24():
    rows = pd.DataFrame(_ult_rows(rounds=(1, 2, 12, 24)))
    rows = rows.drop(index=rows[(rows["round"] == 2) & (rows["slot"] == 4)].index)  # round 2 incomplete
    maps = pd.DataFrame({"match_id": ["m1"], "team_a": ["Alpha"], "team_b": ["Beta"]})
    clean = clean_ult_rows(rows, maps)
    assert sorted(clean["round"].unique()) == [1]
    assert (clean["team_name"] == "Alpha").all()


def test_clean_ult_rows_rejects_an_agent_with_no_sub_role():
    rows = pd.DataFrame(_ult_rows(agents=("omen", "raze", "jett", "sova", "new_agent")))
    maps = pd.DataFrame({"match_id": ["m1"], "team_a": ["A"], "team_b": ["B"]})
    with pytest.raises(ValueError, match="new_agent"):
        clean_ult_rows(rows, maps)


# --- charge transitions ----------------------------------------------------------


def _slot_history(ready, spent, agent="omen"):
    return pd.DataFrame(
        {
            "match_id": "m1", "team": "a", "slot": 0, "agent": agent,
            "round": np.arange(1, len(ready) + 1), "ready": ready, "spent": spent,
        }
    )


def test_charge_transitions_track_age_and_origin():
    # not ready r1-r3, ready r4, used r4, not ready r5-r6, ready r7
    ready = [0, 0, 0, 1, 0, 0, 1]
    spent = [0, 0, 0, 1, 0, 0, 0]
    t = charge_transitions(_slot_history(ready, spent))
    by_round = t.reset_index(drop=True)
    # not-ready rows: r1,r2,r3 (start-of-half run, ages 0,1,2), r5,r6 (post-spend, ages 0,1)
    assert list(by_round["age"]) == [0, 1, 2, 0, 1]
    assert list(by_round["post_spend"]) == [False, False, False, True, True]
    assert list(by_round["next_ready"]) == [0.0, 0.0, 1.0, 0.0, 1.0]


def test_charge_transitions_skip_slots_after_a_gap_in_rounds():
    h = _slot_history([0, 0, 0, 0, 0, 0], [0] * 6)
    h = h[h["round"] != 3]  # round 3 missing -> age after the gap is unknown
    t = charge_transitions(h)
    assert list(t["age"]) == [0]  # round 2 has no next round; rounds 4-6 follow the gap


def test_hazard_uses_the_right_curve_and_caps_age():
    start, post = [0.1, 0.2, 0.3], [0.4, 0.5]
    age = np.array([0, 5, 1, 7])
    got = hazard(start, post, age, np.array([False, False, True, True]), np.ones(4))
    assert np.allclose(got, [0.1, 0.3, 0.5, 0.5])
    assert hazard(start, post, np.array([0]), np.array([True]), np.array([10.0]))[0] <= 0.95


# --- win-boost model --------------------------------------------------------------


def _synthetic_ult_table(n=40_000, seed=1):
    rng = np.random.default_rng(seed)
    role_effect = {"ctrl": 0.05, "duel": 0.15, "init": 0.08, "sent": 0.2}
    true = np.array([role_effect[role_of(AGENT_SUBROLE[a])] for a in AGENTS])
    deltas = rng.integers(-1, 2, size=(n, len(AGENTS))).astype(float) * (rng.random((n, len(AGENTS))) < 0.15)
    delta_loadout = rng.normal(0, 3, n)
    logit = 0.1 + 0.1 * delta_loadout + deltas @ true
    table = pd.DataFrame(deltas, columns=["delta_" + a for a in AGENTS])
    table["match_id"] = rng.integers(0, 600, n).astype(str)
    table["map"] = rng.choice(["a", "b"], n)
    table["delta_loadout"] = delta_loadout
    table["half2"] = (rng.random(n) < 0.5).astype(float)
    table["atk_won"] = (rng.random(n) < sigmoid(logit)).astype(float)
    return table, role_effect


def test_ult_model_recovers_known_role_effects_from_synthetic_rounds():
    table, role_effect = _synthetic_ult_table()
    model = fit_ult_win_model(table, lam_sub=1e4, lam_agent=300.0)
    assert model["n_iter"] <= 15
    for role, truth in role_effect.items():
        mean_effect = np.mean([e for a, e in model["agent_effects"].items() if role_of(AGENT_SUBROLE[a]) == role])
        assert abs(mean_effect - truth) < 0.05, (role, mean_effect, truth)
    assert model["role_effects"]["sent"] > model["role_effects"]["ctrl"]


def test_an_agent_with_no_rounds_gets_exactly_its_group_effect():
    table, _ = _synthetic_ult_table(n=20_000)
    table["delta_hush"] = 0.0  # Hush never appears (true of the real window)
    model = fit_ult_win_model(table, lam_sub=1e3, lam_agent=300.0)
    assert model["agent_deviations"]["hush"] == pytest.approx(0.0, abs=1e-9)
    sub = AGENT_SUBROLE["hush"]
    assert model["agent_effects"]["hush"] == pytest.approx(
        model["role_effects"][role_of(sub)] + model["sub_role_deviations"][sub]
    )


def test_choose_penalties_returns_values_from_the_grid():
    table, _ = _synthetic_ult_table(n=6_000)
    lam_sub, lam_agent, grid = choose_penalties(table, sub_grid=(100.0, 10_000.0), agent_grid=(100.0, 1_000.0), n_folds=3)
    assert lam_sub in (100.0, 10_000.0) and lam_agent in (100.0, 1_000.0)
    assert len(grid) == 4


def test_training_table_signs_attacker_minus_defender(monkeypatch):
    base = pd.DataFrame(
        {"match_id": ["m1"], "round": [2], "map": ["bind"], "atk_won": [1.0], "delta_loadout": [0.0], "half2": [0.0]}
    )
    monkeypatch.setattr(ult_win_model, "build_training_table", lambda *a, **k: base)
    rounds = pd.DataFrame({"match_id": ["m1"], "round": [2], "atk_team": ["Alpha"], "def_team": ["Beta"]})

    def team(name, ready_agents):
        rows = _ult_rows(rounds=(2,), team="a")
        for r, a in zip(rows, ("omen", "raze", "jett", "sova", "cypher")):
            r["ready"] = int(a in ready_agents)
            r["team_name"] = name
        return rows

    ult = pd.DataFrame(team("Alpha", {"raze", "jett"}) + team("Beta", {"raze", "cypher"}))
    out = build_ult_training_table(rounds, pd.DataFrame(), ult)
    row = out.iloc[0]
    assert row["delta_raze"] == 0  # both ready
    assert row["delta_jett"] == 1  # attackers only
    assert row["delta_cypher"] == -1  # defenders only
    assert row["delta_omen"] == 0


def test_training_table_drops_rounds_where_a_team_has_no_ult_record(monkeypatch):
    base = pd.DataFrame(
        {"match_id": ["m1", "m1"], "round": [2, 3], "map": ["bind"] * 2, "atk_won": [1.0, 0.0],
         "delta_loadout": [0.0, 0.0], "half2": [0.0, 0.0]}
    )
    monkeypatch.setattr(ult_win_model, "build_training_table", lambda *a, **k: base)
    rounds = pd.DataFrame(
        {"match_id": ["m1"] * 2, "round": [2, 3], "atk_team": ["Alpha"] * 2, "def_team": ["Beta"] * 2}
    )
    rows = []
    for rnd, teams in ((2, ("Alpha", "Beta")), (3, ("Alpha",))):  # round 3: Beta missing
        for name in teams:
            for r in _ult_rows(rounds=(rnd,)):
                rows.append({**r, "team_name": name})
    out = build_ult_training_table(rounds, pd.DataFrame(), pd.DataFrame(rows))
    assert list(out["round"]) == [2]
