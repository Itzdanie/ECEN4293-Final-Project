import numpy as np
import pandas as pd
import pytest

from winprob.cli import format_fit_report, main
from winprob.data.loading import load_rounds
from winprob.numerics.logistic import sigmoid
from winprob.regression import fit_round_win_model
from winprob.reporting.match import (
    load_match_rounds,
    plot_win_probability,
    step_through,
    win_probability_series,
)


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# --- predict -------------------------------------------------------------------


def test_predict_tier1_says_exact_and_probabilities_add_to_100(capsys):
    code, out, _ = run(capsys, "predict", "--tier", "1", "--score", "5-7")
    assert code == 0
    assert "Tier 1 [exact]" in out and "Monte Carlo" not in out
    assert "Team A  27.28%" in out and "Team B  72.72%" in out


def test_predict_compare_shows_exact_mc_ci_and_difference(capsys):
    code, out, _ = run(capsys, "predict", "--tier", "1", "--score", "6-6", "--compare", "--n", "5000")
    assert code == 0
    assert "[exact]" in out and "[Monte Carlo, N=5,000, seed=0]" in out
    assert "95% CI" in out and "Difference (Monte Carlo - exact)" in out


def test_predict_is_reproducible_with_a_seed(capsys):
    args = ("predict", "--tier", "1", "--score", "4-4", "--method", "mc", "--n", "2000", "--seed", "7")
    assert run(capsys, *args)[1] == run(capsys, *args)[1]


def test_tier_2_is_the_default_and_tier_1_carries_an_economy_caveat(capsys):
    code, out, _ = run(capsys, "predict", "--score", "5-7")  # round 13 is a pistol: no buys needed
    assert code == 0 and "Tier 2 [exact]" in out and "Note:" not in out
    code, out, _ = run(capsys, "predict", "--tier", "1", "--score", "5-7")
    assert "ignores the economy" in out
    code, _, err = run(capsys, "predict", "--score", "5-6")  # not a pistol round: buys are needed
    assert code == 2 and "--buy-a" in err and "--tier 1" in err


def test_predict_tier2_and_tier3(capsys):
    code, out, _ = run(capsys, "predict", "--tier", "2", "--score", "0-0", "--buy-a", "Eco", "--buy-b", "Eco")
    assert code == 0 and "Tier 2 [exact]" in out
    code, out, _ = run(
        capsys, "predict", "--tier", "3", "--score", "5-5", "--bank-a", "19000", "--bank-b", "19000",
        "--agents-a", "chamber,cypher,raze,jett,veto", "--ready-a", "jett", "--n", "2000",
    )
    assert code == 0 and "Tier 3 [Monte Carlo" in out and "95% CI" in out


def test_tier3_has_no_exact_answer_to_compare(capsys):
    code, _, err = run(capsys, "predict", "--tier", "3", "--score", "5-5", "--bank-a", "1", "--bank-b", "1", "--compare")
    assert code == 2 and "Monte Carlo only" in err


@pytest.mark.parametrize(
    "argv,message",
    [
        (["predict", "--score", "14-5"], "past the win condition"),
        (["predict", "--tier", "2", "--score", "5-5", "--buy-a", "Eco", "--buy-b", "Eco", "--streak-a", "9"], "streak_a"),
        (["predict", "--tier", "2", "--score", "5-5"], "--buy-a"),
        (["predict", "--tier", "3", "--score", "5-5", "--bank-a", "-3", "--bank-b", "100"], "negative"),
        (["predict", "--tier", "3", "--score", "5-5", "--bank-a", "99999", "--bank-b", "100"], "credit cap"),
        (["predict", "--tier", "3", "--score", "5-5", "--bank-a", "1", "--bank-b", "1", "--ults-a", "6"], "between 0 and 5"),
        (["predict", "--tier", "3", "--score", "5-5"], "--bank-a"),
        (["predict", "--tier", "3", "--score", "0-0", "--ults-a", "1"], "reset every ult"),
    ],
)
def test_impossible_states_exit_2_with_a_message(capsys, argv, message):
    code, out, err = run(capsys, *argv)
    assert code == 2 and out == ""
    assert err.startswith("error:") and message in err


def test_a_bad_score_format_is_rejected_by_the_parser(capsys):
    with pytest.raises(SystemExit):
        main(["predict", "--score", "seven-five"])
    assert "score must look like" in capsys.readouterr().err


# --- solve / converge / fit -------------------------------------------------------


def test_solve_reports_a_rate_that_really_reaches_the_target(capsys):
    for method in ("bisection", "secant"):
        code, out, _ = run(capsys, "solve", "--score", "3-8", "--target", "0.5", "--method", method)
        assert code == 0
        assert "exact win probability at that rate = 0.500000" in out


def test_solve_rejects_a_target_outside_zero_one(capsys):
    code, _, err = run(capsys, "solve", "--score", "3-8", "--target", "1.5")
    assert code == 2 and "--target" in err


def test_converge_prints_the_slope_and_saves_the_plot(capsys, tmp_path):
    out_png = tmp_path / "c.png"
    code, out, _ = run(capsys, "converge", "--trials", "3", "--max-n", "3000", "--out", str(out_png))
    assert code == 0 and "Log-log slope" in out
    assert out_png.exists() and out_png.stat().st_size > 0


def test_fit_report_lists_coefficients_log_likelihood_and_iterations():
    rng = np.random.default_rng(0)
    n = 4000
    table = pd.DataFrame(
        {
            "delta_loadout": rng.normal(0, 2, n),
            "half2": (rng.random(n) < 0.5).astype(float),
            "map": rng.choice(["abyss", "bind"], n),
        }
    )
    table["atk_won"] = (rng.random(n) < sigmoid(0.1 + 0.1 * table["delta_loadout"])).astype(float)
    report = format_fit_report(fit_round_win_model(table))
    for expected in ("delta_loadout", "half2", "map_bind", "Log-likelihood:", "Newton iterations"):
        assert expected in report


def test_fit_with_missing_files_explains_what_to_do(capsys, tmp_path):
    code, _, err = run(capsys, "fit", "--rounds", str(tmp_path / "nope.csv"), "--buys", str(tmp_path / "nope2.csv"))
    assert code == 2 and "not found" in err


# --- match reporting -----------------------------------------------------------------


def _match_rows(winners, match_id="m1", teams=("Alpha", "Beta"), map_name="bind"):
    """Rounds where Alpha attacks rounds 1-12, Beta attacks 13-24, with
    recorded economy figures."""
    rows = []
    for i, w in enumerate(winners, start=1):
        alpha_attacks = i <= 12
        atk, dfn = (teams[0], teams[1]) if alpha_attacks else (teams[1], teams[0])
        rows.append(
            {
                "match_id": match_id, "date": "2026-01-01", "map": map_name, "round": i,
                "atk_team": atk, "def_team": dfn, "winner": w, "recorded": 1,
                "atk_spent": 3000.0, "def_spent": 1000.0, "atk_bank": 500.0, "def_bank": 200.0,
            }
        )
    return pd.DataFrame(rows)


def test_win_probability_series_tracks_the_scoreline():
    winners = ["Alpha"] * 13  # Alpha sweeps 13-0
    series = win_probability_series(load_match_rounds(_match_rows(winners), "m1"), p_atk=0.5)
    assert len(series) == 14  # before round 1, then after each round
    assert series["p_a"].iloc[0] == pytest.approx(0.5, abs=1e-6)
    assert np.allclose(series["p_a"] + series["p_b"], 1.0, atol=1e-12)
    assert series["p_a"].is_monotonic_increasing
    assert series["p_a"].iloc[-1] == 1.0 and series["p_b"].iloc[-1] == 0.0
    assert series.attrs["team_a"] == "Alpha"  # attacks round 1


def test_a_round_won_by_b_lowers_a_win_probability():
    series = win_probability_series(load_match_rounds(_match_rows(["Alpha", "Beta", "Beta"]), "m1"), p_atk=0.5)
    assert series["p_a"].iloc[2] == pytest.approx(0.5, abs=1e-6)  # 1-1 on a 50% map
    assert series["p_a"].iloc[3] < series["p_a"].iloc[2]


def test_series_handles_a_match_that_goes_to_overtime():
    winners = ["Alpha", "Beta"] * 12 + ["Alpha", "Alpha"]  # 12-12 then Alpha wins by two
    series = win_probability_series(load_match_rounds(_match_rows(winners), "m1"), p_atk=0.5)
    assert series[["score_a", "score_b"]].iloc[-1].tolist() == [14, 12]
    assert series["p_a"].iloc[-1] == 1.0
    assert series["p_a"].iloc[-3] == pytest.approx(0.5, abs=1e-6)  # 12-12 on a 50% map


def test_history_stops_at_a_round_with_no_recorded_winner():
    rows = _match_rows(["Alpha", "Beta", "Alpha", "Beta"])
    rows.loc[2, "winner"] = np.nan
    match = load_match_rounds(rows, "m1")
    assert list(match["round"]) == [1, 2]


def test_unknown_match_id_is_explained():
    with pytest.raises(ValueError, match="isn't in the rounds file"):
        load_match_rounds(_match_rows(["Alpha"]), "nope")


def test_step_through_streaks_credits_and_halftime_reset():
    winners = ["Beta"] * 3 + ["Alpha"] * 9 + ["Alpha"]
    steps = step_through(load_match_rounds(_match_rows(winners), "m1"))
    assert list(steps["streak_a"][:4]) == [0, 1, 2, 3]  # Alpha loses three, then wins
    assert steps["streak_a"].iloc[4] == 0 and steps["streak_b"].iloc[3] == 0
    assert steps["score"].iloc[3] == "0-3"
    # round 1: Alpha attacks, credits = atk_spent + atk_bank; Beta defends
    assert steps["credits_a"].iloc[0] == 3500.0 and steps["credits_b"].iloc[0] == 1200.0
    assert steps["buy_a"].eq("-").all()  # no Buys supplied -> no buy type
    assert steps["streak_a"].iloc[12] == 0  # round 13 is a fresh half


def test_step_through_marks_unrecorded_rounds_and_uses_buys_for_buy_type():
    rows = _match_rows(["Alpha", "Beta"])
    rows.loc[1, "recorded"] = 0
    buys = pd.DataFrame(
        [
            {"match_id": "m1", "round": 1, "team": team, "player": f"{team}{i}", "gun": gun, "armor": armor}
            for team, gun, armor in (("Alpha", "vandal", "heavy"), ("Beta", "classic", None))
            for i in range(5)
        ]
    )
    steps = step_through(load_match_rounds(rows, "m1"), buys)
    assert steps["buy_a"].iloc[0] == "Full buy" and steps["buy_b"].iloc[0] == "Eco"
    assert np.isnan(steps["credits_a"].iloc[1])


def test_plot_is_saved(tmp_path):
    series = win_probability_series(load_match_rounds(_match_rows(["Alpha", "Beta", "Alpha"]), "m1"), p_atk=0.5)
    path = tmp_path / "wp.png"
    plot_win_probability(series, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_match_command_end_to_end_from_a_csv(capsys, tmp_path):
    csv = tmp_path / "rounds.csv"
    _match_rows(["Alpha", "Beta", "Alpha", "Alpha"]).to_csv(csv, index=False)
    code, out, _ = run(capsys, "match", "m1", "--rounds", str(csv), "--step", "--out", str(tmp_path / "p.png"))
    assert code == 0
    assert "Alpha (A) vs Beta (B)" in out and "P(A)" in out and "Plot saved" in out
    assert (tmp_path / "p.png").exists()
    # and the loader reads the same file the command used
    assert len(load_rounds(csv)) == 4
    code, _, err = run(capsys, "match", "zzz", "--rounds", str(csv))
    assert code == 2 and "isn't in the rounds file" in err
