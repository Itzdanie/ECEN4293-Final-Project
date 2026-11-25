"""The Streamlit UI, driven headlessly with Streamlit's own AppTest. The UI is
a thin layer over winprob.predict, so these check wiring and error display,
not the modeling (covered by the other test files)."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from winprob.cli import main  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "winprob" / "ui" / "app.py"


def fresh() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    assert not at.exception
    return at


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_default_view_is_tier_2_exact_and_shows_both_teams():
    at = fresh()
    shown = metrics(at)
    assert shown["Your team"] == "41.26%" and shown["Enemy team"] == "58.74%"  # 5-6
    assert any(s.value == "Exact" for s in at.subheader)


def test_tier_1_shows_the_score_only_answer_and_the_economy_caveat():
    at = fresh()
    at.segmented_control(key="tier").set_value(1).run()
    assert metrics(at)["Your team"] == "37.75%"
    assert any("ignores the economy" in i.value for i in at.info)


def test_impossible_state_shows_an_explanatory_error_not_a_crash():
    at = fresh()
    at.number_input(key="score_a").set_value(14).run()
    at.number_input(key="score_b").set_value(5).run()
    assert not at.exception
    assert any("past the win condition" in e.value for e in at.error)


def test_a_decided_map_is_exact_one_and_zero():
    at = fresh()
    at.segmented_control(key="tier").set_value(1).run()
    at.number_input(key="score_a").set_value(13).run()
    at.number_input(key="score_b").set_value(4).run()
    shown = metrics(at)
    assert shown["Your team"] == "100.00%" and shown["Enemy team"] == "0.00%"


def test_compare_mode_waits_for_run_then_shows_exact_mc_and_the_difference():
    at = fresh()
    at.segmented_control(key="tier").set_value(1).run()
    at.segmented_control(key="method").set_value("both").run()
    at.segmented_control(key="n").set_value(1000).run()
    assert "Monte Carlo minus exact" not in metrics(at)  # nothing until Run is pressed
    at.button(key="run").click().run()
    assert not at.exception
    shown = metrics(at)
    assert "Monte Carlo minus exact" in shown
    assert any("95% confidence interval" in c.value for c in at.caption)


def test_tier_3_runs_monte_carlo_on_the_button_and_labels_it():
    at = fresh()
    at.segmented_control(key="tier").set_value(3).run()
    at.segmented_control(key="n").set_value(1000).run()
    at.button(key="run").click().run()
    assert not at.exception
    assert any(s.value.startswith("Monte Carlo (N=1,000") for s in at.subheader)
    assert metrics(at)["Your team"].endswith("%")


def test_tier_3_agent_picker_and_ready_ults_reach_the_model():
    at = fresh()
    at.segmented_control(key="tier").set_value(3).run()
    at.segmented_control(key="n").set_value(1000).run()
    at.checkbox(key="a_named").set_value(True).run()
    at.number_input(key="score_a").set_value(5).run()
    at.number_input(key="score_b").set_value(5).run()
    at.pills(key="a_ready").set_value(["omen", "raze"]).run()
    at.button(key="run").click().run()
    assert not at.exception and not at.error


def test_tier_3_ready_ults_at_a_pistol_round_are_explained():
    at = fresh()
    at.segmented_control(key="tier").set_value(3).run()
    at.number_input(key="score_a").set_value(0).run()
    at.number_input(key="score_b").set_value(0).run()
    at.segmented_control(key="ults_a").set_value(2).run()
    at.button(key="run").click().run()
    assert any("reset every ult" in e.value for e in at.error)


def test_a_stale_result_is_flagged_when_inputs_change():
    at = fresh()
    at.segmented_control(key="tier").set_value(3).run()
    at.segmented_control(key="n").set_value(1000).run()
    at.button(key="run").click().run()
    at.number_input(key="score_a").set_value(6).run()
    assert any("inputs changed" in w.value for w in at.warning)


def test_scoreboard_plus_and_minus_buttons_change_the_score():
    at = fresh()
    at.button(key="plus_a").click().run()
    at.button(key="plus_b").click().run()
    at.button(key="minus_b").click().run()  # take back an accidental click
    assert (at.session_state["score_a"], at.session_state["score_b"]) == (6, 6)
    at.number_input(key="score_a").set_value(0).run()
    assert at.button(key="minus_a").disabled  # can't go below zero
    at.number_input(key="score_a").set_value(13).run()
    assert at.button(key="plus_a").disabled and not at.button(key="minus_a").disabled


def test_situation_line_names_the_round_side_and_pistol_and_overtime():
    at = fresh()
    assert any("Round 12 · you're on attack" in c.value for c in at.caption)
    at.button(key="plus_a").click().run()  # 6-6: round 13, sides swap
    assert any("Round 13 · you're on defense · pistol round" in c.value for c in at.caption)
    at.number_input(key="score_a").set_value(12).run()
    at.number_input(key="score_b").set_value(12).run()
    assert any("Round 25 · you're on defense · overtime" in c.value for c in at.caption)
    at.number_input(key="score_a").set_value(13).run()
    assert any("your match point" in c.value for c in at.caption)
    at.number_input(key="score_a").set_value(14).run()
    assert any("you won" in c.value for c in at.caption)
    assert at.button(key="plus_a").disabled


def test_map_and_side_are_greyed_out_in_tier_2_only():
    at = fresh()  # Tier 2 has no map or side in its model
    assert at.selectbox(key="map").disabled and at.segmented_control(key="a_started").disabled
    assert any("doesn't use the map" in c.value for c in at.caption)
    for tier in (1, 3):
        at.segmented_control(key="tier").set_value(tier).run()
        assert not at.selectbox(key="map").disabled
        assert not at.segmented_control(key="a_started").disabled


def test_tier_3_run_is_disabled_until_five_agents_are_picked():
    at = fresh()
    at.segmented_control(key="tier").set_value(3).run()
    at.checkbox(key="a_named").set_value(True).run()
    at.pills(key="a_duel").set_value(["raze"]).run()  # drops jett: four agents
    assert at.button(key="run").disabled
    assert any("pick exactly five" in w.value for w in at.warning)


def _write_match(path: Path) -> None:
    rows = []
    for i, w in enumerate(["Alpha", "Beta", "Alpha", "Alpha"], start=1):
        rows.append(
            {"match_id": "m1", "date": "2026-01-01", "map": "bind", "round": i,
             "atk_team": "Alpha", "def_team": "Beta", "winner": w, "recorded": 1,
             "atk_spent": 3000.0, "def_spent": 1000.0, "atk_bank": 500.0, "def_bank": 200.0}
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_match_tab_graphs_a_match_and_explains_a_missing_file_or_id(tmp_path):
    csv = tmp_path / "rounds.csv"
    _write_match(csv)
    at = fresh()
    at.text_input(key="rounds_path").set_value(str(csv)).run()
    at.text_input(key="match_id").set_value("m1").run()
    assert not at.exception and not at.error
    assert len(at.dataframe) == 1  # the round-by-round table
    # one win % per round, taken after that round (Alpha wins, loses, wins, wins)
    assert list(at.dataframe[0].value["Alpha win % after"]) == [57.6, 49.0, 56.9, 65.0]
    at.text_input(key="match_id").set_value("nope").run()
    assert any("isn't in the rounds file" in e.value for e in at.error)
    at.text_input(key="rounds_path").set_value(str(tmp_path / "missing.csv")).run()
    assert any("not found" in w.value for w in at.warning)


def test_ui_command_launches_streamlit_on_the_app_file(monkeypatch):
    import subprocess

    seen = {}
    monkeypatch.setattr(subprocess, "call", lambda cmd: seen.setdefault("cmd", cmd) and 0)
    assert main(["ui", "--port", "8765"]) == 0
    cmd = seen["cmd"]
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("app.py")
    assert cmd[-2:] == ["--server.port", "8765"]
