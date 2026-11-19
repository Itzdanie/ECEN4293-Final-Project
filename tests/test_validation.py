"""Requirement 10: impossible states raise an error that says what's wrong."""

from functools import lru_cache

import numpy as np
import pytest

from winprob.predict import predict_tier1, predict_tier2, predict_tier3
from winprob.tier1.lookup import exact_win_probability
from winprob.tier1.overtime import ot_win_probability
from winprob.tier1.states import other_side, round_win_prob_a, side_for_team_a
from winprob.validation import (
    TEAM_CREDIT_CAP,
    InvalidStateError,
    validate_scores,
    validate_tier2_state,
    validate_tier3_state,
)

T3 = dict(bank_a=19000, bank_b=19000, ults_a=0, ults_b=0, streak_a=0, streak_b=0)


def test_error_type_is_a_value_error():
    assert issubclass(InvalidStateError, ValueError)


@pytest.mark.parametrize("a,b", [(14, 5), (5, 14), (15, 3), (13, 14 + 2), (20, 12), (12, 15)])
def test_scores_past_the_win_condition_are_rejected(a, b):
    with pytest.raises(InvalidStateError, match="past the win condition"):
        validate_scores(a, b)


def test_negative_and_non_integer_scores_are_rejected():
    with pytest.raises(InvalidStateError, match="negative"):
        validate_scores(-1, 3)
    with pytest.raises(InvalidStateError, match="whole number"):
        validate_scores(3.5, 3)


@pytest.mark.parametrize("a,b", [(0, 0), (12, 12), (13, 0), (13, 11), (13, 12), (14, 12), (13, 13), (15, 13), (14, 16 - 2)])
def test_every_reachable_score_is_accepted(a, b):
    validate_scores(a, b)


def test_negative_credits_are_rejected():
    with pytest.raises(InvalidStateError, match=r"bank_a can't be negative"):
        validate_tier3_state(5, 5, **{**T3, "bank_a": -1})


def test_credits_over_the_cap_are_rejected_with_the_cap_in_the_message():
    with pytest.raises(InvalidStateError, match=str(TEAM_CREDIT_CAP)):
        validate_tier3_state(5, 5, **{**T3, "bank_b": TEAM_CREDIT_CAP + 1})
    validate_tier3_state(5, 5, **{**T3, "bank_b": TEAM_CREDIT_CAP})  # exactly at the cap is fine


def test_more_than_five_ults_are_rejected():
    with pytest.raises(InvalidStateError, match="ults_a must be between 0 and 5"):
        validate_tier3_state(5, 5, **{**T3, "ults_a": 6})
    with pytest.raises(InvalidStateError, match="ults_b"):
        validate_tier3_state(5, 5, **{**T3, "ults_b": -1})


def test_ults_at_a_pistol_or_overtime_round_are_rejected():
    for score in [(0, 0), (6, 6), (12, 12), (13, 13)]:
        with pytest.raises(InvalidStateError, match="reset every ult"):
            validate_tier3_state(*score, **{**T3, "ults_a": 1})


def test_other_impossible_inputs_are_described():
    with pytest.raises(InvalidStateError, match="streak_a"):
        validate_tier3_state(5, 5, **{**T3, "streak_a": 4})
    with pytest.raises(InvalidStateError, match="survivors_a"):
        validate_tier3_state(5, 5, **T3, survivors_a=6)
    with pytest.raises(InvalidStateError, match="unknown map"):
        validate_tier3_state(5, 5, **T3, map_name="not_a_map")
    with pytest.raises(InvalidStateError, match="Eco"):
        validate_tier2_state(5, 5, "Huge buy", "Eco", 0, 0)
    with pytest.raises(InvalidStateError, match="Tier 2 stops"):
        validate_tier2_state(13, 13, "Eco", "Eco", 0, 0)


def test_predict_functions_validate_before_computing():
    with pytest.raises(InvalidStateError, match="past the win condition"):
        predict_tier1(14, 5)
    with pytest.raises(InvalidStateError, match="probability"):
        predict_tier1(5, 5, p_atk=1.2)
    with pytest.raises(InvalidStateError, match="streak_b"):
        predict_tier2(5, 5, "Eco", "Eco", 0, 9)
    with pytest.raises(InvalidStateError, match="negative"):
        predict_tier3(5, 5, -5.0, 100.0, n=10)
    with pytest.raises(InvalidStateError, match="agents_a"):
        predict_tier3(5, 5, 100.0, 100.0, ready_agents_a=["jett"], n=10)
    with pytest.raises(InvalidStateError, match="aren't on the team"):
        predict_tier3(5, 5, 100.0, 100.0, agents_a=["omen", "raze", "jett", "sova", "cypher"], ready_agents_a=["fade"], n=10)


# --- exact Tier 1 answers for any score, including deep overtime -----------------


def _brute_force_overtime(lead, k, p_atk, a_started, depth=300):
    """Win-by-two overtime by plain recursion, one round at a time
    (memoized on lead and rounds played, so it stays small)."""
    side_first = side_for_team_a(24, a_started)

    @lru_cache(maxsize=None)
    def f(lead, k):
        if lead == 2:
            return 1.0
        if lead == -2:
            return 0.0
        if k > depth:
            return 0.5
        side = side_first if k % 2 == 0 else other_side(side_first)
        q = round_win_prob_a(p_atk, side)
        return q * f(lead + 1, k + 1) + (1 - q) * f(lead - 1, k + 1)

    return f(lead, k)


@pytest.mark.parametrize("a_started", ["atk", "def"])
@pytest.mark.parametrize("p_atk", [0.45, 0.5, 0.55])
@pytest.mark.parametrize("score", [(12, 12), (13, 12), (12, 13), (13, 13), (14, 13), (13, 14), (15, 15)])
def test_exact_overtime_lookup_matches_round_by_round_recursion(score, p_atk, a_started):
    a, b = score
    expected = _brute_force_overtime(a - b, a + b - 24, p_atk, a_started)
    assert exact_win_probability(a, b, p_atk, a_started) == pytest.approx(expected, abs=1e-9)


def test_a_clinched_map_is_exactly_one_or_zero_and_the_tied_entry_matches_the_closed_form():
    assert exact_win_probability(13, 4, 0.5, "atk") == 1.0
    assert exact_win_probability(2, 13, 0.5, "atk") == 0.0
    assert exact_win_probability(14, 12, 0.5, "def") == 1.0
    _, _, closed_form = ot_win_probability(0.53, side_for_team_a(24, "atk"))
    assert exact_win_probability(12, 12, 0.53, "atk") == pytest.approx(closed_form, abs=1e-12)


def test_predict_exact_probabilities_sum_to_one():
    for score in [(0, 0), (7, 9), (13, 12), (12, 12)]:
        est = predict_tier1(*score).exact
        assert est.p_a + est.p_b == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= est.p_a <= 1.0


def test_decided_maps_are_exact_in_every_tier():
    assert predict_tier1(13, 5).exact.p_a == 1.0
    assert predict_tier2(3, 13, "Eco", "Full buy").exact.p_a == 0.0
    assert predict_tier3(13, 2, 0.0, 0.0, n=10).monte_carlo.p_a == 1.0


def test_compare_mode_reports_the_difference_between_mc_and_exact():
    pred = predict_tier1(6, 6, method="both", n=20_000, seed=3)
    assert pred.exact is not None and pred.monte_carlo is not None
    assert pred.difference == pytest.approx(pred.monte_carlo.p_a - pred.exact.p_a)
    assert abs(pred.difference) < 0.02
    lo, hi = pred.monte_carlo.ci
    assert lo <= pred.monte_carlo.p_a <= hi
    assert np.isfinite(pred.difference)
