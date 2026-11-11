import numpy as np
import pytest

from winprob.tier1.exact import WIN_SCORE, solve_linear_system, solve_recursive
from winprob.tier1.monte_carlo import simulate_win_probability
from winprob.tier1.overtime import ot_win_probability
from winprob.tier1.states import map_winner, other_side, side_for_team_a

P_ATK_VALUES = [0.5, 0.35, 0.7]
A_STARTED_VALUES = ["atk", "def"]


def _dual_table(p_atk, a_started):
    """Solve once as team A, once as team B (who starts on the other side),
    for the Requirement 1 sum-to-one cross-check."""
    table_a = solve_linear_system(p_atk, a_started)
    table_b = solve_linear_system(p_atk, other_side(a_started))
    return table_a, table_b


# --- Requirement 1: probabilities in [0, 1] and sum to 1 within 1e-6 -------


@pytest.mark.parametrize("p_atk", P_ATK_VALUES)
@pytest.mark.parametrize("a_started", A_STARTED_VALUES)
def test_probabilities_bounded_and_sum_to_one(p_atk, a_started):
    table_a, table_b = _dual_table(p_atk, a_started)

    for (score_a, score_b), p_a in table_a.items():
        assert 0.0 <= p_a <= 1.0
        p_b = table_b[(score_b, score_a)]
        assert 0.0 <= p_b <= 1.0
        assert abs((p_a + p_b) - 1.0) < 1e-6


# --- Requirement 2: symmetric 50% round-win gives P = 0.5 at 0-0 ----------


@pytest.mark.parametrize("a_started", A_STARTED_VALUES)
def test_symmetric_round_win_gives_half_at_start(a_started):
    assert abs(solve_linear_system(0.5, a_started)[(0, 0)] - 0.5) < 1e-6
    assert abs(solve_recursive(0.5, a_started)[(0, 0)] - 0.5) < 1e-6


# --- Requirement 3: clinched maps are exactly 1/0; OT matches the closed form --


@pytest.mark.parametrize("p_atk", P_ATK_VALUES)
@pytest.mark.parametrize("a_started", A_STARTED_VALUES)
@pytest.mark.parametrize("solver", [solve_linear_system, solve_recursive])
def test_clinched_states_have_exact_probability(p_atk, a_started, solver):
    table = solver(p_atk, a_started)
    for k in range(WIN_SCORE):
        assert table[(WIN_SCORE, k)] == 1.0
        assert table[(k, WIN_SCORE)] == 0.0


@pytest.mark.parametrize("p_atk", [0.1, 0.35, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("a_started", A_STARTED_VALUES)
def test_prematch_win_probability_is_always_half(p_atk, a_started):
    # Because a map can't be clinched inside a 12-round half, both teams
    # always play a full, equal-length first half before anything can be
    # decided - so the single side-swap at round 13 exactly cancels out any
    # attacker/defender edge, and 0-0 is a coin flip no matter how lopsided
    # p_atk is. (Same reasoning makes 12-12 a coin flip too, since the two
    # overtime rounds split the same way.) Extends Requirement 2's p=0.5
    # case to confirm the side-swap logic itself, not just the input.
    assert abs(solve_linear_system(p_atk, a_started)[(0, 0)] - 0.5) < 1e-6
    assert abs(solve_recursive(p_atk, a_started)[(0, 0)] - 0.5) < 1e-6


def test_overtime_is_half_when_round_win_is_symmetric():
    for a_started in A_STARTED_VALUES:
        assert abs(solve_linear_system(0.5, a_started)[(12, 12)] - 0.5) < 1e-6
        assert abs(solve_recursive(0.5, a_started)[(12, 12)] - 0.5) < 1e-6


@pytest.mark.parametrize("p_atk", [0.35, 0.7])
@pytest.mark.parametrize("a_started", A_STARTED_VALUES)
def test_overtime_matches_hand_derived_closed_form(p_atk, a_started):
    # Re-derive the pair probabilities by hand, independent of winprob.tier1.overtime,
    # to check the implementation against CLAUDE.md's formula rather than against itself.
    side_a_round1 = other_side(a_started)  # one half-swap already happened by total=24
    side_a_round2 = other_side(side_a_round1)  # overtime swaps every round

    p1 = p_atk if side_a_round1 == "atk" else 1.0 - p_atk
    p2 = p_atk if side_a_round2 == "atk" else 1.0 - p_atk
    w = p1 * p2
    l = (1.0 - p1) * (1.0 - p2)
    expected = w / (w + l)

    assert abs(solve_linear_system(p_atk, a_started)[(12, 12)] - expected) < 1e-6
    assert abs(solve_recursive(p_atk, a_started)[(12, 12)] - expected) < 1e-6


def test_ot_win_probability_weights_are_consistent():
    w, l, p = ot_win_probability(0.7, "atk")
    assert 0.0 <= w <= 1.0
    assert 0.0 <= l <= 1.0
    assert abs(p - w / (w + l)) < 1e-12


# --- Requirement 4: linear-system and recursion agree within 1e-9 ---------


@pytest.mark.parametrize("p_atk", P_ATK_VALUES)
@pytest.mark.parametrize("a_started", A_STARTED_VALUES)
def test_linear_system_and_recursion_agree(p_atk, a_started):
    table_linear = solve_linear_system(p_atk, a_started)
    table_recursive = solve_recursive(p_atk, a_started)

    assert table_linear.keys() == table_recursive.keys()
    for state, p_linear in table_linear.items():
        assert abs(p_linear - table_recursive[state]) < 1e-9


# --- side/state helpers ----------------------------------------------------


def test_side_for_team_a_swaps_once_after_round_12():
    assert side_for_team_a(0, "atk") == "atk"
    assert side_for_team_a(11, "atk") == "atk"
    assert side_for_team_a(12, "atk") == "def"
    assert side_for_team_a(23, "atk") == "def"


def test_map_winner_requires_two_round_margin_in_overtime():
    assert map_winner(13, 12) is None  # not decided: only a 1-round margin in OT
    assert map_winner(14, 12) == 1.0
    assert map_winner(12, 14) == 0.0
    assert map_winner(13, 5) == 1.0  # clinched in regulation, no OT margin rule needed


# --- Monte Carlo smoke test -------------------------------------------------


@pytest.mark.parametrize("score_a,score_b", [(0, 0), (10, 8), (12, 12)])
def test_monte_carlo_is_close_to_exact(score_a, score_b):
    rng = np.random.default_rng(0)
    exact = solve_linear_system(0.5, "atk")[(score_a, score_b)]
    p_hat, ci_low, ci_high = simulate_win_probability(0.5, "atk", score_a, score_b, 20_000, rng)
    assert abs(p_hat - exact) < 0.02
    assert ci_low <= p_hat <= ci_high
