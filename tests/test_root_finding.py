import math

import pytest

from winprob.numerics.root_finding import bisection, secant
from winprob.tier1.exact import solve_linear_system
from winprob.tier1.root_finding import (
    required_round_win_rate_bisection,
    required_round_win_rate_secant,
)

# --- generic bisection/secant -----------------------------------------------


def test_bisection_finds_sqrt_two():
    root = bisection(lambda x: x * x - 2.0, 0.0, 2.0, tol=1e-12)
    assert abs(root - math.sqrt(2.0)) < 1e-9


def test_secant_finds_sqrt_two():
    root = secant(lambda x: x * x - 2.0, 1.0, 2.0, tol=1e-12)
    assert abs(root - math.sqrt(2.0)) < 1e-9


def test_bisection_requires_a_sign_change():
    with pytest.raises(ValueError):
        bisection(lambda x: x * x + 1.0, 0.0, 2.0)  # always positive, no root


def test_secant_detects_a_stalled_iteration():
    with pytest.raises(ZeroDivisionError):
        secant(lambda x: 1.0, 0.0, 1.0)  # constant function: f(x0) == f(x1) forever


# --- Tier 1 inverse round-win-rate questions -------------------------------


SAFE_STATES = [(3, 8), (8, 3), (5, 9), (9, 5)]


@pytest.mark.parametrize("score_a,score_b", SAFE_STATES)
@pytest.mark.parametrize("a_started", ["atk", "def"])
def test_bisection_and_secant_agree_and_hit_the_target(score_a, score_b, a_started):
    q_bisection = required_round_win_rate_bisection(score_a, score_b, a_started)
    q_secant = required_round_win_rate_secant(score_a, score_b, a_started)

    assert abs(q_bisection - q_secant) < 1e-6

    p_at_root = solve_linear_system(q_bisection, a_started)[(score_a, score_b)]
    assert abs(p_at_root - 0.5) < 1e-8


@pytest.mark.parametrize("state", [(0, 0), (12, 12)])
def test_degenerate_states_have_no_unique_rate(state):
    # P(A wins) is exactly 0.5 for every q at 0-0 and at 12-12 (see test_tier1.py),
    # so "the rate that reaches 0.5" isn't well-posed - both solvers should say so.
    with pytest.raises(ValueError):
        required_round_win_rate_bisection(*state, "atk")
    with pytest.raises(ValueError):
        required_round_win_rate_secant(*state, "atk")


@pytest.mark.parametrize("state", [(13, 13), (14, 3), (-1, 5)])
def test_invalid_states_are_rejected(state):
    with pytest.raises(ValueError):
        required_round_win_rate_bisection(*state, "atk")


def test_bisection_is_robust_where_secant_can_diverge():
    # Secant isn't a bracketing method: for a lopsided enough score, it can
    # overshoot outside the valid (0, 1) probability range. Bisection, which
    # always stays inside its bracket, still gets the right answer.
    q = required_round_win_rate_bisection(11, 5, "def")
    assert 0.0 <= q <= 1.0
    assert abs(solve_linear_system(q, "def")[(11, 5)] - 0.5) < 1e-8

    with pytest.raises(ValueError, match="did not converge"):
        required_round_win_rate_secant(11, 5, "def")


def test_secant_example_cited_in_the_writeup():
    # A started on defense, so attacking at 11-5: secant escapes to about -2.08,
    # while bisection finds 0.138. (Started on attack, both agree at 0.862.)
    with pytest.raises(ValueError, match=r"q=-2\.08"):
        required_round_win_rate_secant(11, 5, "def")
    assert required_round_win_rate_bisection(11, 5, "def") == pytest.approx(0.138, abs=5e-4)
    assert required_round_win_rate_secant(11, 5, "atk") == pytest.approx(0.862, abs=5e-4)


@pytest.mark.parametrize(
    "solver", [required_round_win_rate_bisection, required_round_win_rate_secant]
)
def test_an_unreachable_target_is_explained_not_faked(solver):
    # From 6-12 (A defends the rest of regulation), even winning every round only
    # reaches a coin-flip overtime: 50% is a limit, never reached for q in (0, 1).
    with pytest.raises(ValueError, match="isn't reached by any round-win rate"):
        solver(6, 12, "atk", target=0.5)


def test_can_solve_for_a_non_default_target():
    q = required_round_win_rate_bisection(2, 0, "atk", target=0.75)
    assert abs(solve_linear_system(q, "atk")[(2, 0)] - 0.75) < 1e-8
