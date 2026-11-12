import numpy as np
import pytest

from winprob.tier2.exact import solve_tier2, win_probability_from_start
from winprob.tier2.monte_carlo import _draw_next_buy_types, simulate_win_probability
from winprob.tier2.states import all_states, boundary_value, equation_terms

# 10+ representative states for Requirement 5's Tier 2 half, spanning
# different scores, buy-type pairs, and loss streaks.
SAMPLE_STATES = [
    (3, 8, "Full buy", "Eco", 0, 2),
    (8, 3, "Eco", "Full buy", 2, 0),
    (5, 5, "Semi-buy", "Semi-eco", 1, 1),
    (10, 10, "Full buy", "Full buy", 0, 0),
    (1, 0, "Semi-buy", "Eco", 0, 1),
    (11, 5, "Full buy", "Semi-buy", 0, 3),
    (6, 9, "Semi-eco", "Full buy", 3, 0),
    (9, 6, "Full buy", "Semi-eco", 0, 1),
    (2, 2, "Eco", "Eco", 1, 1),
    (7, 7, "Semi-buy", "Full buy", 2, 0),
]


@pytest.fixture(scope="module")
def tier2_table():
    table, n_iter = solve_tier2()
    return table, n_iter


# --- state/transition logic -------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        (5, 5, "Full buy", "Eco", 0, 2),
        (0, 1, "Eco", "Full buy", 1, 0),
        (11, 11, "Semi-buy", "Semi-eco", 3, 3),
    ],
)
def test_equation_terms_sum_to_one(state):
    terms = equation_terms(state)
    assert abs(sum(weight for _, weight in terms) - 1.0) < 1e-9


def test_pistol_round_state_has_fewer_terms_but_still_sums_to_one():
    # total == 12: the upcoming round (13) is a pistol
    terms = equation_terms((7, 5, "Semi-buy", "Full buy", 1, 3))
    assert abs(sum(weight for _, weight in terms) - 1.0) < 1e-9


def test_boundary_value_terminal_states():
    assert boundary_value(13, 5) == 1.0
    assert boundary_value(5, 13) == 0.0


def test_boundary_value_overtime_entry_is_a_coin_flip():
    # overtime grants both teams fixed, equal credits - buy type is moot there
    assert boundary_value(12, 12) == 0.5


def test_boundary_value_returns_none_for_genuine_unknowns():
    assert boundary_value(5, 5) is None


def test_pistol_round_resets_loss_streak_regardless_of_prior_streak():
    # Round 13 is a fresh pistol: whatever streak a team carried in doesn't
    # matter, winning/losing it alone decides the next streak (0 or 1).
    rng = np.random.default_rng(0)
    won = np.array([False, False, True])
    prior_streak = np.array([2, 0, 2])
    pistol = np.array([True, True, True])
    _, streak = _draw_next_buy_types(rng, won, prior_streak, pistol)
    assert list(streak) == [1, 1, 0]


# --- exact solver ------------------------------------------------------------


def test_solve_tier2_converges_within_a_reasonable_number_of_iterations(tier2_table):
    _, n_iter = tier2_table
    assert n_iter < 500  # didn't hit the iteration cap


def test_solve_tier2_probabilities_are_bounded(tier2_table):
    table, _ = tier2_table
    for state in SAMPLE_STATES:
        assert 0.0 <= table[state] <= 1.0


def test_win_probability_from_start_is_close_to_half(tier2_table):
    # Tier 2's round-win table has no side/attacker concept (it's pooled
    # over both sides), so the very start of the map should be ~symmetric.
    table, _ = tier2_table
    assert abs(win_probability_from_start(table) - 0.5) < 0.01


# --- Monte Carlo cross-check (Requirement 5, Tier 2 half) -------------------


@pytest.mark.parametrize("state", SAMPLE_STATES)
def test_monte_carlo_matches_exact_within_two_points(tier2_table, state):
    table, _ = tier2_table
    exact = table[state]

    rng = np.random.default_rng(0)
    p_hat, ci_low, ci_high = simulate_win_probability(*state, 100_000, rng)

    assert abs(p_hat - exact) < 0.02
    assert ci_low <= p_hat <= ci_high
