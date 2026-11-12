import numpy as np
import pytest

from winprob.numerics.logistic import fit_gradient_descent, fit_newton, log_likelihood, sigmoid


def _synthetic_rounds(n=20_000, true_beta=(0.1, 0.6, 0.25), seed=0):
    """20,000 synthetic rounds from a known beta: intercept, Delta-loadout
    (scaled), Delta-ults - the "full model with ults" CLAUDE.md asks
    Requirement 8 to recover."""
    rng = np.random.default_rng(seed)
    delta_loadout = rng.normal(0, 2, n)
    delta_ults = rng.integers(-3, 4, n).astype(float)
    X = np.column_stack([np.ones(n), delta_loadout, delta_ults])
    true_beta = np.asarray(true_beta)
    p = sigmoid(X @ true_beta)
    y = (rng.random(n) < p).astype(float)
    return X, y, true_beta


def test_sigmoid_is_bounded_and_monotonic():
    z = np.linspace(-10, 10, 21)
    s = sigmoid(z)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert np.all(np.diff(s) > 0)


def test_newton_converges_fast_and_recovers_known_beta():
    # Requirement 8: Newton converges in <= 25 iterations and recovers
    # each beta within +/- 0.15, on 20,000 synthetic rounds.
    X, y, true_beta = _synthetic_rounds()
    beta_hat, n_iter, history = fit_newton(X, y)

    assert n_iter <= 25
    assert np.all(np.abs(beta_hat - true_beta) < 0.15)
    # log-likelihood should improve (or hold) every iteration, never regress
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:]))


def test_gradient_descent_agrees_with_newton_but_needs_more_iterations():
    X, y, true_beta = _synthetic_rounds()
    beta_newton, n_iter_newton, _ = fit_newton(X, y)
    beta_gd, n_iter_gd, _ = fit_gradient_descent(X, y, learning_rate=0.5, max_iter=20_000)

    assert np.all(np.abs(beta_gd - true_beta) < 0.15)
    assert np.max(np.abs(beta_gd - beta_newton)) < 0.05
    assert n_iter_gd > n_iter_newton  # the whole point of the comparison


def test_log_likelihood_is_maximized_near_the_true_beta():
    X, y, true_beta = _synthetic_rounds()
    beta_hat, _, _ = fit_newton(X, y)
    # the MLE should score at least as well as the true generating beta
    assert log_likelihood(X, y, beta_hat) >= log_likelihood(X, y, true_beta) - 1e-6
