"""Logistic regression by maximum likelihood: Newton's method (IRLS) and
gradient descent, both hand-rolled. ``X`` is a design matrix (first column
is usually an intercept of 1s); ``y`` is a 0/1 outcome vector.
"""

from __future__ import annotations

import numpy as np

from winprob.numerics.linalg import lu_solve


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    z = X @ beta
    return float(np.sum(y * z - np.logaddexp(0.0, z)))  # log(1+exp(z)), computed stably


def fit_newton(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 25,
) -> tuple[np.ndarray, int, list[float]]:
    """Maximum-likelihood logistic fit via Newton's method / IRLS.

    Returns ``(beta, n_iterations, log_likelihood_history)``. Each Newton
    step solves ``(X^T W X) delta = X^T (y - p_hat)`` with the hand-rolled
    LU solver, where ``W = diag(p_hat * (1 - p_hat))``.
    """
    n, p = X.shape
    beta = np.zeros(p)
    history = [log_likelihood(X, y, beta)]

    for iteration in range(1, max_iter + 1):
        p_hat = sigmoid(X @ beta)
        w = p_hat * (1.0 - p_hat)
        gradient = X.T @ (y - p_hat)
        hessian_neg = (X * w[:, None]).T @ X  # X^T W X

        delta = lu_solve(hessian_neg, gradient)
        beta = beta + delta
        history.append(log_likelihood(X, y, beta))

        if np.max(np.abs(delta)) < tol:
            return beta, iteration, history

    return beta, max_iter, history


def fit_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 20_000,
) -> tuple[np.ndarray, int, list[float]]:
    """The same maximum-likelihood fit by plain gradient ascent on the
    log-likelihood, as a slower comparison against :func:`fit_newton`."""
    n, p = X.shape
    beta = np.zeros(p)
    history = [log_likelihood(X, y, beta)]

    for iteration in range(1, max_iter + 1):
        p_hat = sigmoid(X @ beta)
        step = learning_rate * (X.T @ (y - p_hat)) / n
        beta = beta + step
        history.append(log_likelihood(X, y, beta))

        if np.max(np.abs(step)) < tol:
            return beta, iteration, history

    return beta, max_iter, history


def fit_newton_penalized(
    X: np.ndarray,
    y: np.ndarray,
    penalty: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[np.ndarray, int, list[float]]:
    """Ridge-penalized logistic fit by Newton's method.

    Maximizes ``loglik(beta) - 0.5 * sum(penalty_j * beta_j**2)``: each
    coefficient ``j`` is pulled toward 0 with strength ``penalty[j]`` (0 =
    unpenalized, so ``penalty=0`` reproduces :func:`fit_newton`). The Newton
    step solves ``(X^T W X + diag(penalty)) delta = X^T (y - p_hat) - penalty * beta``.
    The penalty also makes the system invertible when columns are redundant
    (e.g. a role column equal to the sum of its sub-role columns).

    Returns ``(beta, n_iterations, penalized_objective_history)``.
    """
    n, p = X.shape
    penalty = np.asarray(penalty, dtype=float)
    beta = np.zeros(p)

    def objective(b: np.ndarray) -> float:
        return log_likelihood(X, y, b) - 0.5 * float(np.sum(penalty * b * b))

    history = [objective(beta)]
    for iteration in range(1, max_iter + 1):
        p_hat = sigmoid(X @ beta)
        w = p_hat * (1.0 - p_hat)
        gradient = X.T @ (y - p_hat) - penalty * beta
        hessian_neg = (X * w[:, None]).T @ X + np.diag(penalty)

        delta = lu_solve(hessian_neg, gradient)
        beta = beta + delta
        history.append(objective(beta))

        if np.max(np.abs(delta)) < tol:
            return beta, iteration, history

    return beta, max_iter, history
