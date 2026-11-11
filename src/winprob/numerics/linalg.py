"""Dense LU decomposition with partial pivoting, implemented from scratch.

SciPy/NumPy's own solvers are for cross-checking in tests only - this is the solver the rest of the package actually calls.
"""

from __future__ import annotations

import numpy as np


def lu_decompose(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor ``a`` as ``P @ a == L @ U`` via Gaussian elimination with
    partial pivoting. Returns ``(l, u, p)``."""
    n = a.shape[0]
    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError(f"expected a square matrix, got shape {a.shape}")

    u = a.astype(float).copy()
    l = np.eye(n)
    p = np.eye(n)

    for k in range(n - 1):
        pivot = k + int(np.argmax(np.abs(u[k:, k])))
        if u[pivot, k] == 0.0:
            raise np.linalg.LinAlgError("matrix is singular")
        if pivot != k:
            u[[k, pivot], :] = u[[pivot, k], :]
            p[[k, pivot], :] = p[[pivot, k], :]
            l[[k, pivot], :k] = l[[pivot, k], :k]

        for i in range(k + 1, n):
            factor = u[i, k] / u[k, k]
            l[i, k] = factor
            u[i, k:] -= factor * u[k, k:]

    return l, u, p


def forward_substitute(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``l @ y == b`` for lower-triangular ``l`` with a unit diagonal."""
    n = l.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - l[i, :i] @ y[:i]
    return y


def back_substitute(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve ``u @ x == y`` for upper-triangular ``u``."""
    n = u.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - u[i, i + 1 :] @ x[i + 1 :]) / u[i, i]
    return x


def lu_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a @ x == b`` via hand-rolled LU decomposition with partial pivoting."""
    l, u, p = lu_decompose(a)
    y = forward_substitute(l, p @ b)
    return back_substitute(u, y)
