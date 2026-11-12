"""Gauss-Seidel iteration for sparse linear systems already solved for x_i:

    x_i = constants[i] + sum_j weight_ij * x_j

(the same balance-equation form the dense Tier 1 system uses, just too
large here for a dense matrix - each row only has a handful of nonzero
weights, given as an adjacency list instead of a full matrix row.)
"""

from __future__ import annotations

import numpy as np


def gauss_seidel(
    neighbors: list[list[tuple[int, float]]],
    constants: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> tuple[np.ndarray, int]:
    """Solve for ``x`` by repeatedly sweeping through every state and
    updating it from the others' latest values (Gauss-Seidel uses each
    neighbor's value as soon as it's updated in the same sweep, which is
    what makes it converge faster than Jacobi iteration).

    Returns ``(x, n_iterations)``.
    """
    n = len(neighbors)
    x = np.zeros(n)

    for iteration in range(1, max_iter + 1):
        max_delta = 0.0
        for i in range(n):
            new_value = constants[i]
            for j, weight in neighbors[i]:
                new_value += weight * x[j]
            max_delta = max(max_delta, abs(new_value - x[i]))
            x[i] = new_value

        if max_delta < tol:
            return x, iteration

    return x, max_iter
