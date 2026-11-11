"""Bisection and secant root-finding, implemented from scratch."""

from __future__ import annotations

from typing import Callable


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Find a root of ``f`` in ``[a, b]`` by bisection.

    Requires ``f(a)`` and ``f(b)`` to have opposite signs (or be zero).
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(
            f"bisection needs a sign change across [{a}, {b}]: f(a)={fa}, f(b)={fb}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if fm == 0.0 or 0.5 * (b - a) < tol:
            return mid
        if fa * fm < 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    return 0.5 * (a + b)


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Find a root of ``f`` near ``x0``/``x1`` by the secant method."""
    f0, f1 = f(x0), f(x1)

    for _ in range(max_iter):
        if abs(f1) < tol:
            return x1
        denom = f1 - f0
        if denom == 0.0:
            raise ZeroDivisionError(
                f"secant method stalled: f({x0})==f({x1})=={f1}, can't divide"
            )
        x2 = x1 - f1 * (x1 - x0) / denom
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)

    return x1
