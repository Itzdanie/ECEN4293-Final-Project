"""Inverse Tier 1 questions: what attacker round-win rate q would give team
A exactly ``target`` win probability from a given score, solved both by
bisection and by the secant method.
"""

from __future__ import annotations

from winprob.numerics.root_finding import bisection, secant
from winprob.tier1.exact import solve_linear_system
from winprob.tier1.states import WIN_SCORE, Side

_EPS = 1e-6  # keep q inside the open interval (0, 1): p=0 or 1 is a degenerate round


def _validate_state(score_a: int, score_b: int) -> None:
    if not (0 <= score_a <= WIN_SCORE and 0 <= score_b <= WIN_SCORE):
        raise ValueError(
            f"({score_a}, {score_b}) is out of range for Tier 1: scores must be 0-{WIN_SCORE}"
        )
    if score_a == WIN_SCORE and score_b == WIN_SCORE:
        raise ValueError(
            f"({score_a}, {score_b}) is a real overtime tie (reachable when the first OT "
            f"pair splits), but solve_linear_system's table stops at the "
            f"{WIN_SCORE - 1}-{WIN_SCORE - 1} overtime entry and doesn't cover deeper ties"
        )


def _objective(score_a: int, score_b: int, a_started: Side, target: float):
    _validate_state(score_a, score_b)

    def g(q: float) -> float:
        return solve_linear_system(q, a_started)[(score_a, score_b)] - target

    lo, hi = g(_EPS), g(1.0 - _EPS)
    if abs(lo - hi) < 1e-9:
        raise ValueError(
            f"({score_a}, {score_b}) has the same win probability "
            f"({lo + target:.6f}) for every round-win rate in (0, 1) - "
            f"there's no unique rate that reaches a target of {target}"
        )
    return g


def required_round_win_rate_bisection(
    score_a: int,
    score_b: int,
    a_started: Side,
    target: float = 0.5,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """The attacker round-win rate q that gives team A exactly ``target``
    win probability from (score_a, score_b), found by bisection."""
    g = _objective(score_a, score_b, a_started, target)
    return bisection(g, _EPS, 1.0 - _EPS, tol=tol, max_iter=max_iter)


def required_round_win_rate_secant(
    score_a: int,
    score_b: int,
    a_started: Side,
    target: float = 0.5,
    x0: float = _EPS,
    x1: float = 1.0 - _EPS,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Same question as :func:`required_round_win_rate_bisection`, solved
    with the secant method instead.

    Unlike bisection, secant isn't a bracketing method: nothing stops it
    from overshooting outside (0, 1) when the two starting guesses don't
    straddle the root well (seen in practice for lopsided scores with
    narrow starting points). We default to the widest possible starting
    guesses to avoid that, and still check the result before returning it.
    """
    g = _objective(score_a, score_b, a_started, target)
    root = secant(g, x0, x1, tol=tol, max_iter=max_iter)

    if not (0.0 <= root <= 1.0) or abs(g(root)) > 1e-6:
        raise ValueError(
            f"secant did not converge to a valid round-win rate for ({score_a}, {score_b}): "
            f"got q={root}, outside (or at the edge of) [0, 1]. Try "
            "required_round_win_rate_bisection instead, which is guaranteed to stay in range."
        )
    return root
