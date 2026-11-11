"""Monte Carlo convergence study: does Tier 1's MC error shrink like 1/sqrt(N)?

Runs the simulator at a range of sample sizes, several independent trials
each, and fits the log-log slope of mean error vs N against the exact
answer (Requirement 6 expects that slope between -0.6 and -0.4).
"""

from __future__ import annotations

import numpy as np

from winprob.numerics.convergence import log_log_slope
from winprob.tier1.exact import solve_linear_system
from winprob.tier1.monte_carlo import simulate_win_probability
from winprob.tier1.states import Side

DEFAULT_NS = np.logspace(2, 5, 10).astype(int)  # 1e2 .. 1e5, 10 points


def convergence_study(
    p_atk: float,
    a_started: Side,
    score_a: int,
    score_b: int,
    ns: np.ndarray = DEFAULT_NS,
    n_trials: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run the Monte Carlo simulator at each sample size in ``ns``,
    ``n_trials`` independent times per size, and measure the mean absolute
    error against the exact answer from :func:`solve_linear_system`.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    exact = solve_linear_system(p_atk, a_started)[(score_a, score_b)]

    mean_errors = np.empty(len(ns))
    for i, n in enumerate(ns):
        errors = np.empty(n_trials)
        for t in range(n_trials):
            p_hat, _, _ = simulate_win_probability(
                p_atk, a_started, score_a, score_b, int(n), rng
            )
            errors[t] = abs(p_hat - exact)
        mean_errors[i] = errors.mean()

    slope, intercept = log_log_slope(ns, mean_errors)
    return {
        "ns": np.asarray(ns),
        "mean_errors": mean_errors,
        "slope": slope,
        "intercept": intercept,
        "exact": exact,
    }


def plot_convergence(result: dict, path: str) -> None:
    """Save a log-log plot of mean error vs N, with a 1/sqrt(N) reference
    line and the fitted slope in the title."""
    import matplotlib.pyplot as plt

    ns = result["ns"]
    errors = result["mean_errors"]
    slope = result["slope"]
    reference = errors[0] * np.sqrt(ns[0] / ns)

    fig, ax = plt.subplots()
    ax.loglog(ns, errors, "o-", label="Monte Carlo mean error")
    ax.loglog(ns, reference, "--", label=r"$1/\sqrt{N}$ reference")
    ax.set_xlabel("N (trials)")
    ax.set_ylabel("mean |MC estimate - exact|")
    ax.set_title(f"Tier 1 Monte Carlo convergence (fitted slope = {slope:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
