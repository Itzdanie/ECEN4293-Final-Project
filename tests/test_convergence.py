import numpy as np
import pytest

from winprob.numerics.convergence import log_log_slope
from winprob.tier1.convergence import convergence_study, plot_convergence


def test_log_log_slope_recovers_a_known_exponent():
    ns = np.array([1e2, 1e3, 1e4, 1e5])
    errors = 3.0 * ns ** (-0.5)  # exact 1/sqrt(N) decay, no noise
    slope, intercept = log_log_slope(ns, errors)
    assert abs(slope - (-0.5)) < 1e-9
    assert abs(intercept - np.log(3.0)) < 1e-9


@pytest.mark.parametrize(
    "p_atk,a_started,score_a,score_b",
    [(0.55, "atk", 5, 8), (0.4, "def", 9, 6), (0.5, "atk", 7, 7)],
)
def test_mc_convergence_slope_is_close_to_one_over_sqrt_n(p_atk, a_started, score_a, score_b):
    # Requirement 6: averaged over 10 trials, N from 1e2 to 1e5, the
    # log-log slope of MC error vs N should be between -0.6 and -0.4.
    result = convergence_study(
        p_atk, a_started, score_a, score_b, n_trials=10, rng=np.random.default_rng(1)
    )
    assert -0.6 < result["slope"] < -0.4


def test_convergence_study_errors_shrink_overall():
    result = convergence_study(
        0.55, "atk", 5, 8, n_trials=10, rng=np.random.default_rng(2)
    )
    # noisy run to run, but the smallest-N error should dwarf the largest-N error
    assert result["mean_errors"][0] > result["mean_errors"][-1]


def test_plot_convergence_writes_a_file(tmp_path):
    result = convergence_study(
        0.55, "atk", 5, 8, n_trials=5, rng=np.random.default_rng(3)
    )
    out = tmp_path / "convergence.png"
    plot_convergence(result, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
