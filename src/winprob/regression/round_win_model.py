"""Fit ``P(attack wins) = sigma(b0 + b1*delta_loadout + b_half + b_map)`` by
maximum likelihood (Newton/IRLS), from the training table in
:mod:`winprob.regression.dataset`.

No ults term: the round-level data doesn't say who had an ult ready each
round (CLAUDE.md's "Open decision: ultimates" - Daniel's call was to fit
without it here and set Tier 3's ult effect from the ``lift_pp`` data
instead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from winprob.numerics.logistic import fit_newton, sigmoid


def _design_matrix(
    table: pd.DataFrame, map_levels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """``[intercept, delta_loadout, half2, map dummies...]``. The first
    entry in ``map_levels`` is the reference level, dropped from the dummies.
    """
    maps = pd.Categorical(table["map"], categories=map_levels)
    map_dummies = pd.get_dummies(maps, prefix="map", drop_first=True).astype(float)

    X = np.column_stack(
        [
            np.ones(len(table)),
            table["delta_loadout"].to_numpy(dtype=float),
            table["half2"].to_numpy(dtype=float),
            map_dummies.to_numpy(dtype=float),
        ]
    )
    feature_names = ["intercept", "delta_loadout", "half2", *map_dummies.columns]
    return X, feature_names


def fit_round_win_model(table: pd.DataFrame) -> dict:
    """Fit the round-win model by Newton's method on ``table`` (as built by
    :func:`winprob.regression.dataset.build_training_table`)."""
    map_levels = sorted(table["map"].unique())
    X, feature_names = _design_matrix(table, map_levels)
    y = table["atk_won"].to_numpy(dtype=float)

    beta, n_iter, history = fit_newton(X, y)
    return {
        "beta": beta,
        "feature_names": feature_names,
        "map_levels": map_levels,
        "n_iter": n_iter,
        "log_likelihood": history[-1],
        "n_rounds": len(table),
    }


def predict_round_win_probability(
    model: dict, delta_loadout: float, half2: bool, map_name: str
) -> float:
    """P(attack wins) for one hypothetical round, from a fitted model."""
    map_levels = model["map_levels"]
    x = [1.0, delta_loadout, float(half2)]
    x.extend(1.0 if map_name == level else 0.0 for level in map_levels[1:])
    return float(sigmoid(np.dot(model["beta"], x)))
