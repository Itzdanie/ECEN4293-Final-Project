"""Fit a straight line through log-log data (ordinary least squares)."""

from __future__ import annotations

import numpy as np


def log_log_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``log(y) = slope * log(x) + intercept`` by least squares.

    Returns ``(slope, intercept)``.
    """
    log_x = np.log(np.asarray(x, dtype=float))
    log_y = np.log(np.asarray(y, dtype=float))

    x_mean = log_x.mean()
    y_mean = log_y.mean()

    slope = np.sum((log_x - x_mean) * (log_y - y_mean)) / np.sum((log_x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)
