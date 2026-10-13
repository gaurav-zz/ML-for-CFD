"""Metrics reported for each aerodynamic target."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float | None]:
    """Report absolute and relative error for one target, in physical units.

    Relative error is what makes a drag number honest: v1's ``Cd`` MAE of
    0.0049 looks small until it is compared with a 1st-percentile ``Cd`` of
    0.006. Read ``median_relative_error`` in preference to the mean, because
    rows whose actual value is near zero — common for ``Cl`` near zero lift —
    dominate the mean. Rows whose actual value is exactly zero are excluded,
    since relative error is undefined there; both relative figures are ``None``
    when no row qualifies.

    ``None`` is used rather than NaN so results stay JSON-serializable with
    ``allow_nan=False``.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    usable = actual != 0
    relative = np.abs(actual[usable] - predicted[usable]) / np.abs(actual[usable])
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
        "mean_relative_error": float(relative.mean()) if relative.size else None,
        "median_relative_error": float(np.median(relative)) if relative.size else None,
    }
