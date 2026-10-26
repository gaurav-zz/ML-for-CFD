"""Fit-free, predeclared aggregation of three physical-coefficient predictors.

Member disagreement is descriptive sample spread, not calibrated uncertainty,
a confidence interval, a physics guarantee, or a certified abstention rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44)
TARGETS = ("Cl", "Cd", "Cm")
METHOD = "mean_Cl_Cm_geometric_mean_Cd"


def _real_array(value, shape):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value)):
        raise ValueError(f"Expected a real numeric ndarray with shape {shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = value.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("All coefficients/spreads must be finite; no rows may be dropped.")
    return array


def aggregate_predictions(member_predictions):
    """Aggregate physical Cl/Cd/Cm arrays in SEEDS=(42,43,44) member order.

    Cl/Cm use arithmetic means, Cd exp(mean(log(Cd))). Disagreement columns
    are sample standard deviations (ddof=1) in Cl/log(Cd)/Cm coordinates.
    Inputs must already be decoded from any training transforms/scalers.
    """
    if (not isinstance(member_predictions, np.ndarray) or member_predictions.ndim != 3
            or member_predictions.shape[0] != 3 or member_predictions.shape[2] != 3
            or member_predictions.shape[1] == 0):
        raise ValueError("Exactly three members and nonempty rows are required: shape (3, n, 3).")
    members = _real_array(member_predictions, member_predictions.shape)
    if (members[:, :, 1] <= 0).any():
        raise ValueError("Physical Cd must be strictly positive for every member and row.")
    coordinates = members.copy()
    coordinates[:, :, 1] = np.log(coordinates[:, :, 1])
    # Normalize before summing/squaring to avoid overflow on finite inputs.
    scale = np.max(np.abs(coordinates), axis=0)
    scale = np.where(scale == 0, 1., scale)
    normalized = coordinates / scale
    center = np.mean(normalized, axis=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prediction = center * scale
        disagreement = np.sqrt(np.sum((normalized - center) ** 2, axis=0) / 2) * scale
        prediction[:, 1] = np.exp(prediction[:, 1])
    # Rounding at the limits of float64 must not leave the members' Cd range.
    prediction[:, 1] = np.clip(prediction[:, 1], members[:, :, 1].min(axis=0), members[:, :, 1].max(axis=0))
    if (not np.isfinite(prediction).all() or not np.isfinite(disagreement).all()
            or (prediction[:, 1] <= 0).any()):
        raise ValueError("Aggregate prediction and disagreement must remain finite with positive Cd.")
    return {"prediction": prediction, "disagreement": disagreement,
            "method": METHOD, "member_seeds": list(SEEDS)}


def disagreement_report(frame, ensemble_result):
    """Score all aligned rows and three prediction-spread-only diagnostic bins.

    Array rows are positional relative to frame; no row is excluded. The fixed
    bins are low <= p50, mid (p50,p90], high > p90 of predicted log-Cd spread.
    Tied quantiles can produce empty bins, whose metrics are explicitly None.
    Spearman correlates absolute log-Cd error with log-Cd sample spread, using
    average ranks without SciPy; either constant variable gives None.
    """
    if (not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique
            or not frame.index.is_unique or not set((*TARGETS, "geometry_id")).issubset(frame.columns)):
        raise ValueError("A nonempty, uniquely indexed frame with geometry IDs and all targets is required.")
    if not all(isinstance(value, str) and value.strip() for value in frame.geometry_id):
        raise ValueError("geometry_id must contain nonmissing, nonempty strings.")
    actual = _real_array(frame[list(TARGETS)].to_numpy(), (len(frame), 3))
    if (actual[:, 1] <= 0).any():
        raise ValueError("Observed Cd must be strictly positive for log-error diagnostics.")
    if (not isinstance(ensemble_result, dict) or ensemble_result.get("method") != METHOD
            or ensemble_result.get("member_seeds") != list(SEEDS)):
        raise ValueError("Diagnostics require the predeclared three-member aggregation result.")
    predicted = _real_array(ensemble_result.get("prediction"), actual.shape)
    spread = _real_array(ensemble_result.get("disagreement"), actual.shape)
    if (predicted[:, 1] <= 0).any() or (spread < 0).any():
        raise ValueError("Predicted Cd must be positive and disagreement nonnegative.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        error = np.abs(predicted[:, 1] - actual[:, 1])
        relative = error / actual[:, 1]
        log_error = np.abs(np.log(predicted[:, 1]) - np.log(actual[:, 1]))
    if not all(np.isfinite(value).all() for value in (error, relative, log_error)):
        raise ValueError("All row errors must remain finite; no rows may be dropped.")
    cd_spread = spread[:, 1]
    correlation = None
    if np.ptp(cd_spread) != 0 and np.ptp(log_error) != 0:
        x = pd.Series(cd_spread).rank(method="average").to_numpy()
        y = pd.Series(log_error).rank(method="average").to_numpy()
        x = x - x.mean()
        y = y - y.mean()
        correlation = float(np.clip(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)), -1., 1.))
    p50, p90 = np.quantile(cd_spread, [.5, .9], method="linear")
    bins = []
    for name, mask in (("low", cd_spread <= p50),
                       ("mid", (cd_spread > p50) & (cd_spread <= p90)),
                       ("high", cd_spread > p90)):
        maximum = float(error[mask].max()) if mask.any() else 0.
        rmse = float(np.sqrt(np.mean((error[mask] / maximum) ** 2)) * maximum) if maximum else 0.
        bins.append({"name": name, "n_rows": int(mask.sum()),
                     "n_geometries": int(frame.loc[mask, "geometry_id"].nunique()),
                     "median_relative_error_Cd": float(np.quantile(relative[mask], .5, method="linear")) if mask.any() else None,
                     "Cd_rmse": rmse if mask.any() else None,
                     "Cd_log_mae": float(np.mean(log_error[mask])) if mask.any() else None})
    return {
        "schema_version": 1, "method": METHOD, "member_seeds": list(SEEDS),
        "n_rows": len(frame), "n_geometries": int(frame.geometry_id.nunique()),
        "definitions": {"prediction_units": "physical_coefficients", "relative_error_units": "fraction",
                        "disagreement_units": ["Cl", "log(Cd)", "Cm"], "disagreement_ddof": 1,
                        "log_error": "abs(log(predicted Cd)-log(observed Cd))",
                        "spearman": "average-rank correlation; either constant variable gives null",
                        "bins": "low <= p50; mid (p50,p90]; high > p90; linear quantiles of predicted log-Cd spread only"},
        "calibrated_uncertainty": False, "confidence_interval": False,
        "physics_guarantee": False, "certified_abstention": False, "row_filtering": False,
        "spearman_abs_log_Cd_error_vs_log_Cd_std": correlation,
        "spread_thresholds": {"p50": float(p50), "p90": float(p90), "source": "prediction_only"},
        "bins": bins,
    }
