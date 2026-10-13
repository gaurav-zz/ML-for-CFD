"""Regime-aware and per-geometry error breakdowns.

One test R-squared per target hides everything a surrogate user needs. v1
reported ``Cd`` R-squared of 0.926 while its mean absolute error was roughly 80%
of a low-drag cruise section's actual drag, and it reported a single number
across pre- and post-stall rows that behave nothing alike. These breakdowns are
what make a generalization claim checkable rather than decorative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.metrics import regression_metrics

# XFOIL polars in this archive run to +/-24 degrees and about 17% of rows lie
# beyond this magnitude, where the flow is separated and the surrogate is
# fitting a different physical regime.
STALL_ALPHA = 12.0

PREDICTED_SUFFIX = "_predicted"
ERROR_SUFFIX = "_error"


def prediction_frame(rows: pd.DataFrame, predictions, target_names) -> pd.DataFrame:
    """Attach physical-unit predictions and signed errors to the evaluated rows.

    ``rows`` keeps its actual target columns, so every breakdown below scores
    ``target`` against ``target + "_predicted"`` in the original coefficient
    units.
    """
    target_names = list(target_names)
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("Provide unique, nonempty target names.")
    missing = [name for name in target_names if name not in rows.columns]
    if missing:
        raise ValueError(f"Evaluated rows are missing actual target columns: {missing}.")
    values = np.asarray(predictions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape != (len(rows), len(target_names)):
        raise ValueError(
            f"Predictions shaped {values.shape} do not match {len(rows)} rows "
            f"and {len(target_names)} targets."
        )
    frame = rows.copy()
    for index, target in enumerate(target_names):
        frame[target + PREDICTED_SUFFIX] = values[:, index]
        frame[target + ERROR_SUFFIX] = values[:, index] - frame[target].to_numpy(dtype=float)
    return frame


def breakdown(frame: pd.DataFrame, target: str, by, label: str | None = None) -> pd.DataFrame:
    """Score one target within each group, in physical units.

    ``by`` is anything ``DataFrame.groupby`` accepts: a column name, or a Series
    of labels such as the one :func:`stall_regime` returns.
    """
    predicted = target + PREDICTED_SUFFIX
    if target not in frame.columns or predicted not in frame.columns:
        raise ValueError(f"Frame must hold both {target!r} and {predicted!r}; call prediction_frame first.")
    if label is None:
        label = by if isinstance(by, str) else getattr(by, "name", None) or "group"
    records = [
        {label: key, "n": len(group), **regression_metrics(group[target], group[predicted])}
        for key, group in frame.groupby(by, observed=True, sort=True)
    ]
    return pd.DataFrame.from_records(records)


def stall_regime(frame: pd.DataFrame, threshold: float = STALL_ALPHA) -> pd.Series:
    """Label each row pre- or post-stall by angle-of-attack magnitude."""
    if "Alpha" not in frame.columns:
        raise ValueError("Frame must contain Alpha to split by flow regime.")
    if not threshold > 0:
        raise ValueError("The stall threshold must be a positive angle in degrees.")
    labels = np.where(frame["Alpha"].abs() > threshold, "post-stall", "pre-stall")
    return pd.Series(labels, index=frame.index, name="regime")


def per_geometry(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Score one target per airfoil, the unit the unseen-geometry claim is about."""
    return breakdown(frame, target, "geometry_id")


def worst_geometries(
    frame: pd.DataFrame, target: str, count: int = 10, column: str = "rmse",
) -> pd.DataFrame:
    """Rank the airfoils the surrogate handles worst, so failures are nameable."""
    if count < 1:
        raise ValueError("Request at least one geometry.")
    scores = per_geometry(frame, target)
    if column not in scores.columns:
        raise ValueError(f"Unknown ranking column {column!r}; choose from {sorted(scores.columns)}.")
    return scores.sort_values([column, "geometry_id"], ascending=[False, True]).head(count)


def lift_to_drag(frame: pd.DataFrame, lift: str = "Cl", drag: str = "Cd") -> pd.DataFrame:
    """Add actual and predicted ``L/D`` plus its error, dropping unusable rows.

    ``L/D`` is the quantity a designer actually compares sections on, and it
    amplifies drag error, so it is the strictest honest check available here.
    Rows whose actual or predicted drag is non-positive cannot form a ratio and
    are dropped rather than producing infinities.
    """
    predicted_lift, predicted_drag = lift + PREDICTED_SUFFIX, drag + PREDICTED_SUFFIX
    required = (lift, drag, predicted_lift, predicted_drag)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing columns needed for L/D: {missing}.")
    usable = (frame[drag] > 0) & (frame[predicted_drag] > 0)
    ratios = frame.loc[usable].copy()
    ratios["lift_to_drag"] = ratios[lift] / ratios[drag]
    ratios["lift_to_drag" + PREDICTED_SUFFIX] = ratios[predicted_lift] / ratios[predicted_drag]
    ratios["lift_to_drag" + ERROR_SUFFIX] = (
        ratios["lift_to_drag" + PREDICTED_SUFFIX] - ratios["lift_to_drag"]
    )
    return ratios


def summarize(frame: pd.DataFrame, target_names, threshold: float = STALL_ALPHA) -> dict:
    """Assemble every breakdown for a run report, as JSON-ready records.

    ``reynolds`` groups on the eight discrete Reynolds numbers in the archive,
    and ``lift_to_drag`` is reported once rather than per target.
    """
    target_names = list(target_names)
    regime = stall_regime(frame, threshold)
    report = {
        "stall_threshold_degrees": threshold,
        "n_rows": int(len(frame)),
        "n_geometries": int(frame["geometry_id"].nunique()) if "geometry_id" in frame else None,
        "targets": {},
    }
    for target in target_names:
        report["targets"][target] = {
            "overall": regression_metrics(frame[target], frame[target + PREDICTED_SUFFIX]),
            "regime": breakdown(frame, target, regime, "regime").to_dict("records"),
            "reynolds": breakdown(frame, target, "Re", "Re").to_dict("records"),
            "worst_geometries": worst_geometries(frame, target).to_dict("records"),
        }
    if {"Cl", "Cd"}.issubset(target_names):
        ratios = lift_to_drag(frame)
        report["lift_to_drag"] = {
            "n_rows": int(len(ratios)),
            "n_dropped_rows": int(len(frame) - len(ratios)),
            **regression_metrics(ratios["lift_to_drag"], ratios["lift_to_drag" + PREDICTED_SUFFIX]),
        }
    return report
