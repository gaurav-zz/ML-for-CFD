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

from src.geometry.descriptors import DESCRIPTOR_NAMES
from src.utils.metrics import regression_metrics

# XFOIL polars in this archive run to +/-24 degrees and about 17% of rows lie
# beyond this magnitude, where the flow is separated and the surrogate is
# fitting a different physical regime.
STALL_ALPHA = 12.0

PREDICTED_SUFFIX = "_predicted"
ERROR_SUFFIX = "_error"
METRIC_NAMES = (
    "mae", "rmse", "r2", "mean_relative_error", "median_relative_error",
)


def _metrics(frame: pd.DataFrame, target: str) -> dict:
    """Keep empty subsets explicit without passing them to sklearn metrics."""
    if frame.empty:
        return dict.fromkeys(METRIC_NAMES)
    return regression_metrics(frame[target], frame[target + PREDICTED_SUFFIX])


def _records(table: pd.DataFrame) -> list[dict]:
    """Pandas may coerce undefined singleton-group metrics from None to NaN."""
    return table.astype(object).where(pd.notna(table), None).to_dict("records")


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
        {label: key, "n": len(group), **_metrics(group, target)}
        for key, group in frame.groupby(by, observed=True, sort=True)
    ]
    return pd.DataFrame.from_records(records, columns=[label, "n", *METRIC_NAMES])


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


def summarize(
    frame: pd.DataFrame, target_names, threshold: float = STALL_ALPHA,
    *, training_frame: pd.DataFrame | None = None,
) -> dict:
    """Assemble every breakdown for a run report, as JSON-ready records.

    Reynolds groups use observed values, never an assumed reference Reynolds
    number. Optional domains use only training geometry descriptor ranges, not
    evaluated target errors, and never remove rows from the pooled headline.
    Row masks are positional boolean lists in evaluated-frame order.
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
        geometry_scores = per_geometry(frame, target)
        concentration = geometry_scores[["geometry_id", "n", "rmse"]].copy()
        concentration["sse"] = concentration["n"] * concentration["rmse"] ** 2
        total_sse = float(concentration["sse"].sum())
        concentration["sse_share"] = (
            concentration["sse"] / total_sse if total_sse > 0 else 0.0
        )
        top = concentration.sort_values(
            ["sse", "geometry_id"], ascending=[False, True],
        ).head(10)
        distributions = {}
        for metric in ("mae", "rmse", "median_relative_error"):
            values = geometry_scores[metric].dropna()
            distributions[metric] = {
                "n_defined": int(len(values)),
                **{
                    name: float(values.quantile(q)) if len(values) else None
                    for name, q in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95))
                },
            }
        report["targets"][target] = {
            "overall": _metrics(frame, target),
            "regime": _records(breakdown(frame, target, regime, "regime")),
            "reynolds": _records(breakdown(frame, target, "Re", "Re")),
            "worst_geometries": _records(worst_geometries(frame, target)),
            "error_concentration": {
                "methodology": "Per-geometry SSE = n * rmse^2; top 10 ranked by SSE. "
                "Shares divide by total evaluated SSE; all shares are zero when total SSE is zero.",
                "total_sse": total_sse,
                "top_10_sse_share": float(top["sse_share"].sum()),
                "top_geometries": _records(top),
            },
            "per_geometry_distribution": {
                "methodology": "Unweighted per-airfoil metric quantiles (linear interpolation); "
                "undefined relative errors are excluded and counted via n_defined.",
                "n_geometries": int(len(geometry_scores)),
                "metrics": distributions,
            },
        }

    missing = [name for name in ("Alpha", "Ncrit", "Re") if name not in frame]
    easy_mask = (
        (frame["Alpha"].abs() <= threshold) & (frame["Ncrit"] == 9)
        if not missing else pd.Series(False, index=frame.index)
    )
    easy = frame.loc[easy_mask]
    report["matched_easy_subset"] = {
        "methodology": "Fixed pre-stall |Alpha| <= threshold and Ncrit = 9; "
        "all observed Reynolds numbers scored separately, with no assumed Re = 1e6. "
        "Missing required columns yield an explicit empty subset; no target-error selection.",
        "criteria": {"max_abs_alpha_degrees": threshold, "Ncrit": 9},
        "missing_columns": missing,
        "row_mask": easy_mask.tolist(),
        "n_rows": int(len(easy)),
        "n_geometries": int(easy["geometry_id"].nunique()),
        "targets": {
            target: {
                "overall": _metrics(easy, target),
                "reynolds": [
                    {"Re": re, "n": int(len(group)),
                     "n_geometries": int(group["geometry_id"].nunique()),
                     **_metrics(group, target)}
                    for re, group in easy.groupby("Re", observed=True, sort=True)
                ],
            }
            for target in target_names
        },
    }

    if training_frame is not None:
        training = training_frame.drop_duplicates("geometry_id")
        descriptors = [name for name in DESCRIPTOR_NAMES if name in training and name in frame]
        ranges = {}
        for name in descriptors:
            values = training[name].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                ranges[name] = {
                    "min": float(values.min()), "max": float(values.max()),
                    "n_geometries": int(len(values)),
                }
        geometries = []
        for geometry in frame.drop_duplicates("geometry_id")[["geometry_id", *descriptors]].to_dict("records"):
            violations, unavailable = [], []
            for name, limits in ranges.items():
                value = float(geometry[name]) if pd.notna(geometry[name]) else None
                if value is None or not np.isfinite(value):
                    unavailable.append(name)
                elif value < limits["min"] or value > limits["max"]:
                    violations.append({"descriptor": name, "value": value,
                                       "min": limits["min"], "max": limits["max"]})
            geometries.append({
                "geometry_id": geometry["geometry_id"],
                "domain": "outside-training-range" if violations else "within-training-range",
                "violations": violations,
                "unavailable_descriptors": unavailable,
            })
        domains = []
        for domain in ("within-training-range", "outside-training-range"):
            names = [record["geometry_id"] for record in geometries if record["domain"] == domain]
            mask = frame["geometry_id"].isin(names)
            subset = frame.loc[mask]
            domains.append({
                "domain": domain, "geometry_ids": names, "row_mask": mask.tolist(),
                "n_rows": int(len(subset)), "n_geometries": int(len(names)),
                "targets": {target: _metrics(subset, target) for target in target_names},
            })
        report["training_domain"] = {
            "methodology": "Univariate scalar DESCRIPTOR_NAMES min/max ranges from training only, "
            "deduplicated by geometry_id (first row); profile features, labels and flow are excluded. "
            "Outside means at least one finite descriptor exceeds a range; within means no observed "
            "violation, not uncertainty or a certified domain. Missing values/ranges provide no evidence. "
            "All evaluated rows remain in pooled and domain metrics; no target-error thresholds.",
            "available": bool(ranges),
            "n_training_rows": int(len(training_frame)),
            "n_training_geometries": int(len(training)),
            "descriptor_ranges": ranges,
            "geometries": geometries,
            "domains": domains,
        }
    if {"Cl", "Cd"}.issubset(target_names):
        ratios = lift_to_drag(frame)
        report["lift_to_drag"] = {
            "n_rows": int(len(ratios)),
            "n_dropped_rows": int(len(frame) - len(ratios)),
            **_metrics(ratios, "lift_to_drag"),
        }
    return report
