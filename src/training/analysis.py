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

# This fixed angle proxy is not a measurement of physical stall or separation.
STALL_ALPHA = 12.0

PREDICTED_SUFFIX = "_predicted"
ERROR_SUFFIX = "_error"
METRIC_NAMES = (
    "mae", "rmse", "r2", "mean_relative_error", "median_relative_error",
)


def _metrics(frame: pd.DataFrame, target: str) -> dict:
    """Score finite pairs only; undefined metrics stay strict-JSON nulls."""
    columns = [target, target + PREDICTED_SUFFIX]
    if any(name not in frame for name in columns):
        return dict.fromkeys(METRIC_NAMES)
    valid = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    if not valid.any():
        return dict.fromkeys(METRIC_NAMES)
    metrics = regression_metrics(frame.loc[valid, target], frame.loc[valid, columns[1]])
    return {name: value if value is not None and np.isfinite(value) else None
            for name, value in metrics.items()}


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
    """Legacy pre/post-stall labels are angle proxies, not measured flow states."""
    if "Alpha" not in frame.columns:
        raise ValueError("Frame must contain Alpha to split by flow regime.")
    if not np.isfinite(threshold) or not threshold > 0:
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


def _error_concentration(frame: pd.DataFrame, target: str) -> dict:
    """Rank finite-pair squared errors by geometry without averaging geometries."""
    columns = [target, target + PREDICTED_SUFFIX]
    valid = (np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
             if all(name in frame for name in columns) else np.zeros(len(frame), dtype=bool))
    usable = frame.loc[valid]
    errors = (usable[["geometry_id"]].copy() if "geometry_id" in usable
              else pd.DataFrame(columns=["geometry_id"]))
    errors["sse"] = (usable[columns[0]] - usable[columns[1]]) ** 2 if len(usable) else 0.0
    concentration = errors.groupby("geometry_id", observed=True, sort=True).agg(
        n=("sse", "size"), sse=("sse", "sum"),
    ).reset_index()
    concentration["rmse"] = np.sqrt(concentration["sse"] / concentration["n"])
    total_sse = float(concentration["sse"].sum())
    concentration["sse_share"] = concentration["sse"] / total_sse if total_sse > 0 else 0.0
    top = concentration.sort_values(["sse", "geometry_id"], ascending=[False, True]).head(10)
    return {
        "methodology": "Per-geometry finite-pair SSE; top 10 ranked by SSE. Shares divide by "
        "this subset's SSE; all shares are zero when total SSE is zero.",
        "total_sse": total_sse,
        "top_10_sse_share": float(top["sse_share"].sum()),
        "top_geometries": _records(top),
    }


def _diagnostic_subset(frame: pd.DataFrame, target: str, poststall_sse: float | None = None) -> dict:
    concentration = _error_concentration(frame, target)
    columns = [target, target + PREDICTED_SUFFIX]
    n_scored = (int(np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1).sum())
                if all(name in frame for name in columns) else 0)
    result = {
        "n_rows": int(len(frame)),
        "n_geometries": int(frame["geometry_id"].nunique()) if "geometry_id" in frame else 0,
        "n_scored_rows": n_scored, "n_unscored_rows": int(len(frame) - n_scored),
        "metrics": _metrics(frame, target), "sse": concentration["total_sse"],
        "error_concentration": concentration,
    }
    if poststall_sse is not None:
        result["sse_share_of_poststall"] = result["sse"] / poststall_sse if poststall_sse > 0 else 0.0
    return result


def poststall_alpha(frame: pd.DataFrame, target: str) -> dict:
    """Fixed angle bands test whether errors cluster near 12 degrees or beyond 20."""
    magnitude = frame["Alpha"].abs()
    post = frame.loc[np.isfinite(magnitude) & (magnitude > STALL_ALPHA)]
    overall = _diagnostic_subset(post, target)
    bands = []
    for label, lower, upper in (("(12,16]", 12, 16), ("(16,20]", 16, 20), (">20", 20, None)):
        mask = np.isfinite(magnitude) & (magnitude > lower)
        if upper is not None:
            mask &= magnitude <= upper
        bands.append({
            "band": label, "lower_exclusive_degrees": lower, "upper_inclusive_degrees": upper,
            **_diagnostic_subset(frame.loc[mask], target, overall["sse"]),
        })
    return {
        "methodology": "Fixed |Alpha| bands, independent of summarize's configurable threshold. "
        "Post-stall denotes only |Alpha| > 12, not measured stall or fully separated flow. "
        "SSE shares use all finite-pair |Alpha| > 12 errors; R2 is subset-specific. "
        "Relative errors are fractions, exclude exactly-zero actuals, and include both mean and median.",
        "threshold_degrees": STALL_ALPHA, "overall": overall, "bands": bands,
        "signed_alpha": [
            {"sign": sign, **_diagnostic_subset(group, target, overall["sse"])}
            for sign, group in (("negative", post.loc[post["Alpha"] < 0]),
                                ("positive", post.loc[post["Alpha"] > 0]))
        ],
        "reynolds": [
            {"Re": re, **_diagnostic_subset(group, target, overall["sse"])}
            for re, group in post.groupby("Re", observed=True, sort=True)
        ] if "Re" in post else [],
    }


def observed_cl_peak(frame: pd.DataFrame, *, min_angles: int = 5) -> dict:
    """Locate signed-Cl maxima on observed half-polars, never infer physical stall.

    Boundary/plateau maxima are censored, sparse curves are excluded, and Cd is
    scored separately around interior and censored peaks. Alpha=0 is excluded
    rather than duplicated into both signs. Positions, not index labels, join
    observations back to curves so duplicate DataFrame indexes are harmless.
    """
    if not isinstance(min_angles, int) or min_angles < 3:
        raise ValueError("min_angles must be an integer of at least 3.")
    keys = ["geometry_id", "Re", "Ncrit"]
    missing = [name for name in [*keys, "Alpha", "Cl"] if name not in frame]
    data = frame.reset_index(drop=True)
    usable = np.zeros(len(data), dtype=bool)
    zero = np.zeros(len(data), dtype=bool)
    if not missing:
        finite = np.isfinite(data[["Re", "Ncrit", "Alpha", "Cl"]].to_numpy(dtype=float)).all(axis=1)
        finite &= data["geometry_id"].notna().to_numpy()
        zero = finite & (data["Alpha"].to_numpy() == 0)
        usable = finite & ~zero
    curves = data.loc[usable, [*keys, "Alpha", "Cl"]].copy() if not missing else pd.DataFrame()
    states = np.full(len(data), "excluded", dtype=object)
    regions = np.full(len(data), "excluded", dtype=object)
    records = []
    if not curves.empty:
        curves["sign"] = np.where(curves["Alpha"] < 0, "negative", "positive")
        curves["magnitude"] = curves["Alpha"].abs()
        curves["signed_cl"] = np.where(curves["Alpha"] < 0, -curves["Cl"], curves["Cl"])
        for key, group in curves.groupby([*keys, "sign"], observed=True, sort=True):
            # Duplicate angles do not make a sparse curve dense; exact ties use
            # the innermost observed angle and expose all tied peak positions.
            angles = group.groupby("magnitude", sort=True)["signed_cl"].max()
            peaks = angles.index[angles == angles.max()].to_numpy(dtype=float)
            peak = float(peaks.min())
            lower, upper = float(angles.index.min()), float(angles.index.max())
            at_lower, at_upper = bool(lower in peaks), bool(upper in peaks)
            boundary = "both" if at_lower and at_upper else "lower" if at_lower else "upper" if at_upper else None
            state = "sparse" if len(angles) < min_angles else "boundary-censored" if boundary else "interior"
            positions = group.index.to_numpy()
            states[positions] = state
            if state != "sparse":
                distance = group["magnitude"].to_numpy() - peak
                regions[positions] = np.where(np.abs(distance) <= 2, "near-peak",
                                              np.where(distance < -2, "farther-pre-peak", "farther-post-peak"))
            records.append({
                **dict(zip([*keys, "sign"], key)), "n_rows": int(len(group)),
                "n_angles": int(len(angles)), "state": state, "peak_boundary": boundary,
                "observed_min_abs_alpha": lower, "observed_max_abs_alpha": upper,
                "peak_alpha": -peak if key[-1] == "negative" else peak,
                "peak_abs_alpha": peak, "max_signed_cl": float(angles.max()),
                "n_tied_peak_angles": int(len(peaks)),
            })
    post_mask = np.isfinite(data["Alpha"]) & (data["Alpha"].abs() > STALL_ALPHA) if "Alpha" in data else usable
    denominator = _error_concentration(data.loc[post_mask], "Cd")["total_sse"]
    comparisons = []
    for state in ("interior", "boundary-censored"):
        comparisons.append({
            "peak_state": state,
            "n_curves": sum(record["state"] == state for record in records),
            "regions": [
                {"region": region, **_diagnostic_subset(data.loc[(states == state) & (regions == region)], "Cd"),
                 "poststall": _diagnostic_subset(
                     data.loc[(states == state) & (regions == region) & post_mask], "Cd", denominator,
                 )}
                for region in ("farther-pre-peak", "near-peak", "farther-post-peak")
            ],
        })
    return {
        "methodology": "Observed per-(geometry_id, Re, Ncrit, angle sign) maximum of sign(Alpha)*Cl; "
        "not true physical Clmax, stall onset, or proof of fully separated flow. Zero-angle and "
        "missing/nonfinite-key/Cl rows are excluded. Curves need at least min_angles distinct angles. "
        "Any tied maximum at either observed angle boundary is boundary-censored; ties use the "
        "innermost peak. Sparse curves are counted but excluded from Cd comparisons. Near-peak "
        "means within +/-2 degrees inclusive; farther pre/post uses increasing |Alpha|. Interior "
        "and censored comparisons are separate. Poststall SSE shares divide by all |Alpha| > 12 Cd SSE. "
        "These are descriptive, target-conditioned diagnostics, never features or tuning criteria.",
        "missing_columns": missing, "missing_cd_columns": [name for name in ("Cd", "Cd_predicted") if name not in data],
        "min_angles": min_angles, "peak_window_degrees": 2.0,
        "n_rows": int(len(data)), "n_usable_rows": int(usable.sum()),
        "n_zero_alpha_rows": int(zero.sum()), "n_missing_or_nonfinite_rows": int((~(usable | zero)).sum()),
        "n_curves": len(records),
        "state_counts": {
            state: {"n_curves": sum(record["state"] == state for record in records),
                    "n_rows": int((states == state).sum())}
            for state in ("interior", "boundary-censored", "sparse")
        },
        "curves": records, "cd_comparisons": comparisons,
    }


def summarize(
    frame: pd.DataFrame, target_names, threshold: float = STALL_ALPHA,
    *, training_frame: pd.DataFrame | None = None, reference_re: float = 1e6,
) -> dict:
    """Assemble every breakdown for a run report, as JSON-ready records.

    Reynolds groups use observed values; the easy reference uses fixed Re=1e6
    unless explicitly selected beforehand. Optional domains use training ranges, not
    evaluated target errors, and never remove rows from the pooled headline.
    Row masks are positional boolean lists in evaluated-frame order.
    """
    target_names = list(target_names)
    if not np.isfinite(reference_re) or reference_re <= 0:
        raise ValueError("reference_re must be finite and positive.")
    regime = stall_regime(frame, threshold)
    report = {
        "stall_threshold_degrees": threshold,
        "n_rows": int(len(frame)),
        "n_geometries": int(frame["geometry_id"].nunique()) if "geometry_id" in frame else None,
        "targets": {},
    }
    for target in target_names:
        geometry_scores = per_geometry(frame, target)
        distributions = {}
        for metric in ("mae", "rmse", "mean_relative_error", "median_relative_error"):
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
            "reynolds": _records(breakdown(frame, target, "Re", "Re")) if "Re" in frame else [],
            "worst_geometries": _records(worst_geometries(frame, target)),
            "error_concentration": _error_concentration(frame, target),
            "poststall_alpha": poststall_alpha(frame, target),
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
        "all observed Reynolds numbers scored separately alongside fixed_re_reference. "
        "Missing required columns yield an explicit empty subset; no target-error selection. "
        "Relative errors are fractions; compare mean to literature means, not median. "
        "This is not an external-paper equality claim: data source, split, and averaging differ.",
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
                ] if "Re" in easy else [],
            }
            for target in target_names
        },
    }
    reference_mask = easy_mask & (frame["Re"] == reference_re) if "Re" in frame else easy_mask
    reference = frame.loc[reference_mask]
    report["matched_easy_subset"]["fixed_re_reference"] = {
        "criteria": {"max_abs_alpha_degrees": threshold, "Ncrit": 9, "Re": float(reference_re)},
        "methodology": "Re fixed in advance (default 1e6), exact equality, never best-test-Re selection. "
        "Missing reference Re produces zero rows and null metrics; no fallback to another Re.",
        "row_mask": reference_mask.tolist(), "n_rows": int(len(reference)),
        "n_geometries": int(reference["geometry_id"].nunique()),
        "targets": {target: _metrics(reference, target) for target in target_names},
    }
    report["observed_cl_peak"] = observed_cl_peak(frame)

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


def main(argv=None) -> None:
    """Diagnose trusted, completed portable artifacts without fitting or Torch."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    import joblib

    from src.data.features import assemble_features
    from src.models.torch_mlp import TorchMLPRegressor
    from src.training.targets import PhysicalPredictor

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("run_dir", type=Path, help="Trusted local completed Kaggle run (joblib is executable).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--frozen-split", type=Path, default=Path("configs/frozen_split.json"))
    parser.add_argument("--reference-re", type=float, default=1e6)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; default is stdout. Never overwrites.")
    args = parser.parse_args(argv)
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("Output must be a new file in an existing directory.")
    run = args.run_dir.resolve()
    manifest = json.loads((run / "run.json").read_text())
    frozen = json.loads(args.frozen_split.read_text())
    metadata = manifest["run_metadata"]
    targets = manifest["target_names"]
    selected = manifest["selected_models"]
    if manifest["status"] != "completed" or set(selected) != set(targets):
        raise ValueError("A completed manifest with frozen winners for all targets is required.")
    partitions = metadata["split_geometries"]
    if partitions != frozen["split_geometries"]:
        raise ValueError("Run partitions do not match the frozen geometry split.")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(partitions[left]) & set(partitions[right]):
            raise ValueError("Frozen geometry partitions overlap.")
    dataset = metadata["dataset"]
    paths, fingerprints = {}, {}
    for name in ("geometry", "polars"):
        path = args.data_dir / dataset[name + "_table"]
        with path.open("rb") as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
            digest = checksum.hexdigest()
        if digest != dataset[name + "_sha256"]:
            raise ValueError(f"{name} table fingerprint differs from the completed run.")
        paths[name], fingerprints[name] = path, digest
    geometry = pd.read_csv(paths["geometry"])
    # Filter narrow polar chunks before attaching 108 geometry columns. Never
    # materialize the full wide train/validation/test design matrix locally.
    chunks, n_polars = [], 0
    for chunk in pd.read_csv(paths["polars"], chunksize=100_000):
        n_polars += len(chunk)
        chunks.append(chunk.loc[chunk["geometry_id"].isin(partitions["test"])])
    polars = pd.concat(chunks, ignore_index=True)
    if len(geometry) != dataset["geometry_rows"] or n_polars != dataset["polar_rows"]:
        raise ValueError("Dataset row counts differ from the completed run.")
    if len(polars) != metadata["split_rows"]["test"] or set(polars["geometry_id"]) != set(partitions["test"]):
        raise ValueError("Test rows/geometries differ from the completed run.")
    rows, columns = assemble_features(polars, geometry)
    if columns != metadata["feature_columns"]:
        raise ValueError("Named feature columns differ from the completed run.")
    predictions = np.empty((len(rows), len(targets)))
    for name in dict.fromkeys(selected.values()):
        result = manifest["results"][name]
        if result["status"] != "success" or name not in ("torch_mlp_mse", "torch_mlp_huber"):
            raise ValueError("Diagnostic CLI only loads successful portable NumPy MLP artifacts.")
        path = (run / result["model_path"]).resolve()
        if not path.is_relative_to(run):
            raise ValueError("Model artifact must be inside the trusted run directory.")
        model = joblib.load(path)
        if not isinstance(model, TorchMLPRegressor) or list(model.target_names) != targets:
            raise ValueError("Artifact is not the expected named-target portable MLP.")
        values = PhysicalPredictor(model, targets, manifest["target_transforms"]).predict(rows[columns])
        for index, target in enumerate(targets):
            if selected[target] == name:
                predictions[:, index] = values[:, index]
    scored = prediction_frame(rows, predictions, targets)
    report = summarize(scored, targets, reference_re=args.reference_re)
    report["diagnostic_provenance"] = {
        "run_dir": str(run), "source_sha256": metadata.get("source_sha256"),
        "dataset_sha256": fingerprints, "split": "frozen geometry-disjoint test",
        "selected_models": selected, "inference": "saved weights, CPU NumPy only; no fitting or Torch",
        "reference_re": args.reference_re,
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print(f"Diagnostics: {len(rows)} frozen test rows -> {args.output}")


if __name__ == "__main__":
    main()
