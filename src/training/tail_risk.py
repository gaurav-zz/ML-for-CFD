"""Physical-unit tail diagnostics and predeclared, paired engineering gates.

No fitting, Torch, target transformation, or exclusion of difficult rows occurs
here. Relative errors are fractions, not percentages. These policies are not
physics guarantees or statistically calibrated confidence bounds.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

import numpy as np
import pandas as pd

from src.utils.metrics import regression_metrics


TARGETS = ("Cl", "Cd", "Cm")
SEEDS = (42, 43, 44)
ROW_KEYS = ("geometry_id", "Alpha", "Re", "Ncrit")
SCORES = ("mae", "rmse", "r2", "mean_relative_error", "median_relative_error")
TAIL_METRICS = ("pooled_rmse", "p95_geometry_rmse", "max_geometry_rmse", "macro_geometry_rmse")
POLICY = {
    "median_fraction_tolerance": .005,
    "mean_tail_relative_tolerance": .05,
    "per_seed_tail_relative_tolerance": .10,
    "epsilon": 1e-12,
    "description": "Predeclared engineering policy; not physics guarantees or derived from old test scores.",
}
DEFINITIONS = {
    "units": "physical_coefficients",
    "relative_error_units": "fraction",
    "relative_error": "abs(prediction-actual)/abs(actual); exactly-zero actual values excluded",
    "pooled_rmse": "sqrt(mean(row squared Cd error)); all rows, no filtering",
    "p95_geometry_rmse": "np.quantile(per-geometry Cd RMSE, 0.95, method='linear'); unweighted geometries",
    "macro_geometry_rmse": "arithmetic mean of per-geometry Cd RMSE; unweighted geometries",
    "stress": "Re <= low_re_max; max_thickness >= thick_min; boundaries inclusive",
}


def _identity(frame):
    if (not isinstance(frame, pd.DataFrame) or frame.empty
            or not frame.columns.is_unique or not frame.index.is_unique):
        raise ValueError("A nonempty frame with unique columns and row indices is required.")
    missing = set(ROW_KEYS).union({"max_thickness"}).difference(frame.columns)
    if missing:
        raise ValueError(f"Tail report lacks required columns: {sorted(missing)}.")
    rows = frame[list(ROW_KEYS)]
    if rows.isna().any().any() or rows.duplicated().any():
        raise ValueError("Operating-point identities must be nonmissing and unique.")
    if not all(isinstance(value, str) and value.strip() for value in rows.geometry_id):
        raise ValueError("geometry_id must contain nonempty strings.")
    numeric = frame[["Alpha", "Re", "Ncrit", "max_thickness"]].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (numeric[:, 1] <= 0).any()
            or (numeric[:, 3] < 0).any()):
        raise ValueError("Identity/stress coordinates must be finite; Re positive and thickness nonnegative.")
    if (frame.groupby("geometry_id")["max_thickness"].nunique() != 1).any():
        raise ValueError("max_thickness must be constant within each geometry.")
    # Same identity-only digest as study._row_hash; row order does not affect it.
    hashes = pd.util.hash_pandas_object(rows, index=False).to_numpy()
    return hashlib.sha256(np.sort(hashes).astype("<u8").tobytes()).hexdigest()


def derive_thresholds(training_frame, *, source="canonical_train_inner_fit"):
    """Derive stress boundaries from training coordinates, never target labels."""
    if source not in ("canonical_train_inner_fit", "canonical_train"):
        raise ValueError("Stress thresholds must be derived from canonical training data only.")
    digest = _identity(training_frame)
    thickness = training_frame.drop_duplicates("geometry_id")["max_thickness"]
    return {
        "source": source,
        "low_re_max": float(training_frame["Re"].min()),
        "thick_min": float(thickness.quantile(.9, interpolation="linear")),
        "thickness_quantile": .9,
        "thickness_quantile_method": "linear_unique_geometries",
        "training_rows": len(training_frame),
        "training_geometries": int(training_frame.geometry_id.nunique()),
        "training_row_sha256": digest,
    }


def tail_report(frame, predictions, targets=TARGETS, *, thresholds):
    """Score aligned *physical* predictions; reject nonfinite or silently lost rows.

    Arrays are positional. A prediction DataFrame must additionally have the
    exact frame index and primary target columns, in order. Empty stress groups
    have explicit null RMSE and cannot pass :func:`paired_tail_gate`.
    """
    if tuple(targets) != TARGETS:
        raise ValueError("Tail reporting requires primary targets Cl, Cd, Cm in order.")
    digest = _identity(frame)
    if not isinstance(thresholds, dict):
        raise ValueError("Explicit training-only stress thresholds are required.")
    for name in ("low_re_max", "thick_min"):
        value = thresholds.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value < 0 or (name == "low_re_max" and value == 0)):
            raise ValueError(f"Invalid stress threshold {name}.")
    json.dumps(thresholds, allow_nan=False)
    if isinstance(predictions, pd.DataFrame):
        if list(predictions.columns) != list(targets) or not predictions.index.equals(frame.index):
            raise ValueError("Prediction columns and row indices must match exactly, in order.")
    if not set(targets).issubset(frame.columns):
        raise ValueError("All three primary labels are required.")
    actual = frame[list(targets)].to_numpy(dtype=float)
    raw = np.asarray(predictions)
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("Predictions must be real numeric physical coefficients.")
    predicted = raw.astype(float, copy=False)
    if actual.shape != predicted.shape:
        raise ValueError("Predictions must have exactly one row per identity and three columns.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("All labels and predictions must be finite; no rows may be dropped.")
    with np.errstate(over="ignore", invalid="ignore"):
        cd_squared = (predicted[:, 1] - actual[:, 1]) ** 2
    if not np.isfinite(cd_squared).all():
        raise ValueError("Cd squared errors must remain finite.")
    metrics = {name: regression_metrics(actual[:, i], predicted[:, i]) for i, name in enumerate(targets)}
    if any(value is not None and not np.isfinite(value) for scores in metrics.values() for value in scores.values()):
        raise ValueError("Primary metrics must be finite when defined.")
    groups = pd.DataFrame({"geometry_id": frame.geometry_id.to_numpy(), "squared": cd_squared})
    grouped = groups.groupby("geometry_id", sort=True).squared.agg(["mean", "size"])
    geometry_rmse = np.sqrt(grouped["mean"].to_numpy())
    stress = {}
    for name, mask in (
        ("low_re", frame["Re"].to_numpy() <= thresholds["low_re_max"]),
        ("thick", frame["max_thickness"].to_numpy() >= thresholds["thick_min"]),
    ):
        selected = frame.loc[mask, list(ROW_KEYS)]
        hashes = pd.util.hash_pandas_object(selected, index=False).to_numpy()
        stress[name] = {
            "n_rows": int(mask.sum()), "n_geometries": int(selected.geometry_id.nunique()),
            "row_sha256": hashlib.sha256(np.sort(hashes).astype("<u8").tobytes()).hexdigest(),
            "missing": not bool(mask.any()),
            "Cd_rmse": float(np.sqrt(cd_squared[mask].mean())) if mask.any() else None,
        }
    return {
        "schema_version": 1, "definitions": deepcopy(DEFINITIONS),
        "n_rows": len(frame), "n_geometries": len(grouped), "row_sha256": digest,
        "geometry_ids": list(grouped.index), "thresholds": deepcopy(thresholds),
        "targets": metrics,
        "cd_tail": {
            "pooled_rmse": float(np.sqrt(cd_squared.mean())),
            "p95_geometry_rmse": float(np.quantile(geometry_rmse, .95, method="linear")),
            "max_geometry_rmse": float(geometry_rmse.max()),
            "macro_geometry_rmse": float(geometry_rmse.mean()),
        },
        "per_geometry": [
            {"geometry_id": name, "n_rows": int(row["size"]), "Cd_rmse": float(np.sqrt(row["mean"]))}
            for name, row in grouped.iterrows()
        ],
        "stress": stress,
    }


def paired_tail_gate(baseline_reports, cdp_reports, *, source="canonical_train_inner_evaluation", audit_only=False):
    """Return an auditable three-seed decision, never a best-seed selection.

    Invalid/missing evidence is inconclusive (eligible=False), not a runtime
    error or a reduced-seed average. At zero baseline error the additive epsilon
    is the only permitted deterioration; relative deltas are explicitly null.
    Canonical-test gates are audit-only and cannot select an experimental arm.
    """
    if source not in ("canonical_train_inner_evaluation", "canonical_test"):
        raise ValueError("Gate source must be inner evaluation or canonical test.")
    if (source == "canonical_test") != audit_only:
        raise ValueError("Canonical test is audit-only; development is the selection source.")
    report = {
        "schema_version": 1, "source": source, "audit_only": audit_only,
        "status": "inconclusive", "passed": False, "eligible": False,
        "seeds": list(SEEDS), "seed_selection": False, "policy": deepcopy(POLICY),
        "reasons": [], "warnings": ["Three paired seeds are descriptive, not a confidence guarantee."],
        "checks": [], "deltas": [],
    }
    arms = {}
    identities = []
    values = {}
    try:
        for arm, supplied in (("baseline", baseline_reports), ("cdp", cdp_reports)):
            if not isinstance(supplied, dict) or set(map(str, supplied)) != set(map(str, SEEDS)) or len(supplied) != 3:
                raise ValueError(f"{arm} requires exactly all three seeds 42, 43, 44.")
            arms[arm] = {int(seed): item for seed, item in supplied.items()}
            for seed, item in arms[arm].items():
                json.dumps(item, allow_nan=False)
                if item["schema_version"] != 1 or item["definitions"] != DEFINITIONS:
                    raise ValueError("Metric definitions/physical units must match the tail-report schema.")
                if "seed" in item and item["seed"] != seed:
                    raise ValueError("Report seed does not match its paired seed identity.")
                if "arm" in item and item["arm"] != arm:
                    raise ValueError("Report arm does not match its paired arm identity.")
                if "source" in item and item["source"] != source:
                    raise ValueError("Report source does not match the gate source.")
                expected_source = "canonical_train" if audit_only else "canonical_train_inner_fit"
                if item["thresholds"].get("source") != expected_source:
                    raise ValueError("Threshold source is not the predeclared training partition.")
                geometry = [[row["geometry_id"], row["n_rows"]] for row in item["per_geometry"]]
                if (type(item["n_rows"]) is not int or type(item["n_geometries"]) is not int
                        or item["n_rows"] <= 0 or item["n_geometries"] <= 0
                        or item["geometry_ids"] != sorted(set(item["geometry_ids"]))
                        or item["geometry_ids"] != [row[0] for row in geometry]
                        or len(geometry) != item["n_geometries"]
                        or sum(row[1] for row in geometry) != item["n_rows"]
                        or any(type(row[1]) is not int or row[1] <= 0 for row in geometry)):
                    raise ValueError("Inconsistent geometry IDs or row counts.")
                geometry_rmse = np.asarray([row["Cd_rmse"] for row in item["per_geometry"]], dtype=float)
                if not np.isfinite(geometry_rmse).all() or (geometry_rmse < 0).any():
                    raise ValueError("Per-geometry Cd RMSE must be finite and nonnegative.")
                maximum = float(geometry_rmse.max())
                weights = np.asarray([row[1] for row in geometry], dtype=float) / item["n_rows"]
                pooled = float(np.sqrt(np.sum(weights * (geometry_rmse / maximum) ** 2)) * maximum) if maximum else 0.
                aggregates = {
                    "pooled_rmse": pooled, "p95_geometry_rmse": float(np.quantile(geometry_rmse, .95, method="linear")),
                    "max_geometry_rmse": maximum, "macro_geometry_rmse": float(np.sum(geometry_rmse / len(geometry_rmse))),
                }
                for metric, expected in aggregates.items():
                    actual = item["cd_tail"][metric]
                    if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not np.isfinite(actual)
                            or not np.isclose(actual, expected, rtol=1e-10, atol=1e-12)):
                        raise ValueError(f"{metric} is inconsistent with per-geometry Cd RMSE/counts.")
                if not np.isclose(item["targets"]["Cd"]["rmse"], pooled, rtol=1e-10, atol=1e-12):
                    raise ValueError("Primary Cd RMSE is inconsistent with per-geometry Cd RMSE/counts.")
                for key in ("low_re_max", "thick_min"):
                    value = item["thresholds"][key]
                    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                            or value < 0 or (key == "low_re_max" and value == 0)):
                        raise ValueError("Stress thresholds must be finite and physically valid.")
                for digest in (item["row_sha256"], item["thresholds"]["training_row_sha256"],
                               *(item["stress"][group]["row_sha256"] for group in ("low_re", "thick"))):
                    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                        raise ValueError("Row identity fingerprints must be SHA256 hex strings.")
                identity = {key: item[key] for key in ("n_rows", "n_geometries", "row_sha256", "geometry_ids", "thresholds")}
                identity["geometry_rows"] = geometry
                identity["stress"] = {}
                for group in ("low_re", "thick"):
                    subset = item["stress"][group]
                    if (type(subset["missing"]) is not bool or subset["missing"]
                            or type(subset["n_rows"]) is not int or type(subset["n_geometries"]) is not int
                            or not 0 < subset["n_rows"] <= item["n_rows"]
                            or not 0 < subset["n_geometries"] <= item["n_geometries"]):
                        raise ValueError(f"Missing or empty {group} stress evidence; gate is inconclusive.")
                    identity["stress"][group] = {key: subset[key] for key in ("n_rows", "n_geometries", "row_sha256")}
                identities.append(identity)
                flat = {f"{target}.{metric}": item["targets"][target][metric] for target in TARGETS for metric in SCORES}
                flat.update({f"Cd.{metric}": item["cd_tail"][metric] for metric in TAIL_METRICS})
                flat.update({f"Cd.{group}_rmse": item["stress"][group]["Cd_rmse"] for group in ("low_re", "thick")})
                for metric, value in flat.items():
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not np.isfinite(value) or (not metric.endswith(".r2") and value < 0)):
                        raise ValueError(f"{arm} seed {seed}: {metric} must be finite and defined.")
                values[arm, seed] = flat
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("All six reports must have identical rows, geometry IDs, stress groups and thresholds.")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
        report["reasons"].append(str(error))
        return report

    report["identity"] = identities[0]
    epsilon = POLICY["epsilon"]
    mean_values = {}
    for metric in values["baseline", SEEDS[0]]:
        for seed in (*SEEDS, "mean"):
            base = float(np.sum([values["baseline", s][metric] / 3 for s in SEEDS])) if seed == "mean" else values["baseline", seed][metric]
            candidate = float(np.sum([values["cdp", s][metric] / 3 for s in SEEDS])) if seed == "mean" else values["cdp", seed][metric]
            delta = candidate - base
            if not np.isfinite(delta):
                report["reasons"].append(f"{metric} has a nonfinite paired delta.")
                report["deltas"] = []
                return report
            relative = delta / base if base != 0 else (0. if delta == 0 else None)
            if relative is not None and not np.isfinite(relative):
                relative = None
            row = {"seed": seed, "metric": metric, "baseline": base, "cdp": candidate, "delta": delta,
                   "relative_delta": relative}
            if metric.endswith("median_relative_error"):
                row["delta_percentage_points"] = 100 * delta
            report["deltas"].append(row)
            if seed == "mean":
                mean_values[metric] = row

    def check(row, rule, bound):
        passed = row["delta"] < bound if rule == "strict_improvement" else row["delta"] <= bound
        entry = {**row, "rule": rule, "allowed_delta": bound, "passed": bool(passed)}
        report["checks"].append(entry)
        if not passed:
            report["reasons"].append(f"seed={row['seed']} {row['metric']}: {rule} failed (delta={row['delta']:.12g}, bound={bound:.12g}).")

    for target in TARGETS:
        check(mean_values[f"{target}.median_relative_error"], "median_fraction_degradation", POLICY["median_fraction_tolerance"] + epsilon)
    check(mean_values["Cd.median_relative_error"], "strict_improvement", -epsilon)
    for metric in ("pooled_rmse", "p95_geometry_rmse", "max_geometry_rmse", "low_re_rmse", "thick_rmse"):
        row = mean_values[f"Cd.{metric}"]
        check(row, "mean_tail_deterioration", row["baseline"] * POLICY["mean_tail_relative_tolerance"] + epsilon)
    for row in report["deltas"]:
        if row["seed"] in SEEDS and row["metric"] in ("Cd.pooled_rmse", "Cd.max_geometry_rmse"):
            check(row, "per_seed_tail_deterioration", row["baseline"] * POLICY["per_seed_tail_relative_tolerance"] + epsilon)
    if any(row["relative_delta"] is None for row in report["deltas"]):
        report["warnings"].append("Zero baseline error or ratio overflow: relative delta undefined; absolute-delta bounds were used.")
    report.update(status="fail" if report["reasons"] else "pass", passed=not report["reasons"], eligible=not report["reasons"])
    return report
