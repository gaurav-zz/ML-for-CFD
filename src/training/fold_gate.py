"""Paired engineering gate for three geometry folds, never seed selection.

Fold means have predeclared equal weight 1/3, not row weighting. These tail
criteria are descriptive engineering limits, not confidence intervals or
physics guarantees. Only canonical-training geometry evaluation is eligible.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re

import numpy as np

from src.models.ensemble import METHOD, SEEDS
from src.training.tail_risk import DEFINITIONS, SCORES, TAIL_METRICS, TARGETS


FOLDS = (0, 1, 2)
SOURCE = "canonical_train_geometry_fold_evaluation"
INPUT_SCALING_STUDY_KIND = "paired_geometry_input_scaling"
INPUT_SCALING_ARMS = ("cdp_column_scaling", "cdp_profile_scaling")
POLICY = {
    "median_fraction_tolerance": .005,
    "mean_tail_relative_tolerance": .05,
    "per_fold_tail_relative_tolerance": .10,
    "epsilon": 1e-12,
    "fold_weight": 1 / 3,
    "primary_rmse_guard_targets": ["Cl", "Cm"],
    "extension": "Before this fold study, add mean Cl/Cm RMSE limits using the same 5% allowance; prior study policies are unchanged.",
    "description": "Predeclared engineering policy; not physics guarantees, confidence intervals, or old-test tuning.",
}


def _number(value, name, *, minimum=0.):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be a finite, defined real number >= {minimum}.")
    return float(value)


def _geometry_ids(value):
    if (not isinstance(value, (list, tuple, set, frozenset)) or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(value) != len(set(value))):
        raise ValueError("Geometry IDs must be nonmissing, nonempty strings without duplicates.")
    return set(value)


def _validate_report(item, fold, arm, *, input_scaling=None):
    if not isinstance(item, dict):
        raise ValueError("Every fold requires a complete tail-report dictionary.")
    json.dumps(item, allow_nan=False)
    if type(item["schema_version"]) is not int or item["schema_version"] != 1 or item["definitions"] != DEFINITIONS:
        raise ValueError("Primary target definitions/physical units must match the tail-report schema.")
    if type(item["fold_id"]) is not int or item["fold_id"] != fold:
        raise ValueError("Integer report fold_id must match its geometry-fold key.")
    if item["source"] != SOURCE:
        raise ValueError("Reports must come from canonical_train_geometry_fold_evaluation, never canonical test.")
    if ("seeds" in item or item.get("unit", "geometry_fold") != "geometry_fold"
            or ("seed" in item and (input_scaling or arm != "reference" or type(item["seed"]) is not int or item["seed"] != 42))):
        raise ValueError("Geometry folds must not be labeled as independent seeds or other units.")
    if input_scaling:
        expected_arm = INPUT_SCALING_ARMS[0 if arm == "reference" else 1]
        expected_scaling = "standard" if arm == "reference" else input_scaling
        if (item["study_kind"] != INPUT_SCALING_STUDY_KIND
                or item["arm"] != expected_arm or item["input_scaling"] != expected_scaling
                or item["predictor"] != f"{expected_arm}_ensemble"
                or item["member_seeds"] != list(SEEDS) or item["aggregation"] != METHOD):
            raise ValueError("Input-scaling gate requires truthful column/profile Cdp ensemble labels, scaling modes, and all seeds [42,43,44].")
    elif arm == "reference":
        if item["predictor"] != "baseline_seed_42" or item["member_seeds"] != [42]:
            raise ValueError("Reference must be predeclared baseline_seed_42 with member_seeds=[42], never best seed.")
    elif (item["predictor"] not in ("baseline_ensemble", "cdp_ensemble")
          or item["member_seeds"] != list(SEEDS) or item["aggregation"] != METHOD):
        raise ValueError("Candidate requires one predeclared ensemble family, seeds [42,43,44], and physical aggregation method.")
    if (not isinstance(item["member_seeds"], list)
            or any(type(seed) is not int for seed in item["member_seeds"])):
        raise ValueError("Member seed metadata must be an explicit list of integers.")
    ids = _geometry_ids(item["geometry_ids"])
    geometry = item["per_geometry"]
    if not isinstance(geometry, list) or any(not isinstance(row, dict) for row in geometry):
        raise ValueError("Complete per-geometry statistics are required.")
    geometry_rows = [[row["geometry_id"], row["n_rows"]] for row in geometry]
    if (type(item["n_rows"]) is not int or item["n_rows"] <= 0
            or type(item["n_geometries"]) is not int or item["n_geometries"] <= 0
            or item["geometry_ids"] != sorted(ids)
            or item["geometry_ids"] != [row[0] for row in geometry_rows]
            or len(geometry) != item["n_geometries"]
            or any(type(row[1]) is not int or row[1] <= 0 for row in geometry_rows)
            or sum(row[1] for row in geometry_rows) != item["n_rows"]):
        raise ValueError("Inconsistent per-geometry IDs or row counts.")
    rmses = np.array([_number(row["Cd_rmse"], "Per-geometry Cd RMSE") for row in geometry])
    maximum = float(rmses.max())
    weights = np.array([row[1] for row in geometry_rows], dtype=float) / item["n_rows"]
    pooled = float(np.sqrt(np.sum(weights * (rmses / maximum) ** 2)) * maximum) if maximum else 0.
    aggregates = {
        "pooled_rmse": pooled,
        "p95_geometry_rmse": float(np.quantile(rmses, .95, method="linear")),
        "max_geometry_rmse": maximum,
        "macro_geometry_rmse": float(np.sum(rmses / len(rmses))),
    }
    for name, expected in aggregates.items():
        if not np.isclose(_number(item["cd_tail"][name], name), expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} is inconsistent with per-geometry Cd RMSE/counts.")
    if not np.isclose(_number(item["targets"]["Cd"]["rmse"], "Primary Cd RMSE"), pooled, rtol=1e-10, atol=1e-12):
        raise ValueError("Primary Cd RMSE is inconsistent with per-geometry Cd RMSE/counts.")
    thresholds = item["thresholds"]
    if (thresholds["source"] != "canonical_train_inner_fit"
            or thresholds["thickness_quantile"] != .9
            or thresholds["thickness_quantile_method"] != "linear_unique_geometries"
            or any(type(thresholds[key]) is not int or thresholds[key] <= 0
                   for key in ("training_rows", "training_geometries"))
            or thresholds["training_geometries"] > thresholds["training_rows"]):
        raise ValueError("Stress thresholds must record the predeclared canonical-training inner-fit derivation.")
    if _number(thresholds["low_re_max"], "low_re_max") == 0:
        raise ValueError("low_re_max must be strictly positive.")
    _number(thresholds["thick_min"], "thick_min")
    identity = {key: deepcopy(item[key]) for key in ("n_rows", "n_geometries", "row_sha256", "geometry_ids", "thresholds")}
    identity["geometry_rows"] = geometry_rows
    identity["stress"] = {}
    digests = [item["row_sha256"], thresholds["training_row_sha256"]]
    for group in ("low_re", "thick"):
        subset = item["stress"][group]
        if (type(subset["missing"]) is not bool or subset["missing"]
                or type(subset["n_rows"]) is not int or not 0 < subset["n_rows"] <= item["n_rows"]
                or type(subset["n_geometries"]) is not int
                or not 0 < subset["n_geometries"] <= min(item["n_geometries"], subset["n_rows"])):
            raise ValueError(f"Missing or empty {group} stress evidence; gate is inconclusive.")
        identity["stress"][group] = {key: subset[key] for key in ("n_rows", "n_geometries", "row_sha256")}
        digests.append(subset["row_sha256"])
    if any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in digests):
        raise ValueError("Row/stress/training identities must record SHA256 hex fingerprints.")
    if set(item["targets"]) != set(TARGETS):
        raise ValueError("Only the physical primary targets Cl, Cd, Cm may be gated.")
    flat = {f"{target}.{score}": _number(item["targets"][target][score], f"{target}.{score}",
                                       minimum=-np.inf if score == "r2" else 0.)
            for target in TARGETS for score in SCORES}
    flat.update({f"Cd.{name}": _number(item["cd_tail"][name], name) for name in TAIL_METRICS})
    flat.update({f"Cd.{group}_rmse": _number(item["stress"][group]["Cd_rmse"], f"{group} Cd RMSE")
                 for group in ("low_re", "thick")})
    return identity, flat


def paired_fold_gate(reference_reports, candidate_reports, expected_geometry_ids=None):
    """Gate exactly three paired geometry folds, returning JSON-safe evidence.

    Keys must be 0/1/2 or their exact string equivalents. Within each fold,
    reference/candidate rows, geometries, stress identities and thresholds must
    match; evaluation geometry sets must be disjoint across folds. When supplied,
    expected_geometry_ids must equal their full union (canonical training only).
    Invalid/incomplete evidence is inconclusive, never a reduced-fold average.
    This helper gates one family; it neither selects a family nor a best fold.
    """
    return _paired_fold_gate(reference_reports, candidate_reports, expected_geometry_ids)


def paired_input_scaling_fold_gate(reference_reports, candidate_reports, expected_geometry_ids=None):
    """Apply the historical 17 relative checks to the frozen scaling ensembles.

    The reference is cdp_column_scaling_ensemble, not a single seed or baseline.
    This relative gate does not certify numerical safety or recommend a model.
    """
    return _paired_fold_gate(reference_reports, candidate_reports, expected_geometry_ids, input_scaling="profile_rms_floor")


def paired_selective_scaling_fold_gate(reference_reports, candidate_reports, expected_geometry_ids=None):
    """Apply the unchanged 17 relative checks to the fixed 1%-RMS candidate.

    Expected modes are fixed here, never inferred from supplied reports. Historical
    full-RMS evidence is not selective evidence; numerical safety remains separate.
    """
    return _paired_fold_gate(reference_reports, candidate_reports, expected_geometry_ids,
                             input_scaling="profile_rms_floor_1pct")


def _paired_fold_gate(reference_reports, candidate_reports, expected_geometry_ids, *, input_scaling=None):
    report = {
        "schema_version": 1, "source": SOURCE, "unit": "geometry_fold", "gated_units": ["geometry_fold"],
        "folds": list(FOLDS), "n_folds": 3, "seed_selection": False, "fold_selection": False,
        "audit_only": False, "status": "inconclusive", "passed": False, "eligible": False,
        "policy": deepcopy(POLICY), "checks": [], "deltas": [], "errors": [], "reasons": [],
        "warnings": ["Three equally weighted geometry folds are not three independent seeds or confidence intervals.",
                     "Tail-report summaries are engineering evidence, not certified physical laws or abstention guarantees."],
        "fold_stats": [], "mean_stats": {},
    }
    if input_scaling:
        report.update(study_kind=INPUT_SCALING_STUDY_KIND, numerical_safety_checked=False)
        if input_scaling == "profile_rms_floor_1pct":
            report["study_variant"] = "selective_profile_floor_1pct"
    arms, values, identities = {}, {}, {}
    try:
        if input_scaling is not None and (not isinstance(input_scaling, str)
                or input_scaling not in ("profile_rms_floor", "profile_rms_floor_1pct")):
            raise ValueError("Fold gate requires a predeclared full-RMS or selective 1%-RMS mode.")
        expected = None if expected_geometry_ids is None else _geometry_ids(expected_geometry_ids)
        for arm, supplied in (("reference", reference_reports), ("candidate", candidate_reports)):
            if (not isinstance(supplied, dict) or len(supplied) != 3
                    or any(not ((type(key) is int and key in FOLDS) or (type(key) is str and key in ("0", "1", "2"))) for key in supplied)
                    or set(map(str, supplied)) != {"0", "1", "2"}):
                raise ValueError(f"{arm} requires exactly geometry folds 0, 1, 2, not seed IDs.")
            arms[arm] = {int(key): value for key, value in supplied.items()}
            for fold in FOLDS:
                identities[arm, fold], values[arm, fold] = _validate_report(arms[arm][fold], fold, arm, input_scaling=input_scaling)
        family = arms["candidate"][0]["predictor"]
        if any(arms["candidate"][fold]["predictor"] != family for fold in FOLDS):
            raise ValueError("All candidate folds must evaluate the same predeclared ensemble family.")
        seen = set()
        for fold in FOLDS:
            if identities["reference", fold] != identities["candidate", fold]:
                raise ValueError(f"Fold {fold} requires identical paired rows, geometries, stress identity and thresholds.")
            ids = set(identities["reference", fold]["geometry_ids"])
            if seen.intersection(ids):
                raise ValueError("Evaluation geometry sets must be disjoint across all three folds.")
            seen.update(ids)
        if expected is not None and seen != expected:
            raise ValueError("Evaluation geometry union must equal declared canonical-training geometry IDs exactly.")
        report["evaluation_geometry_ids"] = sorted(seen)
        report["expected_geometry_ids_validated"] = expected is not None
        if expected is None:
            report["warnings"].append("Canonical-training geometry union was not independently declared; source metadata alone is not a membership proof.")
        report["reference_predictor"] = f"{INPUT_SCALING_ARMS[0]}_ensemble" if input_scaling else "baseline_seed_42"
        report["candidate_predictor"] = family
        report["aggregation"] = METHOD
        report["fold_stats"] = [{"fold_id": fold, "weight": 1 / 3,
                                  "identity": identities["reference", fold],
                                  "reference": deepcopy(arms["reference"][fold]),
                                  "candidate": deepcopy(arms["candidate"][fold])} for fold in FOLDS]
        for metric in values["reference", 0]:
            for fold in (*FOLDS, "mean"):
                base = float(np.sum([values["reference", f][metric] / 3 for f in FOLDS])) if fold == "mean" else values["reference", fold][metric]
                candidate = float(np.sum([values["candidate", f][metric] / 3 for f in FOLDS])) if fold == "mean" else values["candidate", fold][metric]
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = float(np.float64(candidate) - np.float64(base))
                if not np.isfinite(delta):
                    raise ValueError(f"{metric} has a nonfinite paired fold delta.")
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    relative = float(np.float64(delta) / base) if base else (0. if delta == 0 else None)
                if relative is not None and not np.isfinite(relative):
                    relative = None
                row = {"fold_id": fold, "metric": metric, "reference": base, "candidate": candidate,
                       "delta": delta, "relative_delta": relative}
                if metric.endswith("median_relative_error"):
                    with np.errstate(over="ignore"):
                        points = float(np.float64(delta) * 100)
                    if not np.isfinite(points):
                        raise ValueError("Median percentage-point delta must remain finite.")
                    row["delta_percentage_points"] = points
                report["deltas"].append(row)
                if fold == "mean":
                    report["mean_stats"][metric] = row
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
        report["errors"].append(str(error))
        report["reasons"].append(str(error))
        report["checks"], report["deltas"], report["mean_stats"] = [], [], {}
        return report

    epsilon = POLICY["epsilon"]

    def check(row, rule, bound):
        passed = row["delta"] < bound if rule == "strict_improvement" else row["delta"] <= bound
        report["checks"].append({**row, "rule": rule, "allowed_delta": bound, "passed": bool(passed)})
        if not passed:
            report["reasons"].append(f"fold={row['fold_id']} {row['metric']}: {rule} failed (delta={row['delta']:.12g}, bound={bound:.12g}).")

    for target in TARGETS:
        check(report["mean_stats"][f"{target}.median_relative_error"], "median_fraction_degradation", POLICY["median_fraction_tolerance"] + epsilon)
    check(report["mean_stats"]["Cd.median_relative_error"], "strict_improvement", -epsilon)
    for target in POLICY["primary_rmse_guard_targets"]:
        row = report["mean_stats"][f"{target}.rmse"]
        check(row, "mean_primary_rmse_deterioration", row["reference"] * POLICY["mean_tail_relative_tolerance"] + epsilon)
    for metric in ("pooled_rmse", "p95_geometry_rmse", "max_geometry_rmse", "low_re_rmse", "thick_rmse"):
        row = report["mean_stats"][f"Cd.{metric}"]
        check(row, "mean_tail_deterioration", row["reference"] * POLICY["mean_tail_relative_tolerance"] + epsilon)
    for row in report["deltas"]:
        if row["fold_id"] in FOLDS and row["metric"] in ("Cd.pooled_rmse", "Cd.max_geometry_rmse"):
            check(row, "per_fold_tail_deterioration", row["reference"] * POLICY["per_fold_tail_relative_tolerance"] + epsilon)
    if any(row["relative_delta"] is None for row in report["deltas"]):
        report["warnings"].append("Zero reference error or ratio overflow: relative delta undefined; absolute-delta bounds were used.")
    report.update(status="fail" if report["reasons"] else "pass", passed=not report["reasons"], eligible=not report["reasons"])
    return report
