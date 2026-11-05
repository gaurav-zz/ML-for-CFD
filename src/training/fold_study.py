"""Kaggle-only, canonical-train geometry-fold robustness study.

The canonical validation/test frames are inspected for identity and coverage
only. This is a development recommendation, not a production refit or a locked
holdout assessment. Every fold member is fitted afresh on its own inner fit.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.models.torch_mlp import RAW_FEATURE_NAMES, TorchMLPRegressor
from src.training import study
from src.training.auxiliary_study import _auxiliary_values
from src.training.comparison import SCORE_COLUMNS, _write_json, compare_models
from src.training.tail_risk import SEEDS, TAIL_METRICS, derive_thresholds, tail_report
from src.training.targets import DEFAULT_TARGET_TRANSFORMS, INVERSE_LIMITS, PhysicalPredictor, inverse_targets
from src.utils.metrics import regression_metrics


_PARTITION_SEED = 20261026
_ARMS = ("baseline", "cdp")
_INPUT_SCALING_ARMS = ("cdp_column_scaling", "cdp_profile_scaling")
_INPUT_SCALING_MODES = dict(zip(_INPUT_SCALING_ARMS, ("standard", "profile_rms_floor")))
_INPUT_SCALING_STUDY_KIND = "paired_geometry_input_scaling"
_SELECTIVE_STUDY_VARIANT = "selective_profile_floor_1pct"
_SOURCE = "canonical_train_geometry_fold_evaluation"
_GEOMETRY_COUNT = 572
_METHOD = "mean_Cl_Cm_geometric_mean_Cd"


def _geometry_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")).hexdigest()


def _geometry_folds(train):
    """Return identity-sorted original positions and a complete three-fold audit."""
    ids = sorted(train.geometry_id.unique())
    if len(ids) != _GEOMETRY_COUNT:
        raise ValueError("Fold study requires exactly 572 frozen canonical training geometry IDs.")
    evaluation_folds = np.array_split(np.random.default_rng(_PARTITION_SEED).permutation(ids), 3)
    identity = train[study._ROW_KEYS].reset_index(drop=True).sort_values(study._ROW_KEYS)
    positions, folds, records, covered = {}, {}, [], set()
    for fold_id, evaluation_ids in enumerate(evaluation_folds):
        evaluation_set = set(evaluation_ids)
        remaining = sorted(set(ids).difference(evaluation_set))
        stopping_seed = _PARTITION_SEED + fold_id + 1
        shuffled = np.random.default_rng(stopping_seed).permutation(remaining)
        n_stop = max(1, round(.1 * len(remaining)))
        selected = {"fit": shuffled[n_stop:], "stopping": shuffled[:n_stop], "evaluation": evaluation_ids}
        groups = [set(selected[name]) for name in selected]
        if (any(groups[i].intersection(groups[j]) for i in range(3) for j in range(i))
                or set.union(*groups) != set(ids) or covered.intersection(evaluation_set)):
            raise ValueError("Fold geometry identities must be disjoint and cover canonical train exactly.")
        covered.update(evaluation_set)
        positions[fold_id], partitions = {}, {}
        for name, geometry_ids in selected.items():
            pos = identity.index[identity.geometry_id.isin(geometry_ids)].to_numpy()
            rows = train[study._ROW_KEYS].iloc[pos]
            positions[fold_id][name] = pos
            partitions[name] = {
                "geometry_ids": sorted(geometry_ids), "geometry_sha256": _geometry_hash(geometry_ids),
                "n_geometries": len(geometry_ids), "n_rows": len(rows), "row_sha256": study._row_hash(rows),
            }
            recorded = rows.copy()
            recorded.insert(0, "canonical_row_index", rows.index)
            recorded.insert(0, "canonical_train_position", pos)
            recorded.insert(0, "partition", name)
            recorded.insert(0, "fold_id", fold_id)
            records.append(recorded)
        flat = np.concatenate(list(positions[fold_id].values()))
        if len(flat) != len(train) or len(np.unique(flat)) != len(train):
            raise ValueError("Every canonical training row must appear exactly once within each fold.")
        folds[fold_id] = {"fold_id": fold_id, "stopping_seed": stopping_seed, "partitions": partitions}
    if covered != set(ids):
        raise ValueError("Evaluation folds must cover all 572 training geometries without dropping a fold.")
    return positions, {
        "schema_version": 1, "source": "canonical_train_only", "partition_seed": _PARTITION_SEED,
        "n_folds": 3, "geometry_ids": ids, "geometry_sha256": _geometry_hash(ids),
        "evaluation_geometry_counts": [len(group) for group in evaluation_folds],
        "algorithm": "rng.permutation(sorted572IDs), np.array_split into 3 evaluation folds; independent stopping RNG on sorted remaining IDs",
        "stopping_rounding": "max(1, round(0.1*n_remaining)); fit is remainder",
        "row_order": "sorted geometry_id, Alpha, Re, Ncrit; original positions and indices retained",
        "row_indices": "fold_rows.csv", "row_identity_columns": study._ROW_KEYS.copy(),
        "row_hash_algorithm": "sha256(sorted pandas hash_pandas_object(identity, index=False), little-endian uint64)",
        "geometry_hash_algorithm": "sha256(UTF-8 compact JSON sorted geometry ID list)",
        "folds": folds,
    }, pd.concat(records, ignore_index=True)


def _metric_rows(report, **metadata):
    rows = []
    for target, values in report["targets"].items():
        row = {**metadata, "source": _SOURCE, "target": target, **values,
               "n_predictions": report["n_rows"], "row_sha256": report["row_sha256"],
               "geometry_sha256": _geometry_hash(report["geometry_ids"])}
        if "study_kind" in report:
            row.update(study_kind=report["study_kind"], input_scaling=report["input_scaling"])
        if "study_variant" in report:
            row["study_variant"] = report["study_variant"]
        if target == "Cd":
            row.update(report["cd_tail"])
            row.update({f"{name}_rmse": report["stress"][name]["Cd_rmse"] for name in ("low_re", "thick")})
        rows.append(row)
    return rows


def _descriptive_summary(member_rows, ensemble_rows):
    """Keep seed variability within a fold separate from between-fold variation."""
    summaries = []
    members, ensembles = pd.DataFrame(member_rows), pd.DataFrame(ensemble_rows)
    grouped = []
    for (fold_id, arm, target), group in members.groupby(["fold_id", "arm", "target"], sort=False):
        grouped.append(("seed_within_fold", fold_id, arm, f"{arm}_members", target, group, "n_seeds"))
    for (arm, seed, target), group in members.groupby(["arm", "seed", "target"], sort=False):
        grouped.append(("fold_variation", None, arm, f"{arm}_seed_{seed}", target, group, "n_folds"))
    for (arm, target), group in ensembles.groupby(["arm", "target"], sort=False):
        grouped.append(("fold_variation", None, arm, f"{arm}_ensemble", target, group, "n_folds"))
    for variation, fold_id, arm, predictor, target, group, count_key in grouped:
        metrics = [*SCORE_COLUMNS, *TAIL_METRICS, "low_re_rmse", "thick_rmse"] if target == "Cd" else SCORE_COLUMNS
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            complete = len(values) == 3 and np.isfinite(values).all()
            summaries.append({
                "variation": variation, "fold_id": fold_id, "arm": arm, "predictor": predictor,
                "source": _SOURCE, "target": target, "metric": metric, count_key: len(values),
                "n_defined": int(np.isfinite(values).sum()), "ddof": 1,
                "status": "complete" if complete else "inconclusive",
                "mean": float(values.mean()) if complete else None,
                "std": float(values.std(ddof=1)) if complete else None,
                "weighting": "unweighted folds" if count_key == "n_folds" else "unweighted seeds within this fold",
                "predictions_ensembled_across_folds": False,
            })
            if "study_kind" in group:
                summaries[-1].update(study_kind=group["study_kind"].iloc[0], input_scaling=group["input_scaling"].iloc[0])
            if "study_variant" in group:
                summaries[-1]["study_variant"] = group["study_variant"].iloc[0]
    return summaries


def _write_predictions(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _mark_r2_definition(report, actual):
    # sklearn's force_finite default masks undefined constant-target R2 as 0/1.
    for index, values in enumerate(report["targets"].values()):
        status = "undefined_insufficient_rows" if len(actual) < 2 else (
            "undefined_constant_target" if np.all(actual[:, index] == actual[0, index]) else "defined"
        )
        values["r2_status"] = status
        if status != "defined":
            values["r2"] = None


def _pooled_oof_scores(train, targets, fold_predictions):
    """Align one evaluation prediction per original row before direct scoring."""
    if (not isinstance(fold_predictions, dict) or set(fold_predictions) != {0, 1, 2}
            or any(type(fold) is not int for fold in fold_predictions)):
        raise ValueError("Pooled OOF scoring requires exactly all three geometry folds.")
    covered = np.zeros(len(train), dtype=int)
    aligned = np.empty((len(train), len(targets)), dtype=float)
    for fold in range(3):
        positions, predictions = fold_predictions[fold]
        positions, predictions = np.asarray(positions), np.asarray(predictions, dtype=float)
        if (positions.ndim != 1 or positions.dtype.kind not in "iu" or not len(positions)
                or (positions < 0).any() or (positions >= len(train)).any()
                or len(np.unique(positions)) != len(positions)
                or predictions.shape != (len(positions), len(targets)) or not np.isfinite(predictions).all()):
            raise ValueError("Pooled OOF predictions require finite complete arrays and unique in-range canonical-train positions.")
        covered[positions] += 1
        aligned[positions] = predictions
    if not (covered == 1).all():
        raise ValueError("Pooled OOF coverage must include every canonical-train position exactly once, without dropped or duplicated rows.")
    actual = train[targets].to_numpy(dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError("Pooled OOF primary labels must be finite on every canonical-train row.")
    report = {
        "schema_version": 1, "source": _SOURCE, "study_kind": _INPUT_SCALING_STUDY_KIND,
        "score_scope": "pooled_oof_all_canonical_train_rows", "n_rows": len(train),
        "row_sha256": study._row_hash(train), "geometry_ids": sorted(train.geometry_id.unique()),
        "coverage": "each canonical_train_position exactly once", "n_folds": 3,
        "row_filtering": False, "predictions_ensembled_across_folds": False,
        "r2_aggregation": "direct regression_metrics on aligned rows; not average fold R2",
        "targets": {target: regression_metrics(actual[:, index], aligned[:, index])
                    for index, target in enumerate(targets)},
    }
    _mark_r2_definition(report, actual)
    json.dumps(report, allow_nan=False)
    return report


def _input_scaling_safety(ensemble_reports, numerical_reports):
    """Keep absolute sanity screens separate from the relative engineering gate."""
    result = {
        "schema_version": 1, "source": _SOURCE, "study_kind": _INPUT_SCALING_STUDY_KIND,
        "status": "inconclusive", "passed": False, "n_folds": 3, "member_seeds": list(SEEDS),
        "checks": [], "errors": [], "reasons": [], "production_promotion": False,
        "policy": {"primary_r2": "strictly > 0 for every Cl/Cd/Cm ensemble target in every fold",
                   "clipping": "no pre-inverse-log bound crossings in any of the three members on any row",
                   "missing_evidence": "inconclusive", "nonfinite_raw_predictions": "error",
                   "description": "Minimal numerical sanity screens, not physical accuracy guarantees."},
    }
    # Raw nonfinite evidence is an evaluation error, never a completed safety veto.
    if isinstance(numerical_reports, dict):
        for item in numerical_reports.values():
            if isinstance(item, dict) and isinstance(item.get("n_nonfinite"), (int, float)) and item["n_nonfinite"] > 0:
                raise ValueError("Numerical safety received nonfinite raw fitting-space predictions.")
    try:
        for supplied in (ensemble_reports, numerical_reports):
            if (not isinstance(supplied, dict) or set(supplied) != {0, 1, 2}
                    or any(type(fold) is not int for fold in supplied)):
                raise ValueError("Numerical safety requires complete evidence for all three geometry folds.")
        for fold in range(3):
            ensemble, numerical = ensemble_reports[fold], numerical_reports[fold]
            json.dumps((ensemble, numerical), allow_nan=False)
            if (numerical["schema_version"] != 1 or type(numerical["schema_version"]) is not int
                    or numerical["status"] != "complete" or numerical["target_names"] != ["Cl", "Cd", "Cm"]
                    or numerical["member_seeds"] != list(SEEDS)
                    or any(type(seed) is not int for seed in numerical["member_seeds"])
                    or type(numerical["n_nonfinite"]) is not int or numerical["n_nonfinite"] != 0
                    or numerical["inverse_limits"] != list(INVERSE_LIMITS)
                    or numerical["calibrated_uncertainty"] is not False):
                raise ValueError("Numerical safety requires complete finite raw evidence from all ordered members and unchanged inverse limits.")
            for key in ("fold_id", "source", "study_kind", "predictor", "arm", "input_scaling", "n_rows", "n_geometries", "row_sha256"):
                if numerical[key] != ensemble[key]:
                    raise ValueError(f"Fold {fold} numerical and ensemble {key} identities must match.")
            if (ensemble["fold_id"] != fold or type(ensemble["fold_id"]) is not int
                    or ensemble["study_kind"] != _INPUT_SCALING_STUDY_KIND or ensemble["source"] != _SOURCE
                    or ensemble["member_seeds"] != list(SEEDS) or ensemble["aggregation"] != _METHOD
                    or ensemble["predictor"] != ensemble_reports[0]["predictor"]):
                raise ValueError("Numerical safety requires one truthful, complete input-scaling ensemble family across all folds.")
            n_rows = ensemble["n_rows"]
            if type(n_rows) is not int or n_rows <= 0:
                raise ValueError("Numerical safety requires a positive evaluation row count.")
            members = numerical["per_member"]
            if len(members) != 3 or [row["seed"] for row in members] != list(SEEDS):
                raise ValueError("Numerical safety requires clipping counts for every ordered seed.")
            clipped = 0
            for row in members:
                if type(row["seed"]) is not int:
                    raise ValueError("Numerical safety member seeds must be integers.")
                for name in ("n_lower_clipped", "n_upper_clipped"):
                    if type(row[name]) is not int or not 0 <= row[name] <= n_rows:
                        raise ValueError("Numerical safety clipping counts must be complete nonnegative integers.")
                    clipped += row[name]
                if row["n_lower_clipped"] + row["n_upper_clipped"] > n_rows:
                    raise ValueError("Member clipping counts cannot exceed its evaluation row count.")
            geometry = numerical["per_geometry"]
            if (len(geometry) != ensemble["n_geometries"]
                    or [[row["geometry_id"], row["n_rows"]] for row in geometry]
                    != [[row["geometry_id"], row["n_rows"]] for row in ensemble["per_geometry"]]):
                raise ValueError("Numerical safety requires complete matching per-geometry row counts.")
            for row in geometry:
                for name in ("n_rows_any_clipped", "zero_clipped_spread_nonzero_raw_rows"):
                    if type(row[name]) is not int or not 0 <= row[name] <= row["n_rows"]:
                        raise ValueError("Numerical safety per-geometry clipping/spread counts are incomplete.")
            for name in ("n_rows_any_clipped", "zero_clipped_spread_nonzero_raw_rows"):
                if type(numerical[name]) is not int or numerical[name] != sum(row[name] for row in geometry):
                    raise ValueError("Numerical safety row clipping/spread totals must match geometry evidence.")
            if (type(numerical["n_member_values_clipped"]) is not int or numerical["n_member_values_clipped"] != clipped
                    or not numerical["n_rows_any_clipped"] <= clipped <= 3 * numerical["n_rows_any_clipped"]
                    or type(numerical["safe_no_clipping"]) is not bool or numerical["safe_no_clipping"] != (clipped == 0)):
                raise ValueError("Numerical safety no-clipping status must match all member and row counts.")
            for name in ("raw_log_Cd_std_summary", "clipped_log_Cd_std_summary"):
                values = [numerical[name][key] for key in ("min", "mean", "max")]
                if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in values)
                        or not 0 <= values[0] <= values[1] <= values[2]):
                    raise ValueError("Numerical safety requires finite complete raw and clipped spread summaries.")
            for target in ("Cl", "Cd", "Cm"):
                value = ensemble["targets"][target]["r2"]
                if (ensemble["targets"][target]["r2_status"] != "defined" or isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not np.isfinite(value)):
                    raise ValueError("Numerical safety requires defined finite primary ensemble R2 for every target/fold.")
                result["checks"].append({"fold_id": fold, "target": target, "rule": "primary_r2_strictly_positive",
                                         "value": value, "passed": value > 0})
            result["checks"].append({"fold_id": fold, "rule": "no_member_pre_inverse_log_clipping",
                                     "n_member_values_clipped": clipped, "passed": clipped == 0})
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
        result["checks"] = []
        result["errors"].append(str(error))
        result["reasons"].append(str(error))
        return result
    result["reasons"] = [f"fold={row['fold_id']} {row['rule']} failed" + (f" for {row['target']}" if "target" in row else "")
                         for row in result["checks"] if not row["passed"]]
    result.update(status="fail" if result["reasons"] else "pass", passed=not result["reasons"])
    return result


def run_fold_study(split, feature_columns, target_names, output_dir, run_metadata=None, device="cuda"):
    """Run exactly 18 fresh Kaggle fits and return JSON-normalized fold_study.json.

    Audited Cdp sidecar labels must already be attached, with provenance in
    run_metadata['auxiliary'], outside the unchanged primary dataset mapping.
    Only canonical train enters model APIs. All three folds and every member
    are required; failures checkpoint status='error' and propagate immediately.
    No production models, pins, uploads, confirmation fits or holdout scores are
    produced. Explicit CPU still requires the existing Kaggle runtime guard.
    """
    return _run_fold_study(split, feature_columns, target_names, output_dir, run_metadata, device)


def run_input_scaling_fold_study(split, feature_columns, target_names, output_dir, run_metadata=None, device="cuda"):
    """Run the frozen 18-fit column/profile scaling comparison with Cdp in both arms.

    All evaluation rows are canonical train OOF rows. A new recommendation needs
    both the relative fold gate and numerical safety; there is no fallback family
    or production promotion. Raw fitting-space predictions are retained for replay.
    """
    return _run_fold_study(split, feature_columns, target_names, output_dir, run_metadata, device,
                           input_scaling="profile_rms_floor")


def run_selective_scaling_fold_study(split, feature_columns, target_names, output_dir, run_metadata=None, device="cuda"):
    """Run exactly 18 fresh fits comparing standard-Cdp with selective-floor Cdp.

    The candidate fixes input_scaling='profile_rms_floor_1pct': 0.01 times the RMS
    of the same 100 inner-fit/reflected profile SDs. No ratio or sweep is exposed.
    Everything else, including folds, seeds, Cdp weight and safety, is unchanged.
    """
    return _run_fold_study(split, feature_columns, target_names, output_dir, run_metadata, device,
                           input_scaling="profile_rms_floor_1pct")


def _run_fold_study(split, feature_columns, target_names, output_dir, run_metadata, device, *, input_scaling=None):
    if input_scaling is not None and (not isinstance(input_scaling, str)
            or input_scaling not in ("profile_rms_floor", "profile_rms_floor_1pct")):
        raise ValueError("Fold study requires a predeclared full-RMS or selective 1%-RMS mode.")
    output = study._require_kaggle(output_dir, device)
    arms = _INPUT_SCALING_ARMS if input_scaling else _ARMS
    scaling_modes = {**_INPUT_SCALING_MODES, _INPUT_SCALING_ARMS[1]: input_scaling}
    variant_metadata = {"study_variant": _SELECTIVE_STUDY_VARIANT} if input_scaling == "profile_rms_floor_1pct" else {}
    metadata = deepcopy({} if run_metadata is None else run_metadata)
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata must be a mapping with frozen split and auxiliary provenance.")
    canonical = study._validate_split(split, feature_columns, target_names, metadata)
    features, targets = list(feature_columns), list(target_names)
    if (len(features) != 113 or set(features) != set(RAW_FEATURE_NAMES)
            or any("cdp" in name.lower() or "observed" in name.lower() for name in features)):
        raise ValueError("Fold study requires the unchanged standard 113 primary features; no Cdp/mask inputs.")
    if not isinstance(metadata.get("auxiliary"), dict) or not metadata["auxiliary"]:
        raise ValueError("Audited auxiliary provenance is required outside the original dataset mapping.")
    frozen_hashes = metadata.get("split_row_sha256")
    if frozen_hashes is not None and (not isinstance(frozen_hashes, dict)
            or any(frozen_hashes.get(name) != value["row_sha256"] for name, value in canonical.items())):
        raise ValueError("Canonical row identity hashes must match frozen metadata.")
    # Holdout schema/identity inspection ends above. Do not extract holdout data,
    # even after selection; this study has no full-train confirmation phase.
    train = split.train
    missing = {"Cdp", "Cdp_observed", "max_thickness"}.difference(train.columns)
    if missing:
        raise ValueError(f"Canonical train lacks required auxiliary/stress columns: {sorted(missing)}.")
    positions, fold_plan, fold_rows = _geometry_folds(train)
    if input_scaling:
        fold_plan.update(study_kind=_INPUT_SCALING_STUDY_KIND, **variant_metadata)
    for fold_id, pos in positions.items():
        fit = train.iloc[pos["fit"]]
        if not np.isfinite(_auxiliary_values(fit)).any():
            raise ValueError(f"Fold {fold_id} fit requires observed Cdp labels.")
        _auxiliary_values(train.iloc[pos["stopping"]])
        coordinates = fit[[*study._ROW_KEYS, "max_thickness"]]
        fold_plan["folds"][fold_id]["stress_thresholds"] = derive_thresholds(
            coordinates, source="canonical_train_inner_fit",
        )
    del fit, coordinates
    # Helpers are pure NumPy, imported at runtime to keep orchestration importable
    # independently of cloud wiring and to avoid constructing an inference model.
    from src.models.ensemble import aggregate_predictions, disagreement_report
    from src.training.fold_gate import (
        POLICY as FOLD_POLICY, paired_fold_gate, paired_input_scaling_fold_gate, paired_selective_scaling_fold_gate,
    )

    if input_scaling:
        from src.training.numerical_safety import numerical_report

    defaults = TorchMLPRegressor().get_params(deep=False)
    defaults.pop("target_names")
    defaults.update(
        hidden_layers=(128, 128, 64), loss="huber", huber_delta=.05, weight_decay=1e-5,
        max_iter=300, patience=30, batch_size=4096, learning_rate_init=.001,
        target_weights=(1., 2., .5), symmetry=True, min_delta=1e-6,
        lr_patience=10, lr_factor=.5, device=device,
    )
    trials = {}
    for fold_id in range(3):
        for seed in SEEDS:
            for arm in arms:
                trial_id = f"fold_{fold_id}_seed_{seed}_{arm}"
                trials[trial_id] = {
                    "fold_id": fold_id, "seed": seed, "arm": arm, "status": "pending",
                    "config": {"name": "torch_mlp_huber", "scale_targets": False,
                                "parameters": {**defaults, "random_state": seed,
                                               "auxiliary_weight": .1 if input_scaling or arm == "cdp" else 0.,
                                                **({"input_scaling": scaling_modes[arm]} if input_scaling else {})}},
                    "manifest": f"trials/{trial_id}/run.json",
                }
    if input_scaling:
        # Parameter validation is fit-free and must precede output or any member fit.
        for trial in trials.values():
            TorchMLPRegressor(target_names=tuple(targets), **trial["config"]["parameters"])._validate_params()
    manifest = {
        "schema_version": 1, "status": "planned", "phase": "planned", "source": _SOURCE,
        "run_metadata": metadata, "hardware": metadata.get("gpu", {}), "device": device,
        "feature_columns": features, "target_names": targets, "target_transforms": DEFAULT_TARGET_TRANSFORMS.copy(),
        "canonical_split": canonical, "canonical_validation_test_used_for_model_apis": False,
        "plan": {
            "estimated_fits": 18, "n_folds": 3, "seeds": list(SEEDS), "arms": list(arms),
            "trial_order": list(trials), "trial_configs": {key: deepcopy(value["config"]) for key, value in trials.items()},
            "fold_plan": fold_plan, "optimizer": "RAdam", "stopping_metric": "primary_normalized_validation_loss",
            "scalers": "inner-fit only; auxiliary scaler uses observed inner-fit Cdp only",
            "auxiliary": {"target": "Cdp", "transform": "identity", "missing": "NaN", "weight": .1,
                          "input_feature": False, "stopping_labels": "diagnostic only; never model stopping"},
            "reference": "baseline_seed_42", "candidate_order": ["baseline_ensemble", "cdp_ensemble"],
            "selection_metric": "unweighted mean fold Cd median_relative_error among eligible families",
            "tie_break": "baseline_ensemble", "fallback": "baseline_seed_42",
            "aggregation": _METHOD, "gate_unit": "geometry_fold", "gate_policy": deepcopy(FOLD_POLICY),
            "fresh_fold_members": True,
            "seed_policy": "all three predeclared seeds; never choose best seed",
            "holdout_policy": "canonical validation/test identity and coverage checks only, throughout",
            "statistics": "ddof=1 descriptive seed-within-fold and fold variation separately; no confidence calibration",
            "predictions_ensembled_across_folds": False,
            "promotion": "development recommendation only; fresh full-train fitting and locked assessment required later",
        },
        "trials": trials, "completed_fits": 0, "completed_member_evaluations": 0,
        "completed_ensemble_evaluations": 0, "selection": None, "ensemble_reports": {},
        "artifacts": {"folds": "folds.json", "fold_rows": "fold_rows.csv", "members": "member_metrics.csv",
                      "ensembles": "ensemble_metrics.csv", "disagreement": "disagreement.json", "gates": "gates.json",
                      "summary": "summary.csv", "selection": "fold_selection.json"},
    }
    if input_scaling:
        manifest.update(study_kind=_INPUT_SCALING_STUDY_KIND, production_promotion=False,
                        numerical_reports={}, prediction_artifacts={}, pooled_oof=None, **variant_metadata)
        manifest["plan"].update(
            reference=f"{arms[0]}_ensemble", candidate_order=[f"{arms[1]}_ensemble"],
            selection_metric="candidate only when relative gate and numerical safety both pass",
            tie_break=None, fallback=None, controlled_parameter="input_scaling",
            input_scaling_modes=deepcopy(scaling_modes), **variant_metadata,
            numerical_safety="Every fold ensemble must have Cl/Cd/Cm R2 > 0; no raw pre-inverse-log clipping in any member. Missing evidence is inconclusive.",
            pooled_oof="Direct regression_metrics on all canonical-train rows aligned by original position; each row evaluated exactly once; never mean fold R2.",
            evaluation="One raw fitting-space predict call per saved member; unchanged inverse_targets bounds; all rows including OOD retained.",
            prediction_artifact_schema={
                "canonical_train_positions": "original positional indices in canonical train, evaluation row order",
                "member_seeds": "ordered evaluated seeds; partial artifacts contain only completed members",
                "fitting_predictions": "(n_members,n_rows,3), Cl/logCd/Cm before inverse-log clipping",
                "physical_predictions": "(n_members,n_rows,3), unchanged inverse_targets decoding",
                "physical_ensemble_predictions": "(n_rows,3), present only after all members and aggregation succeed",
                "lower_clipped": "(3,n_rows) bool, strict raw logCd < log(1e-6), complete artifacts only",
                "upper_clipped": "(3,n_rows) bool, strict raw logCd > log(1e3), complete artifacts only",
                "raw_log_Cd_std": "(n_rows,), raw fitting-space logCd member std with ddof=1",
                "clipped_log_Cd_std": "(n_rows,), clipped logCd member std with ddof=1, not calibrated uncertainty",
            },
        )
        if variant_metadata:
            manifest["plan"]["prediction_artifact_schema"].update(
                input_scaling="scalar Unicode string, exact arm mode", study_variant="scalar Unicode string, frozen variant",
            )
        manifest["artifacts"].update(numerical_reports="numerical_reports.json", numerical_safety="numerical_safety.json",
                                     pooled_oof="pooled_oof.json", pooled_oof_metrics="pooled_oof_metrics.csv",
                                     predictions="prediction_artifacts.json")
    json.dumps(manifest, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "fold_study.json", manifest)
    current_trial, current_fold = None, None

    def checkpoint():
        _write_json(output / "fold_study.json", manifest)

    member_rows, ensemble_rows, disagreements = [], [], {}
    references, candidates = {}, {arm: {} for arm in arms}
    numerical_reports, oof_predictions = {arm: {} for arm in arms}, {arm: {} for arm in arms}
    try:
        _write_json(output / "folds.json", fold_plan)
        study._write_csv(output / "fold_rows.csv", fold_rows)
        del fold_rows
        for current_fold in range(3):
            pos, fold = positions[current_fold], fold_plan["folds"][current_fold]
            manifest.update(status="running", phase="fold_fitting", fold_id=current_fold)
            checkpoint()
            fit, stopping = train.iloc[pos["fit"]], train.iloc[pos["stopping"]]
            fit_aux, stop_aux = _auxiliary_values(fit), _auxiliary_values(stopping)
            for seed in SEEDS:
                for arm in arms:
                    current_trial = f"fold_{current_fold}_seed_{seed}_{arm}"
                    trial = trials[current_trial]
                    trial["status"] = "running"
                    checkpoint()
                    label = f"[Fold {current_fold + 1}/3 | Fit {manifest['completed_fits'] + 1}/18] seed={seed} arm={arm}"
                    print(f"{label} starting", flush=True)
                    trial_dir = output / "trials" / current_trial
                    trial_metadata = {
                        **deepcopy(metadata), "fold_study_trial": current_trial, "fold_id": current_fold,
                        "study_role": "canonical_train_geometry_fold_development", "study_arm": arm,
                        "fit_split": {
                            "train_rows": len(fit), "validation_rows": len(stopping),
                            "train_row_sha256": fold["partitions"]["fit"]["row_sha256"],
                            "validation_row_sha256": fold["partitions"]["stopping"]["row_sha256"],
                            "train_geometry_sha256": fold["partitions"]["fit"]["geometry_sha256"],
                            "validation_geometry_sha256": fold["partitions"]["stopping"]["geometry_sha256"],
                            "validation_source": "canonical_train_inner_stopping", "fold_rows": "../../fold_rows.csv",
                        },
                        "auxiliary_coverage": {"train_observed": int(np.isfinite(fit_aux).sum()),
                                               "validation_observed": int(np.isfinite(stop_aux).sum()),
                                               "scaler_source": "canonical_train_inner_fit",
                                               "validation_use": "auxiliary diagnostics only"},
                    }
                    if input_scaling:
                        trial_metadata.update(study_kind=_INPUT_SCALING_STUDY_KIND,
                                              input_scaling=scaling_modes[arm], **variant_metadata)
                    compare_models(
                        fit[features], fit[targets], stopping[features], stopping[targets], targets, trial_dir,
                        model_configs=[deepcopy(trial["config"])], run_metadata=trial_metadata,
                        target_transforms=DEFAULT_TARGET_TRANSFORMS.copy(), selection_metrics={"Cd": "median_relative_error"},
                        auxiliary_train=fit_aux.copy() if input_scaling or arm == "cdp" else None,
                        auxiliary_validation=stop_aux.copy() if input_scaling or arm == "cdp" else None,
                    )
                    saved = json.loads((trial_dir / "run.json").read_text(encoding="utf-8"))
                    result = saved["results"][trial["config"]["name"]]
                    trial["result"] = result
                    if saved["status"] != "completed" or result["status"] != "success":
                        raise RuntimeError(f"Fold trial {current_trial} failed: {result.get('error', saved['status'])}")
                    model_path = (trial_dir / result["model_path"]).resolve()
                    if not model_path.is_relative_to(trial_dir.resolve()) or not model_path.is_file():
                        raise RuntimeError(f"Fold trial {current_trial} did not save model weights within its trial directory.")
                    trial.update(status="success", model_path=str(model_path.relative_to(output)))
                    manifest["completed_fits"] += 1
                    checkpoint()
                    training = result.get("training_metadata", {})
                    print(f"{label} finished device={training.get('accelerator', 'unreported')} "
                          f"epochs={training.get('n_iter', 'unreported')}", flush=True)
                    gc.collect()
            del fit, stopping, fit_aux, stop_aux
            current_trial = None
            manifest["phase"] = "fold_evaluation"
            checkpoint()
            evaluation = train.iloc[pos["evaluation"]]
            diagnostics_frame = evaluation[[*study._ROW_KEYS, "max_thickness", *targets]]
            disagreements[current_fold], manifest["ensemble_reports"][current_fold] = {}, {}
            for arm in arms:
                prediction_metadata = {"input_scaling": np.asarray(scaling_modes[arm]),
                                       "study_variant": np.asarray(_SELECTIVE_STUDY_VARIANT)} if variant_metadata else {}
                member_predictions = np.empty((3, len(evaluation), 3), dtype=float)
                raw_predictions = np.empty_like(member_predictions) if input_scaling else None
                directory = output / "folds" / f"fold_{current_fold}"
                if input_scaling:
                    directory.mkdir(parents=True, exist_ok=True)
                    prediction_path = f"folds/fold_{current_fold}/{arm}_predictions.npz"
                for member_index, seed in enumerate(SEEDS):
                    current_trial = f"fold_{current_fold}_seed_{seed}_{arm}"
                    trial = trials[current_trial]
                    model = scored = predictions = None
                    try:
                        model = joblib.load(output / trial["model_path"])
                        if not input_scaling:
                            scored = PhysicalPredictor(model, targets, DEFAULT_TARGET_TRANSFORMS)
                        started = time.perf_counter()
                        if input_scaling:
                            raw = np.asarray(model.predict(evaluation[features]))
                            if (raw.shape != (len(evaluation), 3) or raw.dtype.kind not in "iuf"
                                    or not np.isfinite(raw).all()):
                                raise ValueError("Every input-scaling fold member must return three finite raw fitting-space targets for all evaluation rows.")
                            raw_predictions[member_index] = raw
                            predictions = inverse_targets(raw, targets, DEFAULT_TARGET_TRANSFORMS)
                            del raw
                        else:
                            predictions = scored.predict(evaluation[features])
                        elapsed = time.perf_counter() - started
                        if (np.shape(predictions) != (len(evaluation), 3) or not np.isfinite(predictions).all()
                                or (predictions[:, 1] <= 0).any()):
                            raise ValueError("Every fold member must predict all evaluation rows and exactly three finite physical targets.")
                        member_predictions[member_index] = predictions
                        if input_scaling:
                            _write_predictions(output / prediction_path,
                                               canonical_train_positions=pos["evaluation"],
                                                member_seeds=np.asarray(SEEDS[:member_index + 1]),
                                                fitting_predictions=raw_predictions[:member_index + 1],
                                                physical_predictions=member_predictions[:member_index + 1], **prediction_metadata)
                            manifest["prediction_artifacts"].setdefault(current_fold, {})[arm] = {
                                "path": prediction_path, "status": "partial", "n_members": member_index + 1,
                                "n_rows": len(evaluation), "ensemble_present": False,
                                **({"input_scaling": scaling_modes[arm], **variant_metadata} if variant_metadata else {}),
                            }
                            _write_json(output / "prediction_artifacts.json", manifest["prediction_artifacts"])
                        report = tail_report(evaluation, predictions, targets, thresholds=fold["stress_thresholds"])
                        predictor = f"{arm}_seed_{seed}"
                        report.update(fold_id=current_fold, source=_SOURCE, predictor=predictor,
                                      member_seeds=[seed], seed=seed, arm=arm, trial_id=current_trial)
                        if input_scaling:
                            report.update(study_kind=_INPUT_SCALING_STUDY_KIND, input_scaling=scaling_modes[arm], **variant_metadata)
                            _mark_r2_definition(report, evaluation[targets].to_numpy(dtype=float))
                        provenance = fold["partitions"]["evaluation"]
                        report.update(geometry_sha256=provenance["geometry_sha256"], fold_rows="../../fold_rows.csv")
                        path = f"trials/{current_trial}/fold_tail_report.json"
                        _write_json(output / path, report)
                        rows = _metric_rows(report, fold_id=current_fold, seed=seed, arm=arm, predictor=predictor,
                                            trial_id=current_trial, batch_inference_time_seconds=elapsed)
                        study._write_csv(output / "trials" / current_trial / "fold_metrics.csv", rows)
                        trial.update(tail_report=path, evaluation_status="success")
                        member_rows.extend(rows)
                        study._write_csv(output / "member_metrics.csv", member_rows)
                        manifest["completed_member_evaluations"] += 1
                        if arm == "baseline" and seed == 42:
                            references[current_fold] = report
                        checkpoint()
                    finally:
                        model = scored = predictions = None
                        gc.collect()
                current_trial = None
                result = aggregate_predictions(member_predictions)
                if (result["method"] != _METHOD or result["member_seeds"] != list(SEEDS)
                        or np.shape(result["prediction"]) != (len(evaluation), 3)
                        or np.shape(result["disagreement"]) != (len(evaluation), 3)):
                    raise ValueError("Fold ensembles require all three ordered seeds and the predeclared physical aggregation.")
                report = tail_report(evaluation, result["prediction"], targets, thresholds=fold["stress_thresholds"])
                predictor = f"{arm}_ensemble"
                report.update(fold_id=current_fold, source=_SOURCE, predictor=predictor, arm=arm,
                              member_seeds=list(SEEDS), aggregation=result["method"],
                              geometry_sha256=fold["partitions"]["evaluation"]["geometry_sha256"],
                              fold_rows="../../fold_rows.csv")
                if input_scaling:
                    report.update(study_kind=_INPUT_SCALING_STUDY_KIND, input_scaling=scaling_modes[arm], **variant_metadata)
                    _mark_r2_definition(report, evaluation[targets].to_numpy(dtype=float))
                directory.mkdir(parents=True, exist_ok=True)
                path = f"folds/fold_{current_fold}/{arm}_ensemble_tail_report.json"
                _write_json(output / path, report)
                diagnostics = disagreement_report(diagnostics_frame, result)
                diagnostics.update(fold_id=current_fold, source=_SOURCE, predictor=predictor,
                                   member_seeds=list(SEEDS), aggregation=result["method"])
                if input_scaling:
                    diagnostics.update(study_kind=_INPUT_SCALING_STUDY_KIND, arm=arm, input_scaling=scaling_modes[arm], **variant_metadata)
                disagreements[current_fold][arm] = diagnostics
                _write_json(directory / f"{arm}_disagreement.json", diagnostics)
                _write_json(output / "disagreement.json", disagreements)
                candidates[arm][current_fold] = report
                if input_scaling:
                    if arm == arms[0]:
                        references[current_fold] = report
                    numerical, numerical_arrays = numerical_report(evaluation, raw_predictions)
                    numerical.update(fold_id=current_fold, source=_SOURCE, predictor=predictor, arm=arm,
                                     study_kind=_INPUT_SCALING_STUDY_KIND, input_scaling=scaling_modes[arm], **variant_metadata)
                    numerical_reports[arm][current_fold] = numerical
                    numerical_path = f"folds/fold_{current_fold}/{arm}_numerical_report.json"
                    _write_json(output / numerical_path, numerical)
                    _write_json(output / "numerical_reports.json", numerical_reports)
                    manifest["numerical_reports"].setdefault(current_fold, {})[arm] = numerical_path
                    _write_predictions(output / prediction_path,
                                       canonical_train_positions=pos["evaluation"], member_seeds=np.asarray(SEEDS),
                                        physical_ensemble_predictions=result["prediction"],
                                        **numerical_arrays, **prediction_metadata)
                    manifest["prediction_artifacts"][current_fold][arm].update(status="complete", ensemble_present=True)
                    _write_json(output / "prediction_artifacts.json", manifest["prediction_artifacts"])
                    oof_predictions[arm][current_fold] = (pos["evaluation"].copy(), result["prediction"].copy())
                rows = _metric_rows(report, fold_id=current_fold, arm=arm, predictor=predictor,
                                    aggregation=result["method"], member_seeds="42,43,44")
                ensemble_rows.extend(rows)
                study._write_csv(directory / f"{arm}_ensemble_metrics.csv", rows)
                study._write_csv(output / "ensemble_metrics.csv", ensemble_rows)
                manifest["ensemble_reports"][current_fold][arm] = path
                manifest["completed_ensemble_evaluations"] += 1
                checkpoint()
                del member_predictions, raw_predictions, result
            del evaluation, diagnostics_frame
            print(f"[Fold {current_fold + 1}/3] six fresh members and both ensembles evaluated", flush=True)
            gc.collect()

        current_trial = None
        if (manifest["completed_fits"] != 18 or manifest["completed_member_evaluations"] != 18
                or manifest["completed_ensemble_evaluations"] != 6):
            raise RuntimeError("All 18 fits and every fold/member/ensemble evaluation must finish before selection.")
        manifest["phase"] = "fold_selection"
        checkpoint()
        gate_function = paired_input_scaling_fold_gate if input_scaling else paired_fold_gate
        if variant_metadata:
            gate_function = paired_selective_scaling_fold_gate
        gated_arms = arms[1:] if input_scaling else arms
        gates = {f"{arm}_ensemble": gate_function(references, candidates[arm], expected_geometry_ids=fold_plan["geometry_ids"])
                 for arm in gated_arms}
        _write_json(output / "gates.json", gates)
        family_scores = {}
        for arm in arms:
            values = [candidates[arm][fold_id]["targets"]["Cd"]["median_relative_error"] for fold_id in range(3)]
            finite = all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in values)
            family_scores[f"{arm}_ensemble"] = {
                "mean_fold_Cd_median_relative_error": float(np.mean(values)) if finite else None,
                "n_folds": 3, "weighting": "unweighted folds", "predictions_ensembled_across_folds": False,
            }
        eligible = [name for name in manifest["plan"]["candidate_order"]
                    if gates[name]["eligible"] and family_scores[name]["mean_fold_Cd_median_relative_error"] is not None]
        selected = min(eligible, key=lambda name: family_scores[name]["mean_fold_Cd_median_relative_error"]) if eligible else "baseline_seed_42"
        selection = {
            "schema_version": 1, "status": "frozen", "source": _SOURCE, "selected_family": selected,
            "reference": "baseline_seed_42", "candidate_order": manifest["plan"]["candidate_order"],
            "family_scores": family_scores, "gates": gates, "gate_unit": "geometry_fold", "n_folds": 3,
            "seed_selection": False, "member_seeds": [42] if selected == "baseline_seed_42" else list(SEEDS),
            "canonical_validation_scored": False, "canonical_test_scored": False,
            "completed_fits_before_selection": 18, "promotion": manifest["plan"]["promotion"],
            "fallback_reason": None if eligible else "No eligible ensemble family; failed/inconclusive gates retain predeclared baseline_seed_42.",
            "confidence_calibrated": False, "old_test_improvement_claim": False,
        }
        if input_scaling:
            safety = {f"{arm}_ensemble": _input_scaling_safety(candidates[arm], numerical_reports[arm]) for arm in arms}
            if variant_metadata:
                for arm in arms:
                    safety[f"{arm}_ensemble"].update(input_scaling=scaling_modes[arm], **variant_metadata)
            _write_json(output / "numerical_safety.json", safety)
            pooled = {f"{arm}_ensemble": _pooled_oof_scores(train, targets, oof_predictions[arm]) for arm in arms}
            if variant_metadata:
                for arm in arms:
                    pooled[f"{arm}_ensemble"].update(predictor=f"{arm}_ensemble", input_scaling=scaling_modes[arm], **variant_metadata)
            _write_json(output / "pooled_oof.json", pooled)
            study._write_csv(output / "pooled_oof_metrics.csv", [
                {"predictor": family, "source": _SOURCE, "study_kind": _INPUT_SCALING_STUDY_KIND,
                  "target": target, "n_predictions": report["n_rows"], "row_sha256": report["row_sha256"],
                  "score_scope": "pooled_oof_all_canonical_train_rows", **scores,
                  **({"input_scaling": report["input_scaling"], **variant_metadata} if variant_metadata else {})}
                for family, report in pooled.items() for target, scores in report["targets"].items()
            ])
            manifest["pooled_oof"] = pooled
            candidate_family = f"{arms[1]}_ensemble"
            selected = candidate_family if gates[candidate_family]["passed"] and safety[candidate_family]["passed"] else None
            selection.update(
                study_kind=_INPUT_SCALING_STUDY_KIND, selected_family=selected, reference=f"{arms[0]}_ensemble",
                member_seeds=list(SEEDS) if selected else [], numerical_safety=safety,
                relative_gate_passed=gates[candidate_family]["passed"], candidate_numerically_safe=safety[candidate_family]["passed"],
                reference_numerically_safe=safety[f"{arms[0]}_ensemble"]["passed"],
                production_promotion=False, new_recommendation=selected is not None, fallback_family=None,
                fallback_reason=None if selected else "No new recommendation: candidate relative gate or numerical safety failed/inconclusive; reference safety is reported separately, not assumed.",
                **variant_metadata,
            )
            if variant_metadata:
                selection["input_scaling_modes"] = deepcopy(scaling_modes)
        # Fresh output is required and this decision file is written exactly once.
        _write_json(output / "fold_selection.json", selection)
        manifest["selection"] = selection
        checkpoint()
        study._write_csv(output / "summary.csv", _descriptive_summary(member_rows, ensemble_rows))
        manifest.update(status="completed", phase="completed")
        checkpoint()
        print(f"Fold study completed: 18/18 fits; development recommendation={selected}; "
              "fresh full-train fitting and locked assessment required; no confidence calibration or old-test gain claim.", flush=True)
        return json.loads(json.dumps(manifest, allow_nan=False))
    except BaseException as error:
        if current_trial is not None:
            trial = trials[current_trial]
            if trial["status"] == "running":
                trial["status"] = "error"
            else:
                trial["evaluation_status"] = "error"
        manifest.update(status="error", error={"type": type(error).__name__, "message": str(error),
                                               "trial_id": current_trial, "fold_id": current_fold})
        checkpoint()
        raise
