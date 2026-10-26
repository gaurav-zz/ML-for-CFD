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
from src.training.targets import DEFAULT_TARGET_TRANSFORMS, PhysicalPredictor


_PARTITION_SEED = 20261026
_ARMS = ("baseline", "cdp")
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
    return summaries


def run_fold_study(split, feature_columns, target_names, output_dir, run_metadata=None, device="cuda"):
    """Run exactly 18 fresh Kaggle fits and return JSON-normalized fold_study.json.

    Audited Cdp sidecar labels must already be attached, with provenance in
    run_metadata['auxiliary'], outside the unchanged primary dataset mapping.
    Only canonical train enters model APIs. All three folds and every member
    are required; failures checkpoint status='error' and propagate immediately.
    No production models, pins, uploads, confirmation fits or holdout scores are
    produced. Explicit CPU still requires the existing Kaggle runtime guard.
    """
    output = study._require_kaggle(output_dir, device)
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
    from src.training.fold_gate import POLICY as FOLD_POLICY, paired_fold_gate

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
            for arm in _ARMS:
                trial_id = f"fold_{fold_id}_seed_{seed}_{arm}"
                trials[trial_id] = {
                    "fold_id": fold_id, "seed": seed, "arm": arm, "status": "pending",
                    "config": {"name": "torch_mlp_huber", "scale_targets": False,
                               "parameters": {**defaults, "random_state": seed, "auxiliary_weight": .1 if arm == "cdp" else 0.}},
                    "manifest": f"trials/{trial_id}/run.json",
                }
    manifest = {
        "schema_version": 1, "status": "planned", "phase": "planned", "source": _SOURCE,
        "run_metadata": metadata, "hardware": metadata.get("gpu", {}), "device": device,
        "feature_columns": features, "target_names": targets, "target_transforms": DEFAULT_TARGET_TRANSFORMS.copy(),
        "canonical_split": canonical, "canonical_validation_test_used_for_model_apis": False,
        "plan": {
            "estimated_fits": 18, "n_folds": 3, "seeds": list(SEEDS), "arms": list(_ARMS),
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
    json.dumps(manifest, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "fold_study.json", manifest)
    current_trial, current_fold = None, None

    def checkpoint():
        _write_json(output / "fold_study.json", manifest)

    member_rows, ensemble_rows, disagreements = [], [], {}
    references, candidates = {}, {arm: {} for arm in _ARMS}
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
                for arm in _ARMS:
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
                    compare_models(
                        fit[features], fit[targets], stopping[features], stopping[targets], targets, trial_dir,
                        model_configs=[deepcopy(trial["config"])], run_metadata=trial_metadata,
                        target_transforms=DEFAULT_TARGET_TRANSFORMS.copy(), selection_metrics={"Cd": "median_relative_error"},
                        auxiliary_train=fit_aux.copy() if arm == "cdp" else None,
                        auxiliary_validation=stop_aux.copy() if arm == "cdp" else None,
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
            for arm in _ARMS:
                member_predictions = np.empty((3, len(evaluation), 3), dtype=float)
                for member_index, seed in enumerate(SEEDS):
                    current_trial = f"fold_{current_fold}_seed_{seed}_{arm}"
                    trial = trials[current_trial]
                    model = scored = predictions = None
                    try:
                        model = joblib.load(output / trial["model_path"])
                        scored = PhysicalPredictor(model, targets, DEFAULT_TARGET_TRANSFORMS)
                        started = time.perf_counter()
                        predictions = scored.predict(evaluation[features])
                        elapsed = time.perf_counter() - started
                        if (np.shape(predictions) != (len(evaluation), 3) or not np.isfinite(predictions).all()
                                or (predictions[:, 1] <= 0).any()):
                            raise ValueError("Every fold member must predict all evaluation rows and exactly three finite physical targets.")
                        member_predictions[member_index] = predictions
                        report = tail_report(evaluation, predictions, targets, thresholds=fold["stress_thresholds"])
                        predictor = f"{arm}_seed_{seed}"
                        report.update(fold_id=current_fold, source=_SOURCE, predictor=predictor,
                                      member_seeds=[seed], seed=seed, arm=arm, trial_id=current_trial)
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
                directory = output / "folds" / f"fold_{current_fold}"
                directory.mkdir(parents=True, exist_ok=True)
                path = f"folds/fold_{current_fold}/{arm}_ensemble_tail_report.json"
                _write_json(output / path, report)
                diagnostics = disagreement_report(diagnostics_frame, result)
                diagnostics.update(fold_id=current_fold, source=_SOURCE, predictor=predictor,
                                   member_seeds=list(SEEDS), aggregation=result["method"])
                disagreements[current_fold][arm] = diagnostics
                _write_json(directory / f"{arm}_disagreement.json", diagnostics)
                _write_json(output / "disagreement.json", disagreements)
                candidates[arm][current_fold] = report
                rows = _metric_rows(report, fold_id=current_fold, arm=arm, predictor=predictor,
                                    aggregation=result["method"], member_seeds="42,43,44")
                ensemble_rows.extend(rows)
                study._write_csv(directory / f"{arm}_ensemble_metrics.csv", rows)
                study._write_csv(output / "ensemble_metrics.csv", ensemble_rows)
                manifest["ensemble_reports"][current_fold][arm] = path
                manifest["completed_ensemble_evaluations"] += 1
                checkpoint()
                del member_predictions, result
            del evaluation, diagnostics_frame
            print(f"[Fold {current_fold + 1}/3] six fresh members and both ensembles evaluated", flush=True)
            gc.collect()

        current_trial = None
        if (manifest["completed_fits"] != 18 or manifest["completed_member_evaluations"] != 18
                or manifest["completed_ensemble_evaluations"] != 6):
            raise RuntimeError("All 18 fits and every fold/member/ensemble evaluation must finish before selection.")
        manifest["phase"] = "fold_selection"
        checkpoint()
        gates = {f"{arm}_ensemble": paired_fold_gate(references, candidates[arm], expected_geometry_ids=fold_plan["geometry_ids"])
                 for arm in _ARMS}
        _write_json(output / "gates.json", gates)
        family_scores = {}
        for arm in _ARMS:
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
