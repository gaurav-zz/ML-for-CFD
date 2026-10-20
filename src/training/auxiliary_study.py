"""Kaggle-only Cdp auxiliary experiment with a train-only, immutable decision."""

from __future__ import annotations

from copy import deepcopy
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.training import study
from src.training.comparison import SCORE_COLUMNS, _write_json, compare_models
from src.training.tail_risk import POLICY, SEEDS, TAIL_METRICS, derive_thresholds, paired_tail_gate, tail_report
from src.training.targets import DEFAULT_TARGET_TRANSFORMS, PhysicalPredictor


_PARTITION_SEED = 20261020
_ARMS = ("baseline", "cdp")


def _inner_partition(train):
    """Identity-only geometry partition shared by every arm and model seed."""
    groups = np.random.default_rng(_PARTITION_SEED).permutation(sorted(train.geometry_id.unique()))
    n_stop = n_evaluation = max(1, round(.1 * len(groups)))
    if n_stop + n_evaluation >= len(groups):
        raise ValueError("Auxiliary study requires at least three canonical training geometries.")
    selected = {
        "stopping": groups[:n_stop],
        "evaluation": groups[n_stop:n_stop + n_evaluation],
        "fit": groups[n_stop + n_evaluation:],
    }
    identity = train[study._ROW_KEYS].reset_index(drop=True).sort_values(study._ROW_KEYS)
    positions, partitions, records = {}, {}, []
    for name in ("fit", "stopping", "evaluation"):
        positions[name] = identity.index[identity.geometry_id.isin(selected[name])].to_numpy()
        rows = train[study._ROW_KEYS].iloc[positions[name]]
        partitions[name] = {
            "geometry_ids": sorted(selected[name]), "n_rows": len(rows),
            "n_geometries": len(selected[name]), "row_sha256": study._row_hash(rows),
        }
        recorded = rows.copy()
        recorded.insert(0, "canonical_row_index", rows.index)
        recorded.insert(0, "canonical_train_position", positions[name])
        recorded.insert(0, "partition", name)
        records.append(recorded)
    return positions, {
        "source": "canonical_train_only", "partition_seed": _PARTITION_SEED,
        "geometry_fractions": {"fit": .8, "stopping": .1, "evaluation": .1},
        "rounding": "max(1, round(0.1*n_geometries)) for stopping and evaluation; fit is remainder",
        "algorithm": "seeded permutation of sorted geometry IDs; rows sorted by operating-point identity",
        "row_indices": "inner_rows.csv", "row_identity_columns": study._ROW_KEYS.copy(),
        "hash_algorithm": "sha256(sorted pandas hash_pandas_object(identity, index=False), little-endian uint64)",
        "partitions": partitions,
    }, pd.concat(records, ignore_index=True)


def _auxiliary_values(frame):
    observed = frame["Cdp_observed"]
    if not pd.api.types.is_bool_dtype(observed.dtype) or observed.isna().any():
        raise ValueError("Cdp_observed must be a nonmissing boolean sidecar label.")
    if (not pd.api.types.is_numeric_dtype(frame["Cdp"].dtype)
            or pd.api.types.is_bool_dtype(frame["Cdp"].dtype) or pd.api.types.is_complex_dtype(frame["Cdp"].dtype)):
        raise ValueError("Cdp must be real numeric, with NaN exactly where unobserved.")
    values = frame["Cdp"].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or not np.array_equal(np.isfinite(values), observed.to_numpy(dtype=bool)):
        raise ValueError("Cdp must be finite exactly where Cdp_observed is true; missing values must be NaN.")
    return values


def run_auxiliary_study(split, feature_columns, target_names, output_dir, run_metadata=None, device="cuda"):
    """Run exactly six development fits, freeze the arm, then six confirmations.

    Input frames already carry audited ``Cdp``/``Cdp_observed`` sidecar labels.
    ``run_metadata`` must retain the full frozen split and a nonempty auxiliary
    provenance mapping outside the unchanged primary ``dataset`` mapping.
    Canonical validation/test features and labels are not accessed until the
    decision is persisted. All twelve successful trials are required, even when
    the Cdp gate fails. Failed fits/evaluations fail fast with saved evidence.

    Returns the JSON-normalized final ``auxiliary_study.json`` manifest. No seed
    is selected, production defaults/pins are not changed, and test-tail gates
    are audit-only. No Torch import occurs except the existing Kaggle CUDA guard
    and the actual training implementation, both called only at runtime.
    """
    output = study._require_kaggle(output_dir, device)
    metadata = deepcopy({} if run_metadata is None else run_metadata)
    if not isinstance(metadata, dict):
        raise ValueError("run_metadata must be a mapping of frozen split and dataset provenance.")
    canonical = study._validate_split(split, feature_columns, target_names, metadata)
    features, targets = list(feature_columns), list(target_names)
    if any("cdp" in name.lower() or "observed" in name.lower() for name in features):
        raise ValueError("Auxiliary Cdp and observed-label masks are forbidden input features.")
    from src.models.torch_mlp import RAW_FEATURE_NAMES, TorchMLPRegressor

    if len(features) != 113 or set(features) != set(RAW_FEATURE_NAMES):
        raise ValueError("The auxiliary study requires the unchanged standard 113 primary features.")
    if not isinstance(metadata.get("auxiliary"), dict) or not metadata["auxiliary"]:
        raise ValueError("run_metadata requires audited auxiliary dataset provenance outside dataset.")
    for name in ("train", "validation", "test"):
        frame = getattr(split, name)
        missing = {"Cdp", "Cdp_observed", "max_thickness"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} lacks required auxiliary/stress columns: {sorted(missing)}.")
        frozen_hashes = metadata.get("split_row_sha256")
        if frozen_hashes is not None and (not isinstance(frozen_hashes, dict)
                                          or frozen_hashes.get(name) != canonical[name]["row_sha256"]):
            raise ValueError(f"{name} row identity fingerprint does not match frozen metadata.")
    auxiliary = _auxiliary_values(split.train)
    positions, inner, inner_rows = _inner_partition(split.train)
    if not np.isfinite(auxiliary[positions["fit"]]).any():
        raise ValueError("Inner training fit requires observed Cdp labels.")
    del auxiliary
    coordinates = split.train[[*study._ROW_KEYS, "max_thickness"]]
    thresholds = {
        "development": derive_thresholds(coordinates.iloc[positions["fit"]]),
        "confirmation": derive_thresholds(coordinates, source="canonical_train"),
    }
    del coordinates
    defaults = TorchMLPRegressor().get_params(deep=False)
    defaults.pop("target_names")
    defaults.update(
        hidden_layers=(128, 128, 64), loss="huber", huber_delta=.05, weight_decay=1e-5,
        max_iter=300, patience=30, batch_size=4096, learning_rate_init=.001,
        target_weights=(1., 2., .5), symmetry=True, min_delta=1e-6,
        lr_patience=10, lr_factor=.5, device=device,
    )
    trials = {}
    for phase in ("development", "confirmation"):
        for seed in SEEDS:
            for arm in _ARMS:
                trial_id = f"{phase}_seed_{seed}_{arm}"
                trials[trial_id] = {
                    "phase": phase, "seed": seed, "arm": arm, "status": "pending",
                    "config": {"name": "torch_mlp_huber", "scale_targets": False,
                               "parameters": {**defaults, "random_state": seed, "auxiliary_weight": .1 if arm == "cdp" else 0.}},
                    "manifest": f"trials/{trial_id}/run.json",
                }
    manifest = {
        "schema_version": 1, "status": "planned", "phase": "planned",
        "run_metadata": metadata, "hardware": metadata.get("gpu", {}), "device": device,
        "feature_columns": features, "target_names": targets,
        "target_transforms": DEFAULT_TARGET_TRANSFORMS.copy(), "canonical_split": canonical,
        "plan": {
            "estimated_fits": 12, "development_fits": 6, "confirmation_fits": 6,
            "seeds": list(SEEDS), "arms": list(_ARMS), "trial_order": list(trials),
            "trial_configs": {key: deepcopy(value["config"]) for key, value in trials.items()},
            "inner_split": inner, "stress_thresholds": thresholds, "gate_policy": deepcopy(POLICY),
            "auxiliary": {"target": "Cdp", "transform": "identity", "missing": "NaN",
                          "weight": .1, "weight_source": "fixed experiment choice, not NASA-reported",
                          "input_feature": False, "public_targets": targets},
            "optimizer": "RAdam", "stopping_metric": "primary_normalized_validation_loss",
            "scalers": "training-only; auxiliary scaler uses observed training Cdp only",
            "development_selection_source": "canonical_train_inner_evaluation",
            "confirmation_stopping_source": "canonical_validation",
            "seed_policy": "all three paired seeds; no best-seed selection",
            "test_policy": "score all six confirmations after freezing selection and all confirmation fits",
            "statistics": "descriptive mean and sample std (ddof=1); no reduced-seed or NaN averages",
            "paired_delta": "cdp_minus_baseline; negative error deltas favor cdp",
            "promotion": "not automatically promoted; production default and pinned baseline unchanged",
        },
        "trials": trials, "selection": None,
        "artifacts": {
            "selection": "auxiliary_selection.json", "inner_rows": "inner_rows.csv",
            "development_metrics": "development_metrics.csv", "confirmation_metrics": "confirmation_metrics.csv",
            "paired_deltas": "paired_deltas.csv", "summary": "summary.csv", "tail_audit": "tail_audit.json",
        },
    }
    json.dumps(manifest, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "auxiliary_study.json", manifest)
    current_trial = None

    def checkpoint():
        _write_json(output / "auxiliary_study.json", manifest)

    def fit_trial(trial_id, train, stopping):
        trial = trials[trial_id]
        trial["status"] = "running"
        checkpoint()
        trial_dir = output / "trials" / trial_id
        train_auxiliary = _auxiliary_values(train)
        stopping_auxiliary = _auxiliary_values(stopping)
        trial_metadata = {
            **deepcopy(metadata), "auxiliary_study_trial": trial_id, "study_role": trial["phase"],
            "study_arm": trial["arm"],
            "fit_split": {
                "train_rows": len(train), "validation_rows": len(stopping),
                "train_row_sha256": study._row_hash(train), "validation_row_sha256": study._row_hash(stopping),
                "validation_source": "canonical_train_inner_stopping" if trial["phase"] == "development" else "canonical_validation",
            },
            "auxiliary_coverage": {"train_observed": int(np.isfinite(train_auxiliary).sum()),
                                   "validation_observed": int(np.isfinite(stopping_auxiliary).sum())},
        }
        if trial["phase"] == "confirmation":
            trial_metadata["auxiliary_selection"] = {"path": "../../auxiliary_selection.json",
                                                     "selected_arm": manifest["selection"]["selected_arm"],
                                                     "source": "canonical_train_inner_evaluation", "seed_selection": False}
        try:
            compare_models(
                train[features], train[targets], stopping[features], stopping[targets], targets, trial_dir,
                model_configs=[deepcopy(trial["config"])], run_metadata=trial_metadata,
                target_transforms=DEFAULT_TARGET_TRANSFORMS, selection_metrics={"Cd": "median_relative_error"},
                auxiliary_train=train_auxiliary if trial["arm"] == "cdp" else None,
                auxiliary_validation=stopping_auxiliary if trial["arm"] == "cdp" else None,
            )
            saved = json.loads((trial_dir / "run.json").read_text(encoding="utf-8"))
            result = saved["results"][trial["config"]["name"]]
            trial["result"] = result
            if saved["status"] != "completed" or result["status"] != "success":
                raise RuntimeError(f"Auxiliary study trial {trial_id} failed: {result.get('error', saved['status'])}")
            model_path = trial_dir / result["model_path"]
            if not model_path.is_file():
                raise RuntimeError(f"Auxiliary study trial {trial_id} did not save model weights.")
            trial.update(status="success", model_path=str(model_path.relative_to(output)))
            checkpoint()
        finally:
            train_auxiliary = stopping_auxiliary = None
            gc.collect()

    def score_trial(trial_id, frame):
        trial = trials[trial_id]
        phase = trial["phase"]
        source = "canonical_train_inner_evaluation" if phase == "development" else "canonical_test"
        trial_dir = output / "trials" / trial_id
        model = scored = predictions = None
        try:
            model = joblib.load(output / trial["model_path"])
            scored = PhysicalPredictor(model, targets, DEFAULT_TARGET_TRANSFORMS)
            started = time.perf_counter()
            predictions = scored.predict(frame[features])
            elapsed = time.perf_counter() - started
            report = tail_report(frame, predictions, targets, thresholds=thresholds[phase])
            report.update(trial_id=trial_id, seed=trial["seed"], arm=trial["arm"], source=source)
            report_name = "dev_report.json" if phase == "development" else "test_tail_report.json"
            _write_json(trial_dir / report_name, report)
            trial["tail_report"] = str((trial_dir / report_name).relative_to(output))
            records = [
                {"model": trial["config"]["name"], "target": target, **report["targets"][target],
                 "batch_inference_time_seconds": elapsed, "n_predictions": len(frame)}
                for target in targets
            ]
            if phase == "confirmation":
                saved = json.loads((trial_dir / "run.json").read_text(encoding="utf-8"))
                # Same contract as evaluate_selected_models, without predicting a
                # second time or selecting any model on test performance.
                payload = {key: saved[key] for key in (
                    "selection_metric", "selection_metrics", "selected_models", "target_names", "target_transforms",
                )}
                payload["results"] = records
                _write_json(trial_dir / "test_evaluation.json", payload)
                trial["test_evaluation"] = str((trial_dir / "test_evaluation.json").relative_to(output))
            study._write_csv(trial_dir / ("dev_metrics.csv" if phase == "development" else "test_metrics.csv"), records)
            rows = []
            for record in records:
                row = {"trial_id": trial_id, "phase": phase, "partition": source,
                       "seed": trial["seed"], "arm": trial["arm"], **record}
                if record["target"] == "Cd":
                    row.update(report["cd_tail"])
                    row.update({f"{name}_rmse": report["stress"][name]["Cd_rmse"] for name in ("low_re", "thick")})
                rows.append(row)
            checkpoint()
            return report, rows
        finally:
            model = scored = predictions = None
            gc.collect()

    try:
        study._write_csv(output / "inner_rows.csv", inner_rows)
        del inner_rows
        reports = {phase: {arm: {} for arm in _ARMS} for phase in ("development", "confirmation")}
        all_rows = []
        for phase in ("development", "confirmation"):
            manifest.update(status="running", phase=phase)
            checkpoint()
            phase_rows = []
            if phase == "development":
                train = split.train.iloc[positions["fit"]]
                stopping = split.train.iloc[positions["stopping"]]
                evaluation = split.train.iloc[positions["evaluation"]]
            else:
                train, stopping = split.train, split.validation
            phase_trials = [key for key, value in trials.items() if value["phase"] == phase]
            for index, current_trial in enumerate(phase_trials, start=1):
                trial = trials[current_trial]
                print(f"[{phase.title()} {index}/6] seed={trial['seed']} arm={trial['arm']} starting", flush=True)
                fit_trial(current_trial, train, stopping)
                if phase == "development":
                    report, rows = score_trial(current_trial, evaluation)
                    reports[phase][trial["arm"]][trial["seed"]] = report
                    phase_rows.extend(rows)
                    study._write_csv(output / "development_metrics.csv", phase_rows)
                training = trial["result"].get("training_metadata", {})
                print(f"[{phase.title()} {index}/6] seed={trial['seed']} arm={trial['arm']} finished "
                      f"device={training.get('accelerator', 'unreported')} epochs={training.get('n_iter', 'unreported')}", flush=True)
            current_trial = None
            del train, stopping
            if phase == "development":
                del evaluation
                gate = paired_tail_gate(reports[phase]["baseline"], reports[phase]["cdp"])
                selection = {
                    "schema_version": 1, "status": "frozen", "source": "canonical_train_inner_evaluation",
                    "selected_arm": "cdp" if gate["eligible"] else "baseline", "gate": gate,
                    "seed_selection": False, "seeds": list(SEEDS), "canonical_validation_scored": False,
                    "canonical_test_scored": False, "promotion": "not automatically promoted",
                    "fallback_reason": None if gate["eligible"] else "Cdp failed or had inconclusive development gates; selected baseline.",
                }
                # Written once, before any canonical validation/test features or
                # labels enter an API. Neither test audits nor failures rewrite it.
                _write_json(output / "auxiliary_selection.json", selection)
                manifest["selection"] = selection
                checkpoint()
            else:
                manifest["phase"] = "canonical_test_audit"
                checkpoint()
                for current_trial in phase_trials:
                    trial = trials[current_trial]
                    report, rows = score_trial(current_trial, split.test)
                    reports[phase][trial["arm"]][trial["seed"]] = report
                    phase_rows.extend(rows)
                    study._write_csv(output / "confirmation_metrics.csv", phase_rows)
                audit = paired_tail_gate(reports[phase]["baseline"], reports[phase]["cdp"], source="canonical_test", audit_only=True)
                audit.update(selected_arm=manifest["selection"]["selected_arm"], selection_changed=False,
                             promotion="not automatically promoted", deployment_guard_passed=audit["passed"])
                _write_json(output / "tail_audit.json", audit)
                manifest["tail_audit"] = {"status": audit["status"], "deployment_guard_passed": audit["passed"],
                                          "selection_changed": False}
            all_rows.extend(phase_rows)
            gc.collect()

        current_trial = None
        summaries, deltas = [], []
        table = pd.DataFrame(all_rows)
        for (phase, arm, target), group in table.groupby(["phase", "arm", "target"], sort=False):
            metrics = [*SCORE_COLUMNS, *TAIL_METRICS, "low_re_rmse", "thick_rmse"] if target == "Cd" else SCORE_COLUMNS
            for metric in metrics:
                values = group[metric].to_numpy(dtype=float)
                complete = len(values) == 3 and np.isfinite(values).all()
                summaries.append({"phase": phase, "arm": arm, "target": target, "metric": metric,
                                  "n_seeds": len(values), "n_defined": int(np.isfinite(values).sum()), "ddof": 1,
                                  "status": "complete" if complete else "inconclusive",
                                  "mean": float(values.mean()) if complete else None,
                                  "std": float(values.std(ddof=1)) if complete else None})
        for (phase, seed, target), group in table.groupby(["phase", "seed", "target"], sort=False):
            pair = group.set_index("arm")
            metrics = [*SCORE_COLUMNS, *TAIL_METRICS, "low_re_rmse", "thick_rmse"] if target == "Cd" else SCORE_COLUMNS
            deltas.append({"phase": phase, "seed": seed, "target": target, "direction": "cdp_minus_baseline", **{
                metric: float(pair.loc["cdp", metric] - pair.loc["baseline", metric])
                if np.isfinite([pair.loc["cdp", metric], pair.loc["baseline", metric]]).all() else None for metric in metrics
            }})
        study._write_csv(output / "summary.csv", summaries)
        study._write_csv(output / "paired_deltas.csv", deltas)
        manifest.update(status="completed", phase="completed")
        checkpoint()
        print(f"Auxiliary study completed: 12/12 fits; selected arm={manifest['selection']['selected_arm']} "
              f"from development only; test tail audit={manifest['tail_audit']['status']}; not automatically promoted.", flush=True)
        return json.loads(json.dumps(manifest, allow_nan=False))
    except BaseException as error:
        if current_trial is not None:
            trials[current_trial]["status"] = "failed"
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error), "trial_id": current_trial})
        checkpoint()
        raise
