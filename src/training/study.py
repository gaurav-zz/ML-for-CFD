"""Kaggle-only, predeclared seed/sweep study and train-only leakage diagnostic."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.torch_mlp import TorchMLPRegressor
from src.training.analysis import prediction_frame, summarize
from src.training.comparison import ACCURACY_MODEL_CONFIGS, SCORE_COLUMNS, _write_json, compare_models
from src.training.evaluate import evaluate_model
from src.training.targets import DEFAULT_TARGET_TRANSFORMS, PhysicalPredictor


_KAGGLE_WORKING = Path("/kaggle/working")
_KAGGLE_INPUT = Path("/kaggle/input")
_ROW_KEYS = ["geometry_id", "Alpha", "Re", "Ncrit"]
_SEEDS = (42, 43, 44)


def _require_kaggle(output_dir, device):
    if not _KAGGLE_WORKING.is_dir() or not _KAGGLE_INPUT.is_dir():
        raise RuntimeError("Studies may run only on Kaggle (/kaggle/working and /kaggle/input required).")
    output = Path(output_dir).resolve()
    if not output.is_relative_to(_KAGGLE_WORKING.resolve()):
        raise ValueError("Study output must be inside /kaggle/working.")
    if device not in ("cuda", "cpu"):
        raise ValueError("Study device must be 'cuda' or explicitly 'cpu'; no automatic fallback.")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requires an available Kaggle CUDA GPU.")
    return output


def _row_hash(frame):
    # Identity only, never labels; sorting makes the digest independent of row order.
    hashes = pd.util.hash_pandas_object(frame[_ROW_KEYS], index=False).to_numpy()
    return hashlib.sha256(np.sort(hashes).astype("<u8").tobytes()).hexdigest()


def _validate_split(split, features, targets, metadata):
    if (not isinstance(features, (list, tuple)) or not features
            or any(not isinstance(name, str) or not name.strip() for name in features)
            or len(set(features)) != len(features)):
        raise ValueError("feature_columns must contain unique, nonempty names.")
    if list(targets) != ["Cl", "Cd", "Cm"]:
        raise ValueError("This paired study requires target_names in Cl, Cd, Cm order.")
    if set(features).intersection(["geometry_id", *targets]):
        raise ValueError("Features cannot contain geometry_id or targets.")
    partitions = ("train", "validation", "test")
    geometries, counts = metadata.get("split_geometries"), metadata.get("split_rows")
    if (not isinstance(geometries, dict) or set(geometries) != set(partitions)
            or not isinstance(counts, dict) or set(counts) != set(partitions)):
        raise ValueError("run_metadata must supply exact frozen split_geometries and split_rows.")
    for key, columns in (("feature_columns", features), ("target_columns", targets)):
        if key in metadata and metadata[key] != list(columns):
            raise ValueError(f"{key} must match frozen run_metadata in order.")
    seen, report = set(), {}
    for name in partitions:
        frame, ids = getattr(split, name), geometries[name]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f"{name} frozen geometry IDs must be nonempty and unique.")
        if seen.intersection(ids):
            raise ValueError("Frozen geometry partitions must be disjoint.")
        seen.update(ids)
        if (not isinstance(frame, pd.DataFrame) or frame.empty
                or not frame.columns.is_unique or not frame.index.is_unique):
            raise ValueError(f"{name} must be a nonempty frame with unique columns and row indices.")
        required = set(_ROW_KEYS).union(features, targets)
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} lacks required columns: {sorted(required.difference(frame.columns))}.")
        # Inspect canonical test identity/schema only. Its features/labels are not
        # materialized for prediction or scoring until study_selection.json exists.
        identity = frame[_ROW_KEYS]
        if identity.isna().any().any() or identity.duplicated().any():
            raise ValueError(f"{name} operating-point rows must be nonmissing and unique.")
        if set(identity.geometry_id) != set(ids):
            raise ValueError(f"{name} must contain exactly its frozen geometry IDs.")
        if type(counts[name]) is not int or counts[name] != len(frame):
            raise ValueError(f"{name} row count must match frozen split_rows exactly.")
        report[name] = {"geometry_ids": sorted(ids), "rows": len(frame), "row_sha256": _row_hash(frame)}
    return report


def _leakage_partition(train):
    """Fixed identity-only partition, independent of model seed and target values."""
    rng = np.random.default_rng(42)
    groups = rng.permutation(sorted(train.geometry_id.unique()))
    n_audit, n_stop = max(1, round(.20 * len(groups))), max(1, round(.15 * len(groups)))
    if n_audit + n_stop >= len(groups):
        raise ValueError("Leakage diagnostic requires at least three canonical training geometries.")
    audit_ids, stop_ids = groups[:n_audit], groups[n_audit:n_audit + n_stop]
    # Sort only the small identity table, retaining positions into the full frame.
    identity = train[_ROW_KEYS].reset_index(drop=True).sort_values(_ROW_KEYS)
    base_positions = identity.index[~identity.geometry_id.isin([*audit_ids, *stop_ids])].to_numpy()
    stop_positions = identity.index[identity.geometry_id.isin(stop_ids)].to_numpy()
    audit_positions, candidates = [], []
    for group in sorted(audit_ids):
        positions = rng.permutation(identity.index[identity.geometry_id == group].to_numpy())
        if len(positions) < 2:
            raise ValueError("Each audit geometry needs at least two distinct operating-point rows.")
        half = len(positions) // 2
        audit_positions.extend(positions[:half])
        candidates.extend(positions[half:])
    n_exposed = min(len(candidates), len(base_positions))
    exposed_positions = rng.choice(candidates, size=n_exposed, replace=False)
    kept_positions = rng.choice(base_positions, size=len(base_positions) - n_exposed, replace=False)
    frames = {
        "honest": train.iloc[sorted(base_positions)],
        "exposed": train.iloc[sorted([*kept_positions, *exposed_positions])],
        "stopping": train.iloc[sorted(stop_positions)],
        "audit": train.iloc[sorted(audit_positions)],
    }
    audit_groups, stop_groups = set(audit_ids), set(stop_ids)
    arm_report = {}
    for arm in ("honest", "exposed"):
        frame = frames[arm]
        audit_overlap = len(set(frame.geometry_id).intersection(audit_groups))
        stop_overlap = len(set(frame.geometry_id).intersection(stop_groups))
        if (len(frame) != len(base_positions) or stop_overlap
                or (audit_overlap != 0 if arm == "honest" else audit_overlap == 0)
                or not set(frame.index).isdisjoint(frames["audit"].index)
                or not set(frame.index).isdisjoint(frames["stopping"].index)):
            raise ValueError("Invalid leakage diagnostic: row or geometry separation violated.")
        arm_report[arm] = {
            "rows": len(frame), "row_sha256": _row_hash(frame),
            "geometry_count": int(frame.geometry_id.nunique()),
            "audit_geometry_overlap": audit_overlap, "stopping_geometry_overlap": stop_overlap,
            "exposure_rows": 0 if arm == "honest" else n_exposed,
        }
    report = {
        "purpose": "diagnostic_not_production", "partition_seed": 42,
        "source": "canonical_train_only", "canonical_validation_test_used": False,
        "audit_geometry_ids": sorted(audit_ids), "stopping_geometry_ids": sorted(stop_ids),
        "base_geometry_ids": sorted(frames["honest"].geometry_id.unique()),
        "audit_rows": len(frames["audit"]), "audit_row_sha256": _row_hash(frames["audit"]),
        "audit_row_ids": "leakage_audit_rows.csv", "row_identity_columns": _ROW_KEYS,
        "hash_algorithm": "sha256(sorted pandas hash_pandas_object(identity, index=False), little-endian uint64)",
        "stopping_rows": len(frames["stopping"]), "stopping_row_sha256": _row_hash(frames["stopping"]),
        "candidate_exposure_rows": len(candidates), "arms": arm_report,
    }
    return frames, report


def _write_csv(path, records):
    table = pd.DataFrame(records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)
    return table


def run_study(split, feature_columns, target_names, output_dir, run_metadata=None, device="cuda"):
    """Run all 12 fits on Kaggle and return the final ``study.json`` dictionary.

    ``run_metadata`` must contain the frozen ``split_geometries`` and ``split_rows``.
    Output must be a fresh directory under /kaggle/working. No test-based seed
    selection, cross-product sweep, refit, or production selection on leakage
    metrics is performed. Three-seed summaries are descriptive, not significance
    claims. Any failed trial or evaluation raises and checkpoints status=failed.
    """
    output = _require_kaggle(output_dir, device)
    metadata = deepcopy({} if run_metadata is None else run_metadata)
    canonical = _validate_split(split, feature_columns, target_names, metadata)
    features, targets = list(feature_columns), list(target_names)
    leakage, leakage_report = _leakage_partition(split.train)
    defaults = TorchMLPRegressor().get_params(deep=False)
    defaults.pop("target_names")  # compare_models supplies this from target_names.
    templates = {config["name"]: config for config in ACCURACY_MODEL_CONFIGS}
    trials = {}

    def add_trial(trial_id, role, seed=42, loss="huber", delta=.05, decay=1e-5):
        config = deepcopy(templates[f"torch_mlp_{loss}"])
        config["parameters"] = {
            **defaults, **config["parameters"], "random_state": seed, "device": device,
            "loss": loss, "huber_delta": delta, "weight_decay": decay,
        }
        trials[trial_id] = {
            "role": role, "seed": seed, "config": config, "status": "pending",
            "manifest": f"trials/{trial_id}/run.json",
        }

    base_ids = []
    for seed in _SEEDS:
        for loss in ("mse", "huber"):
            trial_id = f"seed_{seed}_{loss}"
            add_trial(trial_id, "seed", seed=seed, loss=loss)
            base_ids.append(trial_id)
    central_id = "seed_42_huber"
    sweep_ids = [central_id]
    for trial_id, delta, decay in (
        ("sweep_delta_0p02", .02, 1e-5), ("sweep_delta_0p1", .1, 1e-5),
        ("sweep_decay_0", .05, 0.), ("sweep_decay_1e-4", .05, 1e-4),
    ):
        add_trial(trial_id, "sweep", delta=delta, decay=decay)
        sweep_ids.append(trial_id)
    for arm in ("honest", "exposed"):
        add_trial(f"leakage_{arm}", "leakage")

    manifest = {
        "schema_version": 1, "status": "planned", "phase": "planned",
        "run_metadata": metadata, "hardware": metadata.get("gpu", {}), "device": device,
        "feature_columns": features, "target_names": targets,
        "target_transforms": DEFAULT_TARGET_TRANSFORMS.copy(), "canonical_split": canonical,
        "plan": {
            "seeds": list(_SEEDS), "seed_trials": base_ids, "estimated_fits": len(trials),
            "sweep_seed": 42, "sweep_trial_order": sweep_ids,
            "huber_deltas": [.02, .05, .1], "weight_decays": [0., 1e-5, 1e-4],
            "sweep_design": "one_factor_at_a_time; central seed-42 Huber reused",
            "selection_metric": "validation.Cd.median_relative_error",
            "tie_break": "first in sweep_trial_order (central configuration first)",
            "seed_policy": "report all paired seeds; never select a winning seed",
            "test_policy": "all six seed trials plus selected seed-42 sweep configuration only; no refit",
            "statistics": "descriptive mean and sample std (ddof=1); n=3 is not strong statistical evidence",
            "paired_delta": "Huber minus MSE; negative error deltas favor Huber",
            "leakage": leakage_report,
            "trial_configs": {key: deepcopy(value["config"]) for key, value in trials.items()},
        },
        "trials": trials, "selection": None,
        "artifacts": {
            "selection": "study_selection.json", "seed_metrics": "three_seed_metrics.csv",
            "seed_summary": "three_seed_summary.csv", "seed_deltas": "paired_seed_deltas.csv",
            "sweep_validation": "sweep_validation.csv", "sweep_test": "sweep_test_metrics.csv",
            "leakage_split": "leakage_split.json", "leakage_audit_rows": "leakage_audit_rows.csv",
            "leakage_metrics": "leakage_metrics.csv", "leakage_deltas": "leakage_paired_deltas.csv",
            "selected_analysis": "selected_analysis.json",
        },
    }
    json.dumps(manifest, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "study.json", manifest)
    current_trial = None

    def checkpoint():
        _write_json(output / "study.json", manifest)

    def score_trial(trial_id, frame, partition):
        trial = trials[trial_id]
        model = scored = None
        try:
            model = joblib.load(output / trial["model_path"])
            scored = PhysicalPredictor(model, targets, DEFAULT_TARGET_TRANSFORMS)
            evaluation = evaluate_model(scored, frame[features], frame[targets], targets)
            path = f"trials/{trial_id}/{partition}_evaluation.json"
            payload = {"trial_id": trial_id, "partition": partition, **evaluation}
            if partition == "test":
                # Retain the standard report contract for the CI accuracy guard.
                # Each trial comparison contains one predeclared model, not a
                # winner reranked on these test scores.
                saved = json.loads((output / trial["manifest"]).read_text(encoding="utf-8"))
                records = [
                    {"model": trial["config"]["name"], "target": target,
                     **evaluation["targets"][target],
                     "batch_inference_time_seconds": evaluation["inference_time_seconds"],
                     "n_predictions": evaluation["n_predictions"]}
                    for target in targets
                ]
                payload.update({key: saved[key] for key in (
                    "selection_metric", "selection_metrics", "selected_models",
                    "target_names", "target_transforms",
                )})
                payload["results"] = records
                _write_csv(output / f"trials/{trial_id}/test_metrics.csv", records)
            _write_json(output / path, payload)
            trial[f"{partition}_evaluation"] = path
            if partition == "test" and trial_id == manifest["selection"]["selected_trial"]:
                report = summarize(
                    prediction_frame(frame, scored.predict(frame[features]), targets),
                    targets, training_frame=split.train,
                )
                report["study_trial"] = trial_id
                _write_json(output / "selected_analysis.json", report)
            checkpoint()
            return [
                {"trial_id": trial_id, "seed": trial["seed"], "loss": trial["config"]["parameters"]["loss"],
                 "partition": partition, "target": target, **evaluation["targets"][target],
                 "batch_inference_time_seconds": evaluation["inference_time_seconds"],
                 "n_predictions": evaluation["n_predictions"]}
                for target in targets
            ]
        finally:
            model = scored = None
            gc.collect()

    try:
        _write_json(output / "leakage_split.json", leakage_report)
        _write_csv(output / "leakage_audit_rows.csv", leakage["audit"][_ROW_KEYS])
        manifest.update(status="running", phase="fitting")
        checkpoint()
        for current_trial, trial in trials.items():
            trial["status"] = "running"
            checkpoint()
            train, validation = split.train, split.validation
            if trial["role"] == "leakage":
                train = leakage[current_trial.removeprefix("leakage_")]
                validation = leakage["stopping"]
            trial_dir = output / "trials" / current_trial
            trial_metadata = {
                **metadata, "study_trial": current_trial,
                "study_role": "diagnostic_not_production" if trial["role"] == "leakage" else trial["role"],
                "fit_split": {"train_rows": len(train), "validation_rows": len(validation),
                              "train_row_sha256": _row_hash(train), "validation_row_sha256": _row_hash(validation),
                              "validation_source": "canonical_train_stopping_pool" if trial["role"] == "leakage" else "canonical_validation"},
            }
            compare_models(
                train[features], train[targets], validation[features], validation[targets], targets, trial_dir,
                model_configs=[deepcopy(trial["config"])], run_metadata=trial_metadata,
                target_transforms=DEFAULT_TARGET_TRANSFORMS, selection_metrics={"Cd": "median_relative_error"},
            )
            saved = json.loads((trial_dir / "run.json").read_text(encoding="utf-8"))
            result = saved["results"][trial["config"]["name"]]
            trial["result"] = result
            if saved["status"] != "completed" or result["status"] != "success":
                raise RuntimeError(f"Study trial {current_trial} failed: {result.get('error', saved['status'])}")
            model_path = trial_dir / result["model_path"]
            if not model_path.is_file():
                raise RuntimeError(f"Study trial {current_trial} did not save model weights.")
            trial.update(status="success", model_path=str(model_path.relative_to(output)))
            checkpoint()

        current_trial = None
        manifest["phase"] = "validation_selection"
        sweep_rows = []
        for trial_id in sweep_ids:
            trial = trials[trial_id]
            scores, parameters = trial["result"]["targets"], trial["config"]["parameters"]
            sweep_rows.append({
                "trial_id": trial_id, "seed": 42, "huber_delta": parameters["huber_delta"],
                "weight_decay": parameters["weight_decay"],
                "Cd_median_relative_error": scores["Cd"]["median_relative_error"],
                "Cl_rmse": scores["Cl"]["rmse"], "Cm_rmse": scores["Cm"]["rmse"],
            })
        if not all(np.isfinite(row[key]) for row in sweep_rows for key in ("Cd_median_relative_error", "Cl_rmse", "Cm_rmse")):
            raise ValueError("All sweep validation selection and companion metrics must be finite.")
        selected = min(sweep_rows, key=lambda row: row["Cd_median_relative_error"])["trial_id"]
        selection = {
            "status": "frozen", "metric": manifest["plan"]["selection_metric"],
            "candidate_order": sweep_ids, "selected_trial": selected, "seed": 42,
            "selected_config": deepcopy(trials[selected]["config"]), "validation": sweep_rows,
            "seed_trials_to_test": base_ids, "seed_selection": False,
            "canonical_test_scored": False, "leakage_used_for_selection": False,
        }
        # This file is never rewritten, even after scoring. No canonical test
        # feature matrix or labels have entered a fitting/scoring API above.
        _write_json(output / "study_selection.json", selection)
        manifest["selection"] = selection
        _write_csv(output / "sweep_validation.csv", [dict(row, selected=row["trial_id"] == selected) for row in sweep_rows])
        manifest["phase"] = "leakage_audit"
        checkpoint()
        leakage_rows = []
        audit = leakage["audit"]
        for arm in ("honest", "exposed"):
            current_trial = f"leakage_{arm}"
            records = score_trial(current_trial, audit, "audit")
            for row in records:
                values = audit[row["target"]].to_numpy(dtype=float)
                row.update(arm=arm, purpose="diagnostic_not_production",
                           audit_sst=float(np.sum((values - values.mean()) ** 2)))
            leakage_rows.extend(records)
        _write_csv(output / "leakage_metrics.csv", leakage_rows)
        leak_pairs = []
        for target in targets:
            pair = {row["arm"]: row for row in leakage_rows if row["target"] == target}
            leak_pairs.append({"target": target, "direction": "exposed_minus_honest", **{
                metric: None if any(pair[arm][metric] is None for arm in pair)
                else pair["exposed"][metric] - pair["honest"][metric] for metric in SCORE_COLUMNS
            }})
        _write_csv(output / "leakage_paired_deltas.csv", leak_pairs)
        del leakage, audit, train, validation

        manifest["phase"] = "canonical_test"
        checkpoint()
        seed_rows, test_records = [], {}
        for current_trial in base_ids:
            trial = trials[current_trial]
            for target in targets:
                seed_rows.append({"trial_id": current_trial, "seed": trial["seed"],
                                  "loss": trial["config"]["parameters"]["loss"], "partition": "validation",
                                  "target": target, **trial["result"]["targets"][target]})
            test_records[current_trial] = score_trial(current_trial, split.test, "test")
            seed_rows.extend(test_records[current_trial])
            _write_csv(output / "three_seed_metrics.csv", seed_rows)
        current_trial = selected
        if selected not in test_records:
            test_records[selected] = score_trial(selected, split.test, "test")
        _write_csv(output / "sweep_test_metrics.csv", test_records[selected])
        table = pd.DataFrame(seed_rows)
        summaries, pairs = [], []
        for (partition, loss, target), group in table.groupby(["partition", "loss", "target"], sort=False):
            for metric in SCORE_COLUMNS:
                values = group[metric].dropna()
                summaries.append({"partition": partition, "loss": loss, "target": target, "metric": metric,
                                  "n_seeds": len(group), "n_defined": len(values), "ddof": 1,
                                  "mean": values.mean(), "std": values.std(ddof=1)})
        for (partition, seed, target), group in table.groupby(["partition", "seed", "target"], sort=False):
            pair = group.set_index("loss")
            pairs.append({"partition": partition, "seed": seed, "target": target,
                          "direction": "huber_minus_mse", **{
                              metric: pair.loc["huber", metric] - pair.loc["mse", metric] for metric in SCORE_COLUMNS
                          }})
        _write_csv(output / "three_seed_summary.csv", summaries)
        _write_csv(output / "paired_seed_deltas.csv", pairs)
        manifest.update(status="completed", phase="completed")
        checkpoint()
        return json.loads(json.dumps(manifest, allow_nan=False))
    except BaseException as error:
        if current_trial is not None and trials[current_trial]["status"] == "running":
            trials[current_trial]["status"] = "failed"
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error), "trial_id": current_trial})
        checkpoint()
        raise
