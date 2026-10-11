"""Sequential model comparison on a frozen, geometry-disjoint baseline split."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data.preprocessing import SplitData
from src.data.validation import validate_dataframe
from src.training.evaluate import evaluate_model, geometry_metrics
from src.training.train import MODEL_BUILDERS, train_model


# Polynomial expansion is deliberately excluded: the full training matrix alone
# would require about 23.5 GB in float64, before fitting or temporary copies.
DEFAULT_MODEL_CONFIGS = (
    {"name": "linear", "parameters": {}, "scale_targets": False},
    {
        "name": "gradient_boosting",
        "parameters": {
            "max_iter": 200, "learning_rate": .05, "max_leaf_nodes": 31,
            "early_stopping": False, "random_state": 42,
        },
        "scale_targets": True,
    },
    {
        "name": "random_forest",
        "parameters": {
            "n_estimators": 100, "max_depth": 20, "max_leaf_nodes": 2048,
            "min_samples_leaf": 5, "max_samples": .5, "n_jobs": 2,
            "random_state": 42,
        },
        "scale_targets": True,
    },
    {
        "name": "mlp",
        "parameters": {
            "hidden_layers": (64, 32), "max_iter": 100, "batch_size": 1024,
            "early_stopping": False, "random_state": 42,
        },
        "scale_targets": True,
    },
)

MLP_IMPROVEMENT_CONFIGS = tuple(
    {
        "id": f"mlp_{budget}", "name": "mlp", "scale_targets": True,
        "parameters": {
            "hidden_layers": (64, 32), "max_iter": budget, "batch_size": 1024,
            "early_stopping": False, "random_state": 42,
        },
    }
    for budget in (100, 200, 400)
)

METRIC_COLUMNS = [
    "model", "target", "mae", "rmse", "r2",
    "batch_inference_time_seconds", "n_predictions",
]
COMPARISON_COLUMNS = METRIC_COLUMNS[:5] + ["training_time_seconds"] + METRIC_COLUMNS[5:]
GEOMETRY_COLUMNS = ["model", "geometry_id", "target", "n_predictions", "mae", "rmse", "r2"]


def _model_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is None:
        raise ValueError("Model IDs must be safe, nonempty names of at most 64 characters.")
    return value


def model_artifact_path(manifest, model_id):
    """Resolve a named variant or a persisted pre-variant model without arbitrary paths."""
    model_id = _model_id(model_id)
    result = manifest["results"].get(model_id, {})
    if result.get("status") != "success":
        raise ValueError(f"Selected model {model_id!r} has no successful validation result.")
    # Published comparison runs before named variants used the family as the ID.
    family = result.get("family", model_id)
    if not isinstance(family, str) or family not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported selected model family: {family!r}.")
    expected = f"{model_id}/{family}.joblib"
    if result.get("model_path") != expected:
        raise ValueError(f"Model artifact path does not match its ID and family: {model_id}")
    return Path(expected)


def _names(values, label):
    if (
        not isinstance(values, (list, tuple)) or not values
        or not all(isinstance(value, str) and value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{label} must contain unique, nonempty column names.")
    return list(values)


def _validate_targets(x, y, target_names):
    actual = np.asarray(y)
    if actual.ndim not in (1, 2) or not len(actual) or len(x) != len(actual):
        raise ValueError("Features and targets must have matching, nonempty rows.")
    n_targets = 1 if actual.ndim == 1 else actual.shape[1]
    if n_targets != len(target_names):
        raise ValueError("Provide one target name per target column.")
    if hasattr(y, "columns") and list(y.columns) != target_names:
        raise ValueError("Target names must match the target DataFrame columns in order.")


def _write_json(path, value):
    # Replace a complete checkpoint, rather than leaving a truncated JSON file.
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_baseline_split(processed_path, baseline_run_path):
    """Verify the baseline CSV fingerprint and restore its exact geometry groups.

    ``baseline_run_path`` is the notebook02 ``run.json`` file. Rows retain their
    CSV order within each partition; no new random split or scaler is fitted.
    """
    processed_path = Path(processed_path)
    with Path(baseline_run_path).open(encoding="utf-8") as source:
        baseline = json.load(source)
    digest = hashlib.sha256()
    with processed_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != baseline["dataset"]["sha256"]:
        raise ValueError("Processed CSV SHA256 does not match the baseline dataset.")

    features = _names(baseline["feature_columns"], "feature_columns")
    targets = _names(baseline["target_columns"], "target_columns")
    if "geometry_id" in features or set(features).intersection(targets):
        raise ValueError("Features must not contain geometry_id or target columns (leakage).")
    frame = pd.read_csv(processed_path)
    validate_dataframe(
        frame, required_columns=list(dict.fromkeys(["geometry_id", "Re", "Cd", *features, *targets])),
    )
    if type(baseline["dataset"]["rows"]) is not int or len(frame) != baseline["dataset"]["rows"]:
        raise ValueError("Dataset row count does not match the baseline dataset rows.")
    if frame["geometry_id"].isna().any() or frame["geometry_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Dataset geometry_id values must not be missing or empty.")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in features + targets):
        raise ValueError("Feature and target columns must be numeric.")

    partitions = {}
    seen = set()
    for partition in ("train", "validation", "test"):
        ids = baseline["split_geometries"][partition]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{partition} geometry IDs must be a nonempty list.")
        if any(not isinstance(value, (str, int, float)) or pd.isna(value) for value in ids):
            raise ValueError(f"{partition} geometry IDs must not be missing or invalid.")
        groups = set(ids)
        if len(groups) != len(ids):
            raise ValueError(f"{partition} geometry IDs must be unique.")
        if seen.intersection(groups):
            raise ValueError("Baseline geometry partitions overlap; groups must be disjoint.")
        seen.update(groups)
        partitions[partition] = frame.loc[frame["geometry_id"].isin(groups)].copy()
    if seen != set(frame["geometry_id"]):
        raise ValueError("Baseline geometry IDs must exactly cover the dataset, with no missing or extra IDs.")
    for partition, rows in partitions.items():
        count = baseline["split_rows"][partition]
        if type(count) is not int or not len(rows) or len(rows) != count:
            raise ValueError(f"{partition} row count does not match baseline split_rows.")
    return SplitData(**partitions), features, targets, baseline


def compare_models(
    x_train, y_train, x_validation, y_validation, target_names, output_dir,
    model_configs=None, run_metadata=None,
    *, train_groups=None, validation_groups=None,
):
    """Fit one model at a time and select each target's lowest validation RMSE.

    Only training and validation data enter this API. Model failures are recorded
    and do not stop later fits; interrupts propagate with prior checkpoints intact.
    """
    target_names = _names(target_names, "target_names")
    _validate_targets(x_train, y_train, target_names)
    _validate_targets(x_validation, y_validation, target_names)
    if (train_groups is None) != (validation_groups is None):
        raise ValueError("Provide training and validation geometry IDs together.")
    if validation_groups is not None:
        normalized_groups = []
        for features, values in ((x_train, train_groups), (x_validation, validation_groups)):
            if isinstance(values, pd.Series) and isinstance(features, pd.DataFrame) and not values.index.equals(features.index):
                raise ValueError("Geometry ID order must match the feature DataFrame index.")
            groups = np.asarray(values)
            if groups.ndim != 1 or len(groups) != len(features) or pd.isna(groups).any():
                raise ValueError("Provide one nonmissing geometry ID per feature row.")
            if any(not str(value).strip() for value in groups):
                raise ValueError("Geometry IDs must not be empty.")
            normalized_groups.append(groups)
        train_groups, validation_groups = normalized_groups
        if set(train_groups).intersection(validation_groups):
            raise ValueError("Training and validation geometries overlap.")
    configs = list(DEFAULT_MODEL_CONFIGS if model_configs is None else model_configs)
    names = set()
    for config in configs:
        required = {"name", "parameters", "scale_targets"}
        if not isinstance(config, dict) or not required.issubset(config) or set(config) - required - {"id"}:
            raise ValueError("Each model config requires name, parameters, and scale_targets, with an optional id.")
        name = config["name"]
        if not isinstance(name, str) or name not in MODEL_BUILDERS:
            raise ValueError(f"Unsupported model name: {name!r}.")
        model_id = _model_id(config.get("id", name))
        if model_id in names:
            raise ValueError("Model config IDs must be unique.")
        names.add(model_id)
        if not isinstance(config["parameters"], dict) or type(config["scale_targets"]) is not bool:
            raise ValueError("Model parameters must be a dict and scale_targets must be a bool.")
        if {"name", "x_train", "y_train", "output_dir", "scale_targets"}.intersection(config["parameters"]):
            raise ValueError("Model parameters must not override training inputs or artifact settings.")

    manifest = {
        "run_metadata": {} if run_metadata is None else run_metadata,
        "target_names": target_names,
        "model_configs": configs,
        "status": "running",
        "selection_metric": "validation_rmse",
        "selected_models": {},
        "results": {
            config.get("id", config["name"]): {
                "family": config["name"],
                "status": "pending", "parameters": config["parameters"],
                "scale_targets": config["scale_targets"], "warnings": [],
            }
            for config in configs
        },
    }
    # Reject unserializable metadata/settings before creating the fresh directory.
    json.dumps(manifest, allow_nan=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    geometry_rows = []
    best_rmse = {}

    def checkpoint():
        table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
        temporary = output / "comparison.csv.tmp"
        table.to_csv(temporary, index=False)
        temporary.replace(output / "comparison.csv")
        if validation_groups is not None:
            by_geometry = pd.DataFrame(geometry_rows, columns=GEOMETRY_COLUMNS)
            temporary = output / "validation_by_geometry.csv.tmp"
            by_geometry.to_csv(temporary, index=False)
            temporary.replace(output / "validation_by_geometry.csv")
        _write_json(output / "run.json", manifest)
        return table

    checkpoint()
    for index, config in enumerate(configs, start=1):
        family = config["name"]
        name = config.get("id", family)
        result = manifest["results"][name]
        model_dir = output / name
        model_dir.mkdir()
        result["status"] = "running"
        _write_json(model_dir / "result.json", result)
        checkpoint()
        print(f"[{index}/{len(configs)}] Starting {name}", flush=True)
        model = None
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model, elapsed = train_model(
                    family, x_train, y_train, output_dir=model_dir,
                    scale_targets=config["scale_targets"], **config["parameters"],
                )
                evaluation = evaluate_model(model, x_validation, y_validation, target_names)
                per_geometry = None
                if validation_groups is not None:
                    per_geometry = geometry_metrics(
                        y_validation, model.predict(x_validation), validation_groups, target_names,
                    )
                    per_geometry.insert(0, "model", name)
            success = {
                "status": "success", "training_time_seconds": float(elapsed),
                "targets": evaluation["targets"],
                "batch_inference_time_seconds": evaluation["inference_time_seconds"],
                "n_predictions": evaluation["n_predictions"],
                "model_path": f"{name}/{family}.joblib",
            }
            if family == "mlp":
                pipeline = model.regressor_ if config["scale_targets"] else model
                network = pipeline.named_steps["model"]
                success["training"] = {
                    "n_iter": int(network.n_iter_), "max_iter": int(network.max_iter),
                    "hit_iteration_limit": bool(network.n_iter_ >= network.max_iter),
                    "loss_curve": [float(loss) for loss in network.loss_curve_],
                    "final_loss": float(network.loss_),
                    "loss_space": "standardized_targets" if config["scale_targets"] else "original_targets",
                }
            json.dumps(success, allow_nan=False)
            result.update(success)
            if per_geometry is not None:
                geometry_rows.extend(per_geometry.to_dict("records"))
            for target in target_names:
                scores = evaluation["targets"][target]
                rows.append({
                    "model": name, "target": target, **scores,
                    "training_time_seconds": result["training_time_seconds"],
                    "batch_inference_time_seconds": result["batch_inference_time_seconds"],
                    "n_predictions": result["n_predictions"],
                })
                if target not in best_rmse or scores["rmse"] < best_rmse[target]:
                    best_rmse[target] = scores["rmse"]
                    manifest["selected_models"][target] = name
        except Exception as error:
            # Store text, not the exception/traceback, so failed fits can be freed.
            result.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)}})
        finally:
            result["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)} for item in caught
            ]
            del model
            gc.collect()
        _write_json(model_dir / "result.json", result)
        checkpoint()
        if result["status"] == "success":
            print(f"[{index}/{len(configs)}] Finished {name} in {result['training_time_seconds']:.3f}s", flush=True)
        else:
            print(f"[{index}/{len(configs)}] Failed {name}: {result['error']['message']}", flush=True)

    statuses = [result["status"] for result in manifest["results"].values()]
    manifest["status"] = (
        "failed" if "success" not in statuses else
        "completed_with_failures" if "failed" in statuses else "completed"
    )
    return checkpoint()


def evaluate_selected_models(comparison_dir, x_test, y_test, target_names):
    """Evaluate frozen validation winners on test, without selecting on test scores."""
    output = Path(comparison_dir)
    with (output / "run.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    target_names = _names(target_names, "target_names")
    if target_names != manifest["target_names"]:
        raise ValueError("Test target names and order must match the comparison manifest.")
    _validate_targets(x_test, y_test, target_names)
    selected = manifest["selected_models"]
    if not selected or set(selected) != set(target_names):
        raise ValueError("A validation-selected winner is required for every target.")
    if manifest["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Comparison must be completed before evaluating frozen winners on test.")
    artifacts = {name: model_artifact_path(manifest, name) for name in selected.values()}

    rows = {}
    for name in dict.fromkeys(selected[target] for target in target_names):
        print(f"Evaluating selected model {name} on the full test split", flush=True)
        model = None
        try:
            model = joblib.load(output / artifacts[name])
            evaluation = evaluate_model(model, x_test, y_test, target_names)
            for target in target_names:
                if selected[target] == name:
                    rows[target] = {
                        "model": name, "target": target, **evaluation["targets"][target],
                        "batch_inference_time_seconds": evaluation["inference_time_seconds"],
                        "n_predictions": evaluation["n_predictions"],
                    }
        finally:
            del model
            gc.collect()
    records = [rows[target] for target in target_names]
    table = pd.DataFrame(records, columns=METRIC_COLUMNS)
    _write_json(output / "test_evaluation.json", {
        "selection_metric": "validation_rmse", "selected_models": selected,
        "target_names": target_names, "results": records,
    })
    table.to_csv(output / "test_metrics.csv", index=False)
    return table
