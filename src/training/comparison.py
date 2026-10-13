"""Sequential model comparison on a frozen, geometry-disjoint baseline split."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data.features import assemble_features
from src.data.preprocessing import apply_frozen_split
from src.data.validation import validate_dataframe
from src.training.evaluate import evaluate_model
from src.training.targets import PhysicalPredictor, forward_targets, transform_names
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

SCORE_COLUMNS = ["mae", "rmse", "r2", "mean_relative_error", "median_relative_error"]
METRIC_COLUMNS = [
    "model", "target", *SCORE_COLUMNS,
    "batch_inference_time_seconds", "n_predictions",
]
COMPARISON_COLUMNS = [
    "model", "target", *SCORE_COLUMNS, "training_time_seconds",
    "batch_inference_time_seconds", "n_predictions",
]


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


def _digest(path: Path) -> str:
    """Stream a file's SHA256, so a large table is never read into memory whole."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_split(
    geometry_path, polars_path, baseline_run_path, target_columns=("Cl", "Cd", "Cm"),
):
    """Assemble the v2 design matrix and restore v1's frozen geometry partitions.

    v1's monolithic CSV is gone, so its SHA256 guard cannot apply; this records
    a *new* fingerprint over both v2 tables instead, while reusing the exact
    geometry ID lists from the baseline run. Reusing those lists is what keeps
    v2 comparable with v1, and :func:`apply_frozen_split` re-asserts that the
    partitions are disjoint and cover the dataset exactly.

    Returns the split, the ordered feature names, the target names, and the
    provenance metadata to record in ``run.json``.
    """
    geometry_path, polars_path = Path(geometry_path), Path(polars_path)
    with Path(baseline_run_path).open(encoding="utf-8") as source:
        baseline = json.load(source)
    if "split_geometries" not in baseline:
        raise ValueError(f"{baseline_run_path} has no split_geometries to restore.")

    targets = _names(list(target_columns), "target_columns")
    # geometry_id is the join key and the split key: read it as text so an ID
    # that merely looks numeric is not silently coerced to an integer.
    polars = pd.read_csv(polars_path, dtype={"geometry_id": str})
    missing = [name for name in targets if name not in polars.columns]
    if missing:
        raise ValueError(
            f"Polars table lacks target columns {missing}. "
            "Regenerate the tables, or pass target_columns explicitly."
        )
    validate_dataframe(
        polars,
        required_columns=list(dict.fromkeys(["geometry_id", "Alpha", "Re", "Cd", *targets])),
    )
    geometry = pd.read_csv(geometry_path, dtype={"geometry_id": str})
    joined, features = assemble_features(polars, geometry)
    if "geometry_id" in features or set(features).intersection(targets):
        raise ValueError("Features must not contain geometry_id or target columns (leakage).")

    split = apply_frozen_split(joined, baseline["split_geometries"])
    metadata = {
        "dataset": {
            "geometry_table": geometry_path.name,
            "polars_table": polars_path.name,
            "geometry_sha256": _digest(geometry_path),
            "polars_sha256": _digest(polars_path),
            "geometry_rows": int(len(joined["geometry_id"].unique())),
            "polar_rows": int(len(polars)),
        },
        "baseline_run": str(baseline_run_path),
        "baseline_git_commit": baseline.get("git_commit"),
        # v1's row counts are recorded for comparison, not enforced: the v2
        # loader filters non-numeric polars that v1 let through.
        "baseline_split_rows": baseline.get("split_rows") or baseline.get("v1_split_rows"),
        "feature_columns": features,
        "target_columns": targets,
        "split_geometries": {
            name: sorted(baseline["split_geometries"][name])
            for name in ("train", "validation", "test")
        },
        "split_rows": {
            name: int(len(getattr(split, name))) for name in ("train", "validation", "test")
        },
    }
    return split, features, targets, metadata


def compare_models(
    x_train, y_train, x_validation, y_validation, target_names, output_dir,
    model_configs=None, run_metadata=None, target_transforms=None,
):
    """Fit one model at a time and select each target's lowest validation RMSE.

    Only training and validation data enter this API. Model failures are recorded
    and do not stop later fits; interrupts propagate with prior checkpoints intact.

    ``target_transforms`` maps a target to a fitting-space transform, such as
    ``{"Cd": "log"}``. Targets are supplied and scored in physical units either
    way: models are fitted on the transformed values and wrapped so that every
    number in ``comparison.csv`` stays in original coefficient units. Pass
    nothing to fit the targets exactly as given.
    """
    target_names = _names(target_names, "target_names")
    _validate_targets(x_train, y_train, target_names)
    _validate_targets(x_validation, y_validation, target_names)
    transforms = transform_names(target_names, target_transforms)
    transformed = any(name != "identity" for name in transforms.values())
    # Leave the arrays untouched when nothing is transformed, so no copy of a
    # 500k-row training matrix is made just to pass it straight through.
    fitting_targets = y_train
    if transformed:
        fitting_targets = forward_targets(y_train, target_names, target_transforms)
        if len(target_names) == 1:
            fitting_targets = fitting_targets[:, 0]

    configs = list(DEFAULT_MODEL_CONFIGS if model_configs is None else model_configs)
    names = set()
    for config in configs:
        if not isinstance(config, dict) or set(config) != {"name", "parameters", "scale_targets"}:
            raise ValueError("Each model config requires name, parameters, and scale_targets.")
        name = config["name"]
        if not isinstance(name, str) or name not in MODEL_BUILDERS:
            raise ValueError(f"Unsupported model name: {name!r}.")
        if name in names:
            raise ValueError("Model config names must be unique.")
        names.add(name)
        if not isinstance(config["parameters"], dict) or type(config["scale_targets"]) is not bool:
            raise ValueError("Model parameters must be a dict and scale_targets must be a bool.")
        if {"name", "x_train", "y_train", "output_dir", "scale_targets"}.intersection(config["parameters"]):
            raise ValueError("Model parameters must not override training inputs or artifact settings.")

    manifest = {
        "run_metadata": {} if run_metadata is None else run_metadata,
        "target_names": target_names,
        "target_transforms": transforms,
        "model_configs": configs,
        "status": "running",
        "selection_metric": "validation_rmse",
        "selected_models": {},
        "results": {
            config["name"]: {
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
    best_rmse = {}

    def checkpoint():
        table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
        temporary = output / "comparison.csv.tmp"
        table.to_csv(temporary, index=False)
        temporary.replace(output / "comparison.csv")
        _write_json(output / "run.json", manifest)
        return table

    checkpoint()
    for index, config in enumerate(configs, start=1):
        name = config["name"]
        result = manifest["results"][name]
        model_dir = output / name
        model_dir.mkdir()
        result["status"] = "running"
        _write_json(model_dir / "result.json", result)
        checkpoint()
        print(f"[{index}/{len(configs)}] Starting {name}", flush=True)
        model = None
        scored = None
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model, elapsed = train_model(
                    name, x_train, fitting_targets, output_dir=model_dir,
                    scale_targets=config["scale_targets"], **config["parameters"],
                )
                scored = (
                    PhysicalPredictor(model, target_names, target_transforms)
                    if transformed else model
                )
                evaluation = evaluate_model(scored, x_validation, y_validation, target_names)
            success = {
                "status": "success", "training_time_seconds": float(elapsed),
                "targets": evaluation["targets"],
                "batch_inference_time_seconds": evaluation["inference_time_seconds"],
                "n_predictions": evaluation["n_predictions"],
                "model_path": f"{name}/{name}.joblib",
            }
            json.dumps(success, allow_nan=False)
            result.update(success)
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
            # The scoring wrapper holds the model too, so both must go before
            # the next fit allocates.
            del model, scored
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
    """Evaluate frozen validation winners on test, without selecting on test scores.

    Targets are supplied in physical units. Saved models hold whatever fitting
    space they were trained in, so the transforms recorded in ``run.json`` are
    reapplied in reverse before scoring.
    """
    output = Path(comparison_dir)
    with (output / "run.json").open(encoding="utf-8") as source:
        manifest = json.load(source)
    target_names = _names(target_names, "target_names")
    if target_names != manifest["target_names"]:
        raise ValueError("Test target names and order must match the comparison manifest.")
    _validate_targets(x_test, y_test, target_names)
    # A manifest without the key predates target transforms, when nothing was
    # transformed, so identity is the correct reading.
    transforms = manifest.get("target_transforms") or {}
    transformed = any(value != "identity" for value in transforms.values())
    selected = manifest["selected_models"]
    if not selected or set(selected) != set(target_names):
        raise ValueError("A validation-selected winner is required for every target.")
    if manifest["status"] not in ("completed", "completed_with_failures"):
        raise ValueError("Comparison must be completed before evaluating frozen winners on test.")
    for name in selected.values():
        if not isinstance(name, str) or name not in MODEL_BUILDERS:
            raise ValueError(f"Unsupported selected model name: {name!r}.")
        if manifest["results"].get(name, {}).get("status") != "success":
            raise ValueError(f"Selected model {name!r} has no successful validation result.")

    rows = {}
    for name in dict.fromkeys(selected[target] for target in target_names):
        print(f"Evaluating selected model {name} on the full test split", flush=True)
        model = None
        scored = None
        try:
            model = joblib.load(output / name / f"{name}.joblib")
            scored = PhysicalPredictor(model, target_names, transforms) if transformed else model
            evaluation = evaluate_model(scored, x_test, y_test, target_names)
            for target in target_names:
                if selected[target] == name:
                    rows[target] = {
                        "model": name, "target": target, **evaluation["targets"][target],
                        "batch_inference_time_seconds": evaluation["inference_time_seconds"],
                        "n_predictions": evaluation["n_predictions"],
                    }
        finally:
            del model, scored
            gc.collect()
    records = [rows[target] for target in target_names]
    table = pd.DataFrame(records, columns=METRIC_COLUMNS)
    _write_json(output / "test_evaluation.json", {
        "selection_metric": "validation_rmse", "selected_models": selected,
        "target_names": target_names, "target_transforms": transforms, "results": records,
    })
    table.to_csv(output / "test_metrics.csv", index=False)
    return table
