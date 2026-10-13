"""MLP epochs controlled by geometry-disjoint validation, not training-loss patience."""

from __future__ import annotations

import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from src.models.mlp import build_model
from src.training.evaluate import evaluate_model


DEFAULT_POLICY = {"max_epochs": 400, "check_every": 5, "patience": 10, "min_epochs": 100, "min_delta": 0.001}
CONTROLLED_PARAMETERS = {"max_iter", "early_stopping", "tol", "n_iter_no_change"}


def validation_policy(policy):
    if not isinstance(policy, dict) or set(policy) - set(DEFAULT_POLICY):
        raise ValueError("Unknown validation-training policy settings.")
    values = {**DEFAULT_POLICY, **policy}
    for key in ("max_epochs", "check_every", "min_epochs"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if values["min_epochs"] > values["max_epochs"]:
        raise ValueError("min_epochs must not exceed max_epochs.")
    if values["patience"] is not None and (type(values["patience"]) is not int or values["patience"] < 1):
        raise ValueError("patience must be a positive number of validation checks or None.")
    if isinstance(values["min_delta"], bool) or not isinstance(values["min_delta"], (int, float)) or not 0 <= values["min_delta"] < 1:
        raise ValueError("min_delta must be a relative improvement in [0, 1).")
    return values


def validate_geometry_groups(x_train, x_validation, train_groups, validation_groups):
    arrays = []
    for features, groups in ((x_train, train_groups), (x_validation, validation_groups)):
        if isinstance(groups, pd.Series) and isinstance(features, pd.DataFrame) and not groups.index.equals(features.index):
            raise ValueError("Geometry ID order must match the feature DataFrame index.")
        values = np.asarray(groups)
        if values.ndim != 1 or len(values) != len(features) or pd.isna(values).any():
            raise ValueError("Provide one nonmissing geometry ID per feature row.")
        if any(not str(value).strip() for value in values):
            raise ValueError("Geometry IDs must not be empty.")
        arrays.append(values)
    if set(arrays[0]).intersection(arrays[1]):
        raise ValueError("Training and validation geometries overlap.")
    return tuple(arrays)


def _one_epoch(action):
    # Only the one-epoch bootstrap fit emits an expected max_iter warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        action()
    for warning in caught:
        if str(warning.message) == "Training interrupted by user.":
            # sklearn catches KeyboardInterrupt internally; do not start another epoch.
            raise KeyboardInterrupt("MLP training interrupted by user.")
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn(str(warning.message), warning.category, stacklevel=2)


def train_validation_mlp(
    x_train, y_train, x_validation, y_validation, target_names, output_dir,
    *, train_groups, validation_groups, parameters=None, policy=None,
):
    """Fit scalers once, preserve Adam state, and restore the best observed checkpoint.

    One joint checkpoint is selected by the mean validation squared error after
    scaling each target by its TRAINING standard deviation. Returned predictions
    remain in original units. No test data enter this function.
    """
    settings = validation_policy({} if policy is None else policy)
    validate_geometry_groups(x_train, x_validation, train_groups, validation_groups)
    names = list(target_names) if isinstance(target_names, (list, tuple)) else []
    if not names or not all(isinstance(name, str) and name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Target names must be unique, nonempty strings.")
    for features, targets in ((x_train, y_train), (x_validation, y_validation)):
        actual = np.asarray(targets)
        if actual.ndim not in (1, 2) or not len(actual) or len(features) != len(actual):
            raise ValueError("Features and targets must have matching nonempty rows.")
        if (1 if actual.ndim == 1 else actual.shape[1]) != len(names):
            raise ValueError("Provide one target name per target column.")
        if isinstance(targets, pd.DataFrame) and list(targets.columns) != names:
            raise ValueError("Target names must match the DataFrame columns in order.")
        if isinstance(features, pd.DataFrame) and isinstance(targets, (pd.DataFrame, pd.Series)) and not features.index.equals(targets.index):
            raise ValueError("Feature and target row order must match.")
    if isinstance(x_train, pd.DataFrame) and isinstance(x_validation, pd.DataFrame) and list(x_train.columns) != list(x_validation.columns):
        raise ValueError("Training and validation feature columns must match in order.")
    params = dict(parameters or {})
    if CONTROLLED_PARAMETERS.intersection(params):
        raise ValueError("Epoch and stopping settings belong in validation_training, not model parameters.")
    seed = params.get("random_state", 42)
    if type(seed) is not int:
        raise ValueError("Validation-controlled MLP training requires an integer random_state.")
    pipeline = build_model(max_iter=1, early_stopping=False, tol=0.0, n_iter_no_change=settings["max_epochs"] + 1, **params)
    # An RNG object advances between partial_fit calls instead of reseeding each epoch.
    pipeline.set_params(model__random_state=np.random.RandomState(seed))
    model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "mlp.joblib"
    if artifact.exists() or (output / "validation_state.json").exists():
        raise FileExistsError("Validation training requires a fresh model directory.")
    history, losses = [], []
    state = {
        "status": "running", "method": "geometry_validation", "policy": settings,
        "epochs_completed": 0, "best_epoch": None, "best_validation_score": None,
        "stale_checks": 0, "stop_reason": None,
        "checkpoint_metric": "mean_validation_mse_in_train_standardized_target_units",
    }

    def save_progress():
        temporary = output / "validation_state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output / "validation_state.json")
        if history:
            temporary = output / "training_history.csv.tmp"
            pd.DataFrame(history).to_csv(temporary, index=False)
            temporary.replace(output / "training_history.csv")

    started = time.perf_counter()
    progress_score = float("inf")
    save_progress()
    try:
        for epoch in range(1, settings["max_epochs"] + 1):
            if epoch == 1:
                _one_epoch(lambda: model.fit(x_train, y_train))
                fitted = model.regressor_
                network = fitted.named_steps["model"]
                scaled_x = fitted.named_steps["scaler"].transform(x_train)
                targets = np.asarray(y_train).reshape(len(y_train), len(names))
                scaled_y = model.transformer_.transform(targets)
                if len(names) == 1:
                    scaled_y = scaled_y.ravel()
            else:
                _one_epoch(lambda: network.partial_fit(scaled_x, scaled_y))
            loss = float(network.loss_)
            if not np.isfinite(loss):
                raise ValueError("Training loss became nonfinite.")
            losses.append(loss)
            state["epochs_completed"] = epoch
            if epoch != 1 and epoch % settings["check_every"] and epoch != settings["max_epochs"]:
                continue
            evaluation = evaluate_model(model, x_validation, y_validation, names)
            score = float(np.mean([
                (evaluation["targets"][name]["rmse"] / model.transformer_.scale_[index]) ** 2
                for index, name in enumerate(names)
            ]))
            if not np.isfinite(score):
                raise ValueError("Validation score became nonfinite.")
            for name in names:
                history.append({
                    "epoch": epoch, "target": name, "training_loss": loss,
                    "validation_score": score, "elapsed_seconds": time.perf_counter() - started,
                    **evaluation["targets"][name],
                })
            if state["best_validation_score"] is None or score < state["best_validation_score"]:
                temporary = output / "mlp.joblib.tmp"
                joblib.dump(model, temporary)
                temporary.replace(artifact)
                state.update(best_epoch=epoch, best_validation_score=score)
            threshold = progress_score * (1 - settings["min_delta"])
            significant_progress = score < progress_score and (
                score <= threshold or np.isclose(score, threshold, rtol=1e-12, atol=0)
            )
            if significant_progress:
                progress_score = score
                state["stale_checks"] = 0
            else:
                state["stale_checks"] += 1
            print(
                f"  epoch {epoch}: validation score={score:.6g}, best epoch={state['best_epoch']}, "
                f"stale checks={state['stale_checks']}", flush=True,
            )
            if settings["patience"] is not None and epoch >= settings["min_epochs"] and state["stale_checks"] >= settings["patience"]:
                state["stop_reason"] = "validation_patience"
            elif epoch == settings["max_epochs"]:
                state["stop_reason"] = "max_epochs"
            save_progress()
            if state["stop_reason"]:
                break
        state["status"] = "completed"
        save_progress()
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        save_progress()
        raise
    elapsed = time.perf_counter() - started
    best_model = joblib.load(artifact)
    details = {
        **state, "n_iter": state["epochs_completed"], "max_iter": settings["max_epochs"],
        "hit_iteration_limit": state["epochs_completed"] == settings["max_epochs"],
        "tol": 0.0, "n_iter_no_change": settings["max_epochs"] + 1, "early_stopping": False,
        "loss_curve": losses, "final_loss": losses[-1], "loss_space": "standardized_targets",
        "validation_history": history, "timing_scope": "training_validation_checkpointing",
    }
    return best_model, elapsed, details
