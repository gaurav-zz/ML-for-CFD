"""Evaluate saved or in-memory regressors with timing and standard metrics."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.utils.metrics import regression_metrics


def evaluate_model(model, x_test, y_test, target_names: list[str]) -> dict[str, object]:
    """Report each named target separately, with timing for the full prediction batch."""
    actual = np.asarray(y_test)
    if actual.ndim == 1:
        actual = actual[:, None]
    if actual.ndim != 2:
        raise ValueError("Targets must be a one- or two-dimensional array.")
    if (
        isinstance(target_names, str)
        or len(target_names) != actual.shape[1]
        or not all(isinstance(name, str) and name.strip() for name in target_names)
        or len(set(target_names)) != len(target_names)
        or not target_names
    ):
        raise ValueError("Provide one unique, non-empty target name per target column.")
    if hasattr(y_test, "columns") and list(y_test.columns) != list(target_names):
        raise ValueError("Target names must match the target DataFrame columns in order.")

    started = time.perf_counter()
    prediction = np.asarray(model.predict(x_test))
    inference_time = time.perf_counter() - started
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if prediction.shape != actual.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match targets {actual.shape}.")

    return {
        "targets": {
            name: regression_metrics(actual[:, index], prediction[:, index])
            for index, name in enumerate(target_names)
        },
        "inference_time_seconds": inference_time,
        "n_predictions": len(prediction),
    }


def geometry_metrics(actual, predicted, groups, target_names):
    """Report each geometry separately so densely sampled airfoils cannot hide others."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim == 1:
        actual = actual[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    groups = pd.Series(groups)
    if (
        actual.ndim != 2 or predicted.shape != actual.shape
        or len(groups) != len(actual) or groups.isna().any()
        or len(target_names) != actual.shape[1]
    ):
        raise ValueError("Geometry IDs, predictions, and named targets must have matching shapes.")
    records = []
    for geometry_id, indices in groups.groupby(groups, sort=False).indices.items():
        for index, target in enumerate(target_names):
            records.append({
                "geometry_id": geometry_id, "target": target, "n_predictions": len(indices),
                **regression_metrics(actual[indices, index], predicted[indices, index]),
            })
    return pd.DataFrame(records)
