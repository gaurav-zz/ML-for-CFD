"""Evaluate saved or in-memory regressors with timing and standard metrics."""

from __future__ import annotations

import time

import numpy as np

from src.utils.metrics import regression_metrics


def evaluate_model(model, x_test, y_test) -> dict[str, float]:
    started = time.perf_counter()
    prediction = np.asarray(model.predict(x_test))
    inference_time = time.perf_counter() - started
    metrics = regression_metrics(y_test, prediction)
    metrics["inference_time_seconds"] = inference_time
    metrics["n_predictions"] = float(len(prediction))
    return metrics
