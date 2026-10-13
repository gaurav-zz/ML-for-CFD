import json
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.training.evaluate import evaluate_model


def test_evaluation_reports_each_target_and_times_one_prediction_batch():
    x = np.zeros((3, 2))
    actual = pd.DataFrame({"Cl": [0.0, 2.0, 4.0], "Cd": [0.0, 0.2, 0.4]})
    model = Mock()
    model.predict.return_value = np.array([[1.0, 0.0], [3.0, 0.3], [5.0, 0.5]])

    with patch("src.training.evaluate.time.perf_counter", side_effect=[10.0, 10.25]):
        result = evaluate_model(model, x, actual, target_names=["Cl", "Cd"])

    model.predict.assert_called_once_with(x)
    # Relative error skips the zero-valued row, where it is undefined.
    assert result["targets"]["Cl"] == pytest.approx({
        "mae": 1.0,
        "rmse": 1.0,
        "r2": 0.625,
        "mean_relative_error": 0.375,
        "median_relative_error": 0.375,
    })
    assert result["targets"]["Cd"] == pytest.approx({
        "mae": 0.2 / 3,
        "rmse": np.sqrt(0.02 / 3),
        "r2": 0.75,
        "mean_relative_error": 0.375,
        "median_relative_error": 0.375,
    })
    assert result["inference_time_seconds"] == 0.25
    assert result["n_predictions"] == 3


def test_single_observation_metrics_serialize_undefined_r2_as_null():
    model = Mock()
    model.predict.return_value = np.array([[1.5, 0.03]])
    result = evaluate_model(
        model, np.zeros((1, 1)), np.array([[1.0, 0.02]]), target_names=["Cl", "Cd"],
    )

    assert result["targets"]["Cl"] == {
        "mae": 0.5, "rmse": 0.5, "r2": None,
        "mean_relative_error": 0.5, "median_relative_error": 0.5,
    }
    assert result["targets"]["Cd"]["mae"] == pytest.approx(0.01)
    persisted = json.loads(json.dumps(result, allow_nan=False))
    assert persisted["targets"]["Cl"]["r2"] is None
    assert persisted["targets"]["Cd"]["r2"] is None


@pytest.mark.parametrize("column_target", [False, True])
def test_evaluation_supports_a_named_single_target(column_target):
    actual = np.array([1.0, 2.0, 3.0])
    model = Mock()
    model.predict.return_value = actual.copy()
    if column_target:
        actual = actual[:, None]

    result = evaluate_model(model, np.zeros((3, 1)), actual, target_names=["Cl"])

    assert result["targets"] == {"Cl": {
        "mae": 0.0, "rmse": 0.0, "r2": 1.0,
        "mean_relative_error": 0.0, "median_relative_error": 0.0,
    }}


@pytest.mark.parametrize("names", [["Cl"], ["Cl", "Cl"], ["Cl", ""], "Cl"])
def test_evaluation_rejects_missing_or_ambiguous_target_names(names):
    with pytest.raises(ValueError, match="one unique, non-empty target name"):
        evaluate_model(Mock(), np.zeros((3, 1)), np.ones((3, 2)), target_names=names)


def test_evaluation_rejects_mislabeled_dataframe_columns():
    actual = pd.DataFrame({"Cl": [1.0, 2.0], "Cd": [0.1, 0.2]})
    with pytest.raises(ValueError, match="columns in order"):
        evaluate_model(Mock(), np.zeros((2, 1)), actual, target_names=["Cd", "Cl"])


@pytest.mark.parametrize("prediction", [np.ones((3, 1)), np.ones((2, 2))])
def test_evaluation_rejects_prediction_shape_mismatches(prediction):
    model = Mock()
    model.predict.return_value = prediction
    with pytest.raises(ValueError, match="Prediction shape"):
        evaluate_model(model, np.zeros((3, 1)), np.ones((3, 2)), target_names=["Cl", "Cd"])
