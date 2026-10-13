import json

import numpy as np
import pytest

from src.utils.metrics import regression_metrics


def test_relative_error_is_reported_alongside_absolute():
    """v1's Cd MAE of 0.0049 looked small against a 1st-percentile Cd of 0.006."""
    scores = regression_metrics([0.006, 0.5], [0.0109, 0.5049])
    assert scores["mae"] == pytest.approx(0.0049)
    # The same absolute error is 82% of a cruise drag and 1% of a stalled one.
    assert scores["mean_relative_error"] == pytest.approx((0.0049 / 0.006 + 0.0049 / 0.5) / 2)
    assert scores["median_relative_error"] == pytest.approx((0.0049 / 0.006 + 0.0049 / 0.5) / 2)


def test_rows_with_zero_actual_are_excluded_not_counted_as_infinite():
    scores = regression_metrics([0.0, 2.0, 4.0], [1.0, 3.0, 5.0])
    assert np.isfinite(scores["mean_relative_error"])
    assert scores["mean_relative_error"] == pytest.approx((0.5 + 0.25) / 2)
    assert scores["median_relative_error"] == pytest.approx(0.375)


def test_all_zero_actuals_give_null_relative_error_not_nan():
    scores = regression_metrics([0.0, 0.0], [0.1, 0.2])
    assert scores["mean_relative_error"] is None
    assert scores["median_relative_error"] is None
    # run.json is written with allow_nan=False, so None is the only safe value.
    assert json.loads(json.dumps(scores, allow_nan=False))["median_relative_error"] is None


def test_median_resists_the_near_zero_lift_rows_that_dominate_the_mean():
    actual = [1e-4, 1.0, 1.1, 1.2]
    predicted = [0.1, 1.01, 1.11, 1.21]
    scores = regression_metrics(actual, predicted)
    assert scores["mean_relative_error"] > 100
    assert scores["median_relative_error"] < 0.02


def test_r2_is_null_for_a_single_observation():
    assert regression_metrics([1.0], [1.5])["r2"] is None
    assert regression_metrics([1.0, 2.0], [1.0, 2.0])["r2"] == pytest.approx(1.0)


def test_perfect_predictions_score_zero_error():
    assert regression_metrics([1.0, 2.0], [1.0, 2.0]) == {
        "mae": 0.0, "rmse": 0.0, "r2": 1.0,
        "mean_relative_error": 0.0, "median_relative_error": 0.0,
    }
