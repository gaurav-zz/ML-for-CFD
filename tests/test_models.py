import numpy as np
import pytest

from src.models.gradient_boosting import build_model as build_gb
from src.models.linear import build_model as build_linear
from src.models.mlp import build_model as build_mlp
from src.models.polynomial import build_model as build_polynomial
from src.models.random_forest import build_model as build_rf
from src.utils.metrics import regression_metrics


@pytest.mark.parametrize("builder", [build_linear, build_polynomial, build_rf, build_gb, build_mlp])
def test_models_fit_and_predict(builder):
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = np.column_stack((x[:, 0], x[:, 1]))
    model = builder(max_iter=20) if builder is build_mlp else builder()
    model.fit(x, y)
    assert model.predict(x).shape == y.shape


def test_metrics():
    metrics = regression_metrics([1, 2], [1, 3])
    assert set(metrics) == {"mae", "rmse", "r2"}
