import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.models.gradient_boosting import build_model as build_gb
from src.models.linear import build_model as build_linear
from src.models.mlp import build_model as build_mlp
from src.models.polynomial import build_model as build_polynomial
from src.models.random_forest import build_model as build_rf
from src.training.train import train_model
from src.utils.metrics import regression_metrics


@pytest.mark.parametrize("builder", [build_linear, build_polynomial, build_rf, build_gb, build_mlp])
def test_models_fit_and_predict(builder):
    x = np.arange(90, dtype=float).reshape(30, 3)
    y = x[:, 0] if builder is build_gb else np.column_stack((x[:, 0], x[:, 1]))
    model = builder(max_iter=20) if builder is build_mlp else builder()
    model.fit(x, y)
    assert model.predict(x).shape == y.shape


@pytest.mark.parametrize("target_columns", [None, 1, 2], ids=["single-target", "single-column", "multi-target"])
def test_train_gradient_boosting_target_shapes_and_roundtrip(tmp_path, target_columns):
    x = np.arange(120, dtype=float).reshape(40, 3)
    y = x[:, 0] if target_columns is None else x[:, :target_columns]
    model, elapsed = train_model("gradient_boosting", x, y, output_dir=tmp_path)

    if target_columns is None:
        assert isinstance(model, HistGradientBoostingRegressor)
    else:
        assert isinstance(model, MultiOutputRegressor)
        assert isinstance(model.estimator, HistGradientBoostingRegressor)
    predictions = model.predict(x)
    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
    assert elapsed >= 0
    saved_model = joblib.load(tmp_path / "gradient_boosting.joblib")
    np.testing.assert_array_equal(saved_model.predict(x), predictions)


def test_metrics():
    metrics = regression_metrics([1, 2], [1, 3])
    assert set(metrics) == {"mae", "rmse", "r2"}
