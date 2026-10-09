import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

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
@pytest.mark.parametrize("scale_targets", [False, True], ids=["unscaled", "scaled"])
def test_train_gradient_boosting_target_shapes_and_roundtrip(tmp_path, target_columns, scale_targets):
    x = np.arange(120, dtype=float).reshape(40, 3)
    y = x[:, 0] if target_columns is None else x[:, :target_columns]
    options = {"scale_targets": True, "max_iter": 5, "early_stopping": False} if scale_targets else {}
    model, elapsed = train_model("gradient_boosting", x, y, output_dir=tmp_path, **options)

    if target_columns is None:
        estimators = [model]
    else:
        assert type(model) is MultiOutputRegressor
        estimators = model.estimators_
    for index, estimator in enumerate(estimators):
        regressor = estimator
        if scale_targets:
            assert type(estimator) is TransformedTargetRegressor
            assert isinstance(estimator.transformer_, StandardScaler)
            target = y if target_columns is None else y[:, index]
            np.testing.assert_allclose(estimator.transformer_.mean_, target.mean())
            assert estimator.transformer_.n_samples_seen_ == len(y)
            regressor = estimator.regressor_
            assert regressor.n_iter_ == 5
            assert regressor.early_stopping is False
        assert isinstance(regressor, HistGradientBoostingRegressor)
    predictions = model.predict(x)
    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
    assert elapsed >= 0
    saved_model = joblib.load(tmp_path / "gradient_boosting.joblib")
    np.testing.assert_array_equal(saved_model.predict(x), predictions)


@pytest.mark.parametrize("target_columns", [1, 2])
def test_scaled_gradient_boosting_uses_portable_per_target_estimators(tmp_path, target_columns):
    x = np.arange(120, dtype=float).reshape(40, 3)
    y = np.column_stack((1000 + x[:, 0], 0.01 + x[:, 1] / 10000))[:, :target_columns]
    model, _ = train_model(
        "gradient_boosting", x, y, output_dir=tmp_path,
        scale_targets=True, max_iter=3, early_stopping=False,
    )
    assert type(model) is MultiOutputRegressor
    assert type(model.estimator) is TransformedTargetRegressor
    for index, estimator in enumerate(model.estimators_):
        np.testing.assert_allclose(estimator.transformer_.mean_, y[:, index].mean())
        np.testing.assert_allclose(estimator.transformer_.var_, y[:, index].var())
        independent = clone(model.estimator).fit(x, y[:, index])
        np.testing.assert_allclose(model.predict(x)[:, index], independent.predict(x))


@pytest.mark.parametrize("builder, expected", [
    (build_gb, {
        "max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31,
        "early_stopping": "auto", "random_state": 42, "l2_regularization": 1e-4,
    }),
    (build_rf, {
        "n_estimators": 200, "random_state": 42, "n_jobs": -1,
        "min_samples_leaf": 2, "max_depth": None, "max_leaf_nodes": None,
        "max_samples": None,
    }),
    (build_mlp, {
        "hidden_layer_sizes": (128, 64), "max_iter": 500, "random_state": 42,
        "batch_size": "auto", "early_stopping": True, "learning_rate_init": 0.001,
    }),
])
def test_model_factory_defaults_are_preserved(builder, expected):
    model = builder()
    regressor = model.named_steps["model"] if builder is build_mlp else model
    params = regressor.get_params()
    assert {name: params[name] for name in expected} == expected
    if builder is build_mlp:
        assert isinstance(model.named_steps["scaler"], StandardScaler)


@pytest.mark.parametrize("builder, options", [
    (build_gb, {
        "max_iter": 100, "learning_rate": 0.025, "max_leaf_nodes": 15,
        "early_stopping": False, "random_state": 42,
    }),
    (build_rf, {
        "n_estimators": 100, "max_depth": 20, "max_leaf_nodes": 2048,
        "min_samples_leaf": 5, "max_samples": 0.5, "n_jobs": 2, "random_state": 42,
    }),
    (build_mlp, {
        "hidden_layers": (64, 32), "max_iter": 100, "batch_size": 1024,
        "early_stopping": False, "random_state": 42, "learning_rate_init": 0.002,
    }),
])
def test_model_factories_accept_explicit_comparison_limits(builder, options):
    model = builder(**options)
    regressor = model.named_steps["model"] if builder is build_mlp else model
    expected = dict(options)
    if builder is build_mlp:
        expected["hidden_layer_sizes"] = expected.pop("hidden_layers")
    params = regressor.get_params()
    assert {name: params[name] for name in expected} == expected
    if builder is build_gb:
        assert params["l2_regularization"] == 1e-4


@pytest.mark.parametrize("target_columns", [None, 1, 2], ids=["single-target", "single-column", "multi-target"])
def test_target_scaler_training_statistics_and_physical_unit_roundtrip(tmp_path, target_columns):
    x_train = np.linspace(-2, 2, 24).reshape(-1, 1)
    x_holdout = np.array([[5.0], [8.0], [12.0]])
    y_train = np.column_stack((1000 + 80 * x_train[:, 0], 0.03 + 0.004 * x_train[:, 0]))
    y_holdout = np.column_stack((1000 + 80 * x_holdout[:, 0], 0.03 + 0.004 * x_holdout[:, 0]))
    if target_columns is None:
        y_train, y_holdout = y_train[:, 0], y_holdout[:, 0]
    else:
        y_train, y_holdout = y_train[:, :target_columns], y_holdout[:, :target_columns]

    model, elapsed = train_model("linear", x_train, y_train, output_dir=tmp_path, scale_targets=True)

    assert isinstance(model, TransformedTargetRegressor)
    assert isinstance(model.transformer_, StandardScaler)
    assert model.transformer_.n_samples_seen_ == len(y_train)
    np.testing.assert_allclose(model.transformer_.mean_, np.mean(y_train, axis=0))
    np.testing.assert_allclose(model.transformer_.var_, np.var(y_train, axis=0))
    combined_mean = np.mean(np.concatenate((y_train, y_holdout)), axis=0)
    assert not np.allclose(model.transformer_.mean_, combined_mean)
    predictions = model.predict(x_holdout)
    assert predictions.shape == y_holdout.shape
    np.testing.assert_allclose(predictions, y_holdout)
    assert elapsed >= 0

    saved_model = joblib.load(tmp_path / "linear.joblib")
    assert isinstance(saved_model, TransformedTargetRegressor)
    np.testing.assert_array_equal(saved_model.transformer_.mean_, model.transformer_.mean_)
    np.testing.assert_array_equal(saved_model.predict(x_holdout), predictions)
    np.testing.assert_allclose(model.transformer_.mean_, np.mean(y_train, axis=0))


def test_metrics():
    metrics = regression_metrics([1, 2], [1, 3])
    assert set(metrics) == {"mae", "rmse", "r2"}
