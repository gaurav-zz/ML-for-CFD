"""NumPy-only local tests; fitting tests are opt-in on Kaggle CUDA only."""

import builtins
import importlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from src.models.torch_mlp import (
    EXTRA_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    TorchMLPRegressor,
    build_model,
)


def _frame(n=19, seed=12):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, 113)), columns=RAW_FEATURE_NAMES)
    frame.loc[:, "Alpha"] = np.linspace(-16, 20, n)
    frame.loc[:, "alpha_sin"] = np.sin(np.deg2rad(frame["Alpha"]))
    frame.loc[:, "alpha_cos"] = np.cos(np.deg2rad(frame["Alpha"]))
    frame.loc[:, "Ncrit"] = 9.0
    frame.loc[:, "log10_re"] = 6.0
    return frame


def _reflect(frame):
    reflected = frame.copy()
    columns = [name for name in frame if name.startswith("camber_")]
    columns += ["max_camber", "Alpha", "alpha_sin"]
    reflected.loc[:, columns] = -reflected.loc[:, columns]
    return reflected


def _portable_model(frame, *, symmetry=True, target_names=("Cl", "Cd", "Cm")):
    """Construct saved-weight state without torch or any network fitting."""
    model = build_model(hidden_layers=(7,), batch_size=5, symmetry=symmetry, target_names=target_names)
    model.feature_names_in_ = np.asarray(frame.columns, dtype=object)
    model.n_features_in_ = 113
    model.n_outputs_ = 3
    model.symmetry_ = symmetry
    model.reflection_signs_ = model._reflection_signs(frame.columns)
    model.target_signs_ = np.array([-1 if name in ("Cl", "Cm") else 1 for name in target_names], dtype=np.float32)
    raw = model._feature_array(frame)
    model.input_scaler_ = StandardScaler().fit(raw)
    model.target_scaler_ = StandardScaler().fit(np.array([[2, -6, -0.3], [8, -2, 0.1]], dtype=np.float32))
    rng = np.random.default_rng(4)
    model.coefs_ = [rng.normal(0, 0.1, size=(7, 115)).astype(np.float32),
                    rng.normal(0, 0.1, size=(3, 7)).astype(np.float32)]
    model.intercepts_ = [rng.normal(0, 0.1, size=7).astype(np.float32),
                         np.array([0.4, -0.2, 0.7], dtype=np.float32)]
    model.training_metadata_ = {"accelerator": "cuda", "best_epoch": 3, "seed": 42}
    return model


def test_factory_defaults_clone_and_no_fitted_state():
    model = build_model()
    assert is_regressor(model)
    assert clone(model).get_params() == model.get_params()
    assert model.hidden_layers == (128, 128, 64)
    assert model.max_iter == 300
    assert model.batch_size == 4096
    assert model.learning_rate_init == 0.001
    assert model.weight_decay == 1e-5
    assert model.huber_delta == 0.05
    assert model.target_weights == (1, 2, 0.5)
    assert model.symmetry is True
    assert model.target_names == ("Cl", "Cd", "Cm")
    assert model.patience == 30
    assert model.min_delta == 1e-6
    with pytest.raises(NotFittedError):
        model.predict(_frame())


def test_internal_features_and_geometry_reflection():
    frame = _frame()
    original = frame.copy()
    values = TorchMLPRegressor._feature_array(frame)
    reflected = TorchMLPRegressor._feature_array(_reflect(frame))
    assert values.shape == (len(frame), 115)
    assert values.dtype == np.float32
    np.testing.assert_allclose(values[:, -2], np.sin(2 * np.deg2rad(frame["Alpha"])), atol=1e-7)
    np.testing.assert_allclose(values[:, -1], np.sin(np.deg2rad(frame["Alpha"])) ** 2, atol=1e-7)
    signs = TorchMLPRegressor._reflection_signs(frame.columns)
    np.testing.assert_array_equal(reflected, values * signs)
    for name in ("max_camber_x", "max_thickness_x", "thickness_025", "alpha_cos", "Ncrit", "area"):
        assert signs[RAW_FEATURE_NAMES.index(name)] == 1
    assert signs[-2] == -1 and signs[-1] == 1
    pd.testing.assert_frame_equal(frame, original)
    assert EXTRA_FEATURE_NAMES == ("alpha_sin2", "alpha_sin_sq")


@pytest.mark.parametrize("target_names", [("Cl", "Cd", "Cm"), ("Cm", "Cl", "Cd")])
def test_numpy_symmetry_in_unscaled_fitting_coordinates(target_names):
    frame = _frame()
    model = _portable_model(frame, target_names=target_names)
    prediction = model.predict(frame)
    np.testing.assert_allclose(model.predict(_reflect(frame)), prediction * model.target_signs_, atol=1e-6)
    raw = model._feature_array(frame)
    expected = 0.5 * (model._numpy_forward(raw)
                      + model.target_signs_ * model._numpy_forward(raw * model.reflection_signs_))
    np.testing.assert_allclose(prediction, expected, atol=1e-6)
    # Standardization means are deliberately nonzero. Projecting normalized
    # network outputs before inverse-scaling incorrectly retains odd means.
    mean = model.target_scaler_.mean_
    scale = model.target_scaler_.scale_
    wrong = 0.5 * ((model._numpy_forward(raw) - mean) / scale
                   + model.target_signs_ * (model._numpy_forward(raw * model.reflection_signs_) - mean) / scale)
    wrong = wrong * scale + mean
    odd = model.target_signs_ == -1
    assert not np.allclose(prediction[:, odd], wrong[:, odd])


def test_symmetric_geometry_at_zero_alpha_has_exact_zero_odd_targets():
    frame = _frame()
    model = _portable_model(frame)
    frame.loc[:, [name for name in frame if name.startswith("camber_")]] = 0.0
    frame.loc[:, ["max_camber", "Alpha", "alpha_sin"]] = 0.0
    np.testing.assert_array_equal(model.predict(frame)[:, [0, 2]], 0.0)


def test_symmetry_can_be_disabled_and_numpy_silu_is_stable():
    frame = _frame()
    model = _portable_model(frame, symmetry=False)
    model.intercepts_[0][:2] = [-1000.0, 1000.0]
    with np.errstate(over="raise", invalid="raise"):
        prediction = model.predict(frame)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, model._numpy_forward(model._feature_array(frame)), atol=1e-5)


def test_joblib_roundtrip_and_module_import_without_torch(tmp_path, monkeypatch):
    frame = _frame()
    model = _portable_model(frame)
    expected = model.predict(frame)
    path = tmp_path / "portable.joblib"
    joblib.dump(model, path)
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Portable import/loading/predict must not import torch")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_torch)
        module = importlib.import_module("src.models.torch_mlp")
        importlib.reload(module)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(loaded.predict(frame), expected)
        json.dumps(loaded.training_metadata_, allow_nan=False)
    assert loaded.n_features_in_ == 113
    assert loaded.coefs_[0].shape == (7, 115)
    assert not any(type(value).__module__.startswith("torch") for value in vars(loaded).values())


@pytest.mark.parametrize("validation", [None, (), (None, None), (_frame(),), "validation"])
def test_explicit_validation_guard_precedes_torch_import(validation, monkeypatch):
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Invalid fit must fail before torch import or training")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(ValueError, match="validation_data.*required"):
        build_model().fit(_frame(), np.ones((19, 3)), validation_data=validation)


@pytest.mark.parametrize("options", [
    {"max_iter": 0}, {"batch_size": True}, {"patience": -1}, {"lr_patience": 0},
    {"hidden_layers": ()}, {"hidden_layers": (1.5,)}, {"learning_rate_init": 0},
    {"weight_decay": -1}, {"huber_delta": 0}, {"min_delta": np.nan}, {"lr_factor": 1},
    {"loss": "mae"}, {"device": "cuda:0"}, {"symmetry": 1},
    {"target_names": ("Cl", "logCd", "Cm")}, {"target_names": ("Cl", "Cl", "Cm")},
    {"target_names": ["Cl", "Cd", "Cm"]}, {"target_weights": (1, 2)},
    {"target_weights": (1, 0, 1)}, {"random_state": -1},
])
def test_invalid_hyperparameters_fail_without_training(options):
    with pytest.raises(ValueError):
        build_model(**options)._validate_params()


@pytest.mark.parametrize("mutate, message", [
    (lambda x: x.to_numpy(), "DataFrame"),
    (lambda x: x.drop(columns="Alpha"), "113-feature"),
    (lambda x: x.assign(geometry_id="leak"), "113-feature"),
    (lambda x: x.rename(columns={"Alpha": "alpha_sin"}), "113-feature"),
    (lambda x: x.assign(Alpha="text"), "numeric"),
    (lambda x: x.assign(Alpha=1j), "real numeric"),
    (lambda x: x.assign(Alpha=pd.Series([pd.NA] * len(x), dtype="Float64")), "finite"),
    (lambda x: x.assign(Alpha=np.nan), "finite"),
    (lambda x: x.assign(Alpha=np.inf), "finite"),
    (lambda x: x.assign(Alpha=1e100), "float32"),
    (lambda x: x.iloc[:0], "at least one row"),
    (lambda x: x[x.columns[::-1]], "order"),
])
def test_prediction_schema_guard(mutate, message):
    frame = _frame()
    model = _portable_model(frame)
    with pytest.raises(ValueError, match=message):
        model.predict(mutate(frame))


def test_validation_schema_and_target_guards_do_not_fit():
    frame = _frame()
    model = build_model()
    targets = np.ones((len(frame), 3))
    with pytest.raises(ValueError, match="order"):
        model.fit(frame, targets, validation_data=(frame[frame.columns[::-1]], targets))
    with pytest.raises(ValueError, match="shape"):
        model.fit(frame, targets[:, 0], validation_data=(frame, targets))
    with pytest.raises(ValueError, match="finite"):
        model.fit(frame, targets, validation_data=(frame, targets * np.nan))
    with pytest.raises(ValueError, match="target_names order"):
        model.fit(frame, pd.DataFrame(targets, columns=["Cd", "Cl", "Cm"]), validation_data=(frame, targets))
    for invalid in (targets.astype(str), targets.astype(complex) + 1j):
        with pytest.raises(ValueError, match="numeric"):
            model.fit(frame, invalid, validation_data=(frame, targets))


KAGGLE_GPU_ONLY = pytest.mark.skipif(
    os.environ.get("KAGGLE_GPU_TESTS") != "1" or not Path("/kaggle/working").is_dir(),
    reason="Fitting is permitted only on Kaggle; opt in with KAGGLE_GPU_TESTS=1.",
)


@pytest.fixture
def kaggle_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Optional training test requires CUDA on Kaggle.")
    return torch


def _gpu_data():
    train = _frame(96)
    val = _frame(23, seed=33)
    val.loc[:, "max_thickness"] = 20.0
    train_targets = np.column_stack((
        1.2 + 0.08 * train["Alpha"], -4 + 0.002 * train["Alpha"] ** 2,
        -0.2 + 0.01 * train["Alpha"],
    )).astype(np.float32)
    val_targets = np.column_stack((
        2.2 + 0.08 * val["Alpha"], -3 + 0.002 * val["Alpha"] ** 2,
        -0.3 + 0.01 * val["Alpha"],
    )).astype(np.float32)
    return train, train_targets, val, val_targets


@pytest.mark.gpu
@KAGGLE_GPU_ONLY
@pytest.mark.parametrize("loss,symmetry", [("mse", True), ("huber", True), ("mse", False)])
def test_kaggle_cuda_cap_checkpoint_scalers_and_portable_roundtrip(loss, symmetry, kaggle_cuda, tmp_path):
    torch = kaggle_cuda
    train, targets, val, val_targets = _gpu_data()
    model = build_model(hidden_layers=(8,), max_iter=3, batch_size=32, patience=10,
                        lr_patience=1, device="cuda", loss=loss, symmetry=symmetry)
    assert model.fit(train, targets, validation_data=(val, val_targets)) is model
    assert model.n_iter_ == 3
    metadata = model.training_metadata_
    json.dumps(metadata, allow_nan=False)
    assert metadata["accelerator"] == "cuda"
    assert metadata["device_name"] == torch.cuda.get_device_name()
    assert metadata["torch_version"] == str(torch.__version__)
    assert metadata["cuda_version"] == torch.version.cuda
    assert metadata["stop_reason"] == "max_iter"
    assert metadata["restored_best_checkpoint"] is True
    assert 1 <= metadata["best_epoch"] <= metadata["n_iter"]
    assert metadata["best_loss"] == min(row["validation_loss"] for row in metadata["history"])
    assert len(metadata["history"]) <= 300
    assert metadata["n_train"] == len(train)
    assert metadata["n_validation"] == len(val)
    np.testing.assert_allclose(model.target_scaler_.mean_, targets.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(model.target_scaler_.scale_, targets.std(axis=0), atol=1e-6)
    assert not np.allclose(model.target_scaler_.mean_, np.concatenate((targets, val_targets)).mean(axis=0))
    assert model.target_scaler_.n_samples_seen_ == len(train)
    raw = model._feature_array(train)
    reflected_train = raw * model.reflection_signs_
    expected_scaler = StandardScaler().fit(np.concatenate((raw, reflected_train)) if symmetry else raw)
    np.testing.assert_allclose(model.input_scaler_.mean_, expected_scaler.mean_, atol=1e-7)
    np.testing.assert_allclose(model.input_scaler_.scale_, expected_scaler.scale_, atol=1e-7)
    assert model.input_scaler_.n_samples_seen_ == (2 if symmetry else 1) * len(train)
    prediction = model.predict(val)
    difference = ((prediction - model.target_scaler_.mean_) / model.target_scaler_.scale_
                  - (val_targets - model.target_scaler_.mean_) / model.target_scaler_.scale_)
    if loss == "mse":
        elementwise = difference ** 2
    else:
        error = np.abs(difference)
        elementwise = np.where(error <= model.huber_delta, 0.5 * error ** 2,
                               model.huber_delta * (error - 0.5 * model.huber_delta))
    measured_loss = np.average(elementwise, weights=model.target_weights, axis=1).mean()
    np.testing.assert_allclose(measured_loss, model.best_loss_, rtol=2e-5, atol=2e-6)
    if symmetry:
        np.testing.assert_allclose(model.predict(_reflect(val)), prediction * model.target_signs_, atol=1e-6)
    assert all(isinstance(w, np.ndarray) and w.dtype == np.float32 for w in (*model.coefs_, *model.intercepts_))
    path = tmp_path / "trained.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(val), prediction)


@pytest.mark.gpu
@KAGGLE_GPU_ONLY
def test_kaggle_validation_patience_and_no_cuda_fallback(kaggle_cuda, monkeypatch):
    torch = kaggle_cuda
    train, targets, val, val_targets = _gpu_data()
    model = build_model(hidden_layers=(8,), max_iter=8, batch_size=32, patience=2,
                        min_delta=1e6, lr_patience=1, device="cuda")
    model.fit(train, targets, validation_data=(val, val_targets))
    assert model.n_iter_ == 3
    assert model.training_metadata_["stop_reason"] == "validation_patience"
    assert model.training_metadata_["stopped_early"] is True
    assert model.training_metadata_["restored_best_checkpoint"] is True
    with pytest.raises(ValueError, match="validation_data"):
        build_model(device="cuda").fit(train, targets)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an available CUDA GPU"):
        build_model(device="cuda").fit(train, targets, validation_data=(val, val_targets))


@pytest.mark.gpu
@KAGGLE_GPU_ONLY
def test_kaggle_best_checkpoint_is_not_last_epoch(kaggle_cuda, monkeypatch):
    torch = kaggle_cuda
    train, targets, val, _ = _gpu_data()
    val_targets = np.tile(targets.mean(axis=0), (len(val), 1))
    steps = 0

    def worsening_step(optimizer, closure=None):
        nonlocal steps
        steps += 1
        with torch.no_grad():
            for parameter in optimizer.param_groups[0]["params"]:
                parameter.zero_()
            optimizer.param_groups[0]["params"][-1].fill_(steps)

    # Make each epoch deterministically worse, to detect accidentally saving
    # the final network instead of the best-validation checkpoint.
    monkeypatch.setattr(torch.optim.RAdam, "step", worsening_step)
    model = build_model(hidden_layers=(4,), symmetry=False, loss="mse",
                        max_iter=3, patience=10, batch_size=32, device="cuda")
    model.fit(train, targets, validation_data=(val, val_targets))
    assert model.best_epoch_ == 1
    assert model.n_iter_ == 3
    assert model.training_metadata_["history"][-1]["validation_loss"] > model.best_loss_
    np.testing.assert_array_equal(model.intercepts_[-1], np.full(3, 3, dtype=np.float32))
    normalized = (model.predict(val) - model.target_scaler_.mean_) / model.target_scaler_.scale_
    np.testing.assert_allclose(normalized, 3, atol=1e-6)
    np.testing.assert_allclose(model.best_loss_, 9, atol=1e-5)
