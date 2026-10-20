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


def _portable_model(frame, *, symmetry=True, target_names=("Cl", "Cd", "Cm"), auxiliary_weight=0.0):
    """Construct saved-weight state without torch or any network fitting."""
    model = build_model(hidden_layers=(7,), batch_size=5, symmetry=symmetry, target_names=target_names,
                        target_weights=tuple({"Cl": 1, "Cd": 2, "Cm": 0.5}[name] for name in target_names),
                        auxiliary_weight=auxiliary_weight)
    model.feature_names_in_ = np.asarray(frame.columns, dtype=object)
    model.n_features_in_ = 113
    model.n_outputs_ = len(target_names)
    model.symmetry_ = symmetry
    model.reflection_signs_ = model._reflection_signs(frame.columns)
    model.target_signs_ = np.array([-1 if name in ("Cl", "Cm") else 1 for name in target_names], dtype=np.float32)
    raw = model._feature_array(frame)
    model.input_scaler_ = StandardScaler().fit(raw)
    columns = [("Cl", "Cd", "Cm").index(name) for name in target_names]
    model.target_scaler_ = StandardScaler().fit(np.array([[2, -6, -0.3], [8, -2, 0.1]], dtype=np.float32)[:, columns])
    rng = np.random.default_rng(4)
    model.coefs_ = [rng.normal(0, 0.1, size=(7, 115)).astype(np.float32),
                    rng.normal(0, 0.1, size=(model.n_outputs_, 7)).astype(np.float32)]
    model.intercepts_ = [rng.normal(0, 0.1, size=7).astype(np.float32),
                         np.array([0.4, -0.2, 0.7], dtype=np.float32)[columns]]
    if auxiliary_weight:
        model.auxiliary_scaler_ = StandardScaler().fit(np.array([[-0.2], [0], [0.8]], dtype=np.float32))
        model.coefs_[-1] = np.concatenate((model.coefs_[-1], rng.normal(0, 0.1, size=(1, 7)).astype(np.float32)))
        model.intercepts_[-1] = np.append(model.intercepts_[-1], np.float32(-0.4))
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
    assert model.auxiliary_weight == 0.0
    assert clone(build_model(auxiliary_weight=0.25)).auxiliary_weight == 0.25
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


@pytest.mark.parametrize("auxiliary_weight,legacy", [(0.0, False), (0.0, True), (0.25, False)])
def test_joblib_roundtrip_and_module_import_without_torch(auxiliary_weight, legacy, tmp_path, monkeypatch):
    frame = _frame()
    model = _portable_model(frame, auxiliary_weight=auxiliary_weight)
    if legacy:
        del model.auxiliary_weight
        model.training_metadata_["accelerator"] = "cpu"
    expected = model.predict(frame)
    expected_auxiliary = model.predict_auxiliary(frame) if auxiliary_weight else None
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
        if auxiliary_weight:
            np.testing.assert_array_equal(loaded.predict_auxiliary(frame), expected_auxiliary)
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
    {"auxiliary_weight": -0.1}, {"auxiliary_weight": True}, {"auxiliary_weight": np.bool_(False)},
    {"auxiliary_weight": np.nan}, {"auxiliary_weight": np.inf}, {"auxiliary_weight": "0.2"},
    {"auxiliary_weight": 1j},
    {"auxiliary_weight": 0.1, "target_names": ("Cm", "Cl", "Cd")},
    {"auxiliary_weight": 0.1, "target_names": ("Cd",), "target_weights": (2,)},
    {"symmetry": False, "target_names": ("Cl", "Cd", "Cdp")},
])
def test_invalid_hyperparameters_fail_without_training(options):
    with pytest.raises(ValueError):
        build_model(**options)._validate_params()


@pytest.mark.parametrize("mutate, message", [
    (lambda x: x.to_numpy(), "DataFrame"),
    (lambda x: x.drop(columns="Alpha"), "113-feature"),
    (lambda x: x.assign(geometry_id="leak"), "113-feature"),
    (lambda x: x.assign(Cdp=0.1), "113-feature"),
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


@pytest.mark.parametrize("target_names", [("Cd",), ("Cl", "Cm"), ("Cm", "Cl", "Cd")])
def test_baseline_known_primary_subsets_remain_supported(target_names):
    frame = _frame()
    model = _portable_model(frame, target_names=target_names)
    model._validate_params()
    prediction = model.predict(frame)
    assert prediction.shape == (len(frame), len(target_names))
    # Equivalent float32 computations need not be bit-identical across BLAS
    # kernels/layouts. This checks mathematical parity, not serialization.
    np.testing.assert_allclose(model.predict(_reflect(frame)), prediction * model.target_signs_,
                               rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("symmetry", [True, False])
def test_auxiliary_portable_head_is_separate_signed_and_even(symmetry):
    frame = _frame()
    baseline = _portable_model(frame, symmetry=symmetry)
    model = _portable_model(frame, symmetry=symmetry, auxiliary_weight=0.25)
    np.testing.assert_array_equal(model.coefs_[-1][:3], baseline.coefs_[-1])
    np.testing.assert_allclose(model.predict(frame), baseline.predict(frame), atol=1e-6)
    assert model.n_outputs_ == 3
    assert model.predict(frame).shape == (len(frame), 3)
    prediction = model.predict_auxiliary(frame)
    assert prediction.shape == (len(frame), 1)
    raw = model._feature_array(frame)
    direct = model._latent_prediction(raw)[:, 3:] * model.auxiliary_scaler_.scale_ + model.auxiliary_scaler_.mean_
    if symmetry:
        reflected = model._latent_prediction(raw * model.reflection_signs_)[:, 3:]
        reflected = reflected * model.auxiliary_scaler_.scale_ + model.auxiliary_scaler_.mean_
        direct = 0.5 * (direct + reflected)
        np.testing.assert_allclose(model.predict_auxiliary(_reflect(frame)), prediction,
                                   rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(prediction, direct, rtol=2e-6, atol=1e-7)
    # Cdp decoding is identity, not exp/clipping, and retains the nonzero offset.
    model.coefs_[-1][-1] = 0
    model.intercepts_[-1][-1] = -2
    np.testing.assert_allclose(model.predict_auxiliary(frame),
                               -2 * model.auxiliary_scaler_.scale_[0] + model.auxiliary_scaler_.mean_[0], atol=1e-7)
    assert (model.predict_auxiliary(frame) < 0).all()
    np.testing.assert_allclose(model.predict(frame), baseline.predict(frame), atol=1e-6)
    with pytest.raises(NotFittedError):
        baseline.predict_auxiliary(frame)


@pytest.mark.parametrize("values", [
    [-0.2, 0, np.nan, 0.3], np.array([[-0.2], [0], [np.nan], [0.3]]),
    pd.Series([-0.2, 0, np.nan, 0.3], name="Cdp"),
    pd.DataFrame({"Cdp": [-0.2, 0, np.nan, 0.3]}),
])
def test_auxiliary_array_identity_and_nan_missing(values):
    result = TorchMLPRegressor._auxiliary_array(values, 4)
    np.testing.assert_array_equal(result, np.array([[-0.2], [0], [np.nan], [0.3]], dtype=np.float32))
    assert result.dtype == np.float32


@pytest.mark.parametrize("values", [
    np.ones((19, 2)), np.ones(18), np.ones((19, 1, 1)), np.array(0.1),
    np.full(19, np.inf), np.full(19, -np.inf), np.full(19, 1e100),
    np.full(19, True), np.full(19, "0.1"), np.full(19, 0.1 + 0j),
    [True, *([0.1] * 18)], [[False], *([[0.1]] * 18)],
    pd.DataFrame({"Cd": np.ones(19)}),
])
@pytest.mark.parametrize("validation", [False, True])
def test_auxiliary_guards_precede_torch_import(values, validation, monkeypatch):
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Invalid auxiliary labels must fail before Torch import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    frame = _frame()
    targets = np.ones((len(frame), 3))
    options = {"auxiliary_targets": np.zeros(len(frame))}
    options["validation_auxiliary_targets" if validation else "auxiliary_targets"] = values
    with pytest.raises(ValueError, match="auxiliary_targets"):
        build_model(auxiliary_weight=0.25).fit(frame, targets, validation_data=(frame, targets), **options)


@pytest.mark.parametrize("weight,train_auxiliary,val_auxiliary,message", [
    (0.0, np.ones(19), None, "auxiliary_weight"),
    (0.0, None, np.ones(19), "auxiliary_weight"),
    (0.25, None, np.ones(19), "observed training"),
    (0.25, np.full(19, np.nan), None, "observed training"),
])
def test_auxiliary_enablement_guards_do_not_fit(weight, train_auxiliary, val_auxiliary, message, monkeypatch):
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Invalid auxiliary configuration must fail before Torch import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    frame = _frame()
    targets = np.ones((len(frame), 3))
    with pytest.raises(ValueError, match=message):
        build_model(auxiliary_weight=weight).fit(
            frame, targets, validation_data=(frame, targets), auxiliary_targets=train_auxiliary,
            validation_auxiliary_targets=val_auxiliary,
        )


KAGGLE_GPU_ONLY = pytest.mark.skipif(
    os.environ.get("KAGGLE_GPU_TESTS") != "1" or not Path("/kaggle/working").is_dir(),
    reason="Fitting is permitted only on Kaggle; opt in with KAGGLE_GPU_TESTS=1.",
)


@pytest.mark.cloud_fit
@pytest.mark.skipif(
    os.environ.get("KAGGLE_CPU_TESTS") != "1" or not Path("/kaggle/working").is_dir(),
    reason="CPU fitting is permitted only on Kaggle; opt in with KAGGLE_CPU_TESTS=1.",
)
@pytest.mark.parametrize("loss", ["mse", "huber"])
@pytest.mark.parametrize("auxiliary_weight", [0.0, 0.25])
def test_kaggle_cpu_fitting_and_portable_predictions(loss, auxiliary_weight, tmp_path):
    pytest.importorskip("torch")
    train, targets, val, val_targets = _gpu_data()
    model = build_model(hidden_layers=(8,), max_iter=3, batch_size=32, patience=10,
                        device="cpu", loss=loss, auxiliary_weight=auxiliary_weight)
    auxiliary = np.linspace(-0.2, 0.4, len(train), dtype=np.float32)
    auxiliary[::3] = np.nan
    options = {"auxiliary_targets": auxiliary, "validation_auxiliary_targets": np.full(len(val), np.nan)} if auxiliary_weight else {}
    model.fit(train, targets, validation_data=(val, val_targets), **options)
    assert model.training_metadata_["accelerator"] == "cpu"
    assert model.training_metadata_["restored_best_checkpoint"] is True
    assert model.n_iter_ == 3
    json.dumps(model.training_metadata_, allow_nan=False)
    assert model.training_metadata_["stopping_metric"] == "primary_normalized_validation_loss"
    assert all(row["validation_auxiliary_loss"] is None for row in model.training_metadata_["history"])
    if auxiliary_weight:
        assert model.auxiliary_scaler_.n_samples_seen_ == np.isfinite(auxiliary).sum()
        np.testing.assert_allclose(model.auxiliary_scaler_.mean_, np.nanmean(auxiliary), atol=1e-7)
        assert model.training_metadata_["auxiliary_label_coverage"]["validation"]["n_observed"] == 0
        assert model.training_metadata_["auxiliary_label_coverage"]["validation"]["mean"] is None
    prediction = model.predict(val)
    np.testing.assert_allclose(model.predict(_reflect(val)), prediction * model.target_signs_, atol=1e-6)
    error = np.abs((prediction - val_targets) / model.target_scaler_.scale_)
    elementwise = error ** 2 if loss == "mse" else np.where(
        error <= model.huber_delta, 0.5 * error ** 2,
        model.huber_delta * (error - 0.5 * model.huber_delta),
    )
    measured = np.average(elementwise, weights=model.target_weights, axis=1).mean()
    np.testing.assert_allclose(measured, model.best_loss_, rtol=2e-5, atol=2e-6)
    path = tmp_path / "cpu.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(val), prediction)
    if auxiliary_weight:
        np.testing.assert_array_equal(joblib.load(path).predict_auxiliary(val), model.predict_auxiliary(val))


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
@pytest.mark.parametrize("auxiliary_weight", [0.0, 0.25])
def test_kaggle_cuda_cap_checkpoint_scalers_and_portable_roundtrip(loss, symmetry, auxiliary_weight, kaggle_cuda, tmp_path):
    torch = kaggle_cuda
    train, targets, val, val_targets = _gpu_data()
    model = build_model(hidden_layers=(8,), max_iter=3, batch_size=32, patience=10,
                        lr_patience=1, device="cuda", loss=loss, symmetry=symmetry, auxiliary_weight=auxiliary_weight)
    auxiliary = np.linspace(-0.1, 0.3, len(train), dtype=np.float32)
    auxiliary[0] = 0.0
    auxiliary[1::4] = np.nan
    val_auxiliary = np.linspace(1, 2, len(val), dtype=np.float32)
    val_auxiliary[1::3] = np.nan
    options = {"auxiliary_targets": auxiliary, "validation_auxiliary_targets": val_auxiliary} if auxiliary_weight else {}
    assert model.fit(train, targets, validation_data=(val, val_targets), **options) is model
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
    assert metadata["stopping_metric"] == "primary_normalized_validation_loss"
    assert metadata["target_names"] == ["Cl", "Cd", "Cm"]
    assert metadata["target_weights"] == [1, 2, 0.5]
    assert metadata["auxiliary_weight"] == auxiliary_weight
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
    assert prediction.shape == (len(val), 3)
    assert model.n_outputs_ == 3
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
    if auxiliary_weight:
        assert metadata["auxiliary_name"] == "Cdp"
        assert metadata["auxiliary_transform"] == "identity"
        assert metadata["auxiliary_label_coverage"]["train"]["n_observed"] == np.isfinite(auxiliary).sum()
        assert metadata["auxiliary_label_coverage"]["train"]["min"] == float(np.nanmin(auxiliary))
        assert metadata["auxiliary_label_coverage"]["validation"]["n_missing"] == np.isnan(val_auxiliary).sum()
        assert model.auxiliary_scaler_.n_samples_seen_ == np.isfinite(auxiliary).sum()
        np.testing.assert_allclose(model.auxiliary_scaler_.mean_, np.nanmean(auxiliary), atol=1e-7)
        np.testing.assert_allclose(model.auxiliary_scaler_.scale_, np.nanstd(auxiliary), atol=1e-7)
        assert not np.allclose(model.auxiliary_scaler_.mean_, np.nanmean(np.concatenate((auxiliary, val_auxiliary))))
        assert metadata["auxiliary_scaler"]["mean"] == model.auxiliary_scaler_.mean_[0]
        assert metadata["auxiliary_scaler"]["scale"] == model.auxiliary_scaler_.scale_[0]
        auxiliary_prediction = model.predict_auxiliary(val)
        assert auxiliary_prediction.shape == (len(val), 1)
        observed = np.isfinite(val_auxiliary)
        error = np.abs((auxiliary_prediction[observed, 0] - val_auxiliary[observed]) / model.auxiliary_scaler_.scale_[0])
        elements = error ** 2 if loss == "mse" else np.where(
            error <= model.huber_delta, 0.5 * error ** 2,
            model.huber_delta * (error - 0.5 * model.huber_delta),
        )
        np.testing.assert_allclose(elements.mean(), metadata["best_auxiliary_validation_loss"], rtol=2e-5, atol=2e-6)
        if symmetry:
            np.testing.assert_allclose(model.predict_auxiliary(_reflect(val)), auxiliary_prediction,
                                       rtol=2e-6, atol=1e-7)
    else:
        assert metadata["auxiliary_name"] is None
        assert metadata["auxiliary_scaler"] is None
        assert all(row["train_auxiliary_loss"] is None for row in metadata["history"])
        assert all(row["train_total_loss"] == row["train_loss"] for row in metadata["history"])
    assert all(isinstance(w, np.ndarray) and w.dtype == np.float32 for w in (*model.coefs_, *model.intercepts_))
    path = tmp_path / "trained.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(val), prediction)
    if auxiliary_weight:
        np.testing.assert_array_equal(joblib.load(path).predict_auxiliary(val), auxiliary_prediction)


@pytest.mark.gpu
@KAGGLE_GPU_ONLY
@pytest.mark.parametrize("auxiliary_weight", [0.0, 0.25])
def test_kaggle_validation_patience_and_no_cuda_fallback(auxiliary_weight, kaggle_cuda, monkeypatch):
    torch = kaggle_cuda
    train, targets, val, val_targets = _gpu_data()
    model = build_model(hidden_layers=(8,), max_iter=8, batch_size=32, patience=2,
                        min_delta=1e6, lr_patience=1, device="cuda", auxiliary_weight=auxiliary_weight)
    options = {"auxiliary_targets": np.zeros(len(train))} if auxiliary_weight else {}
    model.fit(train, targets, validation_data=(val, val_targets), **options)
    assert model.n_iter_ == 3
    assert model.training_metadata_["stop_reason"] == "validation_patience"
    assert model.training_metadata_["stopped_early"] is True
    assert model.training_metadata_["restored_best_checkpoint"] is True
    assert all(row["validation_auxiliary_loss"] is None for row in model.training_metadata_["history"])
    json.dumps(model.training_metadata_, allow_nan=False)
    with pytest.raises(ValueError, match="validation_data"):
        build_model(device="cuda").fit(train, targets)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires an available CUDA GPU"):
        build_model(device="cuda").fit(train, targets, validation_data=(val, val_targets))


@pytest.mark.gpu
@KAGGLE_GPU_ONLY
@pytest.mark.parametrize("auxiliary_weight", [0.0, 0.25])
def test_kaggle_best_checkpoint_is_not_last_epoch(auxiliary_weight, kaggle_cuda, monkeypatch):
    torch = kaggle_cuda
    train, targets, val, _ = _gpu_data()
    val_targets = np.tile(targets.mean(axis=0), (len(val), 1))
    steps = 0
    scheduled = []
    original_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def record_schedule(scheduler, value, *args, **kwargs):
        scheduled.append(value)
        return original_schedule(scheduler, value, *args, **kwargs)

    def worsening_step(optimizer, closure=None):
        nonlocal steps
        steps += 1
        with torch.no_grad():
            for parameter in optimizer.param_groups[0]["params"]:
                parameter.zero_()
            optimizer.param_groups[0]["params"][-1].fill_(steps)
            if auxiliary_weight:
                optimizer.param_groups[0]["params"][-1][-1] = 1000 - steps

    # Make each epoch deterministically worse, to detect accidentally saving
    # the final network instead of the best-validation checkpoint.
    monkeypatch.setattr(torch.optim.RAdam, "step", worsening_step)
    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", record_schedule)
    model = build_model(hidden_layers=(4,), symmetry=False, loss="mse",
                        max_iter=3, patience=10, lr_patience=1, batch_size=32,
                        device="cuda", auxiliary_weight=auxiliary_weight)
    options = {"auxiliary_targets": np.zeros(len(train)), "validation_auxiliary_targets": np.zeros(len(val))} if auxiliary_weight else {}
    model.fit(train, targets, validation_data=(val, val_targets), **options)
    assert model.best_epoch_ == 1
    assert model.n_iter_ == 3
    assert model.training_metadata_["history"][-1]["validation_loss"] > model.best_loss_
    np.testing.assert_array_equal(model.intercepts_[-1][:3], np.full(3, 3, dtype=np.float32))
    normalized = (model.predict(val) - model.target_scaler_.mean_) / model.target_scaler_.scale_
    np.testing.assert_allclose(normalized, 3, atol=1e-6)
    np.testing.assert_allclose(model.best_loss_, 9, atol=1e-5)
    history = model.training_metadata_["history"]
    assert scheduled == [row["validation_loss"] for row in history]
    if auxiliary_weight:
        # The combined validation objective IMPROVES while primary degrades.
        # Best state and scheduler must nevertheless follow primary alone.
        combined = [row["validation_loss"] + auxiliary_weight * row["validation_auxiliary_loss"] for row in history]
        assert np.all(np.diff(combined) < 0)
        assert model.intercepts_[-1][-1] == 997
        # CUDA's float32 mean reduction may differ from the exact integer by
        # one ULP (observed 994009.0625 versus 994009); checkpoint values above
        # remain exact, so a wrong epoch cannot pass this tolerance.
        np.testing.assert_allclose(model.training_metadata_["best_auxiliary_validation_loss"],
                                   997 ** 2, rtol=2 * np.finfo(np.float32).eps, atol=0)


@pytest.mark.gpu
@KAGGLE_GPU_ONLY
@pytest.mark.parametrize("symmetry", [False, True])
def test_kaggle_auxiliary_initialization_preserves_trunk_primary_and_shuffle(symmetry, kaggle_cuda, monkeypatch):
    torch = kaggle_cuda
    train, targets, val, val_targets = _gpu_data()
    initial_parameters, orders = [], []
    original_init = torch.optim.RAdam.__init__
    original_randperm = torch.randperm

    def capture_init(optimizer, parameters, *args, **kwargs):
        parameters = list(parameters)
        initial_parameters.append([parameter.detach().clone() for parameter in parameters])
        original_init(optimizer, parameters, *args, **kwargs)

    def capture_order(*args, **kwargs):
        order = original_randperm(*args, **kwargs)
        orders.append(order.clone())
        return order

    monkeypatch.setattr(torch.optim.RAdam, "__init__", capture_init)
    monkeypatch.setattr(torch.optim.RAdam, "step", lambda optimizer, closure=None: None)
    monkeypatch.setattr(torch, "randperm", capture_order)
    options = dict(hidden_layers=(8, 5), max_iter=1, batch_size=32, device="cuda", symmetry=symmetry)
    baseline = build_model(**options).fit(train, targets, validation_data=(val, val_targets))
    auxiliary = build_model(**options, auxiliary_weight=0.25).fit(
        train, targets, validation_data=(val, val_targets), auxiliary_targets=np.linspace(-0.2, 0.5, len(train)),
    )
    explicit_zero = build_model(**options, auxiliary_weight=0.0).fit(train, targets, validation_data=(val, val_targets))
    for index, (base, extra, zero) in enumerate(zip(*initial_parameters)):
        assert torch.equal(base, extra[:3] if index >= len(initial_parameters[0]) - 2 else extra)
        assert torch.equal(base, zero)
    assert torch.equal(orders[0], orders[1])
    assert torch.equal(orders[0], orders[2])

    def initial_primary_prediction(parameters):
        # Evaluate the captured primary head at initialization, not a trained
        # checkpoint; matching final trained parameters would be a wrong test.
        raw = torch.as_tensor(baseline._feature_array(val), device="cuda")
        mean = torch.as_tensor(baseline.input_scaler_.mean_, dtype=torch.float32, device="cuda")
        scale = torch.as_tensor(baseline.input_scaler_.scale_, dtype=torch.float32, device="cuda")
        target_mean = torch.as_tensor(baseline.target_scaler_.mean_, dtype=torch.float32, device="cuda")
        target_scale = torch.as_tensor(baseline.target_scaler_.scale_, dtype=torch.float32, device="cuda")

        def forward(raw):
            values = (raw - mean) / scale
            for index in range(0, len(parameters) - 2, 2):
                values = torch.nn.functional.silu(torch.nn.functional.linear(values, parameters[index], parameters[index + 1]))
            return torch.nn.functional.linear(values, parameters[-2][:3], parameters[-1][:3]) * target_scale + target_mean

        prediction = forward(raw)
        if symmetry:
            reflection = torch.as_tensor(baseline.reflection_signs_, device="cuda")
            signs = torch.as_tensor(baseline.target_signs_, device="cuda")
            prediction = 0.5 * (prediction + signs * forward(raw * reflection))
        return prediction

    assert torch.equal(initial_primary_prediction(initial_parameters[0]), initial_primary_prediction(initial_parameters[1]))
    np.testing.assert_array_equal(baseline.predict(val), explicit_zero.predict(val))
    np.testing.assert_allclose(baseline.predict(val), auxiliary.predict(val), rtol=2e-6, atol=1e-6)
    assert baseline.training_metadata_["history"] == explicit_zero.training_metadata_["history"]
    for base, extra in zip(baseline.coefs_[:-1], auxiliary.coefs_[:-1]):
        np.testing.assert_array_equal(base, extra)
    np.testing.assert_array_equal(baseline.coefs_[-1], auxiliary.coefs_[-1][:3])
    np.testing.assert_array_equal(baseline.intercepts_[-1], auxiliary.intercepts_[-1][:3])


@pytest.mark.gpu
@KAGGLE_GPU_ONLY
@pytest.mark.parametrize("loss", ["mse", "huber"])
def test_kaggle_auxiliary_masked_gradient_and_unchanged_primary_denominator(loss, kaggle_cuda, monkeypatch):
    torch = kaggle_cuda
    train, targets, val, val_targets = _gpu_data()
    auxiliary = np.full(len(train), np.nan, dtype=np.float32)
    auxiliary[:5] = [-0.2, 0.0, 0.3, 0.9, -0.1]
    outputs = []
    original_forward = torch.nn.Linear.forward

    def capture_forward(layer, values):
        prediction = original_forward(layer, values)
        if layer.out_features == 4 and torch.is_grad_enabled():
            prediction.retain_grad()
            outputs.append(prediction)
        return prediction

    def no_step(optimizer, closure=None):
        assert all(torch.isfinite(parameter.grad).all() for parameter in optimizer.param_groups[0]["params"])

    monkeypatch.setattr(torch.nn.Linear, "forward", capture_forward)
    monkeypatch.setattr(torch.optim.RAdam, "step", no_step)
    model = build_model(hidden_layers=(7,), max_iter=1, batch_size=16, weight_decay=0,
                        loss=loss, symmetry=False, device="cuda", auxiliary_weight=0.25)
    model.fit(train, targets, validation_data=(val, val_targets), auxiliary_targets=auxiliary)
    generator = torch.Generator(device="cpu").manual_seed(model.random_state)
    order = torch.randperm(len(train), generator=generator).numpy()
    primary_sum, auxiliary_sum, total_sum, n_observed = 0.0, 0.0, 0.0, 0
    empty_batches = 0

    def elementwise(error):
        if loss == "mse":
            return error ** 2, 2 * error
        absolute = np.abs(error)
        return (np.where(absolute <= model.huber_delta, 0.5 * error ** 2,
                         model.huber_delta * (absolute - 0.5 * model.huber_delta)),
                np.clip(error, -model.huber_delta, model.huber_delta))

    for start, output in zip(range(0, len(train), model.batch_size), outputs):
        indices = order[start:start + model.batch_size]
        prediction = output.detach().cpu().numpy()
        gradient = output.grad.cpu().numpy()
        normalized = ((targets[indices] - model.target_scaler_.mean_.astype(np.float32))
                      / model.target_scaler_.scale_.astype(np.float32))
        elements, derivative = elementwise(prediction[:, :3] - normalized)
        weights = np.asarray(model.target_weights, dtype=np.float32)
        # Exactly the pre-auxiliary denominator, never 4 or sum(weights)+lambda.
        primary_value = (elements * weights).sum(axis=1).mean() / weights.sum()
        np.testing.assert_allclose(gradient[:, :3], derivative * weights / (len(indices) * weights.sum()), rtol=2e-6, atol=1e-7)
        mask = np.isfinite(auxiliary[indices])
        count = int(mask.sum())
        np.testing.assert_array_equal(gradient[~mask, 3], 0.0)
        if count:
            normalized_auxiliary = ((auxiliary[indices][mask] - np.float32(model.auxiliary_scaler_.mean_[0]))
                                    / np.float32(model.auxiliary_scaler_.scale_[0]))
            elements, derivative = elementwise(prediction[mask, 3] - normalized_auxiliary)
            auxiliary_value = elements.mean()
            np.testing.assert_allclose(gradient[mask, 3], model.auxiliary_weight * derivative / count, rtol=2e-6, atol=1e-7)
        else:
            auxiliary_value = 0.0
            empty_batches += 1
        primary_sum += float(primary_value) * len(indices)
        auxiliary_sum += float(auxiliary_value) * count
        total_sum += float(primary_value + model.auxiliary_weight * auxiliary_value) * len(indices)
        n_observed += count
    assert len(outputs) == len(train) // model.batch_size
    assert empty_batches > 0
    assert n_observed == 5
    history = model.training_metadata_["history"][0]
    np.testing.assert_allclose(history["train_loss"], primary_sum / len(train), rtol=2e-6)
    np.testing.assert_allclose(history["train_auxiliary_loss"], auxiliary_sum / n_observed, rtol=2e-6)
    np.testing.assert_allclose(history["train_total_loss"], total_sum / len(train), rtol=2e-6)
    assert history["train_primary_loss"] == history["train_loss"]
    assert history["validation_primary_loss"] == history["validation_loss"]
    assert history["validation_auxiliary_loss"] is None
    json.dumps(model.training_metadata_, allow_nan=False)
