from copy import deepcopy
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.neural_network import MLPRegressor

from src.training import validation_mlp as training
from src.training.comparison import VALIDATION_MLP_CONFIGS, compare_models


@pytest.fixture
def data():
    x = pd.DataFrame({"Alpha": np.linspace(-1, 1, 18), "shape": np.linspace(0, .1, 18)})
    y = pd.DataFrame({"Cl": x.Alpha / 2, "Cd": .02 + .01 * x.Alpha**2})
    return x.iloc[:12], y.iloc[:12], x.iloc[12:], y.iloc[12:]


def run(data, output, **kwargs):
    x, y, xv, yv = data
    return training.train_validation_mlp(
        x, y, xv, yv, list(y.columns), output,
        train_groups=["train"] * len(x), validation_groups=["validation"] * len(xv),
        parameters={"hidden_layers": (4,), "batch_size": 4, "random_state": 42}, **kwargs,
    )


def test_full_budget_passes_107_epochs_without_resetting_adam_or_refitting_scalers(data, tmp_path, monkeypatch):
    original = MLPRegressor.partial_fit
    optimizer_steps = []

    def observed_partial_fit(self, x, y, *args, **kwargs):
        optimizer = self._optimizer
        result = original(self, x, y, *args, **kwargs)
        assert self._optimizer is optimizer
        assert isinstance(self.random_state, np.random.RandomState)
        optimizer_steps.append((id(optimizer), optimizer.t))
        return result

    monkeypatch.setattr(MLPRegressor, "partial_fit", observed_partial_fit)
    model, elapsed, details = run(data, tmp_path / "model", policy={
        "max_epochs": 110, "min_epochs": 1, "check_every": 10, "patience": None,
    })
    assert len(optimizer_steps) == 109
    assert len({identity for identity, _ in optimizer_steps}) == 1
    assert all(right[1] > left[1] for left, right in zip(optimizer_steps, optimizer_steps[1:]))
    assert details["epochs_completed"] == 110
    assert details["stop_reason"] == "max_epochs"
    assert len(details["loss_curve"]) == 110
    assert details["n_iter_no_change"] == 111
    assert elapsed > 0
    x, y, xv, _ = data
    assert type(model) is TransformedTargetRegressor
    np.testing.assert_allclose(model.transformer_.mean_, y.mean().to_numpy())
    np.testing.assert_allclose(model.regressor_.named_steps["scaler"].mean_, x.mean().to_numpy())
    assert model.transformer_.n_samples_seen_ == len(y)
    restored = joblib.load(tmp_path / "model" / "mlp.joblib")
    np.testing.assert_allclose(restored.predict(xv), model.predict(xv))
    history = pd.read_csv(tmp_path / "model" / "training_history.csv")
    assert history.epoch.max() == 110
    assert set(history.target) == {"Cl", "Cd"}


def test_validation_patience_restores_best_checkpoint_not_last_epoch(data, tmp_path, monkeypatch):
    scores = iter([1.0, .5, .8, .9])
    predictions = []

    def validation_curve(model, x, y, names):
        predictions.append(model.predict(x).copy())
        score = next(scores)
        return {"targets": {
            name: {"mae": 0., "rmse": np.sqrt(score) * model.transformer_.scale_[index], "r2": 0.}
            for index, name in enumerate(names)
        }}

    monkeypatch.setattr(training, "evaluate_model", validation_curve)
    model, _, details = run(data, tmp_path / "model", policy={
        "max_epochs": 8, "min_epochs": 1, "check_every": 1, "patience": 2, "min_delta": 0.,
    })
    assert details["epochs_completed"] == 4
    assert details["best_epoch"] == 2
    assert details["best_validation_score"] == pytest.approx(.5)
    assert details["stop_reason"] == "validation_patience"
    np.testing.assert_allclose(model.predict(data[2]), predictions[1])
    assert not np.allclose(predictions[1], predictions[-1])
    state = json.loads((tmp_path / "model" / "validation_state.json").read_text())
    assert state["status"] == "completed"
    assert state["best_epoch"] == 2


def test_validation_failure_preserves_checkpoint_and_records_error(data, tmp_path, monkeypatch):
    original = training.evaluate_model
    calls = 0

    def fail_after_checkpoint(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated validation failure")
        return original(*args)

    monkeypatch.setattr(training, "evaluate_model", fail_after_checkpoint)
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="simulated validation failure"):
        run(data, output, policy={"max_epochs": 4, "min_epochs": 1, "check_every": 1})
    assert (output / "mlp.joblib").is_file()
    state = json.loads((output / "validation_state.json").read_text())
    assert state["status"] == "failed"
    assert state["best_epoch"] == 1
    assert pd.read_csv(output / "training_history.csv").epoch.max() == 1


def test_patience_accepts_exact_relative_improvement_and_honors_min_epochs(data, tmp_path, monkeypatch):
    scores = iter([1., .9, .81, .81, .81, .81])

    def validation_curve(model, x, y, names):
        score = next(scores)
        return {"targets": {
            name: {"mae": 0., "rmse": np.sqrt(score) * model.transformer_.scale_[index], "r2": 0.}
            for index, name in enumerate(names)
        }}

    monkeypatch.setattr(training, "evaluate_model", validation_curve)
    _, _, details = run(data, tmp_path / "model", policy={
        "max_epochs": 8, "min_epochs": 6, "check_every": 1, "patience": 1, "min_delta": .1,
    })
    assert details["epochs_completed"] == 6
    assert details["best_epoch"] == 3
    assert details["stale_checks"] == 3
    assert details["stop_reason"] == "validation_patience"


@pytest.mark.parametrize("policy", [
    {"max_epochs": 0}, {"check_every": 0}, {"patience": 0}, {"min_delta": 1},
    {"max_epochs": 5, "min_epochs": 6}, {"unknown": 1},
])
def test_invalid_stopping_policies_are_rejected(policy):
    with pytest.raises(ValueError):
        training.validation_policy(policy)


def test_controller_requires_disjoint_geometries(data, tmp_path):
    x, y, xv, yv = data
    with pytest.raises(ValueError, match="overlap"):
        training.train_validation_mlp(
            x, y, xv, yv, ["Cl", "Cd"], tmp_path / "model",
            train_groups=["same"] * len(x), validation_groups=["same"] * len(xv),
        )
    assert not (tmp_path / "model").exists()


def test_sklearn_interrupt_warning_stops_the_epoch_loop(data, tmp_path, monkeypatch):
    def interrupted(self, *args, **kwargs):
        warnings.warn("Training interrupted by user.")
        return self

    monkeypatch.setattr(MLPRegressor, "partial_fit", interrupted)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run(data, output, policy={"max_epochs": 4, "min_epochs": 1, "check_every": 1})
    state = json.loads((output / "validation_state.json").read_text())
    assert state["status"] == "interrupted"
    assert state["epochs_completed"] == 1
    assert state["best_epoch"] == 1
    assert (output / "mlp.joblib").exists()


@pytest.mark.parametrize("column", [False, True])
def test_epoch_training_preserves_single_target_prediction_shape(data, tmp_path, column):
    x, y, xv, yv = data
    y = y[["Cl"]] if column else y.Cl
    yv = yv[["Cl"]] if column else yv.Cl
    # Some supported sklearn versions warn when adapting an (n, 1) target.
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        model, _, _ = training.train_validation_mlp(
            x, y, xv, yv, ["Cl"], tmp_path / "single",
            train_groups=["train"] * len(x), validation_groups=["validation"] * len(xv),
            parameters={"hidden_layers": (4,), "batch_size": 4},
            policy={"max_epochs": 3, "min_epochs": 1, "check_every": 1, "patience": None},
        )
    assert all(issubclass(item.category, DataConversionWarning) for item in emitted)
    assert model.predict(xv).shape == yv.shape


def test_controlled_variants_integrate_with_comparison_and_saved_history(data, tmp_path):
    x, y, xv, yv = data
    configs = deepcopy(VALIDATION_MLP_CONFIGS)
    for config in configs:
        config["parameters"].update(hidden_layers=(4,), batch_size=4)
        config["validation_training"].update(max_epochs=3, min_epochs=1, check_every=1, patience=None)
    output = tmp_path / "comparison"
    metrics = compare_models(
        x, y, xv, yv, ["Cl", "Cd"], output, configs,
        train_groups=["train"] * len(x), validation_groups=["validation"] * len(xv),
    )
    assert len(metrics) == 4
    report = json.loads((output / "run.json").read_text())
    for name, result in report["results"].items():
        assert result["status"] == "success"
        assert result["training"]["method"] == "geometry_validation"
        assert result["training"]["epochs_completed"] == 3
        history = pd.read_csv(output / name / "training_history.csv")
        best = history[history.epoch == result["training"]["best_epoch"]]
        for target in ("Cl", "Cd"):
            assert result["targets"][target]["rmse"] == pytest.approx(best.loc[best.target == target, "rmse"].iloc[0])
    assert len(pd.read_csv(output / "validation_history.csv")) == 12
    assert not (output / "test_metrics.csv").exists()
