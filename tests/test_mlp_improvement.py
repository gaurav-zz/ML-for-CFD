from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.training.comparison import (
    DEFAULT_MODEL_CONFIGS, MLP_IMPROVEMENT_CONFIGS, compare_models,
    evaluate_selected_models, model_artifact_path,
)
from src.training.evaluate import geometry_metrics


@pytest.fixture
def data():
    x = pd.DataFrame({"Alpha": np.linspace(-2, 2, 36), "geom_000": np.repeat(np.arange(9) / 100, 4)})
    y = pd.DataFrame({"Cl": x.Alpha / 10, "Cd": .02 + .003 * x.Alpha**2})
    groups = pd.Series([f"foil-{index}" for index in np.repeat(np.arange(9), 4)])
    return x, y, groups


def test_improvement_presets_change_only_budget_from_current_mlp():
    baseline = next(config for config in DEFAULT_MODEL_CONFIGS if config["name"] == "mlp")
    for config, budget in zip(MLP_IMPROVEMENT_CONFIGS, (100, 200, 400)):
        assert config["id"] == f"mlp_{budget}"
        assert config["name"] == "mlp" and config["scale_targets"] is True
        assert config["parameters"] == {**baseline["parameters"], "max_iter": budget}


def test_named_variants_keep_separate_artifacts_losses_and_unseen_geometry_metrics(data, tmp_path):
    x, y, groups = data
    configs = deepcopy(MLP_IMPROVEMENT_CONFIGS[:2])
    for config, budget in zip(configs, (2, 3)):
        config["id"] = f"mlp_{budget}"
        config["parameters"].update(max_iter=budget, hidden_layers=(4,), batch_size=8)
    output = tmp_path / "improvement"
    table = compare_models(
        x.iloc[:24], y.iloc[:24], x.iloc[24:32], y.iloc[24:32], ["Cl", "Cd"], output,
        model_configs=configs, train_groups=groups.iloc[:24], validation_groups=groups.iloc[24:32],
    )
    assert set(table.model) == {"mlp_2", "mlp_3"}
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "completed"
    for model_id, budget in (("mlp_2", 2), ("mlp_3", 3)):
        result = report["results"][model_id]
        assert result["family"] == "mlp"
        assert result["model_path"] == f"{model_id}/mlp.joblib"
        assert (output / model_artifact_path(report, model_id)).is_file()
        assert result["training"]["n_iter"] == budget
        assert len(result["training"]["loss_curve"]) == budget
        assert result["training"]["hit_iteration_limit"] is True
        assert result["training"]["tol"] == 1e-4
        assert result["training"]["n_iter_no_change"] == 10
        assert result["training"]["early_stopping"] is False
    by_geometry = pd.read_csv(output / "validation_by_geometry.csv")
    assert set(by_geometry.geometry_id) == set(groups.iloc[24:32])
    assert set(by_geometry.geometry_id).isdisjoint(groups.iloc[:24])
    for (model_id, target), subset in by_geometry.groupby(["model", "target"]):
        assert subset.n_predictions.sum() == 8
        global_rmse = table.loc[(table.model == model_id) & (table.target == target), "rmse"].iloc[0]
        assert np.sqrt(np.average(subset.rmse**2, weights=subset.n_predictions)) == pytest.approx(global_rmse)
    assert not (output / "test_metrics.csv").exists()
    test = evaluate_selected_models(output, x.iloc[32:], y.iloc[32:], ["Cl", "Cd"])
    assert set(test.model).issubset({"mlp_2", "mlp_3"})
    assert (test.n_predictions == 4).all()


@pytest.mark.parametrize("bad_id", ["", "../outside", "nested/run", "x" * 65])
def test_unsafe_variant_ids_are_rejected_before_training(data, tmp_path, bad_id):
    x, y, _ = data
    config = {**MLP_IMPROVEMENT_CONFIGS[0], "id": bad_id}
    with pytest.raises(ValueError, match="Model IDs"):
        compare_models(x, y, x, y, ["Cl", "Cd"], tmp_path / "run", [config])
    assert not (tmp_path / "run").exists()


def test_duplicate_ids_and_geometry_leakage_are_rejected(data, tmp_path):
    x, y, groups = data
    with pytest.raises(ValueError, match="unique"):
        compare_models(x, y, x, y, ["Cl", "Cd"], tmp_path / "duplicate", [MLP_IMPROVEMENT_CONFIGS[0]] * 2)
    with pytest.raises(ValueError, match="overlap"):
        compare_models(x, y, x, y, ["Cl", "Cd"], tmp_path / "leaked", train_groups=groups, validation_groups=groups)
    with pytest.raises(ValueError, match="together"):
        compare_models(x, y, x, y, ["Cl", "Cd"], tmp_path / "missing", validation_groups=groups)
    with pytest.raises(ValueError, match="order"):
        compare_models(
            x.iloc[:24], y.iloc[:24], x.iloc[24:], y.iloc[24:], ["Cl", "Cd"], tmp_path / "reordered",
            train_groups=groups.iloc[:24], validation_groups=groups.iloc[24:][::-1],
        )


def test_published_pre_variant_artifacts_remain_loadable(data, tmp_path):
    x, y, _ = data
    output = tmp_path / "legacy"
    compare_models(x.iloc[:24], y.iloc[:24], x.iloc[24:32], y.iloc[24:32], ["Cl", "Cd"], output, DEFAULT_MODEL_CONFIGS[:1])
    report = json.loads((output / "run.json").read_text())
    del report["results"]["linear"]["family"]
    (output / "run.json").write_text(json.dumps(report))
    assert model_artifact_path(report, "linear").as_posix() == "linear/linear.joblib"
    result = evaluate_selected_models(output, x.iloc[32:], y.iloc[32:], ["Cl", "Cd"])
    assert len(result) == 2
    report["results"]["linear"]["model_path"] = "../outside.joblib"
    with pytest.raises(ValueError, match="artifact path"):
        model_artifact_path(report, "linear")


def test_geometry_metrics_keeps_singleton_r2_undefined():
    actual = np.array([[0., .01], [1., .02], [2., .03]])
    predicted = actual + [1., .001]
    result = geometry_metrics(actual, predicted, ["a", "a", "b"], ["Cl", "Cd"])
    assert len(result) == 4
    assert result.loc[result.geometry_id == "b", "r2"].isna().all()
    np.testing.assert_allclose(result.loc[result.target == "Cl", "mae"], 1.)
    np.testing.assert_allclose(result.loc[result.target == "Cd", "mae"], .001)
