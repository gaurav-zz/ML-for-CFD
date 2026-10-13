import json
from pathlib import Path
import warnings
import weakref

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning

from src.data.preprocessing import SplitData
from src.geometry.descriptors import DESCRIPTOR_NAMES
from src.training import comparison


class OffsetRegressor:
    """Pickleable deterministic predictions for cheap workflow-only checks."""

    def __init__(self, offsets):
        self.offsets = np.asarray(offsets)

    def predict(self, x):
        return np.asarray(x) + self.offsets


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


@pytest.fixture
def v2_tables(tmp_path):
    geometry = pd.DataFrame({
        "geometry_id": ["foil-a", "foil-b", "foil-c", "foil-d"],
        "airfoil_name": list("ABCD"),
        "thickness_000": [0.0, 0.0, 0.0, 0.0],
        "thickness_001": [.10, .12, .14, .16],
        "camber_000": [0.0, 0.0, 0.0, 0.0],
        "camber_001": [.01, .02, .03, .04],
        **{name: [1., 2., 3., 4.] for name in DESCRIPTOR_NAMES},
        "clipped_stations": [0, 0, 0, 0],
    })
    polars = pd.DataFrame({
        "geometry_id": ["foil-a", "foil-a", "foil-b", "foil-c", "foil-d", "foil-d"],
        "Alpha": [0., 4., 2., 6., 1., 8.],
        "Re": [1_000_000.] * 6,
        "Ncrit": [9.] * 6,
        "Cl": [.1, .2, .3, .4, .5, .6],
        "Cd": [.001, .002, .003, .004, .005, .006],
        "Cm": [-.01, -.02, -.03, -.04, -.05, -.06],
    })
    geometry_path, polars_path = tmp_path / "airfoil_geometry.csv", tmp_path / "airfoil_polars.csv"
    geometry.to_csv(geometry_path, index=False)
    polars.to_csv(polars_path, index=False)
    frozen = tmp_path / "frozen_split.json"
    frozen.write_text(json.dumps({
        "git_commit": "a" * 40,
        "split_geometries": {
            "train": ["foil-a", "foil-d"], "validation": ["foil-b"], "test": ["foil-c"],
        },
        "split_rows": {"train": 4, "validation": 1, "test": 1},
    }))
    return geometry_path, polars_path, frozen, polars


def test_frozen_split_assembles_v2_features_and_records_a_new_fingerprint(v2_tables):
    """v1's CSV hash guard cannot apply once the tables change shape."""
    geometry_path, polars_path, frozen, polars = v2_tables
    split, features, targets, metadata = comparison.load_frozen_split(
        geometry_path, polars_path, frozen,
    )
    assert isinstance(split, SplitData)
    assert targets == ["Cl", "Cd", "Cm"]
    assert features[:4] == ["thickness_000", "thickness_001", "camber_000", "camber_001"]
    assert features[-5:] == ["Alpha", "alpha_sin", "alpha_cos", "log10_re", "Ncrit"]
    assert set(DESCRIPTOR_NAMES).issubset(features)
    # geometry_id defines the split groups and must never become a feature.
    assert "geometry_id" not in features and not set(features).intersection(targets)

    assert split.train.geometry_id.tolist() == ["foil-a", "foil-a", "foil-d", "foil-d"]
    assert metadata["split_rows"] == {"train": 4, "validation": 1, "test": 1}
    assert metadata["dataset"]["polar_rows"] == len(polars)
    assert metadata["dataset"]["geometry_rows"] == 4
    assert len(metadata["dataset"]["geometry_sha256"]) == 64
    # Explicitly not v1's fingerprint, which described the deleted 1.7 GB CSV.
    assert metadata["dataset"]["polars_sha256"] != (
        "b4429f3c97a3d99169d1a17e83606218cdfcebbde2c3bb68ea41fd24e79ee0d4"
    )
    assert metadata["baseline_git_commit"] == "a" * 40
    assert metadata["split_geometries"]["train"] == ["foil-a", "foil-d"]


def test_fingerprints_track_each_table_separately(v2_tables):
    geometry_path, polars_path, frozen, polars = v2_tables
    before = comparison.load_frozen_split(geometry_path, polars_path, frozen)[3]["dataset"]
    polars.loc[0, "Cl"] = .999
    polars.to_csv(polars_path, index=False)
    after = comparison.load_frozen_split(geometry_path, polars_path, frozen)[3]["dataset"]
    assert before["polars_sha256"] != after["polars_sha256"]
    assert before["geometry_sha256"] == after["geometry_sha256"]


def test_fingerprints_are_streamed_not_read_whole(v2_tables, monkeypatch):
    geometry_path, polars_path, frozen, _ = v2_tables

    def no_full_read(*args, **kwargs):
        pytest.fail("Table fingerprints must be streamed, not read_bytes().")

    monkeypatch.setattr(Path, "read_bytes", no_full_read)
    comparison.load_frozen_split(geometry_path, polars_path, frozen)


def test_cm_is_available_as_a_third_target_but_can_be_dropped(v2_tables):
    geometry_path, polars_path, frozen, _ = v2_tables
    _, _, targets, _ = comparison.load_frozen_split(
        geometry_path, polars_path, frozen, target_columns=("Cl", "Cd"),
    )
    assert targets == ["Cl", "Cd"]


@pytest.mark.parametrize("targets,error", [
    (("Cl", "Missing"), "lacks target columns"),
    (("Cl", "Cl"), "unique"),
    ((), "nonempty"),
])
def test_frozen_split_rejects_bad_target_requests(v2_tables, targets, error):
    geometry_path, polars_path, frozen, _ = v2_tables
    with pytest.raises(ValueError, match=error):
        comparison.load_frozen_split(geometry_path, polars_path, frozen, targets)


@pytest.mark.parametrize("column,value,error", [
    ("Re", -1., "positive"),
    ("Cd", -.1, "negative"),
    ("Alpha", np.inf, "infinite"),
    ("geometry_id", None, "no geometry row"),
])
def test_frozen_split_validates_polars_content(v2_tables, column, value, error):
    geometry_path, polars_path, frozen, polars = v2_tables
    if value is None:
        polars = polars.astype({column: object})
    polars.loc[0, column] = value
    polars.to_csv(polars_path, index=False)
    with pytest.raises(ValueError, match=error):
        comparison.load_frozen_split(geometry_path, polars_path, frozen)


def test_frozen_split_requires_a_split_definition(v2_tables, tmp_path):
    geometry_path, polars_path, _, _ = v2_tables
    without = tmp_path / "nosplit.json"
    without.write_text(json.dumps({"git_commit": "a" * 40}))
    with pytest.raises(ValueError, match="no split_geometries"):
        comparison.load_frozen_split(geometry_path, polars_path, without)


def test_frozen_split_refuses_a_geometry_table_that_misses_airfoils(v2_tables):
    geometry_path, polars_path, frozen, _ = v2_tables
    pd.read_csv(geometry_path).head(2).to_csv(geometry_path, index=False)
    with pytest.raises(ValueError, match="have no geometry row"):
        comparison.load_frozen_split(geometry_path, polars_path, frozen)


def test_frozen_split_refuses_partitions_that_do_not_cover_the_tables(v2_tables):
    geometry_path, polars_path, frozen, _ = v2_tables
    definition = json.loads(frozen.read_text())
    definition["split_geometries"]["train"] = ["foil-a"]
    frozen.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match="exactly cover"):
        comparison.load_frozen_split(geometry_path, polars_path, frozen)


def test_log_drag_is_fitted_in_log_space_but_scored_in_physical_units(tmp_path):
    """The whole point of logging Cd is lost if the metrics stay in log space."""
    x = np.arange(1, 9, dtype=float)[:, None]
    # Drag grows geometrically, so a linear fit only works in log space.
    y = np.column_stack([x[:, 0], 0.001 * 2.0 ** x[:, 0]])
    output = tmp_path / "comparison"
    table = comparison.compare_models(
        x, y, x, y, ["Cl", "Cd"], output,
        [{"name": "linear", "parameters": {}, "scale_targets": False}],
        target_transforms={"Cd": "log"},
    )
    manifest = read_json(output / "run.json")
    assert manifest["target_transforms"] == {"Cl": "identity", "Cd": "log"}
    # Physical units: drag RMSE sits on the scale of Cd itself, not of log(Cd).
    drag = table.loc[table.target == "Cd"].iloc[0]
    assert drag["rmse"] < 0.01
    assert drag["median_relative_error"] < 1e-6

    restored = joblib.load(output / "linear" / "linear.joblib")
    # The saved model still predicts in log space; the wrapper does the inverting.
    np.testing.assert_allclose(restored.predict(x[:1]), [[1.0, np.log(0.002)]], rtol=1e-6)
    test = comparison.evaluate_selected_models(output, x, y, ["Cl", "Cd"])
    np.testing.assert_allclose(test.rmse, table.rmse, rtol=1e-9)
    assert read_json(output / "test_evaluation.json")["target_transforms"] == {
        "Cl": "identity", "Cd": "log",
    }


def test_untransformed_runs_pass_the_target_arrays_straight_through(tmp_path, monkeypatch):
    """No copy of a 500k-row target matrix just to apply identity."""
    seen = {}

    def trainer(name, x, y, output_dir, **kwargs):
        seen["y"] = y
        model = OffsetRegressor([0.0])
        joblib.dump(model, output_dir / f"{name}.joblib")
        return model, .01

    monkeypatch.setattr(comparison, "train_model", trainer)
    y = np.array([[1.0], [2.0]])
    comparison.compare_models(
        y, y, y, y, ["Cl"], tmp_path / "comparison",
        [{"name": "linear", "parameters": {}, "scale_targets": False}],
    )
    assert seen["y"] is y


def test_default_queue_is_sequential_and_bounded():
    configs = comparison.DEFAULT_MODEL_CONFIGS
    assert [config["name"] for config in configs] == ["linear", "gradient_boosting", "random_forest", "mlp"]
    assert configs[0] == {"name": "linear", "parameters": {}, "scale_targets": False}
    assert all(config["scale_targets"] for config in configs[1:])
    assert configs[1]["parameters"] == {
        "max_iter": 200, "learning_rate": .05, "max_leaf_nodes": 31,
        "early_stopping": False, "random_state": 42,
    }
    assert configs[2]["parameters"] == {
        "n_estimators": 100, "max_depth": 20, "max_leaf_nodes": 2048,
        "min_samples_leaf": 5, "max_samples": .5, "n_jobs": 2, "random_state": 42,
    }
    assert configs[3]["parameters"] == {
        "hidden_layers": (64, 32), "max_iter": 100, "batch_size": 1024,
        "early_stopping": False, "random_state": 42,
    }


def test_sequential_checkpoints_warnings_failure_and_model_release(tmp_path, monkeypatch, capsys):
    output = tmp_path / "nested" / "comparison"
    x = np.array([[1000., .001], [2000., .002], [3000., .003]])
    y = x.copy()
    configs = [
        {"name": name, "parameters": {}, "scale_targets": name != "linear"}
        for name in ("linear", "gradient_boosting", "random_forest")
    ]
    calls, references = [], []

    def trainer(name, x_train, y_train, output_dir, scale_targets, **parameters):
        assert x_train is x and y_train is y  # No hidden outer-data subsampling.
        assert all(reference() is None for reference in references)
        assert output_dir == output / name
        assert scale_targets == (name != "linear")
        assert parameters == {}
        manifest = read_json(output / "run.json")
        assert manifest["status"] == "running"
        assert manifest["results"][name]["status"] == "running"
        checkpoint = pd.read_csv(output / "comparison.csv")
        if calls:
            assert set(checkpoint.model) == {"linear"}
            assert manifest["selected_models"] == {"Cl": "linear", "Cd": "linear"}
            previous = calls[-1]
            assert read_json(output / previous / "result.json") == manifest["results"][previous]
        else:
            assert checkpoint.empty
        calls.append(name)
        warnings.warn(f"{name} reached iteration limit", ConvergenceWarning)
        if name == "gradient_boosting":
            raise ValueError("intentional model failure")
        model = OffsetRegressor([10., .0004] if name == "linear" else [20., .0001])
        references.append(weakref.ref(model))
        joblib.dump(model, output_dir / f"{name}.joblib")
        return model, .125

    monkeypatch.setattr(comparison, "train_model", trainer)
    table = comparison.compare_models(x, y, x.copy(), y.copy(), ["Cl", "Cd"], output, configs, {"baseline": "frozen"})
    assert calls == [config["name"] for config in configs]
    assert all(reference() is None for reference in references)
    assert table.columns.tolist() == comparison.COMPARISON_COLUMNS
    assert table.model.tolist() == ["linear", "linear", "random_forest", "random_forest"]
    assert table.n_predictions.tolist() == [len(x)] * 4
    manifest = read_json(output / "run.json")
    assert manifest["status"] == "completed_with_failures"
    assert manifest["run_metadata"] == {"baseline": "frozen"}
    assert manifest["selection_metric"] == "validation_rmse"
    assert manifest["selected_models"] == {"Cl": "linear", "Cd": "random_forest"}
    for name, result in manifest["results"].items():
        assert result["warnings"] == [{"category": "ConvergenceWarning", "message": f"{name} reached iteration limit"}]
        assert result == read_json(output / name / "result.json")
    assert manifest["results"]["gradient_boosting"]["error"] == {"type": "ValueError", "message": "intentional model failure"}
    assert not (output / "gradient_boosting" / "gradient_boosting.joblib").exists()
    pd.testing.assert_frame_equal(pd.read_csv(output / "comparison.csv"), table)
    progress = capsys.readouterr().out
    steps = ["Starting linear", "Finished linear", "Starting gradient_boosting", "Failed gradient_boosting", "Starting random_forest", "Finished random_forest"]
    assert [progress.index(step) for step in steps] == sorted(progress.index(step) for step in steps)
    before = (output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        comparison.compare_models(x, y, x, y, ["Cl", "Cd"], output, configs)
    assert (output / "run.json").read_bytes() == before
    assert len(calls) == 3


@pytest.mark.parametrize("configs", [[], [{"name": "linear", "parameters": {}, "scale_targets": False}]])
def test_empty_and_failed_queues_leave_diagnostics(tmp_path, monkeypatch, configs):
    def fail(*args, **kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(comparison, "train_model", fail)
    output = tmp_path / "comparison"
    table = comparison.compare_models([[1]], [[2]], [[3]], [[4]], ["Cl"], output, configs)
    assert table.empty and table.columns.tolist() == comparison.COMPARISON_COLUMNS
    assert pd.read_csv(output / "comparison.csv").empty
    manifest = read_json(output / "run.json")
    assert manifest["status"] == "failed"
    assert manifest["selected_models"] == {}
    if configs:
        assert manifest["results"]["linear"]["error"]["message"] == "fit failed"
        assert (output / "linear" / "result.json").is_file()
    with pytest.raises(ValueError, match="winner"):
        comparison.evaluate_selected_models(output, [[1]], [[2]], ["Cl"])


def test_interrupt_is_not_swallowed(tmp_path, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(comparison, "train_model", interrupt)
    output = tmp_path / "comparison"
    with pytest.raises(KeyboardInterrupt):
        comparison.compare_models([[1]], [[2]], [[3]], [[4]], ["Cl"], output, comparison.DEFAULT_MODEL_CONFIGS[:1])
    manifest = read_json(output / "run.json")
    assert manifest["status"] == "running"
    assert manifest["results"]["linear"]["status"] == "running"


@pytest.mark.parametrize("nonfinite", [False, True])
def test_evaluation_failures_do_not_pollute_checkpoints(tmp_path, monkeypatch, nonfinite):
    original_evaluate = comparison.evaluate_model

    def trainer(name, x, y, output_dir, **kwargs):
        model = OffsetRegressor([1. if name == "linear" else 2.])
        joblib.dump(model, output_dir / f"{name}.joblib")
        return model, .01

    def evaluate(model, *args):
        if model.offsets[0] == 1.:
            if not nonfinite:
                raise ValueError("prediction failed")
            result = original_evaluate(model, *args)
            result["targets"]["Cl"]["r2"] = float("nan")
            return result
        return original_evaluate(model, *args)

    monkeypatch.setattr(comparison, "train_model", trainer)
    monkeypatch.setattr(comparison, "evaluate_model", evaluate)
    output = tmp_path / "comparison"
    table = comparison.compare_models(
        [[1.], [2.]], [[1.], [2.]], [[3.], [4.]], [[3.], [4.]], ["Cl"],
        output, comparison.DEFAULT_MODEL_CONFIGS[:2],
    )
    assert table.model.tolist() == ["gradient_boosting"]
    manifest = read_json(output / "run.json")
    assert manifest["status"] == "completed_with_failures"
    assert manifest["selected_models"] == {"Cl": "gradient_boosting"}
    assert manifest["results"]["linear"]["status"] == "failed"
    assert "targets" not in manifest["results"]["linear"]
    assert read_json(output / "linear" / "result.json")["error"]["type"] == "ValueError"


@pytest.mark.parametrize("configs,error", [
    ([{"name": "../linear", "parameters": {}, "scale_targets": False}], "Unsupported"),
    ([{"name": "unknown", "parameters": {}, "scale_targets": False}], "Unsupported"),
    ([{"name": "linear", "parameters": {}, "scale_targets": False}] * 2, "unique"),
    ([{"name": "linear", "parameters": {"output_dir": "outside"}, "scale_targets": False}], "override"),
    ([{"name": "linear", "parameters": {}, "scale_targets": "false"}], "bool"),
])
def test_invalid_configs_do_not_create_artifacts(tmp_path, configs, error):
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match=error):
        comparison.compare_models([[1]], [[2]], [[3]], [[4]], ["Cl"], output, configs)
    assert not output.exists()


def test_real_scaled_fit_metrics_and_singleton_r2_are_in_target_units(tmp_path):
    x = np.arange(8, dtype=float)[:, None]
    y = np.column_stack([1000. + 200. * x[:, 0], .001 + .0002 * x[:, 0]])
    x_validation = np.array([[8.]])
    y_validation = np.array([[2603., .002604]])
    output = tmp_path / "comparison"
    table = comparison.compare_models(
        x, y, x_validation, y_validation, ["Cl", "Cd"], output,
        [{"name": "linear", "parameters": {}, "scale_targets": True}],
    )
    np.testing.assert_allclose(table.mae, [3., .000004], rtol=1e-8)
    np.testing.assert_allclose(table.rmse, [3., .000004], rtol=1e-8)
    assert table.r2.isna().all()
    manifest = read_json(output / "run.json")
    assert manifest["status"] == "completed"
    assert manifest["results"]["linear"]["targets"]["Cl"]["r2"] is None
    restored = joblib.load(output / "linear" / "linear.joblib")
    assert isinstance(restored, TransformedTargetRegressor)
    np.testing.assert_allclose(restored.transformer_.mean_, y.mean(axis=0))
    np.testing.assert_allclose(restored.predict(x_validation), [[2600., .0026]])
    test = comparison.evaluate_selected_models(output, x_validation, y_validation, ["Cl", "Cd"])
    np.testing.assert_allclose(test.mae, table.mae)
    assert all(row["r2"] is None for row in read_json(output / "test_evaluation.json")["results"])
    assert pd.read_csv(output / "test_metrics.csv").r2.isna().all()


@pytest.mark.parametrize("same_winner", [False, True])
def test_test_evaluation_only_loads_frozen_validation_winners(tmp_path, monkeypatch, same_winner):
    offsets = {"linear": [10., .0004], "random_forest": [20., .0001], "gradient_boosting": [30., .0005]}
    if same_winner:
        offsets["linear"] = [10., .00001]
    configs = [{"name": name, "parameters": {}, "scale_targets": False} for name in offsets]

    def trainer(name, x, y, output_dir, **kwargs):
        model = OffsetRegressor(offsets[name])
        joblib.dump(model, output_dir / f"{name}.joblib")
        return model, .01

    monkeypatch.setattr(comparison, "train_model", trainer)
    x = np.array([[1000., .001], [2000., .002], [3000., .003]])
    output = tmp_path / "comparison"
    comparison.compare_models(x, x.copy(), x.copy(), x.copy(), ["Cl", "Cd"], output, configs)
    before = (output / "run.json").read_bytes()
    loaded, references = [], []
    original_load = joblib.load

    def load(path):
        assert all(reference() is None for reference in references)
        loaded.append(path.stem)
        model = original_load(path)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(comparison.joblib, "load", load)
    # The unselected GB would be perfect on test, but must never be loaded/ranked.
    y_test = x + offsets["gradient_boosting"]
    table = comparison.evaluate_selected_models(output, x, y_test, ["Cl", "Cd"])
    assert loaded == (["linear"] if same_winner else ["linear", "random_forest"])
    assert all(reference() is None for reference in references)
    assert table.columns.tolist() == comparison.METRIC_COLUMNS
    assert table.target.tolist() == ["Cl", "Cd"]
    assert table.model.tolist() == ["linear", "linear" if same_winner else "random_forest"]
    np.testing.assert_allclose(table.rmse, [20., .00049 if same_winner else .0004])
    assert table.n_predictions.tolist() == [len(x)] * 2
    assert (output / "run.json").read_bytes() == before
    report = read_json(output / "test_evaluation.json")
    assert report["selected_models"] == read_json(output / "run.json")["selected_models"]
    assert report["selection_metric"] == "validation_rmse"
    pd.testing.assert_frame_equal(pd.read_csv(output / "test_metrics.csv"), table)
    with pytest.raises(ValueError, match="order"):
        comparison.evaluate_selected_models(output, x, y_test, ["Cd", "Cl"])
    assert loaded == (["linear"] if same_winner else ["linear", "random_forest"])
    manifest = read_json(output / "run.json")
    manifest["status"] = "running"
    (output / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed"):
        comparison.evaluate_selected_models(output, x, y_test, ["Cl", "Cd"])
    assert loaded == (["linear"] if same_winner else ["linear", "random_forest"])
