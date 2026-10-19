"""Study orchestration checks only: all trainers are mocked, including on Kaggle."""

import builtins
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import SplitData
from src.training import comparison, study


FEATURES = ["Alpha", "shape", "log10_re", "Ncrit"]
TARGETS = ["Cl", "Cd", "Cm"]


def physical_targets(x):
    signal = x["shape"].to_numpy() + x.Alpha.to_numpy() / 10
    return np.column_stack([.2 + .02 * signal, .01 + .0002 * signal, -.02 - .001 * signal])


class MockPredictor:
    """No fit method or torch dependency; saved solely to exercise disk inference."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.training_metadata_ = {
            "accelerator": parameters["device"], "device_name": "mock accelerator",
            "n_iter": 9, "best_epoch": 6, "stop_reason": "validation_patience",
            "history": [{"epoch": 6, "validation_loss": .1}],
        }

    def predict(self, x):
        physical = physical_targets(x)
        fitting = physical.copy()
        fitting[:, 1] = np.log(physical[:, 1])
        p = self.parameters
        drag_error = {42: .03, 43: .04, 44: .02}[p["random_state"]]
        if p["loss"] == "mse":
            drag_error += .02
        elif p["huber_delta"] == .02:
            drag_error = .001  # Validation selects this condition, not a seed.
        elif p["huber_delta"] == .1:
            drag_error = .07
        elif p["weight_decay"] == 0:
            drag_error = .025
        elif p["weight_decay"] == 1e-4:
            drag_error = .024
        # The selected sweep configuration is intentionally worse on test.
        if p["huber_delta"] == .02:
            drag_error = np.where(x["shape"].to_numpy() >= 23, .2, drag_error)
        fitting[:, 0] += .001 * (p["random_state"] - 40)
        fitting[:, 1] += drag_error
        fitting[:, 2] += .0001
        return fitting


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    working, inputs = tmp_path / "working", tmp_path / "input"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setattr(study, "_KAGGLE_WORKING", working)
    monkeypatch.setattr(study, "_KAGGLE_INPUT", inputs)
    return working


@pytest.fixture
def frozen():
    rows = [
        {"geometry_id": f"g{geometry:02d}", "Alpha": float(alpha), "Re": 1e6,
         "Ncrit": 9., "log10_re": 6., "shape": float(geometry)}
        for geometry in range(26) for alpha in range(6)
    ]
    frame = pd.DataFrame(rows)
    frame[TARGETS] = physical_targets(frame)
    split = SplitData(frame.iloc[:120].copy(), frame.iloc[120:138].copy(), frame.iloc[138:].copy())
    metadata = {
        "split_geometries": {name: sorted(getattr(split, name).geometry_id.unique()) for name in ("train", "validation", "test")},
        "split_rows": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
        "feature_columns": FEATURES.copy(), "target_columns": TARGETS.copy(),
        "gpu": {"device_name": "mock reported GPU", "torch_version": "mock", "device_count": 1},
        "git_commit": "a" * 40,
    }
    return split, metadata


@pytest.fixture
def mocked_training(monkeypatch):
    state = {"calls": [], "evaluations": [], "fail_at": None, "failure": RuntimeError,
             "save_weights": True, "selection_snapshots": []}
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Mock study tests must never import/install/train torch.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)

    def trainer(name, x, y, output_dir, scale_targets, **parameters):
        root = output_dir.parent.parent.parent
        plan = read_json(root / "study.json")
        assert plan["status"] == "running" and plan["phase"] == "fitting"
        assert plan["plan"]["estimated_fits"] == 12
        assert len(plan["plan"]["trial_configs"]) == 12
        assert not (root / "study_selection.json").exists()
        assert name in ("torch_mlp_mse", "torch_mlp_huber")
        assert scale_targets is False and parameters["target_names"] == tuple(TARGETS)
        x_val, y_val = parameters.pop("validation_data")
        expected, expected_val = physical_targets(x), physical_targets(x_val)
        expected[:, 1], expected_val[:, 1] = np.log(expected[:, 1]), np.log(expected_val[:, 1])
        np.testing.assert_allclose(y, expected)
        np.testing.assert_allclose(y_val, expected_val)
        state["calls"].append({"name": name, "x": x.copy(), "x_val": x_val.copy(),
                               "parameters": parameters.copy(), "trial_id": output_dir.parent.name})
        if len(state["calls"]) == state["fail_at"]:
            raise state["failure"]("intentional mock failure")
        model = MockPredictor(parameters)
        if state["save_weights"]:
            joblib.dump(model, output_dir / f"{name}.joblib")
        return model, .125

    original_evaluate = study.evaluate_model

    def evaluate(model, x, y, targets):
        root = state["output"]
        selection_path = root / "study_selection.json"
        assert selection_path.is_file(), "Selection must freeze before audit/test scoring."
        selection = read_json(selection_path)
        assert selection["seed_selection"] is False
        assert selection["canonical_test_scored"] is False
        state["selection_snapshots"].append(selection_path.read_bytes())
        state["evaluations"].append({"x": x.copy(), "y": y.copy(), "parameters": model.model.parameters})
        assert isinstance(model, study.PhysicalPredictor)
        return original_evaluate(model, x, y, targets)

    monkeypatch.setattr(comparison, "train_model", trainer)
    monkeypatch.setattr(study, "evaluate_model", evaluate)
    return state


def run_mock(cloud, frozen, state, name="study"):
    split, metadata = frozen
    state["output"] = cloud / name
    return study.run_study(split, FEATURES, TARGETS, state["output"], run_metadata=metadata, device="cpu")


def test_full_study_runs_exact_12_fits_and_records_existing_artifacts(cloud, frozen, mocked_training):
    split, metadata = frozen
    before = deepcopy(metadata)
    original_frames = {name: getattr(split, name).copy() for name in ("train", "validation", "test")}
    result = run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    assert result == read_json(output / "study.json")
    assert result["status"] == result["phase"] == "completed"
    assert result["hardware"] == metadata["gpu"]
    assert result["target_transforms"] == {"Cl": "identity", "Cd": "log", "Cm": "identity"}
    assert result["plan"]["seeds"] == [42, 43, 44]
    assert len(mocked_training["calls"]) == 12
    assert len({call["trial_id"] for call in mocked_training["calls"]}) == 12
    for path in result["artifacts"].values():
        assert (output / path).is_file()
    for trial in result["trials"].values():
        assert trial["status"] == "success"
        assert (output / trial["manifest"]).is_file()
        assert (output / trial["model_path"]).is_file()
        assert trial["result"]["training_metadata"]["best_epoch"] == 6
        assert trial["result"]["training_metadata"]["stop_reason"] == "validation_patience"
        assert trial["result"]["training_metadata"]["device_name"] == "mock accelerator"
    assert metadata == before
    for name, frame in original_frames.items():
        pd.testing.assert_frame_equal(getattr(split, name), frame)
    assert not list(output.rglob("*.tmp"))


def test_pairs_change_only_loss_and_sweep_is_not_a_cross_product(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    trials = result["trials"]
    for seed in (42, 43, 44):
        mse = dict(trials[f"seed_{seed}_mse"]["config"]["parameters"])
        huber = dict(trials[f"seed_{seed}_huber"]["config"]["parameters"])
        assert mse.pop("loss") == "mse" and huber.pop("loss") == "huber"
        assert mse == huber
        assert mse["hidden_layers"] == [128, 128, 64]
        assert mse["max_iter"] == 300 and mse["symmetry"] is True
        assert mse["huber_delta"] == .05 and mse["weight_decay"] == 1e-5
        assert mse["random_state"] == seed
    conditions = set()
    for trial_id in result["plan"]["sweep_trial_order"]:
        p = trials[trial_id]["config"]["parameters"]
        assert p["random_state"] == 42 and p["loss"] == "huber"
        conditions.add((p["huber_delta"], p["weight_decay"]))
    assert conditions == {(.02, 1e-5), (.05, 1e-5), (.1, 1e-5), (.05, 0.), (.05, 1e-4)}
    calls = mocked_training["calls"]
    assert sum(call["trial_id"] == "seed_42_huber" for call in calls) == 1
    for call in calls[:10]:
        assert set(call["x"].index) == set(frozen[0].train.index)
        assert set(call["x_val"].index) == set(frozen[0].validation.index)


def test_selection_freezes_validation_winner_without_testing_unselected_sweep(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    selection = result["selection"]
    assert selection["selected_trial"] == "sweep_delta_0p02"
    assert selection["metric"] == "validation.Cd.median_relative_error"
    assert selection["seed"] == 42 and selection["seed_selection"] is False
    assert selection["leakage_used_for_selection"] is False
    analysis = read_json(mocked_training["output"] / "selected_analysis.json")
    assert analysis["study_trial"] == selection["selected_trial"]
    assert analysis["n_rows"] == len(frozen[0].test)
    assert "poststall_alpha" in analysis["targets"]["Cd"]
    assert analysis["matched_easy_subset"]["fixed_re_reference"]["criteria"]["Re"] == 1e6
    trial = mocked_training["output"] / "trials" / selection["selected_trial"]
    test_report = read_json(trial / "test_evaluation.json")
    test_metrics = pd.read_csv(trial / "test_metrics.csv")
    assert test_report["selection_metric"] == "validation_per_target"
    assert list(test_metrics.target) == TARGETS
    assert test_report["selected_models"] == dict.fromkeys(TARGETS, "torch_mlp_huber")
    for row in selection["validation"]:
        assert {"Cl_rmse", "Cm_rmse", "Cd_median_relative_error"}.issubset(row)
    assert len(set(mocked_training["selection_snapshots"])) == 1
    assert (mocked_training["output"] / "study_selection.json").read_bytes() == mocked_training["selection_snapshots"][0]
    test_ids = {key for key, value in result["trials"].items() if "test_evaluation" in value}
    assert test_ids == {*result["plan"]["seed_trials"], "sweep_delta_0p02"}
    tests = [entry for entry in mocked_training["evaluations"] if entry["x"]["shape"].min() >= 23]
    assert len(tests) == 7
    for entry in tests:
        assert set(entry["x"].index) == set(frozen[0].test.index)
    metrics = pd.read_csv(mocked_training["output"] / "sweep_test_metrics.csv")
    assert set(metrics.trial_id) == {"sweep_delta_0p02"}
    assert len(metrics) == 3


def test_seed_summary_uses_sample_std_and_paired_huber_minus_mse(cloud, frozen, mocked_training):
    run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    metrics = pd.read_csv(output / "three_seed_metrics.csv")
    summary = pd.read_csv(output / "three_seed_summary.csv")
    deltas = pd.read_csv(output / "paired_seed_deltas.csv")
    assert len(metrics) == 36 and len(deltas) == 18
    assert set(metrics.seed) == {42, 43, 44} and set(metrics.loss) == {"mse", "huber"}
    for row in summary.itertuples():
        values = metrics.loc[(metrics.partition == row.partition) & (metrics.loss == row.loss)
                             & (metrics.target == row.target), row.metric]
        assert row.n_seeds == row.n_defined == 3 and row.ddof == 1
        assert row.mean == pytest.approx(values.mean())
        assert row.std == pytest.approx(values.std(ddof=1))
    for row in deltas.itertuples():
        pair = metrics.loc[(metrics.partition == row.partition) & (metrics.seed == row.seed)
                           & (metrics.target == row.target)].set_index("loss")
        assert row.direction == "huber_minus_mse"
        assert row.median_relative_error == pytest.approx(pair.loc["huber", "median_relative_error"] - pair.loc["mse", "median_relative_error"])


def test_leakage_arms_have_equal_rows_fixed_audit_common_stop_and_no_canonical_holdouts(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    split = frozen[0]
    honest, exposed = mocked_training["calls"][-2:]
    assert honest["trial_id"] == "leakage_honest" and exposed["trial_id"] == "leakage_exposed"
    assert honest["parameters"] == exposed["parameters"]
    assert len(honest["x"]) == len(exposed["x"])
    pd.testing.assert_frame_equal(honest["x_val"], exposed["x_val"])
    audit_evaluations = mocked_training["evaluations"][:2]
    pd.testing.assert_frame_equal(audit_evaluations[0]["x"], audit_evaluations[1]["x"])
    pd.testing.assert_frame_equal(audit_evaluations[0]["y"], audit_evaluations[1]["y"])
    audit_indices = set(audit_evaluations[0]["x"].index)
    audit_geometries = set(split.train.loc[sorted(audit_indices), "geometry_id"])
    stop_geometries = set(split.train.loc[honest["x_val"].index, "geometry_id"])
    for call in (honest, exposed):
        train_indices, stop_indices = set(call["x"].index), set(call["x_val"].index)
        assert train_indices.isdisjoint(audit_indices | stop_indices)
        assert stop_indices.isdisjoint(audit_indices)
        assert (train_indices | stop_indices | audit_indices).issubset(split.train.index)
        assert set(split.train.loc[call["x"].index, "geometry_id"]).isdisjoint(stop_geometries)
    assert set(split.train.loc[honest["x"].index, "geometry_id"]).isdisjoint(audit_geometries)
    assert set(split.train.loc[exposed["x"].index, "geometry_id"]) & audit_geometries
    report = read_json(mocked_training["output"] / "leakage_split.json")
    assert report == result["plan"]["leakage"]
    assert report["source"] == "canonical_train_only" and report["canonical_validation_test_used"] is False
    assert report["arms"]["honest"]["audit_geometry_overlap"] == 0
    assert report["arms"]["exposed"]["audit_geometry_overlap"] > 0
    assert report["arms"]["exposed"]["exposure_rows"] > 0
    assert report["arms"]["honest"]["rows"] == report["arms"]["exposed"]["rows"]
    audit_frame = split.train.loc[sorted(audit_indices)]
    assert study._row_hash(audit_frame) == report["audit_row_sha256"]
    audit_csv = pd.read_csv(mocked_training["output"] / "leakage_audit_rows.csv")
    pd.testing.assert_frame_equal(audit_csv, audit_frame[study._ROW_KEYS].reset_index(drop=True))
    metrics = pd.read_csv(mocked_training["output"] / "leakage_metrics.csv")
    for target in TARGETS:
        rows = metrics.loc[metrics.target == target]
        assert rows.audit_sst.nunique() == 1
        assert rows.audit_sst.iloc[0] == pytest.approx(np.sum((audit_frame[target] - audit_frame[target].mean()) ** 2))
    assert set(metrics.purpose) == {"diagnostic_not_production"}
    assert not any("test_evaluation" in result["trials"][f"leakage_{arm}"] for arm in ("honest", "exposed"))


def test_leakage_partition_is_label_and_input_order_independent(frozen):
    train = frozen[0].train
    first, report = study._leakage_partition(train)
    changed = train.iloc[::-1].copy()
    changed[TARGETS] = -999
    second, second_report = study._leakage_partition(changed)
    assert report == second_report
    for arm in first:
        assert set(first[arm].index) == set(second[arm].index)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_at", [2, 12])
def test_failed_or_interrupted_trial_stops_immediately_and_never_scores_test(cloud, frozen, mocked_training, failure, fail_at):
    mocked_training.update(fail_at=fail_at, failure=failure)
    with pytest.raises(failure, match="intentional mock failure"):
        run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    manifest = read_json(output / "study.json")
    assert manifest["status"] == "failed" and manifest["phase"] == "fitting"
    assert manifest["trials"]["seed_42_mse"]["status"] == "success"
    trial_order = list(manifest["trials"])
    assert manifest["trials"][trial_order[fail_at - 1]]["status"] == "failed"
    if fail_at < 12:
        assert manifest["trials"][trial_order[fail_at]]["status"] == "pending"
    assert len(mocked_training["calls"]) == fail_at and not mocked_training["evaluations"]
    assert not (output / "study_selection.json").exists()
    assert (output / manifest["trials"]["seed_42_mse"]["model_path"]).is_file()


def test_missing_weights_is_failure_not_completed_study(cloud, frozen, mocked_training):
    mocked_training["save_weights"] = False
    with pytest.raises(RuntimeError, match="did not save model weights"):
        run_mock(cloud, frozen, mocked_training)
    assert len(mocked_training["calls"]) == 1
    assert read_json(mocked_training["output"] / "study.json")["status"] == "failed"
    assert not mocked_training["evaluations"]


def test_audit_evaluation_failure_never_reaches_canonical_test(cloud, frozen, mocked_training, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("audit scoring failed")

    monkeypatch.setattr(study, "evaluate_model", fail)
    with pytest.raises(RuntimeError, match="audit scoring failed"):
        run_mock(cloud, frozen, mocked_training)
    manifest = read_json(mocked_training["output"] / "study.json")
    assert manifest["status"] == "failed" and manifest["phase"] == "leakage_audit"
    assert len(mocked_training["calls"]) == 12
    assert all("test_evaluation" not in trial for trial in manifest["trials"].values())


@pytest.mark.parametrize("bad", ["metadata", "overlap", "geometry", "row_count", "duplicate_ids", "duplicate_rows", "index", "features", "targets"])
def test_bad_frozen_inputs_are_rejected_before_output_or_training(cloud, frozen, mocked_training, bad):
    split, metadata = frozen
    features, targets = FEATURES.copy(), TARGETS.copy()
    if bad == "metadata":
        metadata = None
    elif bad == "overlap":
        metadata["split_geometries"]["test"].append(metadata["split_geometries"]["train"][0])
    elif bad == "geometry":
        split.validation.loc[split.validation.index[0], "geometry_id"] = "foreign"
    elif bad == "row_count":
        metadata["split_rows"]["train"] += 1
    elif bad == "duplicate_ids":
        metadata["split_geometries"]["train"].append(metadata["split_geometries"]["train"][0])
    elif bad == "duplicate_rows":
        split.train.iloc[1] = split.train.iloc[0]
    elif bad == "index":
        split.train.index = [0] * len(split.train)
    elif bad == "features":
        features.append("Cd")
    else:
        targets.reverse()
    output = cloud / "invalid"
    with pytest.raises(ValueError):
        study.run_study(split, features, targets, output, metadata, device="cpu")
    assert not output.exists() and not mocked_training["calls"]


@pytest.mark.parametrize("missing", ["working", "input"])
def test_kaggle_directories_are_required_before_metadata_or_output(tmp_path, monkeypatch, missing):
    working, inputs = tmp_path / "working", tmp_path / "input"
    if missing != "working":
        working.mkdir()
    if missing != "input":
        inputs.mkdir()
    monkeypatch.setattr(study, "_KAGGLE_WORKING", working)
    monkeypatch.setattr(study, "_KAGGLE_INPUT", inputs)
    output = working / "forbidden"
    with pytest.raises(RuntimeError, match="only on Kaggle"):
        study.run_study(None, FEATURES, TARGETS, output)
    assert not output.exists()


def test_cuda_unavailable_is_rejected_before_output_without_real_torch(cloud, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    output = cloud / "no-gpu"
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        study.run_study(None, FEATURES, TARGETS, output)
    assert not output.exists()


def test_cuda_available_guard_uses_lazy_mock_import(cloud, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: calls.append(True) or True)))
    assert study._require_kaggle(cloud / "gpu", "cuda") == cloud / "gpu"
    assert calls == [True] and not (cloud / "gpu").exists()


@pytest.mark.parametrize("outside", ["sibling", "parent", "symlink"])
def test_output_cannot_escape_working_directory(cloud, tmp_path, monkeypatch, outside):
    if outside == "symlink":
        link = cloud / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "escaped"
    elif outside == "parent":
        output = cloud / ".." / "escaped"
    else:
        output = tmp_path / "working-other" / "escaped"
    with pytest.raises(ValueError, match="inside /kaggle/working"):
        study.run_study(None, FEATURES, TARGETS, output)
    assert not output.exists()


def test_automatic_device_fallback_is_not_supported(cloud):
    with pytest.raises(ValueError, match="no automatic fallback"):
        study.run_study(None, FEATURES, TARGETS, cloud / "fallback", device="auto")
    assert not (cloud / "fallback").exists()


def test_existing_directory_is_not_overwritten(cloud, frozen, mocked_training):
    output = cloud / "study"
    output.mkdir()
    sentinel = output / "study.json"
    sentinel.write_text('{"keep": true}')
    with pytest.raises(FileExistsError):
        run_mock(cloud, frozen, mocked_training)
    assert sentinel.read_text() == '{"keep": true}'
    assert not mocked_training["calls"]


def test_unserializable_metadata_fails_before_output(cloud, frozen, mocked_training):
    frozen[1]["bad"] = object()
    with pytest.raises(TypeError):
        run_mock(cloud, frozen, mocked_training)
    assert not mocked_training["output"].exists()
    assert not mocked_training["calls"]


def test_canonical_test_labels_cannot_be_read_before_selection(cloud, frozen, mocked_training):
    output = cloud / "study"

    class GuardedTestFrame(pd.DataFrame):
        def __getitem__(self, key):
            columns = [key] if isinstance(key, str) else key
            if any(column in TARGETS or column == "shape" for column in columns):
                assert (output / "study_selection.json").is_file(), "Premature canonical test access"
            return super().__getitem__(key)

    frozen[0].test = GuardedTestFrame(frozen[0].test)
    result = run_mock(cloud, frozen, mocked_training)
    assert result["status"] == "completed"


def test_validation_tie_reuses_central_test_without_refit_or_duplicate_scoring(cloud, frozen, mocked_training, monkeypatch):
    original = comparison.evaluate_model

    def tied_validation(*args, **kwargs):
        evaluated = original(*args, **kwargs)
        evaluated["targets"]["Cd"]["median_relative_error"] = .123
        return evaluated

    monkeypatch.setattr(comparison, "evaluate_model", tied_validation)
    result = run_mock(cloud, frozen, mocked_training)
    assert result["selection"]["selected_trial"] == "seed_42_huber"
    assert len(mocked_training["calls"]) == 12
    assert len(mocked_training["evaluations"]) == 8  # Two audits plus all six seed tests.
    assert sum("test_evaluation" in trial for trial in result["trials"].values()) == 6
    sweep_test = pd.read_csv(mocked_training["output"] / "sweep_test_metrics.csv")
    assert set(sweep_test.trial_id) == {"seed_42_huber"}


def test_test_labels_do_not_change_validation_selection(cloud, frozen, mocked_training):
    first = run_mock(cloud, frozen, mocked_training, "first")
    frozen[0].test[TARGETS] *= 100
    second = run_mock(cloud, frozen, mocked_training, "second")
    assert first["selection"] == second["selection"]
    assert first["canonical_split"] == second["canonical_split"]


def test_insufficient_training_geometry_fails_before_fit_or_output(cloud, frozen, mocked_training):
    split, metadata = frozen
    split.train = split.train.iloc[:12].copy()
    metadata["split_geometries"]["train"] = sorted(split.train.geometry_id.unique())
    metadata["split_rows"]["train"] = len(split.train)
    with pytest.raises(ValueError, match="at least three"):
        run_mock(cloud, frozen, mocked_training)
    assert not mocked_training["output"].exists() and not mocked_training["calls"]


def test_json_checkpoints_include_predeclared_plan_and_are_atomic(cloud, frozen, mocked_training, monkeypatch):
    transitions = []
    original_replace = Path.replace

    def replace(source, target):
        if target.name == "study.json":
            assert source.name == "study.json.tmp"
            checkpoint = read_json(source)
            transitions.append(checkpoint["status"])
            if len(transitions) == 1:
                assert not mocked_training["calls"] and not mocked_training["evaluations"]
                assert checkpoint["status"] == "planned"
                assert len(checkpoint["plan"]["trial_configs"]) == 12
                assert all(trial["status"] == "pending" for trial in checkpoint["trials"].values())
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    run_mock(cloud, frozen, mocked_training)
    assert transitions[0] == "planned" and transitions[-1] == "completed"
    assert set(transitions[1:-1]) == {"running"}
