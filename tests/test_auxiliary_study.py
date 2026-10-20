"""Orchestration tests using mocked trainers and portable synthetic predictors."""

import builtins
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import SplitData
from src.models.torch_mlp import RAW_FEATURE_NAMES
from src.training import auxiliary_study, comparison, study


FEATURES = list(RAW_FEATURE_NAMES)
TARGETS = ["Cl", "Cd", "Cm"]


def physical_targets(x):
    signal = x["thickness_000"].to_numpy() * 1000 + x.Alpha.to_numpy() / 10
    return np.column_stack([.2 + .002 * signal, .01 + .00002 * signal, -.02 - .0001 * signal])


class MockPredictor:
    """No fitting or Torch dependency; predictions mimic the log-Cd contract."""

    def __init__(self, parameters, development_factor, test_factor):
        self.parameters = parameters
        self.development_factor = development_factor
        self.test_factor = test_factor
        self.training_metadata_ = {
            "accelerator": parameters["device"], "device_name": "mock device", "n_iter": 11,
            "best_epoch": 8, "stop_reason": "validation_patience", "max_iter": 300,
            "stopping_metric": "primary_normalized_validation_loss",
            "auxiliary_weight": parameters["auxiliary_weight"],
            "history": [{"epoch": 8, "validation_loss": .001}],
        }

    def predict(self, x):
        actual = physical_targets(x)
        primary_error = .05 + (self.parameters["random_state"] - 42) * .005
        factor = primary_error
        if self.parameters["auxiliary_weight"]:
            factor = np.where(x["thickness_000"].to_numpy() >= .023, self.test_factor, self.development_factor)
        physical = actual * (1 + np.asarray(factor).reshape(-1, 1))
        physical[:, 1] = np.log(physical[:, 1])
        return physical


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Auxiliary orchestration tests must never import Torch or fit real models.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def frozen():
    records = []
    for geometry in range(26):
        for alpha in range(5):
            row = dict.fromkeys(FEATURES, .01)
            row.update(geometry_id=f"g{geometry:02d}", Alpha=float(alpha), Re=1e5 if alpha == 0 else 1e6,
                       Ncrit=9., log10_re=5. if alpha == 0 else 6., max_thickness=.2,
                       thickness_000=geometry / 1000,
                       Cdp=.001 * geometry - .0001 * alpha, Cdp_observed=(geometry + alpha) % 7 != 0)
            if not row["Cdp_observed"]:
                row["Cdp"] = np.nan
            records.append(row)
    frame = pd.DataFrame(records)
    frame[TARGETS] = physical_targets(frame)
    split = SplitData(frame.iloc[:100].copy(), frame.iloc[100:115].copy(), frame.iloc[115:].copy())
    metadata = {
        "feature_columns": FEATURES.copy(), "target_columns": TARGETS.copy(),
        "split_geometries": {name: sorted(getattr(split, name).geometry_id.unique()) for name in ("train", "validation", "test")},
        "split_rows": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
        "split_row_sha256": {name: study._row_hash(getattr(split, name)) for name in ("train", "validation", "test")},
        "dataset": {"geometry_sha256": "a" * 64, "polars_sha256": "b" * 64, "polar_rows": len(frame)},
        "auxiliary": {"raw_sha256": "c" * 64, "sidecar_sha256": "d" * 64, "polars_sha256": "b" * 64},
        "git_commit": "e" * 40, "source_sha256": "f" * 64,
        "gpu": {"device_name": "mock runtime", "torch_version": "mock", "device_count": 1},
    }
    return split, metadata


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    working, inputs = tmp_path / "working", tmp_path / "input"
    working.mkdir()
    inputs.mkdir()
    monkeypatch.setattr(study, "_KAGGLE_WORKING", working)
    monkeypatch.setattr(study, "_KAGGLE_INPUT", inputs)
    return working


@pytest.fixture
def mocked_training(monkeypatch, frozen):
    split, _ = frozen
    state = {"calls": [], "scores": [], "fail_at": None, "failure": RuntimeError,
             "save_weights": True, "development_factor": .02, "test_factor": .3,
             "selection_snapshots": []}

    def train(name, x, y, output_dir, scale_targets, **parameters):
        root = output_dir.parent.parent.parent
        plan = read_json(root / "auxiliary_study.json")
        selection = root / "auxiliary_selection.json"
        phase = plan["phase"]
        assert plan["status"] == "running" and phase in ("development", "confirmation")
        assert plan["plan"]["estimated_fits"] == len(plan["plan"]["trial_configs"]) == 12
        assert selection.exists() is (phase == "confirmation")
        if selection.exists():
            state["selection_snapshots"].append(selection.read_bytes())
        assert name == "torch_mlp_huber" and scale_targets is False
        assert parameters["target_names"] == tuple(TARGETS)
        x_val, y_val = parameters.pop("validation_data")
        assert list(x.columns) == list(x_val.columns) == FEATURES
        assert not {"Cdp", "Cdp_observed"}.intersection(x.columns)
        expected, expected_val = physical_targets(x), physical_targets(x_val)
        expected[:, 1], expected_val[:, 1] = np.log(expected[:, 1]), np.log(expected_val[:, 1])
        np.testing.assert_allclose(y, expected)
        np.testing.assert_allclose(y_val, expected_val)
        source = split.train if phase == "development" else split.validation
        if parameters["auxiliary_weight"]:
            np.testing.assert_allclose(parameters.pop("auxiliary_targets"), split.train.loc[x.index, "Cdp"], equal_nan=True)
            np.testing.assert_allclose(parameters.pop("validation_auxiliary_targets"), source.loc[x_val.index, "Cdp"], equal_nan=True)
        else:
            assert "auxiliary_targets" not in parameters and "validation_auxiliary_targets" not in parameters
        state["calls"].append({"trial_id": output_dir.parent.name, "phase": phase,
                               "x": x.copy(), "x_val": x_val.copy(), "parameters": deepcopy(parameters)})
        if len(state["calls"]) == state["fail_at"]:
            raise state["failure"]("intentional mock failure")
        model = MockPredictor(parameters, state["development_factor"], state["test_factor"])
        if state["save_weights"]:
            joblib.dump(model, output_dir / f"{name}.joblib")
        return model, .125

    original_tail_report = auxiliary_study.tail_report

    def score(frame, prediction, targets, *, thresholds):
        if thresholds["source"] == "canonical_train":
            assert len(state["calls"]) == 12, "All six confirmations must finish before any test scoring."
            selection = state["output"] / "auxiliary_selection.json"
            assert selection.is_file()
            state["selection_snapshots"].append(selection.read_bytes())
            assert set(frame.index) == set(split.test.index)
        else:
            assert not (state["output"] / "auxiliary_selection.json").exists()
            assert set(frame.index).issubset(split.train.index)
        state["scores"].append({"index": frame.index.copy(), "thresholds": deepcopy(thresholds)})
        return original_tail_report(frame, prediction, targets, thresholds=thresholds)

    monkeypatch.setattr(comparison, "train_model", train)
    monkeypatch.setattr(auxiliary_study, "tail_report", score)
    return state


def run_mock(cloud, frozen, state, name="study"):
    state["output"] = cloud / name
    return auxiliary_study.run_auxiliary_study(frozen[0], FEATURES, TARGETS, state["output"], frozen[1], device="cpu")


def test_exact_twelve_fit_order_progress_configs_and_provenance(cloud, frozen, mocked_training, capsys):
    metadata_before = deepcopy(frozen[1])
    frames_before = {name: getattr(frozen[0], name).copy() for name in ("train", "validation", "test")}
    result = run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    assert result == read_json(output / "auxiliary_study.json")
    assert result["status"] == result["phase"] == "completed"
    expected = [f"{phase}_seed_{seed}_{arm}" for phase in ("development", "confirmation")
                for seed in (42, 43, 44) for arm in ("baseline", "cdp")]
    assert [call["trial_id"] for call in mocked_training["calls"]] == expected
    assert result["plan"]["trial_order"] == expected
    assert len(mocked_training["scores"]) == 12
    for phase in ("development", "confirmation"):
        for seed in (42, 43, 44):
            base = deepcopy(result["trials"][f"{phase}_seed_{seed}_baseline"]["config"])
            cdp = deepcopy(result["trials"][f"{phase}_seed_{seed}_cdp"]["config"])
            assert base["parameters"].pop("auxiliary_weight") == 0.
            assert cdp["parameters"].pop("auxiliary_weight") == .1
            assert base == cdp
            p = base["parameters"]
            assert p["max_iter"] == 300 and p["patience"] == 30 and p["loss"] == "huber"
            assert p["hidden_layers"] == [128, 128, 64] and p["target_weights"] == [1., 2., .5]
            assert p["huber_delta"] == .05 and p["weight_decay"] == 1e-5 and p["random_state"] == seed
    for trial in result["trials"].values():
        assert trial["status"] == "success"
        saved = read_json(output / trial["manifest"])
        forwarded = saved["run_metadata"]
        for key in ("dataset", "auxiliary", "split_rows", "split_geometries", "feature_columns", "target_columns", "git_commit", "source_sha256"):
            assert forwarded[key] == metadata_before[key]
        assert forwarded["fit_split"]["validation_source"] == (
            "canonical_validation" if trial["phase"] == "confirmation" else "canonical_train_inner_stopping")
        assert (output / trial["model_path"]).is_file()
        assert (output / trial["tail_report"]).is_file()
        assert trial["result"]["training_metadata"]["device_name"] == "mock device"
        assert trial["result"]["training_metadata"]["best_epoch"] == 8
    for artifact in result["artifacts"].values():
        assert (output / artifact).is_file()
    assert result["run_metadata"] == metadata_before == frozen[1]
    for name, frame in frames_before.items():
        pd.testing.assert_frame_equal(getattr(frozen[0], name), frame)
    text = capsys.readouterr().out
    assert "[Development 1/6] seed=42 arm=baseline" in text
    assert "[Development 6/6] seed=44 arm=cdp" in text
    assert "[Confirmation 1/6] seed=42 arm=baseline" in text
    assert "[Confirmation 6/6] seed=44 arm=cdp" in text
    assert "12/12 fits" in text and "not automatically promoted" in text
    assert not list(output.rglob("*.tmp"))


def test_common_inner_rows_fixed_seed_no_leakage_and_full_confirmation(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    split = frozen[0]
    inner = result["plan"]["inner_split"]
    ids = np.random.default_rng(20261020).permutation(sorted(split.train.geometry_id.unique()))
    assert inner["partition_seed"] == 20261020
    assert inner["partitions"]["stopping"]["geometry_ids"] == sorted(ids[:2])
    assert inner["partitions"]["evaluation"]["geometry_ids"] == sorted(ids[2:4])
    assert inner["partitions"]["fit"]["geometry_ids"] == sorted(ids[4:])
    recorded = pd.read_csv(mocked_training["output"] / "inner_rows.csv")
    assert set(recorded.canonical_row_index) == set(split.train.index)
    assert set(recorded.canonical_train_position) == set(range(len(split.train)))
    assert len(recorded) == len(split.train)
    assert not {"Cdp", "Cl", "Cd", "Cm"}.intersection(recorded.columns)
    for partition, rows in recorded.groupby("partition"):
        original = split.train.iloc[rows.canonical_train_position.to_numpy()]
        assert study._row_hash(original) == inner["partitions"][partition]["row_sha256"]
        assert list(original.index) == list(rows.canonical_row_index)
    fit, stopping, evaluation = [set(recorded.loc[recorded.partition == name, "canonical_row_index"]) for name in ("fit", "stopping", "evaluation")]
    assert not (fit & stopping or fit & evaluation or stopping & evaluation)
    for call in mocked_training["calls"][:6]:
        assert set(call["x"].index) == fit and set(call["x_val"].index) == stopping
    for call in mocked_training["calls"][6:]:
        assert set(call["x"].index) == set(split.train.index)
        assert set(call["x_val"].index) == set(split.validation.index)
    thresholds = result["plan"]["stress_thresholds"]
    assert thresholds["development"]["training_row_sha256"] == inner["partitions"]["fit"]["row_sha256"]
    assert thresholds["confirmation"]["training_row_sha256"] == study._row_hash(split.train)


def test_partition_is_independent_of_labels_and_input_order(frozen):
    train = frozen[0].train
    _, first, rows = auxiliary_study._inner_partition(train)
    changed = train.iloc[::-1].copy()
    changed[TARGETS + ["Cdp"]] = -999
    _, second, changed_rows = auxiliary_study._inner_partition(changed)
    assert first == second
    pd.testing.assert_frame_equal(rows.drop(columns="canonical_train_position"), changed_rows.drop(columns="canonical_train_position"))


def test_both_holdouts_are_untouched_before_frozen_decision(cloud, frozen, mocked_training, monkeypatch):
    output = cloud / "study"
    writes = []
    original_write = auxiliary_study._write_json

    class GuardedHoldout(pd.DataFrame):
        def __getitem__(self, key):
            columns = [key] if isinstance(key, str) else key
            protected = set(FEATURES + TARGETS + ["Cdp", "Cdp_observed"]).difference(study._ROW_KEYS)
            if any(column in protected for column in columns):
                assert (output / "auxiliary_selection.json").is_file(), "Premature holdout features/labels"
            return super().__getitem__(key)

    def write(path, payload):
        if path.name == "auxiliary_selection.json":
            assert len(mocked_training["calls"]) == 6
            assert len(mocked_training["scores"]) == 6
            writes.append(deepcopy(payload))
        return original_write(path, payload)

    frozen[0].validation = GuardedHoldout(frozen[0].validation)
    frozen[0].test = GuardedHoldout(frozen[0].test)
    monkeypatch.setattr(auxiliary_study, "_write_json", write)
    result = run_mock(cloud, frozen, mocked_training)
    assert len(writes) == 1 and writes[0] == result["selection"]
    assert result["selection"]["selected_arm"] == "cdp"
    assert result["selection"]["canonical_validation_scored"] is False
    assert result["selection"]["canonical_test_scored"] is False
    assert result["tail_audit"]["status"] == "fail"  # Deliberately worse outer scores.
    assert result["tail_audit"]["selection_changed"] is False
    assert len(set(mocked_training["selection_snapshots"])) == 1
    assert (output / "auxiliary_selection.json").read_bytes() == mocked_training["selection_snapshots"][0]


def test_failed_development_gate_falls_back_and_outer_success_cannot_rewrite(cloud, frozen, mocked_training):
    mocked_training.update(development_factor=.09, test_factor=.01)
    result = run_mock(cloud, frozen, mocked_training)
    assert len(mocked_training["calls"]) == 12
    selection = result["selection"]
    assert selection["selected_arm"] == "baseline" and selection["gate"]["status"] == "fail"
    assert "selected baseline" in selection["fallback_reason"]
    assert result["tail_audit"]["status"] == "pass"
    assert len(set(mocked_training["selection_snapshots"])) == 1
    assert read_json(mocked_training["output"] / "auxiliary_selection.json") == selection


def test_standard_confirmation_contract_all_seeds_mean_std_and_tail_deltas(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    for trial in result["trials"].values():
        directory = (output / trial["manifest"]).parent
        if trial["phase"] == "development":
            assert not (directory / "test_evaluation.json").exists()
            continue
        saved = read_json(directory / "run.json")
        evaluation = read_json(directory / "test_evaluation.json")
        csv = pd.read_csv(directory / "test_metrics.csv")
        for key in ("selected_models", "selection_metric", "selection_metrics", "target_names", "target_transforms"):
            assert evaluation[key] == saved[key]
        assert evaluation["selected_models"] == dict.fromkeys(TARGETS, "torch_mlp_huber")
        assert list(csv.target) == TARGETS and set(csv.n_predictions) == {len(frozen[0].test)}
        pd.testing.assert_frame_equal(csv, pd.DataFrame(evaluation["results"]), check_exact=False)
    metrics = pd.concat([pd.read_csv(output / f"{phase}_metrics.csv") for phase in ("development", "confirmation")])
    assert len(metrics) == 36
    summary = pd.read_csv(output / "summary.csv")
    for row in summary.itertuples():
        values = metrics.loc[(metrics.phase == row.phase) & (metrics.arm == row.arm) & (metrics.target == row.target), row.metric]
        assert row.n_seeds == row.n_defined == 3 and row.ddof == 1
        assert row.mean == pytest.approx(values.mean()) and row.std == pytest.approx(values.std(ddof=1))
    deltas = pd.read_csv(output / "paired_deltas.csv")
    assert len(deltas) == 18
    assert set(deltas.direction) == {"cdp_minus_baseline"}
    for row in deltas.itertuples():
        pair = metrics.loc[(metrics.phase == row.phase) & (metrics.seed == row.seed) & (metrics.target == row.target)].set_index("arm")
        assert row.median_relative_error == pytest.approx(pair.loc["cdp", "median_relative_error"] - pair.loc["baseline", "median_relative_error"])
        if row.target == "Cd":
            assert row.max_geometry_rmse == pytest.approx(pair.loc["cdp", "max_geometry_rmse"] - pair.loc["baseline", "max_geometry_rmse"])


@pytest.mark.parametrize("bad", ["metadata", "aux_provenance", "geometry", "row_count", "row_hash", "duplicate_rows", "index", "aux_feature", "mask_feature", "short_features", "targets", "missing_label", "missing_mask", "missing_thickness", "invalid_mask", "mask_mismatch", "infinite_cdp", "boolean_cdp", "all_missing", "unserializable"])
def test_bad_inputs_fail_before_outputs_or_fits(cloud, frozen, mocked_training, bad):
    split, metadata = frozen
    features, targets = FEATURES.copy(), TARGETS.copy()
    if bad == "metadata":
        metadata = None
    elif bad == "aux_provenance":
        metadata.pop("auxiliary")
    elif bad == "geometry":
        split.test.loc[split.test.index[0], "geometry_id"] = "foreign"
    elif bad == "row_count":
        metadata["split_rows"]["train"] += 1
    elif bad == "row_hash":
        split.train.loc[0, "Alpha"] += .1
    elif bad == "duplicate_rows":
        split.train.iloc[1] = split.train.iloc[0]
    elif bad == "index":
        split.train.index = [0] * len(split.train)
    elif bad in ("aux_feature", "mask_feature", "short_features"):
        features = [*features, "Cdp" if bad == "aux_feature" else "Cdp_observed"] if bad != "short_features" else features[:-1]
        metadata["feature_columns"] = features.copy()
    elif bad == "targets":
        targets.append("Cdp")
    elif bad in ("missing_label", "missing_mask", "missing_thickness"):
        column = {"missing_label": "Cdp", "missing_mask": "Cdp_observed", "missing_thickness": "max_thickness"}[bad]
        split.test = split.test.drop(columns=column)
    elif bad == "invalid_mask":
        split.train["Cdp_observed"] = "True"
    elif bad == "mask_mismatch":
        split.train.loc[0, "Cdp"] = .1  # This row is marked unobserved.
    elif bad == "infinite_cdp":
        split.train.loc[1, "Cdp"] = np.inf
    elif bad == "boolean_cdp":
        split.train["Cdp"] = True
        split.train["Cdp_observed"] = True
    elif bad == "all_missing":
        split.train["Cdp"] = np.nan
        split.train["Cdp_observed"] = False
    else:
        metadata["bad"] = object()
    output = cloud / "invalid"
    with pytest.raises((TypeError, ValueError)):
        auxiliary_study.run_auxiliary_study(split, features, targets, output, metadata, device="cpu")
    assert not output.exists() and not mocked_training["calls"]


@pytest.mark.parametrize("fail_at", [2, 8, 12])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_model_failure_is_fail_fast_preserves_evidence_and_never_scores_test(cloud, frozen, mocked_training, fail_at, failure):
    mocked_training.update(fail_at=fail_at, failure=failure)
    with pytest.raises(failure, match="intentional mock failure"):
        run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    manifest = read_json(output / "auxiliary_study.json")
    assert manifest["status"] == "failed"
    assert len(mocked_training["calls"]) == fail_at
    failed_id = manifest["plan"]["trial_order"][fail_at - 1]
    assert manifest["error"]["trial_id"] == failed_id
    assert "intentional mock failure" in manifest["error"]["message"]
    assert manifest["trials"][failed_id]["status"] == "failed"
    assert (output / "auxiliary_selection.json").exists() is (fail_at > 6)
    assert all(score["thresholds"]["source"] == "canonical_train_inner_fit" for score in mocked_training["scores"])
    assert not list(output.rglob("test_evaluation.json"))
    if fail_at < 12:
        assert manifest["trials"][manifest["plan"]["trial_order"][fail_at]]["status"] == "pending"


def test_missing_saved_weights_fails_instead_of_silently_continuing(cloud, frozen, mocked_training):
    mocked_training["save_weights"] = False
    with pytest.raises(RuntimeError, match="did not save model weights"):
        run_mock(cloud, frozen, mocked_training)
    assert len(mocked_training["calls"]) == 1 and not mocked_training["scores"]
    assert read_json(mocked_training["output"] / "auxiliary_study.json")["status"] == "failed"


def test_atomic_planned_checkpoint_precedes_fitting_and_existing_output_is_preserved(cloud, frozen, mocked_training, monkeypatch):
    transitions = []
    original = Path.replace

    def replace(path, destination):
        if destination.name == "auxiliary_study.json":
            assert path.name == "auxiliary_study.json.tmp"
            payload = read_json(path)
            transitions.append(payload["status"])
            if len(transitions) == 1:
                assert not mocked_training["calls"]
                assert payload["status"] == "planned"
                assert len(payload["plan"]["trial_configs"]) == 12
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    run_mock(cloud, frozen, mocked_training)
    assert transitions[0] == "planned" and transitions[-1] == "completed"
    assert set(transitions[1:-1]) == {"running"}
    before = (mocked_training["output"] / "auxiliary_study.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_mock(cloud, frozen, mocked_training)
    assert (mocked_training["output"] / "auxiliary_study.json").read_bytes() == before
    assert len(mocked_training["calls"]) == 12


def test_existing_kaggle_guard_runs_before_output_or_split_access(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_KAGGLE_WORKING", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="only on Kaggle"):
        auxiliary_study.run_auxiliary_study(None, FEATURES, TARGETS, tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


def test_empty_development_stress_falls_back_and_summary_never_averages_missing_values(cloud, frozen, mocked_training):
    positions, _, _ = auxiliary_study._inner_partition(frozen[0].train)
    train = frozen[0].train
    train["max_thickness"] = .1
    train.loc[train.index[positions["evaluation"]], "max_thickness"] = .01
    result = run_mock(cloud, frozen, mocked_training)
    assert result["status"] == "completed" and len(mocked_training["calls"]) == 12
    assert result["selection"]["selected_arm"] == "baseline"
    assert result["selection"]["gate"]["status"] == "inconclusive"
    summary = pd.read_csv(mocked_training["output"] / "summary.csv")
    rows = summary.loc[(summary.phase == "development") & (summary.metric == "thick_rmse")]
    assert len(rows) == 2 and set(rows.n_seeds) == {3} and set(rows.n_defined) == {0}
    assert set(rows.status) == {"inconclusive"} and rows["mean"].isna().all() and rows["std"].isna().all()


def test_scoring_failure_stops_before_further_fits_and_records_error(cloud, frozen, mocked_training, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("nonfinite prediction evidence")

    monkeypatch.setattr(auxiliary_study, "tail_report", fail)
    with pytest.raises(ValueError, match="nonfinite prediction"):
        run_mock(cloud, frozen, mocked_training)
    saved = read_json(mocked_training["output"] / "auxiliary_study.json")
    assert saved["status"] == "failed" and saved["error"]["type"] == "ValueError"
    assert len(mocked_training["calls"]) == 1
    assert not (mocked_training["output"] / "auxiliary_selection.json").exists()
    assert saved["trials"]["development_seed_42_baseline"]["result"]["status"] == "success"
    assert saved["trials"]["development_seed_42_baseline"]["status"] == "failed"


def test_canonical_test_label_changes_cannot_change_frozen_arm(cloud, frozen, mocked_training):
    first = run_mock(cloud, frozen, mocked_training, "first")
    frozen[0].test[TARGETS] *= 10
    for name in ("calls", "scores", "selection_snapshots"):
        mocked_training[name].clear()
    second = run_mock(cloud, frozen, mocked_training, "second")
    assert first["selection"] == second["selection"]
    assert first["canonical_split"] == second["canonical_split"]
