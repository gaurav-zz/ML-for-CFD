"""Fold orchestration tests: mocked training, portable NumPy prediction only."""

import builtins
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import SplitData
from src.models import ensemble
from src.models.torch_mlp import RAW_FEATURE_NAMES, TorchMLPRegressor
from src.training import comparison, fold_gate, fold_study, study
from src.training.targets import DEFAULT_TARGET_TRANSFORMS, forward_targets


FEATURES = list(RAW_FEATURE_NAMES)
TARGETS = ["Cl", "Cd", "Cm"]
SOURCE = "canonical_train_geometry_fold_evaluation"
EVENTS = {}


def physical_targets(x):
    geometry = x.thickness_000.to_numpy() * 1000
    return np.column_stack([.2 + geometry * .0003 + x.Alpha.to_numpy() * .001,
                            .01 + geometry * .00001 + x.Alpha.to_numpy() * .000001,
                            -.02 - geometry * .000002 - x.Alpha.to_numpy() * .00001])


class MockPredictor:
    """Saved predictions are fitting-space Cl/log(Cd)/Cm, never fitted locally."""

    def __init__(self, parameters, fold_id, stopping_indices):
        self.parameters = parameters
        self.fold_id = fold_id
        self.stopping_indices = stopping_indices
        self.training_metadata_ = {"accelerator": parameters["device"], "device_name": "mock device",
                                   "n_iter": 12, "best_epoch": 7,
                                   "stopping_metric": "primary_normalized_validation_loss"}

    def predict(self, x):
        indices = set(x.index)
        evaluation = indices != self.stopping_indices
        if evaluation:
            assert len(EVENTS["calls"]) == (self.fold_id + 1) * 6
            assert list(x.index) == list(EVENTS["split"].train.iloc[EVENTS["positions"][self.fold_id]["evaluation"]].index)
            assert not (EVENTS["output"] / "fold_selection.json").exists()
            EVENTS["predictions"].append((self.fold_id, self.parameters["random_state"], self.parameters["auxiliary_weight"], list(x.index)))
        actual = physical_targets(x)
        seed_index = self.parameters["random_state"] - 42
        arm = "cdp" if self.parameters["auxiliary_weight"] else "baseline"
        factors = EVENTS["factors"][arm]
        physical = actual * (1 + factors[seed_index])
        physical[:, 1] = np.log(physical[:, 1])
        if evaluation and EVENTS["bad_prediction"] == "shape":
            return physical[:, :2]
        if evaluation and EVENTS["bad_prediction"] == "nonfinite":
            physical[0, 0] = np.nan
        return physical


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Fold tests must not import Torch, fit real models, or install dependencies.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def frozen():
    records = []
    for geometry in range(578):
        for alpha in range(4):
            row = dict.fromkeys(FEATURES, .01)
            row.update(geometry_id=f"g{geometry:03d}", Alpha=float(alpha), Re=5e4 if alpha == 0 else 1e6,
                       Ncrit=9., log10_re=np.log10(5e4 if alpha == 0 else 1e6),
                       thickness_000=geometry / 1000, max_thickness=.1 + geometry * .00025,
                       Cdp=geometry * .00002 - alpha * .0001, Cdp_observed=(geometry + alpha) % 7 != 0)
            if not row["Cdp_observed"]:
                row["Cdp"] = np.nan
            records.append(row)
    frame = pd.DataFrame(records)
    frame[TARGETS] = physical_targets(frame)
    frame.index = np.arange(len(frame)) * 11 + 3
    split = SplitData(frame.iloc[:572 * 4].sample(frac=1, random_state=13),
                      frame.iloc[572 * 4:575 * 4].copy(), frame.iloc[575 * 4:].copy())
    metadata = {
        "feature_columns": FEATURES.copy(), "target_columns": TARGETS.copy(),
        "split_geometries": {name: sorted(getattr(split, name).geometry_id.unique()) for name in ("train", "validation", "test")},
        "split_rows": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
        "split_row_sha256": {name: study._row_hash(getattr(split, name)) for name in ("train", "validation", "test")},
        "dataset": {"geometry_sha256": "a" * 64, "polars_sha256": "b" * 64, "polar_rows": len(frame)},
        "auxiliary": {"raw_sha256": "c" * 64, "sidecar_sha256": "d" * 64},
        "git_commit": "e" * 40, "source_sha256": "f" * 64,
        "gpu": {"device_name": "mock runtime", "device_count": 1},
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
    state = {"calls": [], "predictions": [], "reports": [], "aggregates": [], "gate_calls": [],
             "split": split, "positions": fold_study._geometry_folds(split.train)[0],
             "fail_at": None, "failure": RuntimeError, "save_weights": True, "bad_prediction": None,
             "factors": {"baseline": [.05, .06, .07], "cdp": [.02, .021, .022]}}
    global EVENTS
    EVENTS = state

    def train(name, x, y, output_dir, scale_targets, **parameters):
        root = output_dir.parent.parent.parent
        plan = read_json(root / "fold_study.json")
        assert plan["status"] == "running" and plan["phase"] == "fold_fitting"
        assert plan["plan"]["estimated_fits"] == len(plan["plan"]["trial_configs"]) == 18
        assert (root / "folds.json").is_file() and (root / "fold_rows.csv").is_file()
        assert not (root / "fold_selection.json").exists()
        assert name == "torch_mlp_huber" and scale_targets is False
        trial_id = output_dir.parent.name
        fold_id = plan["trials"][trial_id]["fold_id"]
        x_stop, y_stop = parameters.pop("validation_data")
        assert parameters.pop("target_names") == tuple(TARGETS)
        assert list(x.columns) == list(x_stop.columns) == FEATURES
        np.testing.assert_allclose(y, forward_targets(physical_targets(x), TARGETS, DEFAULT_TARGET_TRANSFORMS))
        np.testing.assert_allclose(y_stop, forward_targets(physical_targets(x_stop), TARGETS, DEFAULT_TARGET_TRANSFORMS))
        expected_fit = split.train.iloc[state["positions"][fold_id]["fit"]]
        expected_stop = split.train.iloc[state["positions"][fold_id]["stopping"]]
        assert list(x.index) == list(expected_fit.index)
        assert list(x_stop.index) == list(expected_stop.index)
        if parameters["auxiliary_weight"]:
            np.testing.assert_allclose(parameters.pop("auxiliary_targets"), expected_fit.Cdp, equal_nan=True)
            np.testing.assert_allclose(parameters.pop("validation_auxiliary_targets"), expected_stop.Cdp, equal_nan=True)
        else:
            assert "auxiliary_targets" not in parameters and "validation_auxiliary_targets" not in parameters
        state["calls"].append({"trial_id": trial_id, "fold_id": fold_id, "fit_indices": list(x.index),
                               "stop_indices": list(x_stop.index), "parameters": deepcopy(parameters)})
        if len(state["calls"]) == state["fail_at"]:
            raise state["failure"]("intentional mock failure")
        model = MockPredictor(parameters, fold_id, set(x_stop.index))
        if state["save_weights"]:
            joblib.dump(model, output_dir / f"{name}.joblib")
        return model, .125

    original_report, original_aggregate = fold_study.tail_report, ensemble.aggregate_predictions
    original_gate = fold_gate.paired_fold_gate

    def report(frame, predictions, targets, *, thresholds):
        assert set(frame.index).issubset(split.train.index)
        assert len(state["calls"]) % 6 == 0
        fold_id = len(state["calls"]) // 6 - 1
        assert list(frame.index) == list(split.train.iloc[state["positions"][fold_id]["evaluation"]].index)
        assert thresholds["source"] == "canonical_train_inner_fit"
        fit = split.train.iloc[state["positions"][fold_id]["fit"]]
        assert thresholds["training_row_sha256"] == study._row_hash(fit)
        state["reports"].append({"fold_id": fold_id, "indices": list(frame.index), "predictions": predictions.copy()})
        return original_report(frame, predictions, targets, thresholds=thresholds)

    def aggregate(members):
        assert members.shape[0] == members.shape[2] == 3
        state["aggregates"].append(members.copy())
        return original_aggregate(members)

    def gate(reference, candidate, expected_geometry_ids):
        assert len(state["calls"]) == len(state["predictions"]) == 18
        assert len(state["reports"]) == 24 and len(state["aggregates"]) == 6
        assert not (state["output"] / "fold_selection.json").exists()
        assert set(reference) == set(candidate) == {0, 1, 2}
        assert expected_geometry_ids == sorted(split.train.geometry_id.unique())
        state["gate_calls"].append((deepcopy(reference), deepcopy(candidate)))
        return original_gate(reference, candidate, expected_geometry_ids)

    monkeypatch.setattr(comparison, "train_model", train)
    monkeypatch.setattr(fold_study, "tail_report", report)
    monkeypatch.setattr(ensemble, "aggregate_predictions", aggregate)
    monkeypatch.setattr(fold_gate, "paired_fold_gate", gate)
    return state


def run_mock(cloud, frozen, state, name="study"):
    state["output"] = cloud / name
    return fold_study.run_fold_study(frozen[0], FEATURES, TARGETS, state["output"], frozen[1], device="cpu")


def test_exact_18_fit_order_configs_and_complete_artifacts(cloud, frozen, mocked_training, capsys):
    before_metadata = deepcopy(frozen[1])
    before_frames = {name: getattr(frozen[0], name).copy() for name in ("train", "validation", "test")}
    result = run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    assert result == read_json(output / "fold_study.json")
    assert result["status"] == result["phase"] == "completed"
    expected = [f"fold_{fold}_seed_{seed}_{arm}" for fold in range(3)
                for seed in (42, 43, 44) for arm in ("baseline", "cdp")]
    assert result["plan"]["trial_order"] == [call["trial_id"] for call in mocked_training["calls"]] == expected
    assert result["completed_fits"] == result["completed_member_evaluations"] == 18
    assert result["completed_ensemble_evaluations"] == 6
    assert result["selection"]["selected_family"] == "cdp_ensemble"
    default_parameters = TorchMLPRegressor().get_params(deep=False)
    default_parameters.pop("target_names")
    default_parameters["device"] = "cpu"
    for trial_id, trial in result["trials"].items():
        p = deepcopy(trial["config"]["parameters"])
        assert p.pop("auxiliary_weight") == (.1 if trial["arm"] == "cdp" else 0.)
        assert p.pop("random_state") == trial["seed"]
        expected_parameters = deepcopy(default_parameters)
        expected_parameters.pop("auxiliary_weight")
        expected_parameters.pop("random_state")
        assert p == json.loads(json.dumps(expected_parameters))
        assert p["huber_delta"] == .05 and p["weight_decay"] == 1e-5
        assert p["hidden_layers"] == [128, 128, 64] and p["target_weights"] == [1., 2., .5]
        assert p["max_iter"] == 300 and p["patience"] == 30 and p["batch_size"] == 4096
        assert p["symmetry"] is True and p["learning_rate_init"] == .001
        assert trial["status"] == trial["evaluation_status"] == "success"
        saved = read_json(output / trial["manifest"])
        assert saved["status"] == "completed" and saved["target_transforms"] == DEFAULT_TARGET_TRANSFORMS
        for key, value in before_metadata.items():
            assert saved["run_metadata"][key] == value
        assert saved["run_metadata"]["fit_split"]["validation_source"] == "canonical_train_inner_stopping"
        assert saved["run_metadata"]["auxiliary_coverage"]["scaler_source"] == "canonical_train_inner_fit"
        assert (output / trial["model_path"]).is_file() and (output / trial["tail_report"]).is_file()
        assert trial["result"]["training_metadata"]["best_epoch"] == 7
    for path in result["artifacts"].values():
        assert (output / path).is_file()
    assert not list(output.rglob("test_evaluation.json")) and not list(output.rglob("*.tmp"))
    assert result["run_metadata"] == before_metadata == frozen[1]
    for name, frame in before_frames.items():
        pd.testing.assert_frame_equal(frame, getattr(frozen[0], name))
    text = capsys.readouterr().out
    assert "[Fold 1/3 | Fit 1/18] seed=42 arm=baseline" in text
    assert "[Fold 3/3 | Fit 18/18] seed=44 arm=cdp" in text
    assert "18/18 fits" in text and "locked assessment required" in text


def test_deterministic_folds_snapshot_exact_rows_and_fit_only_thresholds(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    train = frozen[0].train
    plan = read_json(mocked_training["output"] / "folds.json")
    assert plan == result["plan"]["fold_plan"]
    ids = sorted(train.geometry_id.unique())
    expected_folds = np.array_split(np.random.default_rng(20261026).permutation(ids), 3)
    assert plan["evaluation_geometry_counts"] == [191, 191, 190]
    snapshot = pd.read_csv(mocked_training["output"] / "fold_rows.csv")
    assert len(snapshot) == 3 * len(train)
    assert not set(TARGETS + ["Cdp", "Cdp_observed"]).intersection(snapshot.columns)
    all_evaluation_ids = set()
    for fold_id, expected_eval in enumerate(expected_folds):
        fold = plan["folds"][str(fold_id)]
        remaining = sorted(set(ids).difference(expected_eval))
        shuffled = np.random.default_rng(20261026 + fold_id + 1).permutation(remaining)
        assert fold["partitions"]["stopping"]["geometry_ids"] == sorted(shuffled[:38])
        assert fold["partitions"]["fit"]["geometry_ids"] == sorted(shuffled[38:])
        assert fold["partitions"]["fit"]["n_geometries"] == (344 if fold_id == 2 else 343)
        assert fold["partitions"]["evaluation"]["geometry_ids"] == sorted(expected_eval)
        assert not all_evaluation_ids.intersection(expected_eval)
        all_evaluation_ids.update(expected_eval)
        fold_snapshot = snapshot.loc[snapshot.fold_id == fold_id]
        assert set(fold_snapshot.canonical_train_position) == set(range(len(train)))
        assert set(fold_snapshot.canonical_row_index) == set(train.index)
        groups = []
        for name, part in fold["partitions"].items():
            rows = fold_snapshot.loc[fold_snapshot.partition == name]
            original = train.iloc[rows.canonical_train_position.to_numpy()]
            assert list(original.index) == list(rows.canonical_row_index)
            pd.testing.assert_frame_equal(original[study._ROW_KEYS].reset_index(drop=True),
                                          rows[study._ROW_KEYS].reset_index(drop=True), check_dtype=False)
            pd.testing.assert_frame_equal(original[study._ROW_KEYS].reset_index(drop=True),
                                          original[study._ROW_KEYS].sort_values(study._ROW_KEYS).reset_index(drop=True))
            assert study._row_hash(original) == part["row_sha256"]
            assert fold_study._geometry_hash(original.geometry_id.unique()) == part["geometry_sha256"]
            groups.append(set(original.geometry_id))
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert set.union(*groups) == set(ids)
        assert fold["stress_thresholds"]["source"] == "canonical_train_inner_fit"
        assert fold["stress_thresholds"]["training_row_sha256"] == fold["partitions"]["fit"]["row_sha256"]
    assert all_evaluation_ids == set(ids)


def test_partition_ignores_labels_and_retains_arbitrary_indices(frozen):
    train = frozen[0].train.copy()
    train.index = [f"original_{index}" for index in train.index]
    _, first, rows = fold_study._geometry_folds(train)
    changed = train.iloc[::-1].copy()
    changed[TARGETS + ["Cdp"]] = -999
    _, second, changed_rows = fold_study._geometry_folds(changed)
    assert first == second
    pd.testing.assert_frame_equal(rows.drop(columns="canonical_train_position"),
                                  changed_rows.drop(columns="canonical_train_position"))


def test_holdouts_poisoned_forever_and_selection_written_once_after_every_evaluation(cloud, frozen, mocked_training, monkeypatch):
    writes = []
    original_write = fold_study._write_json

    class PoisonHoldout(pd.DataFrame):
        @property
        def _constructor(self):
            return pd.DataFrame

        def __getitem__(self, key):
            names = [key] if isinstance(key, str) else list(key)
            assert set(names).issubset(study._ROW_KEYS), "Canonical holdout feature/label access is forbidden forever."
            return super().__getitem__(key)

        @property
        def iloc(self):
            raise AssertionError("Do not materialize canonical holdout rows.")

        def to_numpy(self, *args, **kwargs):
            assert set(self.columns).issubset(study._ROW_KEYS)
            return super().to_numpy(*args, **kwargs)

    for name in ("validation", "test"):
        frame = getattr(frozen[0], name).copy()
        frame[list(set(FEATURES + TARGETS + ["Cdp"]).difference(study._ROW_KEYS))] = np.nan
        setattr(frozen[0], name, PoisonHoldout(frame))

    def write(path, payload):
        if path.name == "fold_selection.json":
            assert len(mocked_training["calls"]) == len(mocked_training["predictions"]) == 18
            assert len(mocked_training["reports"]) == 24 and len(mocked_training["gate_calls"]) == 2
            assert not path.exists()
            writes.append(deepcopy(payload))
        return original_write(path, payload)

    monkeypatch.setattr(fold_study, "_write_json", write)
    result = run_mock(cloud, frozen, mocked_training)
    assert len(writes) == 1 and writes[0] == result["selection"]
    assert result["selection"]["canonical_validation_scored"] is False
    assert result["selection"]["canonical_test_scored"] is False
    assert result["canonical_validation_test_used_for_model_apis"] is False


def test_member_predictions_physical_cd_and_ensemble_provenance(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    assert len(mocked_training["predictions"]) == 18
    assert len(set((fold, seed, weight) for fold, seed, weight, _ in mocked_training["predictions"])) == 18
    for i, members in enumerate(mocked_training["aggregates"]):
        fold_id, arm = i // 2, ("baseline", "cdp")[i % 2]
        frame = frozen[0].train.iloc[mocked_training["positions"][fold_id]["evaluation"]]
        actual = physical_targets(frame)
        for j, factor in enumerate(mocked_training["factors"][arm]):
            np.testing.assert_allclose(members[j], actual * (1 + factor))
        report = read_json(mocked_training["output"] / result["ensemble_reports"][str(fold_id)][arm])
        assert report["source"] == SOURCE and report["fold_id"] == fold_id
        assert report["predictor"] == f"{arm}_ensemble" and report["member_seeds"] == [42, 43, 44]
        assert report["aggregation"] == "mean_Cl_Cm_geometric_mean_Cd"
        assert "seed" not in report and "seeds" not in report
        scored = mocked_training["reports"][fold_id * 8 + (3 if arm == "baseline" else 7)]["predictions"]
        np.testing.assert_allclose(scored[:, [0, 2]], members[:, :, [0, 2]].mean(axis=0))
        np.testing.assert_allclose(scored[:, 1], np.exp(np.log(members[:, :, 1]).mean(axis=0)))
        assert not np.allclose(scored[:, 1], np.exp(members[:, :, 1].mean(axis=0)))
        diagnostics = read_json(mocked_training["output"] / f"folds/fold_{fold_id}/{arm}_disagreement.json")
        assert diagnostics["spread_thresholds"]["source"] == "prediction_only"
        assert diagnostics["calibrated_uncertainty"] is diagnostics["row_filtering"] is False
    for references, candidates in mocked_training["gate_calls"]:
        for fold_id in range(3):
            assert references[fold_id]["predictor"] == "baseline_seed_42"
            assert references[fold_id]["member_seeds"] == [42]
            assert references[fold_id]["seed"] == 42
            assert references[fold_id]["row_sha256"] == candidates[fold_id]["row_sha256"]
            assert references[fold_id]["thresholds"] == candidates[fold_id]["thresholds"]


def test_summary_separates_three_seeds_from_three_unweighted_folds(cloud, frozen, mocked_training):
    result = run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    members = pd.read_csv(output / "member_metrics.csv")
    ensembles = pd.read_csv(output / "ensemble_metrics.csv")
    summary = pd.read_csv(output / "summary.csv")
    assert len(members) == 54 and len(ensembles) == 18
    assert set(summary.variation) == {"seed_within_fold", "fold_variation"}
    for row in summary.itertuples():
        if row.variation == "seed_within_fold":
            values = members.loc[(members.fold_id == row.fold_id) & (members.arm == row.arm) & (members.target == row.target), row.metric]
            assert row.n_seeds == 3 and np.isnan(row.n_folds)
        else:
            source = ensembles if row.predictor.endswith("ensemble") else members
            values = source.loc[(source.predictor == row.predictor) & (source.target == row.target), row.metric]
            assert row.n_folds == 3 and np.isnan(row.n_seeds)
        assert row.n_defined == 3 and row.ddof == 1 and row.status == "complete"
        assert row.mean == pytest.approx(values.mean()) and row.std == pytest.approx(values.std(ddof=1))
        assert row.predictions_ensembled_across_folds is False
    for family, score in result["selection"]["family_scores"].items():
        values = ensembles.loc[(ensembles.predictor == family) & (ensembles.target == "Cd"), "median_relative_error"]
        assert score["mean_fold_Cd_median_relative_error"] == pytest.approx(values.mean())
        assert score["n_folds"] == 3 and score["weighting"] == "unweighted folds"


@pytest.mark.parametrize("factors,expected", [({"baseline": [.05, .01, .02], "cdp": [.04, .041, .042]}, "baseline_ensemble"),
                                           ({"baseline": [.05, .06, .07], "cdp": [.08, .09, .10]}, "baseline_seed_42"),
                                           ({"baseline": [.05, .01, .02], "cdp": [.05, .01, .02]}, "baseline_ensemble")])
def test_eligible_lowest_mean_ties_and_predeclared_fallback(cloud, frozen, mocked_training, factors, expected):
    mocked_training["factors"] = factors
    result = run_mock(cloud, frozen, mocked_training)
    assert len(mocked_training["calls"]) == 18
    selection = result["selection"]
    assert selection["selected_family"] == expected
    assert selection["seed_selection"] is False
    assert selection["confidence_calibrated"] is selection["old_test_improvement_claim"] is False
    assert selection == read_json(mocked_training["output"] / "fold_selection.json")
    if expected == "baseline_seed_42":
        assert selection["member_seeds"] == [42] and selection["fallback_reason"]
    else:
        assert selection["member_seeds"] == [42, 43, 44]


def test_empty_stress_is_inconclusive_no_reduced_fold_average_or_selection_rewrite(cloud, frozen, mocked_training):
    train = frozen[0].train
    train.loc[train.geometry_id.isin(mocked_training["split"].train.iloc[mocked_training["positions"][0]["evaluation"]].geometry_id), "max_thickness"] = .001
    result = run_mock(cloud, frozen, mocked_training)
    assert result["status"] == "completed" and result["completed_fits"] == 18
    assert result["selection"]["selected_family"] == "baseline_seed_42"
    assert all(gate["status"] == "inconclusive" for gate in result["selection"]["gates"].values())
    summary = pd.read_csv(mocked_training["output"] / "summary.csv")
    rows = summary.loc[(summary.variation == "fold_variation") & (summary.metric == "thick_rmse")]
    assert len(rows) == 8 and set(rows.n_folds) == {3} and (rows.n_defined < 3).all()
    assert set(rows.status) == {"inconclusive"} and rows["mean"].isna().all() and rows["std"].isna().all()


@pytest.mark.parametrize("fail_at", [1, 7, 18])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_member_fit_checkpoints_and_stops_before_selection(cloud, frozen, mocked_training, fail_at, failure):
    mocked_training.update(fail_at=fail_at, failure=failure)
    with pytest.raises(failure, match="intentional mock failure"):
        run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    manifest = read_json(output / "fold_study.json")
    assert manifest["status"] == "error"
    assert len(mocked_training["calls"]) == fail_at
    failed = manifest["plan"]["trial_order"][fail_at - 1]
    assert manifest["error"]["trial_id"] == failed and manifest["trials"][failed]["status"] == "error"
    assert manifest["completed_fits"] == fail_at - 1
    assert not (output / "fold_selection.json").exists() and not mocked_training["gate_calls"]
    assert (output / "folds.json").is_file() and (output / "fold_rows.csv").is_file()


def test_missing_weights_fails_without_reduced_member_ensemble(cloud, frozen, mocked_training):
    mocked_training["save_weights"] = False
    with pytest.raises(RuntimeError, match="did not save model weights"):
        run_mock(cloud, frozen, mocked_training)
    manifest = read_json(mocked_training["output"] / "fold_study.json")
    assert manifest["status"] == "error" and len(mocked_training["calls"]) == 1
    assert not mocked_training["predictions"] and not mocked_training["aggregates"]


def test_member_disappearing_after_fits_fails_not_two_member_average(cloud, frozen, mocked_training, monkeypatch):
    original_load = fold_study.joblib.load

    def load(path):
        if "seed_43_baseline" in str(path):
            raise FileNotFoundError("missing evaluation member")
        return original_load(path)

    monkeypatch.setattr(fold_study.joblib, "load", load)
    with pytest.raises(FileNotFoundError, match="missing evaluation member"):
        run_mock(cloud, frozen, mocked_training)
    assert len(mocked_training["calls"]) == 6 and len(mocked_training["predictions"]) == 1
    assert not mocked_training["aggregates"]
    manifest = read_json(mocked_training["output"] / "fold_study.json")
    assert manifest["status"] == "error" and manifest["completed_fits"] == 6
    assert manifest["trials"]["fold_0_seed_43_baseline"]["evaluation_status"] == "error"


@pytest.mark.parametrize("kind", ["shape", "nonfinite"])
def test_bad_predictions_fail_at_six_fit_barrier_with_partial_evidence(cloud, frozen, mocked_training, kind):
    mocked_training["bad_prediction"] = kind
    with pytest.raises(ValueError):
        run_mock(cloud, frozen, mocked_training)
    manifest = read_json(mocked_training["output"] / "fold_study.json")
    assert manifest["status"] == "error" and manifest["completed_fits"] == 6
    assert manifest["trials"]["fold_0_seed_42_baseline"]["status"] == "success"
    assert manifest["trials"]["fold_0_seed_42_baseline"]["evaluation_status"] == "error"
    assert not mocked_training["aggregates"] and not mocked_training["gate_calls"]


def test_finalization_error_preserves_frozen_selection(cloud, frozen, mocked_training, monkeypatch):
    original_csv = study._write_csv

    def csv(path, records):
        if path.name == "summary.csv":
            raise OSError("summary disk error")
        return original_csv(path, records)

    monkeypatch.setattr(study, "_write_csv", csv)
    with pytest.raises(OSError, match="summary disk error"):
        run_mock(cloud, frozen, mocked_training)
    output = mocked_training["output"]
    selection = read_json(output / "fold_selection.json")
    manifest = read_json(output / "fold_study.json")
    assert manifest["status"] == "error" and manifest["selection"] == selection
    before = (output / "fold_selection.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_mock(cloud, frozen, mocked_training)
    assert (output / "fold_selection.json").read_bytes() == before
    assert len(mocked_training["calls"]) == 18


@pytest.mark.parametrize("bad", ["metadata", "auxiliary", "geometry_count", "targets", "features", "aux_feature", "rows", "hash", "duplicate", "missing_cdp", "mask", "unobserved", "all_missing", "unserializable"])
def test_bad_inputs_fail_before_fit_or_output(cloud, frozen, mocked_training, bad):
    split, metadata = frozen
    features, targets = FEATURES.copy(), TARGETS.copy()
    if bad == "metadata":
        metadata = None
    elif bad == "auxiliary":
        metadata.pop("auxiliary")
    elif bad == "geometry_count":
        split.train = split.train.loc[split.train.geometry_id != "g000"]
        metadata["split_geometries"]["train"].remove("g000")
        metadata["split_rows"]["train"] = len(split.train)
        metadata["split_row_sha256"]["train"] = study._row_hash(split.train)
    elif bad == "targets":
        targets = ["Cl", "Cm", "Cd"]
    elif bad in ("features", "aux_feature"):
        features = features[:-1] if bad == "features" else [*features, "Cdp"]
        metadata["feature_columns"] = features.copy()
    elif bad == "rows":
        metadata["split_rows"]["test"] += 1
    elif bad == "hash":
        metadata["split_row_sha256"]["train"] = "0" * 64
    elif bad == "duplicate":
        split.train.index = [0] * len(split.train)
    elif bad == "missing_cdp":
        split.train = split.train.drop(columns="Cdp")
    elif bad == "mask":
        split.train["Cdp_observed"] = "true"
    elif bad == "unobserved":
        split.train["Cdp"] = .1
    elif bad == "all_missing":
        split.train["Cdp"] = np.nan
        split.train["Cdp_observed"] = False
    else:
        metadata["bad"] = object()
    output = cloud / "invalid"
    with pytest.raises((ValueError, TypeError)):
        fold_study.run_fold_study(split, features, targets, output, metadata, device="cpu")
    assert not output.exists() and not mocked_training["calls"]


def test_stress_derivation_never_receives_labels_and_auxiliary_never_evaluation(cloud, frozen, mocked_training, monkeypatch):
    original_thresholds, original_aux = fold_study.derive_thresholds, fold_study._auxiliary_values
    seen_thresholds = []
    allowed_indices = [set(frozen[0].train.iloc[pos[name]].index) for pos in mocked_training["positions"].values()
                       for name in ("fit", "stopping")]

    def thresholds(frame, *, source):
        assert list(frame.columns) == [*study._ROW_KEYS, "max_thickness"]
        assert source == "canonical_train_inner_fit"
        seen_thresholds.append(study._row_hash(frame))
        return original_thresholds(frame, source=source)

    def auxiliary(frame):
        assert set(frame.index) in allowed_indices
        return original_aux(frame)

    monkeypatch.setattr(fold_study, "derive_thresholds", thresholds)
    monkeypatch.setattr(fold_study, "_auxiliary_values", auxiliary)
    run_mock(cloud, frozen, mocked_training)
    assert len(seen_thresholds) == 3


def test_atomic_planned_checkpoint_before_fit_and_existing_output_preserved(cloud, frozen, mocked_training, monkeypatch):
    statuses = []
    original_replace = Path.replace

    def replace(path, destination):
        if destination.name == "fold_study.json":
            assert path.name == "fold_study.json.tmp"
            payload = read_json(path)
            statuses.append(payload["status"])
            if len(statuses) == 1:
                assert not mocked_training["calls"] and payload["status"] == "planned"
                assert len(payload["plan"]["trial_configs"]) == 18
                assert len(payload["plan"]["fold_plan"]["folds"]) == 3
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", replace)
    run_mock(cloud, frozen, mocked_training)
    assert statuses[0] == "planned" and statuses[-1] == "completed" and set(statuses[1:-1]) == {"running"}
    output = mocked_training["output"]
    before = (output / "fold_study.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_mock(cloud, frozen, mocked_training)
    assert (output / "fold_study.json").read_bytes() == before


def test_kaggle_guard_precedes_split_access_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_KAGGLE_WORKING", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="only on Kaggle"):
        fold_study.run_fold_study(None, FEATURES, TARGETS, tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()
