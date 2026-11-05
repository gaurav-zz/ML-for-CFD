"""Fit-free saved-array and referral-accounting tests; no model APIs."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.training import reliability_analysis as ra


@pytest.fixture
def train():
    return pd.DataFrame({"geometry_id": np.repeat(list("abcdef"), 2),
                         "Alpha": [1., -1.] * 6, "Re": 1e6, "Ncrit": 9.,
                         "Cl": np.linspace(-1, 1, 12), "Cd": np.linspace(.01, .03, 12), "Cm": .02})


def saved_run(path, train, polars_hash, mode):
    path.mkdir()
    candidate = "profile_rms_floor" if mode == "input_scaling" else "profile_rms_floor_1pct"
    modes = dict(zip(ra.ARMS, ("standard", candidate)))
    manifest = {"schema_version": 1, "status": "completed", "phase": "completed", "source": ra.SOURCE,
        "completed_fits": 18, "completed_member_evaluations": 18, "completed_ensemble_evaluations": 6,
        "run_metadata": {"study_mode": mode, "dataset": {"polars_sha256": polars_hash}, "source_sha256": "fixture"},
        "study_kind": "paired_geometry_input_scaling", "target_names": ra.TARGETS,
        "canonical_validation_test_used_for_model_apis": False,
        "canonical_split": {"train": {"rows": len(train), "geometry_ids": sorted(train.geometry_id.unique()),
                                      "row_sha256": ra._row_hash(train)}},
        "plan": {"seeds": [42, 43, 44], "arms": list(ra.ARMS), "n_folds": 3,
                 "aggregation": "mean_Cl_Cm_geometric_mean_Cd", "input_scaling_modes": modes,
                 "fold_plan": {"folds": {}}, "trial_configs": {}}, "trials": {}, "prediction_artifacts": {}}
    if mode == "selective_scaling":
        manifest["study_variant"] = "selective_profile_floor_1pct"
    for fold, evaluation_ids in enumerate((list("ab"), list("cd"), list("ef"))):
        fold = str(fold)
        remaining = sorted(set(train.geometry_id) - set(evaluation_ids))
        partitions = {}
        for partition, ids in (("evaluation", evaluation_ids), ("fit", remaining[:2]), ("stopping", remaining[2:])):
            rows = train.loc[train.geometry_id.isin(ids)]
            partitions[partition] = {"geometry_ids": ids, "n_rows": len(rows), "row_sha256": ra._row_hash(rows)}
        manifest["plan"]["fold_plan"]["folds"][fold] = {"partitions": partitions}
        manifest["prediction_artifacts"][fold] = {}
        expected = train.loc[train.geometry_id.isin(evaluation_ids)].sort_values(ra.KEYS).index.to_numpy()
        raw = np.broadcast_to(train[ra.TARGETS].to_numpy()[expected], (3, len(expected), 3)).copy()
        raw[:, :, 1] = np.log(raw[:, :, 1])
        raw[:, :, 1] += np.array([-.01, 0, .01])[:, None]
        physical = raw.copy()
        physical[:, :, 1] = np.exp(raw[:, :, 1])
        ensemble = physical.mean(axis=0)
        ensemble[:, 1] = np.exp(np.log(physical[:, :, 1]).mean(axis=0))
        for arm in ra.ARMS:
            arrays = {"canonical_train_positions": expected, "member_seeds": np.array([42, 43, 44]),
                "fitting_predictions": raw, "physical_predictions": physical, "physical_ensemble_predictions": ensemble,
                "lower_clipped": np.zeros((3, len(expected)), bool), "upper_clipped": np.zeros((3, len(expected)), bool),
                "raw_log_Cd_std": raw[:, :, 1].std(axis=0, ddof=1), "clipped_log_Cd_std": raw[:, :, 1].std(axis=0, ddof=1)}
            if mode == "selective_scaling":
                arrays.update(input_scaling=np.asarray(modes[arm]), study_variant=np.asarray(manifest["study_variant"]))
            filename = f"{fold}_{arm}.npz"
            np.savez(path / filename, **arrays)
            manifest["prediction_artifacts"][fold][arm] = {"path": filename, "status": "complete", "n_members": 3,
                                                           "n_rows": len(expected), "ensemble_present": True}
            for seed in (42, 43, 44):
                trial_id = f"fold_{fold}_seed_{seed}_{arm}"
                config = {"parameters": {"random_state": seed, "input_scaling": modes[arm], "auxiliary_weight": .1}}
                manifest["plan"]["trial_configs"][trial_id] = config
                manifest["trials"][trial_id] = {"config": config, "status": "success", "evaluation_status": "success"}
    (path / "fold_study.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def inputs(tmp_path, train):
    polars, split = tmp_path / "polars.csv", tmp_path / "split.json"
    holdouts = pd.DataFrame({"geometry_id": ["v", "t"], "Alpha": [0., 0.], "Re": [1e6, 1e6],
                            "Ncrit": [9., 9.], "Cl": [np.nan, np.nan], "Cd": [0., -1.], "Cm": [np.nan, np.nan]})
    pd.concat([train, holdouts]).to_csv(polars, index=False)
    split.write_text(json.dumps({"split_geometries": {"train": list("abcdef"), "validation": ["v"], "test": ["t"]},
                                "v1_split_rows": {"train": 12}, "geometry_counts": {"train": 6}}))
    canonical = ra.load_train(polars, split)
    broad, selective = tmp_path / "broad", tmp_path / "selective"
    for path, mode in ((broad, "input_scaling"), (selective, "selective_scaling")):
        saved_run(path, canonical, ra._digest(polars), mode)
    return dict(broad_run=broad, selective_run=selective, polars=polars, split=split, output_dir=tmp_path / "output")


def rewrite_array(path, key, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[key] = change(arrays[key])
    np.savez(path, **arrays)


@pytest.mark.parametrize("positions", [[1, 0], [0, 0], [0], [-1, 1], [0, 12], [0., 1.], [False, True]])
def test_alignment_rejects_wrong_order_duplicates_missing_or_invalid(train, positions):
    with pytest.raises(ValueError, match="alignment"):
        ra.validate_alignment(train, positions, np.array([0, 1]))


def test_alignment_accepts_exact_original_positions(train):
    ra.validate_alignment(train, [1, 0, 3, 2], np.array([1, 0, 3, 2]))


@pytest.mark.parametrize("key", ["raw", "clipped", "distance"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1., 1j])
def test_scores_reject_nonfinite_negative_or_complex(key, value):
    args = dict(raw=np.ones(3), clipped=np.ones(3), distance=np.ones(3))
    args[key] = np.array([0, value, 1])
    with pytest.raises(ValueError):
        ra.prediction_scores(**args)


def test_ties_and_prediction_only_percentiles():
    scores = ra.prediction_scores([1., 1., 1., 3.], [0., 0., 0., 0.], [2., 2., 2., 1.])
    np.testing.assert_array_equal(scores["max_raw_geometry_percentile"], [1., 1., 1., 1.])
    retained, cutoff = ra.retained_mask(scores["raw_log_Cd_std"], .5, np.zeros(4, bool))
    assert cutoff == 1
    np.testing.assert_array_equal(retained, [True, True, True, False])
    all_tied, _ = ra.retained_mask(scores["max_raw_geometry_percentile"], .5, np.zeros(4, bool))
    assert all_tied.all()


def record(train, hard=None):
    return {"prediction": train[ra.TARGETS].to_numpy() * np.array([1.1, 1.5, 1.2]),
            "raw": np.arange(len(train), dtype=float), "clipped": np.arange(len(train), dtype=float),
            "hard": np.zeros(len(train), bool) if hard is None else hard}


def test_ranking_independent_of_truth(train):
    predictions = record(train)
    altered = train.copy()
    altered[ra.TARGETS] *= 4
    before = ra.analyze(train, {"sample": predictions})
    after = ra.analyze(altered, {"sample": predictions})
    fields = ["method", "requested_coverage", "score_cutoff", "actual_coverage", "retained_rows",
              "fully_retained_geometries", "partially_retained_geometries", "fully_referred_geometries"]
    pd.testing.assert_frame_equal(before["risk_coverage"][fields], after["risk_coverage"][fields])
    assert not before["risk_coverage"]["Cd_rmse"].equals(after["risk_coverage"]["Cd_rmse"])


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
def test_actual_cd_must_be_positive_even_on_referred_rows(bad):
    actual = np.array([[0., 1., 0.], [0., bad, 0.]])
    with pytest.raises(ValueError):
        ra.error_metrics(actual, np.ones((2, 3)), np.array([True, False]))


def test_no_hidden_near_zero_clamp():
    _, relative = ra.error_metrics([[0., 1e-100, 0.]], [[0., 1e-99, 0.]], np.array([True]))
    assert relative[0] == pytest.approx(9)


def test_fp_fn_convention_and_strict_budget_boundary():
    result = ra.sensitivity([.1, .2, .3, .01], np.array([True, True, False, False]), .1)
    assert {key: result[key] for key in ("TP", "FN", "FP", "TN")} == dict(TP=1, FN=1, FP=1, TN=1)
    assert result["FN_rate_among_unsafe"] == .5
    assert result["false_accept_fraction_among_retained"] == .5
    assert result["unnecessary_referrals"] == 1
    assert result["actual_coverage"] == .5


def test_hard_referrals_full_row_scores_and_empty_retained(train):
    hard = np.ones(len(train), bool)
    tables = ra.analyze(train, {"all_hard": record(train, hard), "one_hard": record(train, np.arange(12) == 0)})
    summary = tables["full_row_summary"].set_index("configuration")
    assert summary.loc["all_hard", "n_rows"] == 12
    assert summary.loc["all_hard", "Cd_rmse"] > 0
    empty = tables["risk_coverage"].query("configuration == 'all_hard'")
    assert (empty.actual_coverage == 0).all() and empty.Cd_rmse.isna().all()
    assert (empty.fully_referred_geometries == 6).all()
    matrix = tables["illustrative_sensitivity"].query("configuration == 'all_hard'").iloc[0]
    assert matrix.FN == matrix.TN == 0 and matrix.TP + matrix.FP == 12
    assert pd.isna(matrix.false_accept_fraction_among_retained)
    one = tables["risk_coverage"].query("configuration == 'one_hard' and requested_coverage == 1").iloc[0]
    assert one.actual_coverage == 11 / 12 and one.partially_retained_geometries == 1
    assert one.fully_retained_geometries == 5


def test_no_unsafe_or_referrals_has_explicit_undefined_rates():
    matrix = ra.sensitivity([0., .01], np.ones(2, bool), .1)
    assert matrix["FN_rate_among_unsafe"] is None
    assert matrix["unnecessary_fraction_among_referrals"] is None
    assert matrix["false_accept_fraction_among_retained"] == 0


def coverage_table():
    return pd.DataFrame({"fold_id": [0, 0, 1, 1, 2, 2], "geometry_id": list("abcdef"),
                         ra.DISTANCE: np.arange(6) / 10, ra.OUTSIDE: np.arange(6)})


def test_geometry_coverage_aligns_without_sorting_rows(train):
    table = coverage_table().iloc[::-1]
    result = ra.geometry_support(table, train, np.repeat([0, 1, 2], 4))
    np.testing.assert_array_equal(result.geometry_id, train.geometry_id)
    np.testing.assert_array_equal(result[ra.OUTSIDE], np.repeat(np.arange(6), 2))


@pytest.mark.parametrize("kind", ["duplicate", "missing", "extra", "wrong_fold", "nonfinite", "negative", "fractional_count"])
def test_bad_geometry_coverage_rejected(train, kind):
    table = coverage_table()
    if kind == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    elif kind == "missing":
        table = table.iloc[:-1]
    elif kind == "extra":
        table.loc[0, "geometry_id"] = "unknown"
    elif kind == "wrong_fold":
        table.loc[0, "fold_id"] = 2
    elif kind in ("nonfinite", "negative"):
        table.loc[0, ra.DISTANCE] = np.nan if kind == "nonfinite" else -1
    else:
        table[ra.OUTSIDE] = .5
    with pytest.raises(ValueError):
        ra.geometry_support(table, train, np.repeat([0, 1, 2], 4))


def test_end_to_end_deduplicates_control_and_never_scores_holdouts(inputs):
    tables, report = ra.run_analysis(**inputs)
    assert report["n_rows"] == 12 and report["n_geometries"] == 6
    assert report["control_repeat"]["exact_numeric_equivalence"]
    assert report["control_repeat"]["independent_replications"] is False
    assert not report["holdouts_scored"] and report["production_policy_recommendation"] is None
    assert list(tables["full_row_summary"].configuration) == ["control", "broad", "selective"]
    assert len(tables["illustrative_sensitivity"]) == 3 * 2 * 5 * 3
    full = tables["risk_coverage"].query("requested_coverage == 1")
    assert (full.actual_coverage == 1).all() and (full.fully_retained_geometries == 6).all()
    assert set(path.suffix for path in inputs["output_dir"].iterdir()) == {".csv", ".json"}
    json.loads((inputs["output_dir"] / "analysis.json").read_text())
    with pytest.raises(ValueError, match="fresh"):
        ra.run_analysis(**inputs)


def test_existing_empty_output_rejected_before_loading(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        ra.run_analysis(None, None, None, None, output)


@pytest.mark.parametrize("key", ["fitting_predictions", "physical_predictions", "physical_ensemble_predictions",
                                "raw_log_Cd_std", "clipped_log_Cd_std"])
def test_loader_nonfinite_arrays_rejected(inputs, key):
    path = inputs["broad_run"] / f"0_{ra.ARMS[0]}.npz"
    rewrite_array(path, key, lambda values: np.full_like(values, np.nan))
    with pytest.raises(ValueError, match="finite"):
        ra.run_analysis(**inputs)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("key,change", [
    ("canonical_train_positions", lambda v: v[::-1]),
    ("member_seeds", lambda v: v[::-1]),
    ("member_seeds", lambda v: v.astype(float)),
    ("lower_clipped", lambda v: ~v),
    ("raw_log_Cd_std", lambda v: v + 1),
    ("physical_ensemble_predictions", lambda v: v + 1),
])
def test_loader_bad_evidence_rejected(inputs, key, change):
    rewrite_array(inputs["broad_run"] / f"0_{ra.ARMS[0]}.npz", key, change)
    with pytest.raises(ValueError):
        ra.run_analysis(**inputs)


@pytest.mark.parametrize("kind", ["mode", "hash", "missing_fold", "duplicate_fold", "trial_config", "repeat_config", "counts", "auxiliary"])
def test_loader_metadata_or_repeat_mismatch_rejected(inputs, kind):
    path = inputs["selective_run"] / "fold_study.json"
    manifest = json.loads(path.read_text())
    if kind == "mode":
        manifest["run_metadata"]["study_mode"] = "input_scaling"
    elif kind == "hash":
        manifest["run_metadata"]["dataset"]["polars_sha256"] = "wrong"
    elif kind == "missing_fold":
        del manifest["prediction_artifacts"]["2"]
    elif kind == "duplicate_fold":
        manifest["plan"]["fold_plan"]["folds"]["2"] = copy.deepcopy(manifest["plan"]["fold_plan"]["folds"]["0"])
    elif kind == "counts":
        manifest["completed_member_evaluations"] = 17
    elif kind == "auxiliary":
        trial = f"fold_0_seed_42_{ra.ARMS[1]}"
        for config in (manifest["trials"][trial]["config"], manifest["plan"]["trial_configs"][trial]):
            config["parameters"]["auxiliary_weight"] = 0
    else:
        trial = f"fold_0_seed_42_{ra.ARMS[0]}"
        manifest["trials"][trial]["config"]["parameters"]["learning_rate"] = .2
        if kind == "repeat_config":
            manifest["plan"]["trial_configs"][trial] = copy.deepcopy(manifest["trials"][trial]["config"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ra.run_analysis(**inputs)


def test_repeat_numerical_mismatch_not_counted_as_new_replication(inputs):
    path = inputs["selective_run"] / f"0_{ra.ARMS[0]}.npz"
    # Within arithmetic replay tolerance, but not exactly the same saved evidence.
    rewrite_array(path, "raw_log_Cd_std", lambda v: np.nextafter(v, np.inf))
    with pytest.raises(ValueError, match="Repeated control"):
        ra.run_analysis(**inputs)


def test_geometry_optional_four_methods_when_supplied(inputs):
    path = inputs["output_dir"].parent / "coverage.csv"
    coverage_table().to_csv(path, index=False)
    tables, report = ra.run_analysis(**inputs, geometry_coverage=path)
    assert len(tables["risk_coverage"]) == 3 * 4 * 5
    assert report["geometry_coverage"]["sha256"] == ra._digest(path)


@pytest.mark.parametrize("raw_logs", [[-20., -16., 8.], [8., 9., 10.], [-20., -16., -15.]])
def test_loaded_raw_member_clipping_is_hard_even_with_plausible_ensemble(inputs, raw_logs):
    path = inputs["broad_run"] / f"0_{ra.ARMS[0]}.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    raw = arrays["fitting_predictions"]
    raw[:, 0, 1] = raw_logs
    low, high = np.log([1e-6, 1e3])
    clipped = np.clip(raw[:, :, 1], low, high)
    arrays["physical_predictions"] = raw.copy()
    arrays["physical_predictions"][:, :, 1] = np.exp(clipped)
    ensemble = arrays["physical_predictions"].mean(axis=0)
    ensemble[:, 1] = np.exp(np.log(arrays["physical_predictions"][:, :, 1]).mean(axis=0))
    arrays.update(physical_ensemble_predictions=ensemble, lower_clipped=raw[:, :, 1] < low,
                  upper_clipped=raw[:, :, 1] > high, raw_log_Cd_std=raw[:, :, 1].std(axis=0, ddof=1),
                  clipped_log_Cd_std=clipped.std(axis=0, ddof=1))
    np.savez(path, **arrays)
    train = ra.load_train(inputs["polars"], inputs["split"])
    configs, _, _ = ra.load_run(inputs["broad_run"], train, ra._digest(inputs["polars"]), "input_scaling")
    record = configs["standard"]
    position = arrays["canonical_train_positions"][0]
    assert record["hard"].sum() == 1 and record["hard"][position]
    retained, _ = ra.retained_mask(record["clipped"], 1., record["hard"])
    assert retained.sum() == len(train) - 1 and not retained[position]
    if raw_logs[0] > high or raw_logs[-1] < low:
        assert record["clipped"][position] == 0 and record["raw"][position] > 0


def test_candidate_non_scaling_config_change_rejected(inputs):
    path = inputs["broad_run"] / "fold_study.json"
    manifest = json.loads(path.read_text())
    trial = f"fold_0_seed_42_{ra.ARMS[1]}"
    for config in (manifest["trials"][trial]["config"], manifest["plan"]["trial_configs"][trial]):
        config["parameters"]["learning_rate"] = .2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="beyond input scaling"):
        ra.run_analysis(**inputs)


@pytest.mark.parametrize("coverage", [0., -1., 1.1, np.nan, np.inf])
def test_invalid_requested_coverage_rejected(coverage):
    with pytest.raises(ValueError):
        ra.retained_mask([0., 1.], coverage, np.zeros(2, bool))


def test_zero_actual_cd_rejected_during_train_reconstruction(inputs):
    frame = pd.read_csv(inputs["polars"])
    frame.loc[0, "Cd"] = 0
    frame.to_csv(inputs["polars"], index=False)
    with pytest.raises(ValueError, match="positive"):
        ra.load_train(inputs["polars"], inputs["split"])


def test_digest_does_not_require_python_311_file_digest(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"reliability")
    monkeypatch.delattr(ra.hashlib, "file_digest", raising=False)
    assert ra._digest(path) == ra.hashlib.sha256(b"reliability").hexdigest()
