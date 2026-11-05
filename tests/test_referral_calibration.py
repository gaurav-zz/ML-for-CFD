"""Cloud-first, fit-free tests of frozen calibration and audit separation."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.training import referral_calibration as rc
from src.training import reliability_analysis as ra


def _save_arrays(path, raw, positions, mode):
    low, high = np.log([1e-6, 1e3])
    clipped = np.clip(raw[:, :, 1], low, high)
    physical = raw.copy()
    physical[:, :, 1] = np.exp(clipped)
    ensemble = physical.mean(axis=0)
    ensemble[:, 1] = np.exp(np.log(physical[:, :, 1]).mean(axis=0))
    np.savez(path, canonical_train_positions=positions, member_seeds=np.array([42, 43, 44]),
        fitting_predictions=raw, physical_predictions=physical, physical_ensemble_predictions=ensemble,
        lower_clipped=raw[:, :, 1] < low, upper_clipped=raw[:, :, 1] > high,
        raw_log_Cd_std=raw[:, :, 1].std(axis=0, ddof=1), clipped_log_Cd_std=clipped.std(axis=0, ddof=1),
        input_scaling=np.asarray(mode), study_variant=np.asarray(rc.SOURCE_VARIANT))


@pytest.fixture
def study(tmp_path):
    """All 18 saved-array artifacts, but only two synthetic rows per original-size geometry fold."""
    ids = [f"g{i:03}" for i in range(572)]
    train = pd.DataFrame({"geometry_id": np.repeat(ids, 2), "Alpha": [-1., 1.] * 572,
        "Re": 1e6, "Ncrit": 9., "Cl": np.linspace(-1, 1, 1144), "Cd": .02, "Cm": .01})
    polars, split, source, coverage = (tmp_path / name for name in ("polars.csv", "split.json", "source", "coverage.csv"))
    holdouts = pd.DataFrame({"geometry_id": ["v", "t"], "Alpha": 0., "Re": 1e6, "Ncrit": 9.,
                             "Cl": np.nan, "Cd": [0., -1.], "Cm": np.nan})
    pd.concat([train, holdouts]).to_csv(polars, index=False)
    groups = {"train": ids, "validation": ["v"], "test": ["t"]}
    split.write_text(json.dumps({"split_geometries": groups, "v1_split_rows": {"train": len(train)},
                                "geometry_counts": {"train": len(ids)}}))
    train = ra.load_train(polars, split)
    source.mkdir()
    modes = dict(zip(ra.ARMS, ("standard", rc.CONFIGURATION)))
    manifest = {"schema_version": 1, "status": "completed", "phase": "completed", "source": ra.SOURCE,
        "completed_fits": 18, "completed_member_evaluations": 18, "completed_ensemble_evaluations": 6,
        "run_metadata": {"study_mode": "selective_scaling", "dataset": {"polars_sha256": ra._digest(polars)},
                         "source_sha256": "synthetic_fixture", "created_at_utc": rc.SOURCE_CREATED_AT},
        "study_variant": rc.SOURCE_VARIANT, "study_kind": "paired_geometry_input_scaling", "target_names": ra.TARGETS,
        "canonical_validation_test_used_for_model_apis": False,
        "canonical_split": {"train": {"rows": len(train), "geometry_ids": ids, "row_sha256": ra._row_hash(train)}},
        "plan": {"seeds": [42, 43, 44], "arms": list(ra.ARMS), "n_folds": 3,
                 "aggregation": "mean_Cl_Cm_geometric_mean_Cd", "input_scaling_modes": modes,
                 "fold_plan": {"folds": {}}, "trial_configs": {}}, "trials": {}, "prediction_artifacts": {}}
    folds = np.repeat(np.repeat([0, 1, 2], [191, 191, 190]), 2)
    for fold in range(3):
        evaluation = sorted(train.loc[folds == fold, "geometry_id"].unique())
        remaining = sorted(set(ids) - set(evaluation))
        partitions = {}
        for name, members in (("evaluation", evaluation), ("stopping", remaining[:38]), ("fit", remaining[38:])):
            frame = train.loc[train.geometry_id.isin(members)]
            partitions[name] = {"geometry_ids": members, "n_rows": len(frame), "row_sha256": ra._row_hash(frame)}
        manifest["plan"]["fold_plan"]["folds"][str(fold)] = {"partitions": partitions}
        manifest["prediction_artifacts"][str(fold)] = {}
        positions = train.loc[folds == fold].sort_values(ra.KEYS).index.to_numpy()
        raw = np.broadcast_to(train.iloc[positions][ra.TARGETS].to_numpy(), (3, len(positions), 3)).copy()
        raw[:, :, 1] = np.log(raw[:, :, 1]) + np.array([-.01, 0., .01])[:, None]
        for arm, mode in modes.items():
            filename = f"{fold}_{arm}.npz"
            _save_arrays(source / filename, raw, positions, mode)
            manifest["prediction_artifacts"][str(fold)][arm] = {"path": filename, "status": "complete",
                "ensemble_present": True, "n_members": 3, "n_rows": len(positions)}
            for seed in (42, 43, 44):
                trial = f"fold_{fold}_seed_{seed}_{arm}"
                config = {"parameters": {"input_scaling": mode, "random_state": seed, "auxiliary_weight": .1}}
                manifest["plan"]["trial_configs"][trial] = config
                manifest["trials"][trial] = {"status": "success", "evaluation_status": "success", "config": config}
    (source / "fold_study.json").write_text(json.dumps(manifest))
    pd.DataFrame({"geometry_id": ids, "fold_id": folds[::2], ra.DISTANCE: np.arange(572, dtype=float),
                  ra.OUTSIDE: 0}).to_csv(coverage, index=False)
    return dict(selective_run=source, polars=polars, split=split, geometry_coverage=coverage, output_dir=tmp_path / "output")


def _partitions(study):
    train = ra.load_train(study["polars"], study["split"])
    manifest = json.loads((study["selective_run"] / "fold_study.json").read_text())
    folds = np.repeat(np.repeat([0, 1, 2], [191, 191, 190]), 2)
    groups = json.loads(study["split"].read_text())["split_geometries"]
    return train, folds, manifest["plan"]["fold_plan"]["folds"], groups


def test_partition_is_deterministic_whole_geometry_disjoint_and_complete(study):
    train, folds, plan, groups = _partitions(study)
    rows, identities = rc.partition_geometries(train, folds, plan, groups)
    reversed_plan = copy.deepcopy(plan)
    for spec in reversed_plan.values():
        spec["partitions"]["evaluation"]["geometry_ids"].reverse()
    again, _ = rc.partition_geometries(train, folds, reversed_plan, groups)
    pd.testing.assert_frame_equal(rows, again)
    assert rows.groupby("geometry_id").partition.nunique().eq(1).all()
    assert list(rows.canonical_train_position) == list(range(len(train)))
    for fold in range(3):
        identity = identities[str(fold)]
        shuffled = np.random.default_rng(20261105 + fold).permutation(sorted(plan[str(fold)]["partitions"]["evaluation"]["geometry_ids"]))
        assert identity["calibration"]["geometry_ids"] == shuffled[:95].tolist()
        assert identity["audit"]["geometry_ids"] == shuffled[95:].tolist()
        calibration, audit = set(shuffled[:95]), set(shuffled[95:])
        forbidden = set(groups["validation"] + groups["test"])
        forbidden.update(plan[str(fold)]["partitions"]["fit"]["geometry_ids"])
        forbidden.update(plan[str(fold)]["partitions"]["stopping"]["geometry_ids"])
        assert not calibration & audit and not (calibration | audit) & forbidden


@pytest.mark.parametrize("invalid", ["fit_overlap", "stopping_overlap", "holdout_overlap", "row_fold", "duplicate_evaluation"])
def test_partition_rejects_invalid_membership(study, invalid):
    train, folds, plan, groups = _partitions(study)
    first = plan["0"]["partitions"]["evaluation"]["geometry_ids"][0]
    if invalid in ("fit_overlap", "stopping_overlap"):
        plan["0"]["partitions"][invalid.split("_")[0]]["geometry_ids"][0] = first
    elif invalid == "holdout_overlap":
        groups["test"].append(first)
    elif invalid == "duplicate_evaluation":
        plan["0"]["partitions"]["evaluation"]["geometry_ids"].append(first)
    else:
        folds[0] = 1
    with pytest.raises(ValueError):
        rc.partition_geometries(train, folds, plan, groups)


def test_fixed_lower_quantiles_exclude_hard_rows_and_retain_ties():
    np.testing.assert_array_equal(rc.QUANTILES, np.linspace(0., 1., 21))
    candidates = rc.candidate_thresholds([1., 2., 5., 100.], np.array([False, False, False, True]))
    assert len(candidates) == 21 and [item["quantile"] for item in candidates] == list(rc.QUANTILES)
    assert candidates[1]["threshold"] == 1 and candidates[10]["threshold"] == 2 and candidates[-1]["threshold"] == 5
    policy, grid = rc.select_threshold([1., 1., 1., 3.], np.zeros(4, bool), [0., 0., 0., 1.], list("abcd"), .1, 0., min_support=3)
    assert policy["threshold"] == 1 and policy["calibration_accepted_rows"] == 3 and policy["quantile"] == 0
    assert sum(item["selected"] for item in grid) == 1


def test_nonmonotone_risk_all_candidates_and_maximum_row_coverage():
    args = (np.arange(6.), np.zeros(6, bool), [0., .2, 0., 0., 0., .2], list("abcdef"), .1, .25)
    policy, grid = rc.select_threshold(*args, min_support=2)
    assert policy["threshold"] == 4 and policy["calibration_accepted_rows"] == 5
    assert policy["calibration_conditional_risk"] == .2
    assert any(not item["eligible"] and item["threshold"] < 4 for item in grid)
    unsupported, _ = rc.select_threshold(*args, min_support=6)
    assert unsupported["threshold"] is None and unsupported["status"] == "no_eligible_threshold"
    duplicated_ids, _ = rc.select_threshold(np.arange(30.), np.zeros(30, bool), np.zeros(30), ["a"] * 30, .1, .1)
    assert duplicated_ids["threshold"] is None


def test_cdfs_are_row_weighted_and_audit_is_not_reranked():
    cdfs = rc.fit_cdfs([1., 1., 2., 4., 100.], [1., 1., 2., 4., 100.], np.array([False] * 4 + [True]))
    np.testing.assert_array_equal(cdfs["raw"], [1., 1., 2., 4.])
    before, _ = rc.transform_scores([1., 3.], [1., 3.], np.zeros(2, bool), cdfs)
    after, _ = rc.transform_scores([1., 3., 0., 999.], [1., 3., 0., 999.], np.zeros(4, bool), cdfs)
    np.testing.assert_array_equal(before[rc.METHODS[2]], [.5, .75])
    np.testing.assert_array_equal(before[rc.METHODS[2]], after[rc.METHODS[2]][:2])


def test_constant_cdf_saturation_cannot_hide_either_out_of_support_component():
    cdfs = rc.fit_cdfs([1., 1., 100.], [2., 2., 100.], np.array([False, False, True]))
    raw, geo, hard = [0., 1., 1.001, 1., 1.], [1., 2., 2., 2.001, 2.], np.array([False] * 4 + [True])
    scores, outside = rc.transform_scores(raw, geo, hard, cdfs)
    np.testing.assert_array_equal(scores[rc.METHODS[2]], [0., 1., 1., 1., 1.])
    np.testing.assert_array_equal(outside, [True, False, True, True, False])
    accepted = rc.apply_policy({"method": rc.METHODS[2], "threshold": 1.}, raw, geo, hard, cdfs)
    np.testing.assert_array_equal(accepted, [False, True, False, False, False])
    raw_only = rc.apply_policy({"method": rc.METHODS[0], "threshold": 1.}, raw, geo, hard, cdfs)
    assert not raw_only[2] and raw_only[3] and not raw_only[4]
    geo_only = rc.apply_policy({"method": rc.METHODS[1], "threshold": 2.}, raw, geo, hard, cdfs)
    assert geo_only[2] and not geo_only[3] and not geo_only[4]


@pytest.mark.parametrize("n", [0, 3])
def test_empty_eligible_no_policy_all_referred_and_undefined_metrics(n):
    actual = np.ones((n, 3))
    cdfs, policies, candidates = rc.calibrate_fold(np.zeros(n), np.zeros(n), np.ones(n, bool), ["a"] * n, actual, actual, min_support=1)
    assert len(policies) == 18 and not candidates and len(cdfs["raw"]) == 0
    for policy in policies:
        assert policy["threshold"] is None and policy["calibration_conditional_risk"] is None
        assert not rc.apply_policy(policy, np.zeros(n), np.zeros(n), np.ones(n, bool), cdfs).any()
    result = rc.diagnostics(actual, actual, ["a"] * n, np.zeros(n, bool), .1)
    assert result["conditional_risk"] is None and result["missed_referral_rate"] is None
    assert result["geometry_macro_retained_risk"] is None and result["Cd_rmse"] is None
    assert result["actual_coverage"] == (0 if n else None)
    assert result["FPR"] == (1 if n else None)


def test_confusion_mapping_and_distinct_denominators_and_geometry_risk():
    actual = np.ones((5, 3))
    prediction = actual.copy()
    prediction[:, 1] += [0., .25, .5, 0., .75]
    result = rc.diagnostics(actual, prediction, list("aabcd"), np.array([True, True, False, False, False]), .125)
    assert {key: result[key] for key in ("TP", "FN", "FP", "TN")} == dict(TP=2, FN=1, FP=1, TN=1)
    assert result["conditional_risk"] == .5 and result["missed_referral_rate"] == pytest.approx(1 / 3)
    assert result["FPR"] == .5 and result["unnecessary_fraction_among_referrals"] == pytest.approx(1 / 3)
    assert result["fully_retained_geometries"] == 1 and result["fully_referred_geometries"] == 3
    assert result["geometry_macro_retained_risk"] == .5 and result["Cd_median_absolute_error"] == .125
    boundary = rc.diagnostics([[1., 8., 1.]], [[1., 9., 1.]], ["a"], np.ones(1, bool), .125)
    assert boundary["TN"] == 1 and boundary["FN"] == 0
    partial = rc.diagnostics(actual, prediction, list("aabcd"), [True, False, True, False, False], .125)
    assert partial["fully_retained_geometries"] == partial["partially_retained_geometries"] == 1
    assert partial["fully_referred_geometries"] == 2 and partial["geometry_max_retained_risk"] == 1


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf])
def test_positive_finite_truth_required_even_if_referred(bad):
    with pytest.raises(ValueError):
        rc.calibrate_fold([0.], [0.], np.ones(1, bool), ["a"], [[0., bad, 0.]], [[0., 1., 0.]], min_support=1)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., 1j])
def test_invalid_score_inputs_cannot_be_accepted(bad):
    with pytest.raises(ValueError):
        rc.fit_cdfs([bad], [0.], np.ones(1, bool))


def test_study_freezes_partitions_and_all_policies_before_audit_and_reports_all54(study, monkeypatch):
    original, calls = ra.error_metrics, []
    def tracked(actual, predicted, retained):
        calls.append(len(actual))
        assert (study["output_dir"] / "partitions.csv").is_file()
        assert (study["output_dir"] / "frozen_plan.json").is_file()
        assert (study["output_dir"] / "selected_policies.json").exists() == (len(calls) > 3)
        return original(actual, predicted, retained)
    monkeypatch.setattr(ra, "error_metrics", tracked)
    tables, report = rc.run_calibration(**study)
    assert calls[:3] == [190, 190, 190]
    assert report["n_policies"] == len(tables["audit_summary"]) == 54
    assert len(tables["audit_baseline"]) == 9 and len(tables["full_row_summary"]) == 12
    assert len(tables["audit_rows"]) == 574 and len(tables["audit_rows"].filter(like="accept_").columns) == 18
    assert not report["qualified_confidence"] and not report["holdouts_scored"] and not report["surrogate_fitted"]
    assert report["production_policy_recommendation"] is None and report["status"] == "complete_development_only"
    assert tables["audit_summary"].groupby("fold_id").size().tolist() == [18, 18, 18]
    assert set(tables["audit_summary"].row_sha256) == set(tables["audit_baseline"].row_sha256)
    candidates = pd.read_csv(study["output_dir"] / "calibration_candidates.csv")
    assert len(candidates) == 54 * 21
    saved = json.loads((study["output_dir"] / "summary.json").read_text())
    for name, digest in saved["artifact_sha256"].items():
        assert ra._digest(study["output_dir"] / name) == digest


@pytest.mark.parametrize("change", ["audit_labels", "audit_inputs", "calibration_labels"])
def test_calibration_label_and_input_isolation(study, change):
    _, before = rc.run_calibration(**study)
    partition = pd.read_csv(study["output_dir"] / "partitions.csv")
    selected_partition = "calibration" if change == "calibration_labels" else "audit"
    selected = partition.loc[partition.partition == selected_partition, "canonical_train_position"].to_numpy()
    if change.endswith("labels"):
        frame = pd.read_csv(study["polars"])
        frame.loc[selected, "Cd"] /= 2
        frame.to_csv(study["polars"], index=False)
        path = study["selective_run"] / "fold_study.json"
        manifest = json.loads(path.read_text())
        manifest["run_metadata"]["dataset"]["polars_sha256"] = ra._digest(study["polars"])
        path.write_text(json.dumps(manifest))
    else:
        for fold in range(3):
            path = study["selective_run"] / f"{fold}_{ra.ARMS[1]}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                raw, positions = arrays["fitting_predictions"], arrays["canonical_train_positions"]
            raw[:, np.isin(positions, selected), 1] += np.array([-.1, 0., .1])[:, None]
            _save_arrays(path, raw, positions, rc.CONFIGURATION)
        coverage = pd.read_csv(study["geometry_coverage"])
        coverage.loc[coverage.geometry_id.isin(partition.loc[partition.partition == "audit", "geometry_id"]), ra.DISTANCE] = 10_000.
        coverage.to_csv(study["geometry_coverage"], index=False)
    new_output = study["output_dir"].with_name("changed")
    _, after = rc.run_calibration(**{**study, "output_dir": new_output})
    with np.load(study["output_dir"] / "score_cdfs.npz", allow_pickle=False) as first, np.load(new_output / "score_cdfs.npz", allow_pickle=False) as second:
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
    first = pd.read_csv(study["output_dir"] / "calibration_candidates.csv")
    second = pd.read_csv(new_output / "calibration_candidates.csv")
    pd.testing.assert_frame_equal(first[["method", "quantile", "threshold"]], second[["method", "quantile", "threshold"]])
    before_policies = json.loads((study["output_dir"] / "selected_policies.json").read_text())["policies"]
    after_policies = json.loads((new_output / "selected_policies.json").read_text())["policies"]
    if change == "calibration_labels":
        assert all(policy["status"] == "no_eligible_threshold" for policy in after_policies)
        assert before_policies != after_policies
        assert all(row["conditional_risk"] is None and row["actual_coverage"] == 0 for row in after["audit_summary"])
    else:
        assert before_policies == after_policies
        pd.testing.assert_frame_equal(first, second)
        if change == "audit_labels":
            assert before["audit_baseline"][0]["Cd_rmse"] != after["audit_baseline"][0]["Cd_rmse"]
        else:
            assert all(row["actual_coverage"] == 0 for row in after["audit_summary"])


@pytest.mark.parametrize("invalid", ["source", "mode", "variant", "timestamp", "seeds", "completion", "positions", "npz_mode", "coverage"])
def test_source_mode_and_alignment_invalids_rejected_before_output(study, invalid):
    path = study["selective_run"] / "fold_study.json"
    manifest = json.loads(path.read_text())
    if invalid == "source":
        manifest["source"] = "canonical_test"
    elif invalid == "mode":
        manifest["run_metadata"]["study_mode"] = "input_scaling"
    elif invalid == "variant":
        manifest["study_variant"] = "different_candidate"
    elif invalid == "timestamp":
        manifest["run_metadata"]["created_at_utc"] = "another_run"
    elif invalid == "seeds":
        manifest["plan"]["seeds"] = [1, 2, 3]
    elif invalid == "completion":
        manifest["completed_fits"] = 17
    elif invalid == "coverage":
        table = pd.read_csv(study["geometry_coverage"])
        table.loc[0, "fold_id"] = 2
        table.to_csv(study["geometry_coverage"], index=False)
    else:
        artifact = study["selective_run"] / f"0_{ra.ARMS[1]}.npz"
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if invalid == "positions":
            arrays["canonical_train_positions"] = arrays["canonical_train_positions"][::-1]
        else:
            arrays["input_scaling"] = np.asarray("standard")
        np.savez(artifact, **arrays)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        rc.run_calibration(**study)
    assert not study["output_dir"].exists()


def test_fresh_output_guard_precedes_loading(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        rc.run_calibration(tmp_path / "absent", None, None, None, output)
