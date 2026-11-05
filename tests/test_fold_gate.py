"""Fit-free geometry-fold policy, metadata, and malformed-evidence tests."""

import builtins
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import METHOD
from src.training.fold_gate import (
    FOLDS, SOURCE, paired_fold_gate, paired_input_scaling_fold_gate, paired_selective_scaling_fold_gate,
)
from src.training.tail_risk import derive_thresholds, tail_report


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Fold-gate tests must never import Torch.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def fold_frame(fold, rows=5):
    return pd.DataFrame([
        {"geometry_id": f"f{fold}_g{g}", "Alpha": float(a), "Re": 1e5 if a == 0 else 1e6,
         "Ncrit": 9., "max_thickness": .1 * (g + 1),
         "Cl": .2 + .03 * a, "Cd": .01 + .001 * a, "Cm": -.02 - .001 * a}
        for g in range(4) for a in range(rows)
    ])


def score(frame, fold, arm, factor=.1, *, thresholds=None, prediction=None, family="baseline_ensemble"):
    if thresholds is None:
        fit = frame.copy()
        fit.geometry_id = "fit_" + fit.geometry_id
        thresholds = derive_thresholds(fit)
    if prediction is None:
        prediction = frame[["Cl", "Cd", "Cm"]].to_numpy() * (1 + factor)
    report = tail_report(frame, prediction, thresholds=thresholds)
    report.update(fold_id=fold, source=SOURCE, predictor="baseline_seed_42" if arm == "reference" else family,
                  member_seeds=[42] if arm == "reference" else [42, 43, 44])
    if arm == "candidate":
        report["aggregation"] = METHOD
    return report


def paired(factors=(.05, .05, .05), rows=(5, 5, 5), family="baseline_ensemble"):
    reference, candidate = {}, {}
    for fold in FOLDS:
        frame = fold_frame(fold, rows[fold])
        reference[fold] = score(frame, fold, "reference")
        candidate[fold] = score(frame, fold, "candidate", factors[fold], family=family)
    return reference, candidate


def expected_ids(reference):
    return sorted(name for report in reference.values() for name in report["geometry_ids"])


def assert_inconclusive(reference, candidate, expected=None):
    gate = paired_fold_gate(reference, candidate, expected)
    assert gate["status"] == "inconclusive" and not gate["eligible"] and not gate["passed"]
    assert gate["errors"] and gate["reasons"]
    assert gate["checks"] == gate["deltas"] == []
    json.dumps(gate, allow_nan=False)
    return gate


@pytest.mark.parametrize("family", ["baseline_ensemble", "cdp_ensemble"])
def test_all_folds_paired_physical_equal_weight_and_no_seed_selection(family):
    reference, candidate = paired(family=family)
    before = deepcopy((reference, candidate))
    gate = paired_fold_gate(reference, candidate, expected_ids(reference))
    assert gate["status"] == "pass" and gate["eligible"] and gate["passed"]
    assert gate["source"] == SOURCE and gate["unit"] == "geometry_fold"
    assert gate["gated_units"] == ["geometry_fold"]
    assert gate["folds"] == [0, 1, 2] and gate["n_folds"] == 3
    assert gate["seed_selection"] is gate["fold_selection"] is False
    assert "seeds" not in gate and "selected_family" not in gate
    assert gate["reference_predictor"] == "baseline_seed_42"
    assert gate["candidate_predictor"] == family and gate["aggregation"] == METHOD
    assert gate["expected_geometry_ids_validated"] is True
    assert len(gate["checks"]) == 17 and len(gate["deltas"]) == 84
    assert gate["policy"]["primary_rmse_guard_targets"] == ["Cl", "Cm"]
    assert all(check["passed"] for check in gate["checks"])
    assert [row["fold_id"] for row in gate["fold_stats"]] == [0, 1, 2]
    assert all(row["weight"] == 1 / 3 for row in gate["fold_stats"])
    assert gate["mean_stats"]["Cd.median_relative_error"]["delta"] == pytest.approx(-.05)
    assert gate["mean_stats"]["Cd.median_relative_error"]["delta_percentage_points"] == pytest.approx(-5.)
    assert all(row["reference"]["definitions"]["relative_error_units"] == "fraction" for row in gate["fold_stats"])
    assert any("not three independent seeds" in warning for warning in gate["warnings"])
    assert "not physics guarantees" in gate["policy"]["description"]
    json.dumps(gate, allow_nan=False)
    assert (reference, candidate) == before


def test_string_fold_keys_and_truthful_reference_member_seed_are_allowed():
    reference, candidate = paired()
    for report in reference.values():
        report["seed"] = 42
    gate = paired_fold_gate({str(k): v for k, v in reference.items()}, {str(k): v for k, v in candidate.items()})
    assert gate["passed"]
    assert gate["expected_geometry_ids_validated"] is False
    assert any("not independently declared" in warning for warning in gate["warnings"])


def test_different_fold_row_counts_are_accepted_and_means_not_row_weighted():
    reference, candidate = paired((.01, .03, .09), rows=(3, 5, 12))
    gate = paired_fold_gate(reference, candidate, expected_ids(reference))
    assert gate["passed"]
    assert [row["identity"]["n_rows"] for row in gate["fold_stats"]] == [12, 20, 48]
    mean = gate["mean_stats"]["Cd.median_relative_error"]["candidate"]
    assert mean == pytest.approx(np.mean([.01, .03, .09]))
    assert mean != pytest.approx(np.average([.01, .03, .09], weights=[12, 20, 48]))
    assert len({json.dumps(report["thresholds"], sort_keys=True) for report in reference.values()}) == 3


def test_single_outlier_median_improvement_cannot_hide_p95_failure():
    reference, candidate = paired((.01, .01, .01))
    frame = fold_frame(2)
    prediction = frame[["Cl", "Cd", "Cm"]].to_numpy() * 1.01
    prediction[-1, 1] = .9
    candidate[2] = score(frame, 2, "candidate", prediction=prediction)
    gate = paired_fold_gate(reference, candidate, expected_ids(reference))
    assert gate["status"] == "fail" and not gate["eligible"]
    assert next(check for check in gate["checks"] if check["rule"] == "strict_improvement")["passed"]
    assert not next(check for check in gate["checks"] if check["fold_id"] == "mean" and check["metric"] == "Cd.p95_geometry_rmse")["passed"]
    assert any("max_geometry_rmse" in reason for reason in gate["reasons"])


def test_one_fold_over_ten_percent_is_not_hidden_by_good_equal_fold_mean():
    reference, candidate = paired((.01, .01, .113))
    gate = paired_fold_gate(reference, candidate)
    assert gate["status"] == "fail" and not gate["eligible"]
    assert all(check["passed"] for check in gate["checks"] if check["rule"] != "per_fold_tail_deterioration")
    failures = [check for check in gate["checks"] if not check["passed"]]
    assert {check["fold_id"] for check in failures} == {2}
    assert {check["metric"] for check in failures} == {"Cd.pooled_rmse", "Cd.max_geometry_rmse"}
    assert all(check["rule"] == "per_fold_tail_deterioration" for check in failures)


@pytest.mark.parametrize("target,column", [("Cl", 0), ("Cm", 2)])
def test_improved_medians_cannot_hide_primary_rmse_regression(target, column):
    reference, candidate = paired()
    for fold in FOLDS:
        frame = fold_frame(fold)
        prediction = frame[["Cl", "Cd", "Cm"]].to_numpy() * 1.05
        prediction[-1, column] += 1.
        candidate[fold] = score(frame, fold, "candidate", prediction=prediction)
    gate = paired_fold_gate(reference, candidate, expected_ids(reference))
    assert not gate["eligible"] and gate["status"] == "fail"
    failures = [check for check in gate["checks"] if not check["passed"]]
    assert len(failures) == 1
    assert failures[0]["metric"] == f"{target}.rmse"
    assert failures[0]["rule"] == "mean_primary_rmse_deterioration"
    assert all(check["passed"] for check in gate["checks"] if check["rule"] == "median_fraction_degradation")


@pytest.mark.parametrize("target", ["Cl", "Cm"])
@pytest.mark.parametrize("factor,passes", [(1.05, True), (1.05000001, False)])
def test_primary_rmse_uses_predeclared_five_percent_allowance(target, factor, passes):
    reference, candidate = paired()
    for fold in FOLDS:
        candidate[fold]["targets"][target]["rmse"] = reference[fold]["targets"][target]["rmse"] * factor
    assert paired_fold_gate(reference, candidate)["passed"] is passes


@pytest.mark.parametrize("factor,passes", [(.11, True), (.11000001, False)])
def test_per_fold_ten_percent_boundary(factor, passes):
    assert paired_fold_gate(*paired((.01, .01, factor)))["passed"] is passes


@pytest.mark.parametrize("factor,passes", [(1.05, True), (1.05000001, False)])
def test_mean_tail_five_percent_boundary(factor, passes):
    reference, candidate = paired((.1 * factor,) * 3)
    for report in candidate.values():
        for target in ("Cl", "Cd", "Cm"):
            report["targets"][target]["median_relative_error"] = .09
    assert paired_fold_gate(reference, candidate)["passed"] is passes


@pytest.mark.parametrize("delta,passes", [(.005, True), (.00500000001, False)])
def test_primary_median_half_percentage_point_limit(delta, passes):
    reference, candidate = paired()
    for fold in FOLDS:
        candidate[fold]["targets"]["Cl"]["median_relative_error"] = reference[fold]["targets"]["Cl"]["median_relative_error"] + delta
    assert paired_fold_gate(reference, candidate)["passed"] is passes


@pytest.mark.parametrize("gain,passes", [(0., False), (5e-13, False), (2e-12, True)])
def test_cd_strict_improvement_is_predeclared_epsilon(gain, passes):
    reference, candidate = paired()
    for fold in FOLDS:
        candidate[fold]["targets"]["Cd"]["median_relative_error"] = reference[fold]["targets"]["Cd"]["median_relative_error"] - gain
    assert paired_fold_gate(reference, candidate)["passed"] is passes


def test_zero_reference_is_finite_absolute_epsilon_and_no_no_gain_pass():
    reference, candidate = paired((.00001,) * 3)
    for fold in FOLDS:
        reference[fold] = score(fold_frame(fold), fold, "reference", 0.)
    gate = paired_fold_gate(reference, candidate)
    assert gate["status"] == "fail"
    assert any(row["relative_delta"] is None for row in gate["deltas"])
    assert any("Zero reference" in warning for warning in gate["warnings"])
    for fold in FOLDS:
        candidate[fold] = score(fold_frame(fold), fold, "candidate", 0.)
    assert paired_fold_gate(reference, candidate)["status"] == "fail"
    json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("bad", ["missing_fold", "extra_fold", "seed_keys", "float_keys", "bool_keys", "duplicate_fold_keys", "missing_report", "fold_metadata", "fold_type", "reference_best_seed", "reference_members", "candidate_seed_label", "reference_fold_seed_label", "candidate_members", "mixed_family", "aggregation", "source", "units", "primary_order", "threshold_source", "threshold_metadata", "threshold_value", "schema_type", "unit"])
def test_malformed_metadata_cannot_be_fold_or_seed_selected(bad):
    reference, candidate = paired()
    item = candidate[2]
    if bad == "missing_fold":
        del candidate[2]
    elif bad == "extra_fold":
        candidate[3] = deepcopy(item)
    elif bad == "seed_keys":
        candidate = {42 + key: value for key, value in candidate.items()}
    elif bad == "float_keys":
        candidate = {float(key): value for key, value in candidate.items()}
    elif bad == "bool_keys":
        candidate = {False: candidate[0], 1: candidate[1], 2: candidate[2]}
    elif bad == "duplicate_fold_keys":
        candidate = {0: candidate[0], "0": candidate[0], 2: candidate[2]}
    elif bad == "missing_report":
        candidate[2] = None
    elif bad == "fold_metadata":
        item["fold_id"] = 0
    elif bad == "fold_type":
        item["fold_id"] = "2"
    elif bad == "reference_best_seed":
        reference[2]["predictor"] = "baseline_seed_44"
    elif bad == "reference_members":
        reference[2]["member_seeds"] = [44]
    elif bad == "candidate_seed_label":
        item["seed"] = 42
    elif bad == "reference_fold_seed_label":
        reference[2]["seed"] = 2
    elif bad == "candidate_members":
        item["member_seeds"] = [0, 1, 2]
    elif bad == "mixed_family":
        item["predictor"] = "cdp_ensemble"
    elif bad == "aggregation":
        item["aggregation"] = "arithmetic_mean_Cd"
    elif bad == "source":
        item["source"] = "canonical_test"
    elif bad == "units":
        item["definitions"]["relative_error_units"] = "percent"
    elif bad == "primary_order":
        item["definitions"]["units"] = "normalized_log_Cd"
    elif bad == "threshold_source":
        item["thresholds"]["source"] = "canonical_test"
    elif bad == "threshold_metadata":
        item["thresholds"]["training_geometries"] = None
    elif bad == "threshold_value":
        item["thresholds"]["low_re_max"] = 0.
    elif bad == "schema_type":
        item["schema_version"] = True
    else:
        item["unit"] = "seed"
    assert_inconclusive(reference, candidate)


@pytest.mark.parametrize("bad", ["nan", "inf", "none", "negative", "bool", "geometry_nan", "geometry_none", "geometry_negative", "count", "geometry_count", "duplicate_geometry", "missing_geometry", "none_geometry", "stress_count", "row_hash", "stress_hash", "hash_format", "pooled", "p95", "max", "macro", "primary_rmse", "primary_missing", "wrong_primary"])
def test_incomplete_nonfinite_or_tampered_counts_and_statistics_are_inconclusive(bad):
    reference, candidate = paired()
    item = candidate[2]
    if bad in ("nan", "inf", "none", "negative", "bool"):
        item["targets"]["Cl"]["mae"] = {"nan": np.nan, "inf": np.inf, "none": None, "negative": -.1, "bool": True}[bad]
    elif bad.startswith("geometry_") and bad not in ("geometry_count",):
        item["per_geometry"][0]["Cd_rmse"] = {"geometry_nan": np.nan, "geometry_none": None, "geometry_negative": -.1}[bad]
    elif bad == "count":
        item["n_rows"] -= 1
    elif bad == "geometry_count":
        item["per_geometry"][0]["n_rows"] += 1
    elif bad == "duplicate_geometry":
        item["geometry_ids"][1] = item["geometry_ids"][0]
    elif bad == "missing_geometry":
        item["per_geometry"].pop()
    elif bad == "none_geometry":
        item["geometry_ids"][0] = None
    elif bad == "stress_count":
        item["stress"]["low_re"]["n_rows"] -= 1
    elif bad == "row_hash":
        item["row_sha256"] = "f" * 64
    elif bad == "stress_hash":
        item["stress"]["thick"]["row_sha256"] = "f" * 64
    elif bad == "hash_format":
        item["thresholds"]["training_row_sha256"] = "not-a-digest"
    elif bad in ("pooled", "p95", "max", "macro"):
        item["cd_tail"][{"pooled": "pooled_rmse", "p95": "p95_geometry_rmse", "max": "max_geometry_rmse", "macro": "macro_geometry_rmse"}[bad]] *= 2
    elif bad == "primary_rmse":
        item["targets"]["Cd"]["rmse"] *= 2
    elif bad == "primary_missing":
        del item["targets"]["Cm"]
    else:
        item["targets"]["Cdp"] = item["targets"].pop("Cm")
    assert_inconclusive(reference, candidate)


def test_per_geometry_error_tamper_is_caught_even_if_aggregate_claims_pass():
    reference, candidate = paired()
    candidate[2]["per_geometry"][0]["Cd_rmse"] *= 100
    gate = assert_inconclusive(reference, candidate)
    assert "inconsistent with per-geometry" in gate["errors"][0]


def test_cross_fold_evaluation_geometry_overlap_is_not_three_fold_evidence():
    reference, candidate = paired()
    frame = fold_frame(2)
    frame.loc[frame.geometry_id == "f2_g0", "geometry_id"] = "f0_g0"
    reference[2] = score(frame, 2, "reference")
    candidate[2] = score(frame, 2, "candidate", .05)
    gate = assert_inconclusive(reference, candidate)
    assert "disjoint" in gate["errors"][0]


@pytest.mark.parametrize("bad", ["missing", "extra", "duplicate", "none", "nan", "string"])
def test_declared_canonical_training_union_rejects_missing_extra_or_bad_ids(bad):
    reference, candidate = paired()
    ids = expected_ids(reference)
    if bad == "missing":
        ids.pop()
    elif bad == "extra":
        ids.append("canonical_test_geometry")
    elif bad == "duplicate":
        ids.append(ids[0])
    elif bad == "none":
        ids[0] = None
    elif bad == "nan":
        ids[0] = np.nan
    else:
        ids = "not-a-geometry-collection"
    assert_inconclusive(reference, candidate, ids)


def test_foreign_test_geometry_cannot_pass_declared_canonical_train_ids():
    reference, candidate = paired()
    ids = expected_ids(reference)
    frame = fold_frame(2)
    frame.loc[frame.geometry_id == "f2_g0", "geometry_id"] = "old_canonical_test_geometry"
    reference[2] = score(frame, 2, "reference")
    candidate[2] = score(frame, 2, "candidate", .05)
    gate = assert_inconclusive(reference, candidate, ids)
    assert "canonical-training geometry IDs" in gate["errors"][0]


@pytest.mark.parametrize("group", ["low_re", "thick"])
def test_empty_stress_null_metrics_is_inconclusive_without_filtering(group):
    reference, candidate = paired()
    frame = fold_frame(2)
    thresholds = deepcopy(reference[2]["thresholds"])
    thresholds["low_re_max" if group == "low_re" else "thick_min"] = 1.
    reference[2] = score(frame, 2, "reference", thresholds=thresholds)
    candidate[2] = score(frame, 2, "candidate", .05, thresholds=thresholds)
    assert candidate[2]["stress"][group]["Cd_rmse"] is None
    assert candidate[2]["n_rows"] == len(frame)
    assert_inconclusive(reference, candidate)


def test_exactly_zero_primary_labels_are_missing_relative_evidence_not_faux_pass():
    reference, candidate = paired()
    frame = fold_frame(2)
    frame.Cl = 0.
    reference[2] = score(frame, 2, "reference")
    candidate[2] = score(frame, 2, "candidate", .05)
    assert reference[2]["targets"]["Cl"]["median_relative_error"] is None
    assert_inconclusive(reference, candidate)


@pytest.mark.parametrize("bad", ["missing_aggregation", "two_members", "four_members", "tuple_members", "float_members", "missing_source", "missing_predictor", "missing_fold_id", "missing_threshold_hash"])
def test_required_provenance_and_exact_integer_member_metadata(bad):
    reference, candidate = paired()
    item = candidate[2]
    if bad == "missing_aggregation":
        del item["aggregation"]
    elif bad == "two_members":
        item["member_seeds"] = [42, 43]
    elif bad == "four_members":
        item["member_seeds"] = [42, 43, 44, 45]
    elif bad == "tuple_members":
        item["member_seeds"] = (42, 43, 44)
    elif bad == "float_members":
        item["member_seeds"] = [42., 43., 44.]
    elif bad == "missing_threshold_hash":
        del item["thresholds"]["training_row_sha256"]
    else:
        del item[{"missing_source": "source", "missing_predictor": "predictor", "missing_fold_id": "fold_id"}[bad]]
    assert_inconclusive(reference, candidate)


def test_low_re_stress_deterioration_is_gated_even_when_all_geometry_tails_improve():
    reference, candidate = paired()
    for fold in FOLDS:
        frame = fold_frame(fold)
        prediction = frame[["Cl", "Cd", "Cm"]].to_numpy() * 1.05
        low_re = frame.Re.to_numpy() == 1e5
        prediction[low_re, 1] = frame.loc[low_re, "Cd"] * 1.11
        candidate[fold] = score(frame, fold, "candidate", prediction=prediction)
    gate = paired_fold_gate(reference, candidate)
    failures = [check for check in gate["checks"] if not check["passed"]]
    assert gate["status"] == "fail"
    assert len(failures) == 1
    assert failures[0]["metric"] == "Cd.low_re_rmse"
    assert failures[0]["rule"] == "mean_tail_deterioration"


@pytest.mark.parametrize("group", ["low_re", "thick"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf, -.001, True])
def test_invalid_stress_metrics_are_inconclusive(group, value):
    reference, candidate = paired()
    candidate[2]["stress"][group]["Cd_rmse"] = value
    assert_inconclusive(reference, candidate)


def test_reallocated_geometry_counts_must_match_pair_even_if_totals_match():
    reference, candidate = paired()
    item = candidate[2]
    item["per_geometry"][0]["n_rows"] += 1
    item["per_geometry"][1]["n_rows"] -= 1
    # Equal errors across geometries mean these reallocated counts leave RMSE unchanged.
    gate = assert_inconclusive(reference, candidate)
    assert "identical paired rows" in gate["errors"][0]


def test_nonfinite_delta_from_finite_evidence_does_not_leave_partial_averages():
    reference, candidate = paired()
    maximum = np.finfo(float).max
    reference[2]["targets"]["Cl"]["r2"] = -maximum
    candidate[2]["targets"]["Cl"]["r2"] = maximum
    gate = assert_inconclusive(reference, candidate)
    assert "nonfinite paired fold delta" in gate["errors"][0]


def scaling_paired(candidate_mode="profile_rms_floor", factors=(.05, .05, .05)):
    reference, candidate = paired(factors)
    for reports, arm, scaling in ((reference, "cdp_column_scaling", "standard"),
                                  (candidate, "cdp_profile_scaling", candidate_mode)):
        for item in reports.values():
            item.update(predictor=f"{arm}_ensemble", arm=arm, input_scaling=scaling,
                         study_kind="paired_geometry_input_scaling", member_seeds=[42, 43, 44], aggregation=METHOD)
            if candidate_mode == "profile_rms_floor_1pct":
                item["study_variant"] = "selective_profile_floor_1pct"
    return reference, candidate


@pytest.mark.parametrize("candidate_mode", ["profile_rms_floor", "profile_rms_floor_1pct"])
def test_input_scaling_gate_shares_17_limits_without_relabeling_or_seed_reference(candidate_mode):
    reference, candidate = scaling_paired(candidate_mode)
    before = deepcopy((reference, candidate))
    old = paired_fold_gate(*paired())
    wrapper = paired_selective_scaling_fold_gate if candidate_mode == "profile_rms_floor_1pct" else paired_input_scaling_fold_gate
    gate = wrapper(reference, candidate, expected_ids(reference))
    assert gate["status"] == "pass" and gate["passed"] and gate["eligible"]
    assert gate["reference_predictor"] == "cdp_column_scaling_ensemble"
    assert gate["candidate_predictor"] == "cdp_profile_scaling_ensemble"
    assert gate["study_kind"] == "paired_geometry_input_scaling"
    assert gate["numerical_safety_checked"] is False
    if candidate_mode == "profile_rms_floor_1pct":
        assert gate["study_variant"] == "selective_profile_floor_1pct"
    else:
        assert "study_variant" not in gate
    assert gate["policy"] == old["policy"] and gate["checks"] == old["checks"]
    assert gate["deltas"] == old["deltas"] and len(gate["checks"]) == 17
    assert all(row["reference"]["member_seeds"] == [42, 43, 44] for row in gate["fold_stats"])
    assert all(row["reference"]["predictor"] == "cdp_column_scaling_ensemble" for row in gate["fold_stats"])
    assert (reference, candidate) == before
    assert paired_fold_gate(reference, candidate)["status"] == "inconclusive"
    json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("arm", ["reference", "candidate"])
@pytest.mark.parametrize("candidate_mode", ["profile_rms_floor", "profile_rms_floor_1pct"])
@pytest.mark.parametrize("bad", ["single_seed", "seed_label", "seed_order", "float_seed", "old_label", "swapped_label",
                                "wrong_arm", "wrong_scaling", "missing_kind", "wrong_kind", "wrong_aggregation"])
def test_input_scaling_gate_rejects_untruthful_metadata_and_single_reference(arm, bad, candidate_mode):
    reference, candidate = scaling_paired(candidate_mode)
    item = (reference if arm == "reference" else candidate)[2]
    if bad == "single_seed":
        item["member_seeds"] = [42]
    elif bad == "seed_label":
        item["seed"] = 42
    elif bad == "seed_order":
        item["member_seeds"] = [44, 43, 42]
    elif bad == "float_seed":
        item["member_seeds"] = [42., 43., 44.]
    elif bad == "old_label":
        item["predictor"] = "baseline_seed_42" if arm == "reference" else "cdp_ensemble"
    elif bad == "swapped_label":
        item["predictor"] = "cdp_profile_scaling_ensemble" if arm == "reference" else "cdp_column_scaling_ensemble"
    elif bad == "wrong_arm":
        item["arm"] = "cdp"
    elif bad == "wrong_scaling":
        item["input_scaling"] = "per_feature_std"
    elif bad == "missing_kind":
        del item["study_kind"]
    elif bad == "wrong_kind":
        item["study_kind"] = "seed_study"
    else:
        item["aggregation"] = "arithmetic_mean_Cd"
    wrapper = paired_selective_scaling_fold_gate if candidate_mode == "profile_rms_floor_1pct" else paired_input_scaling_fold_gate
    gate = wrapper(reference, candidate, expected_ids(reference))
    assert gate["status"] == "inconclusive" and not gate["passed"] and not gate["eligible"]
    assert gate["errors"] and gate["checks"] == gate["deltas"] == []


@pytest.mark.parametrize("candidate_mode", ["profile_rms_floor", "profile_rms_floor_1pct"])
def test_input_scaling_relative_gate_is_not_absolute_r2_safety(candidate_mode):
    reference, candidate = scaling_paired(candidate_mode)
    wrapper = paired_selective_scaling_fold_gate if candidate_mode == "profile_rms_floor_1pct" else paired_input_scaling_fold_gate
    candidate[2]["targets"]["Cd"]["r2"] = -10.
    assert wrapper(reference, candidate)["passed"]
    candidate[2]["cd_tail"]["max_geometry_rmse"] *= 10
    assert wrapper(reference, candidate)["status"] == "inconclusive"


@pytest.mark.parametrize("wrapper,wrong_mode", [(paired_selective_scaling_fold_gate, "profile_rms_floor"),
                                             (paired_input_scaling_fold_gate, "profile_rms_floor_1pct")])
@pytest.mark.parametrize("mixed", [False, True])
def test_full_and_selective_gates_reject_other_mode_never_infer_from_reports(wrapper, wrong_mode, mixed):
    right_mode = "profile_rms_floor_1pct" if wrapper is paired_selective_scaling_fold_gate else "profile_rms_floor"
    reference, candidate = scaling_paired(right_mode if mixed else wrong_mode)
    candidate[2]["input_scaling"] = wrong_mode
    # Even a claimed variant matching the gate cannot make the wrong mode truthful.
    for item in (*reference.values(), *candidate.values()):
        item["study_variant"] = "selective_profile_floor_1pct"
    gate = wrapper(reference, candidate, expected_ids(reference))
    assert gate["status"] == "inconclusive" and not gate["passed"] and not gate["eligible"]
    assert gate["errors"] and gate["checks"] == gate["deltas"] == []
    json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("rule,value,passes", [
    ("primary_rmse", 1.05, True), ("primary_rmse", 1.05000001, False),
    ("per_fold_tail", .11, True), ("per_fold_tail", .11000001, False),
    ("mean_tail", 1.05, True), ("mean_tail", 1.05000001, False),
    ("primary_median", .005, True), ("primary_median", .00500000001, False),
    ("strict_improvement", 0., False), ("strict_improvement", 5e-13, False),
    ("strict_improvement", 2e-12, True),
])
def test_selective_boundaries_match_historical_17_checks_exactly(rule, value, passes):
    factors = (.01, .01, value) if rule == "per_fold_tail" else ((.1 * value,) * 3 if rule == "mean_tail" else (.05,) * 3)
    reference, candidate = scaling_paired("profile_rms_floor_1pct", factors)
    for fold in FOLDS:
        if rule == "primary_rmse":
            for target in ("Cl", "Cm"):
                candidate[fold]["targets"][target]["rmse"] = reference[fold]["targets"][target]["rmse"] * value
        elif rule == "mean_tail":
            for target in ("Cl", "Cd", "Cm"):
                candidate[fold]["targets"][target]["median_relative_error"] = .09
        elif rule == "primary_median":
            candidate[fold]["targets"]["Cl"]["median_relative_error"] = reference[fold]["targets"]["Cl"]["median_relative_error"] + value
        elif rule == "strict_improvement":
            candidate[fold]["targets"]["Cd"]["median_relative_error"] = reference[fold]["targets"]["Cd"]["median_relative_error"] - value
    selective = paired_selective_scaling_fold_gate(reference, candidate, expected_ids(reference))
    for item in candidate.values():
        item["input_scaling"] = "profile_rms_floor"
    historical = paired_input_scaling_fold_gate(reference, candidate, expected_ids(reference))
    assert selective["passed"] is historical["passed"] is passes
    assert len(selective["checks"]) == 17
    assert selective["checks"] == historical["checks"]
    assert selective["deltas"] == historical["deltas"]
    assert selective["policy"] == historical["policy"]


@pytest.mark.parametrize("bad", ["missing_fold", "missing_r2", "nonfinite", "holdout", "row_identity", "missing_stress", "coverage"])
def test_selective_missing_or_invalid_evidence_has_historical_inconclusive_semantics(bad):
    reference, candidate = scaling_paired("profile_rms_floor_1pct")
    ids = expected_ids(reference)
    if bad == "missing_fold":
        del candidate[2]
    elif bad == "missing_r2":
        del candidate[2]["targets"]["Cl"]["r2"]
    elif bad == "nonfinite":
        candidate[2]["targets"]["Cl"]["r2"] = np.inf
    elif bad == "holdout":
        candidate[2]["source"] = "canonical_test"
    elif bad == "row_identity":
        candidate[2]["row_sha256"] = "f" * 64
    elif bad == "missing_stress":
        candidate[2]["stress"]["low_re"]["missing"] = True
    else:
        ids.pop()
    selective = paired_selective_scaling_fold_gate(reference, candidate, ids)
    for item in candidate.values():
        item["input_scaling"] = "profile_rms_floor"
    historical = paired_input_scaling_fold_gate(reference, candidate, ids)
    assert selective["status"] == historical["status"] == "inconclusive"
    assert selective["errors"] == historical["errors"]
    assert selective["reasons"] == historical["reasons"]
    assert selective["checks"] == selective["deltas"] == []
    assert not selective["passed"] and not selective["eligible"]
    json.dumps(selective, allow_nan=False)
