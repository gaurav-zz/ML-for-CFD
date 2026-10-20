"""Pure NumPy diagnostics/policy tests; no fitting or Torch imports."""

import builtins
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.training import study
from src.training.tail_risk import SEEDS, derive_thresholds, paired_tail_gate, tail_report


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Tail-risk tests must never import Torch.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def frame():
    return pd.DataFrame([
        {"geometry_id": f"g{g}", "Alpha": float(a), "Re": 1e5 if a == 0 else 1e6,
         "Ncrit": 9., "max_thickness": .1 * (g + 1),
         "Cl": .2 + .03 * a, "Cd": .01 + .001 * a, "Cm": -.02 - .001 * a}
        for g in range(4) for a in range(5)
    ])


def score(frame, factor=.1, thresholds=None):
    prediction = frame[["Cl", "Cd", "Cm"]].to_numpy() * (1 + factor)
    return tail_report(frame, prediction, thresholds=derive_thresholds(frame) if thresholds is None else thresholds)


def paired(frame, factors=(.05, .05, .05)):
    return ({seed: score(frame) for seed in SEEDS},
            {seed: score(frame, factor) for seed, factor in zip(SEEDS, factors)})


def test_physical_metrics_geometry_quantiles_and_inclusive_stress(frame):
    thresholds = derive_thresholds(frame)
    assert thresholds["low_re_max"] == 1e5
    assert thresholds["thick_min"] == pytest.approx(.37)
    assert thresholds["training_row_sha256"] == study._row_hash(frame)
    thresholds["thick_min"] = .3
    prediction = frame[["Cl", "Cd", "Cm"]].to_numpy().copy()
    errors = np.repeat([.001, .002, .004, .008], 5)
    prediction[:, 1] += errors
    report = tail_report(frame, prediction, thresholds=thresholds)
    assert report["row_sha256"] == study._row_hash(frame)
    assert report["n_rows"] == 20 and report["n_geometries"] == 4
    assert report["cd_tail"]["pooled_rmse"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert report["cd_tail"]["p95_geometry_rmse"] == pytest.approx(np.quantile([.001, .002, .004, .008], .95, method="linear"))
    assert report["cd_tail"]["max_geometry_rmse"] == pytest.approx(.008)
    assert report["cd_tail"]["macro_geometry_rmse"] == pytest.approx(.00375)
    assert report["targets"]["Cd"]["rmse"] == pytest.approx(report["cd_tail"]["pooled_rmse"])
    assert report["stress"]["low_re"]["n_rows"] == 4
    assert report["stress"]["low_re"]["n_geometries"] == 4
    assert report["stress"]["thick"]["n_rows"] == 10
    assert report["stress"]["thick"]["n_geometries"] == 2
    assert report["stress"]["thick"]["Cd_rmse"] == pytest.approx(np.sqrt(np.mean([.004 ** 2, .008 ** 2])))
    assert "linear" in report["definitions"]["p95_geometry_rmse"]
    json.dumps(report, allow_nan=False)


def test_thresholds_use_unique_geometry_not_row_weighting_or_labels(frame):
    first = derive_thresholds(frame)
    extra = frame[frame.geometry_id == "g0"].copy()
    extra.Alpha += 100
    changed = pd.concat([frame, extra], ignore_index=True)
    changed[["Cl", "Cd", "Cm"]] = np.nan
    second = derive_thresholds(changed)
    assert first["thick_min"] == second["thick_min"]
    assert first["low_re_max"] == second["low_re_max"]
    with pytest.raises(ValueError, match="training data only"):
        derive_thresholds(frame, source="canonical_test")


def test_relative_errors_are_fractions_and_zero_actual_is_explicit(frame):
    report = score(frame, .02)
    assert report["targets"]["Cd"]["median_relative_error"] == pytest.approx(.02)
    assert report["definitions"]["relative_error_units"] == "fraction"
    frame["Cl"] = 0.
    report = score(frame, 0)
    assert report["targets"]["Cl"]["median_relative_error"] is None
    gate = paired_tail_gate(dict.fromkeys(SEEDS, report), dict.fromkeys(SEEDS, report))
    assert gate["status"] == "inconclusive" and not gate["passed"]
    assert "finite and defined" in gate["reasons"][0]


def test_all_three_paired_seeds_pass_and_report_every_delta(frame):
    gate = paired_tail_gate(*paired(frame))
    assert gate["status"] == "pass" and gate["eligible"] and gate["passed"]
    assert gate["source"] == "canonical_train_inner_evaluation"
    assert gate["seed_selection"] is False
    assert len(gate["checks"]) == 15
    assert len(gate["deltas"]) == 21 * 4
    assert all(check["passed"] for check in gate["checks"])
    assert any("not physics" in value for value in [gate["policy"]["description"]])
    median = next(row for row in gate["deltas"] if row["seed"] == "mean" and row["metric"] == "Cd.median_relative_error")
    assert median["delta"] == pytest.approx(-.05)
    assert median["delta_percentage_points"] == pytest.approx(-5.)
    json.dumps(gate, allow_nan=False)


def test_each_seed_limit_catches_hidden_seed44_rmse_explosion(frame):
    gate = paired_tail_gate(*paired(frame, (.01, .01, .113)))
    assert gate["status"] == "fail" and not gate["eligible"]
    assert all(check["passed"] for check in gate["checks"] if check["rule"] != "per_seed_tail_deterioration")
    failed = [check for check in gate["checks"] if not check["passed"]]
    assert {check["seed"] for check in failed} == {44}
    assert {check["metric"] for check in failed} == {"Cd.pooled_rmse", "Cd.max_geometry_rmse"}


@pytest.mark.parametrize("last_error,passes", [(.11, True), (.110001, False)])
def test_per_seed_ten_percent_boundary_is_inclusive(frame, last_error, passes):
    assert paired_tail_gate(*paired(frame, (.01, .01, last_error)))["passed"] is passes


def test_median_gain_cannot_hide_single_geometry_catastrophe(frame):
    baseline, candidate = paired(frame)
    prediction = frame[["Cl", "Cd", "Cm"]].to_numpy() * 1.01
    prediction[-1, 1] = .9
    candidate[44] = tail_report(frame, prediction, thresholds=derive_thresholds(frame))
    gate = paired_tail_gate(baseline, candidate)
    assert gate["status"] == "fail"
    assert next(check for check in gate["checks"] if check["rule"] == "strict_improvement")["passed"]
    assert any("max_geometry_rmse" in reason for reason in gate["reasons"])
    assert any("pooled_rmse" in reason for reason in gate["reasons"])


@pytest.mark.parametrize("which", ["low_re", "thick"])
def test_empty_stress_is_null_and_inconclusive_not_silently_passed(frame, which):
    thresholds = derive_thresholds(frame)
    thresholds["low_re_max" if which == "low_re" else "thick_min"] = 1 if which == "low_re" else 1.
    base = score(frame, thresholds=thresholds)
    aux = score(frame, .01, thresholds)
    assert base["stress"][which]["n_rows"] == base["stress"][which]["n_geometries"] == 0
    assert base["stress"][which]["Cd_rmse"] is None
    assert base["stress"][which]["missing"] is True
    gate = paired_tail_gate(dict.fromkeys(SEEDS, base), dict.fromkeys(SEEDS, aux))
    assert gate["status"] == "inconclusive" and not gate["eligible"]
    assert "empty" in gate["reasons"][0]


@pytest.mark.parametrize("delta,passes", [(.005, True), (.00500000001, False)])
def test_median_fraction_boundary_is_half_percentage_point(frame, delta, passes):
    baseline, candidate = paired(frame)
    for seed in SEEDS:
        candidate[seed]["targets"]["Cl"]["median_relative_error"] = baseline[seed]["targets"]["Cl"]["median_relative_error"] + delta
    gate = paired_tail_gate(baseline, candidate)
    assert gate["passed"] is passes


@pytest.mark.parametrize("factor,passes", [(1.05, True), (1.05000001, False)])
def test_mean_tail_boundary_is_five_percent_plus_epsilon(frame, factor, passes):
    baseline, candidate = paired(frame, (.1 * factor,) * 3)
    for seed in SEEDS:
        candidate[seed]["targets"]["Cd"]["median_relative_error"] = .09
        candidate[seed]["targets"]["Cl"]["median_relative_error"] = .09
        candidate[seed]["targets"]["Cm"]["median_relative_error"] = .09
    assert paired_tail_gate(baseline, candidate)["passed"] is passes


@pytest.mark.parametrize("gain,passes", [(0., False), (5e-13, False), (2e-12, True)])
def test_cd_improvement_must_exceed_numerical_epsilon(frame, gain, passes):
    baseline, candidate = paired(frame)
    for seed in SEEDS:
        candidate[seed]["targets"]["Cd"]["median_relative_error"] = baseline[seed]["targets"]["Cd"]["median_relative_error"] - gain
    assert paired_tail_gate(baseline, candidate)["passed"] is passes


def test_zero_error_baseline_never_divides_by_zero_or_passes_no_gain(frame):
    baseline = {seed: score(frame, 0) for seed in SEEDS}
    candidate = {seed: score(frame, .00001) for seed in SEEDS}
    gate = paired_tail_gate(baseline, candidate)
    assert gate["status"] == "fail"
    assert any(row["relative_delta"] is None for row in gate["deltas"])
    assert any("Zero baseline" in warning for warning in gate["warnings"])
    assert paired_tail_gate(baseline, baseline)["status"] == "fail"  # No strict Cd improvement.
    json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("bad", ["seed", "row_hash", "geometry_id", "threshold", "source", "units", "count", "nan", "infinity", "none"])
def test_invalid_evidence_is_inconclusive_without_partial_average(frame, bad):
    baseline, candidate = paired(frame)
    item = candidate[44]
    if bad == "seed":
        candidate[45] = candidate.pop(44)
    elif bad == "row_hash":
        item["row_sha256"] = "f" * 64
    elif bad == "geometry_id":
        item["geometry_ids"][0] = "foreign"
    elif bad == "threshold":
        item["thresholds"]["low_re_max"] = 123.
    elif bad == "source":
        item["thresholds"]["source"] = "canonical_test"
    elif bad == "units":
        item["definitions"]["relative_error_units"] = "percent"
    elif bad == "count":
        item["n_rows"] -= 1
    else:
        item["cd_tail"]["pooled_rmse"] = {"nan": np.nan, "infinity": np.inf, "none": None}[bad]
    gate = paired_tail_gate(baseline, candidate)
    assert gate["status"] == "inconclusive" and not gate["eligible"]
    assert gate["reasons"] and not gate["deltas"]
    json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("bad", ["nan", "infinity", "shape", "index", "columns", "complex"])
def test_predictions_must_be_finite_exactly_aligned_and_primary_only(frame, bad):
    prediction = frame[["Cl", "Cd", "Cm"]].copy()
    if bad in ("nan", "infinity"):
        prediction.iloc[0, 1] = np.nan if bad == "nan" else np.inf
    elif bad == "shape":
        prediction = prediction.to_numpy()[:-1]
    elif bad == "index":
        prediction = prediction.iloc[::-1]
    elif bad == "columns":
        prediction = prediction[["Cd", "Cl", "Cm"]]
    else:
        prediction = prediction.to_numpy().astype(complex)
    with pytest.raises(ValueError):
        tail_report(frame, prediction, thresholds=derive_thresholds(frame))


@pytest.mark.parametrize("bad", ["missing_thickness", "inconsistent_thickness", "missing_id", "duplicate", "empty", "nan_re"])
def test_invalid_identity_or_stress_coordinates_are_rejected(frame, bad):
    if bad == "missing_thickness":
        frame = frame.drop(columns="max_thickness")
    elif bad == "inconsistent_thickness":
        frame.loc[0, "max_thickness"] += .1
    elif bad == "missing_id":
        frame.loc[0, "geometry_id"] = None
    elif bad == "duplicate":
        frame.iloc[1] = frame.iloc[0]
    elif bad == "empty":
        frame = frame.iloc[:0]
    else:
        frame.loc[0, "Re"] = np.nan
    with pytest.raises(ValueError):
        derive_thresholds(frame)


def test_gate_is_pure_and_canonical_test_is_audit_only(frame):
    baseline, candidate = paired(frame)
    before = deepcopy((baseline, candidate))
    paired_tail_gate(baseline, candidate)
    assert (baseline, candidate) == before
    with pytest.raises(ValueError, match="audit-only"):
        paired_tail_gate(baseline, candidate, source="canonical_test")
    for arm in (baseline, candidate):
        for report in arm.values():
            report["thresholds"]["source"] = "canonical_train"
    audit = paired_tail_gate(baseline, candidate, source="canonical_test", audit_only=True)
    assert audit["passed"] and audit["audit_only"]


@pytest.mark.parametrize("bad", ["per_geometry_nan", "per_geometry_negative", "per_geometry_value", "pooled", "p95", "macro", "max", "primary_rmse", "hash_format"])
def test_stale_or_internally_inconsistent_tail_artifacts_are_inconclusive(frame, bad):
    baseline, candidate = paired(frame)
    report = candidate[44]
    if bad.startswith("per_geometry"):
        report["per_geometry"][0]["Cd_rmse"] = {
            "per_geometry_nan": np.nan, "per_geometry_negative": -.1, "per_geometry_value": .1,
        }[bad]
    elif bad == "primary_rmse":
        report["targets"]["Cd"]["rmse"] *= 2
    elif bad == "hash_format":
        for arm in (baseline, candidate):
            for item in arm.values():
                item["row_sha256"] = "not-a-digest"
    else:
        key = {"pooled": "pooled_rmse", "p95": "p95_geometry_rmse", "macro": "macro_geometry_rmse", "max": "max_geometry_rmse"}[bad]
        report["cd_tail"][key] *= 2
    gate = paired_tail_gate(baseline, candidate)
    assert gate["status"] == "inconclusive" and not gate["passed"]
    assert gate["reasons"] and not gate["deltas"]
