"""Pure NumPy ensemble/diagnostic tests, without models, fitting, or Torch."""

import builtins
from copy import deepcopy
import importlib
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import METHOD, SEEDS, aggregate_predictions, disagreement_report


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Ensemble tests must never import Torch.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def members():
    return np.array([
        [[-.2, .01, -.04], [0., .02, .01]],
        [[.4, .04, -.02], [-.3, .08, -.01]],
        [[.7, .16, .03], [-.6, .32, -.03]],
    ])


def diagnostic_frame(n=10):
    return pd.DataFrame({"geometry_id": [f"g{i // 2}" for i in range(n)],
                         "Cl": np.linspace(-.2, .2, n), "Cd": np.full(n, .02),
                         "Cm": np.linspace(-.03, 0., n)}, index=np.arange(n) * 3 + 10)


def diagnostic_result(frame, spreads=None, log_errors=None):
    spread = np.arange(len(frame), dtype=float) / 10 if spreads is None else np.asarray(spreads)
    log_error = np.arange(len(frame), dtype=float) / 20 if log_errors is None else np.asarray(log_errors)
    prediction = frame[["Cl", "Cd", "Cm"]].to_numpy().copy()
    prediction[:, 1] *= np.exp(log_error)
    disagreement = np.zeros_like(prediction)
    disagreement[:, 1] = spread
    return {"prediction": prediction, "disagreement": disagreement,
            "method": METHOD, "member_seeds": list(SEEDS)}


def test_physical_arithmetic_cl_cm_geometric_cd_and_sample_log_spread(members):
    before = members.copy()
    result = aggregate_predictions(members)
    assert set(result) == {"prediction", "disagreement", "method", "member_seeds"}
    assert result["method"] == METHOD
    assert result["member_seeds"] == [42, 43, 44]
    assert result["prediction"].shape == result["disagreement"].shape == (2, 3)
    assert result["prediction"].dtype == result["disagreement"].dtype == np.float64
    np.testing.assert_allclose(result["prediction"][:, [0, 2]], members.mean(axis=0)[:, [0, 2]])
    np.testing.assert_allclose(result["prediction"][:, 1], [.04, .08])
    assert not np.allclose(result["prediction"][:, 1], members[:, :, 1].mean(axis=0))
    coordinates = members.copy()
    coordinates[:, :, 1] = np.log(coordinates[:, :, 1])
    np.testing.assert_allclose(result["disagreement"], coordinates.std(axis=0, ddof=1))
    assert not np.allclose(result["disagreement"][:, 1], members.std(axis=0, ddof=1)[:, 1])
    np.testing.assert_array_equal(members, before)


def test_decode_scale_roundtrip_prevents_mixing_normalized_or_raw_drag_means(members):
    physical = members.copy()
    transformed = physical.copy()
    transformed[:, :, 1] = np.log(transformed[:, :, 1])
    mean = np.array([.3, -5., -.1])
    scale = np.array([2., .7, .2])
    normalized = (transformed - mean) / scale
    decoded = normalized * scale + mean
    decoded[:, :, 1] = np.exp(decoded[:, :, 1])
    result = aggregate_predictions(decoded)
    np.testing.assert_allclose(result["prediction"], aggregate_predictions(physical)["prediction"], rtol=1e-14, atol=1e-15)
    expected = (normalized.mean(axis=0) * scale + mean)
    expected[:, 1] = np.exp(expected[:, 1])
    np.testing.assert_allclose(result["prediction"], expected)
    assert not np.allclose(result["prediction"][:, 1], physical[:, :, 1].mean(axis=0))
    with pytest.raises(ValueError, match="strictly positive"):
        aggregate_predictions(transformed)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_member_permutations_are_invariant_to_roundoff(members, order):
    expected = aggregate_predictions(members)
    actual = aggregate_predictions(members[list(order)])
    for key in ("prediction", "disagreement"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-15, atol=1e-15)
    assert actual["member_seeds"] == [42, 43, 44]


def test_negative_cl_cm_and_exact_zero_spread_are_valid():
    members = np.broadcast_to(np.array([[-.4, .02, -.03], [0., 1., 0.]]), (3, 2, 3)).copy()
    result = aggregate_predictions(members)
    np.testing.assert_allclose(result["prediction"], members[0])
    np.testing.assert_array_equal(result["disagreement"], np.zeros((2, 3)))


@pytest.mark.parametrize("bad", ["two", "four", "empty", "axes", "targets", "list", "object", "complex", "bool", "nan", "inf", "zero_cd", "negative_cd"])
def test_exact_three_members_real_finite_positive_drag_required(members, bad):
    if bad == "two":
        members = members[:2]
    elif bad == "four":
        members = np.concatenate([members, members[:1]])
    elif bad == "empty":
        members = members[:, :0]
    elif bad == "axes":
        members = members.transpose(1, 0, 2)
    elif bad == "targets":
        members = members[:, :, :2]
    elif bad == "list":
        members = members.tolist()
    elif bad in ("object", "complex", "bool"):
        members = members.astype({"object": object, "complex": complex, "bool": bool}[bad])
    else:
        members[0, 0, 1] = {"nan": np.nan, "inf": np.inf, "zero_cd": 0., "negative_cd": -.01}[bad]
    with pytest.raises(ValueError):
        aggregate_predictions(members)


def test_stable_finite_means_and_spreads_do_not_overflow_squares():
    maximum = np.finfo(np.float64).max
    tiny = np.nextafter(0., 1.)
    members = np.array([[[maximum, maximum, -maximum], [1e200, tiny, -1e200]]] * 3)
    result = aggregate_predictions(members)
    assert np.isfinite(result["prediction"]).all()
    np.testing.assert_allclose(result["prediction"][0], [maximum, maximum, -maximum], rtol=1e-13)
    assert result["prediction"][1, 1] == tiny
    np.testing.assert_array_equal(result["disagreement"], np.zeros((2, 3)))
    members[:, 0, 0] = [1e200, 2e200, 3e200]
    result = aggregate_predictions(members)
    assert result["prediction"][0, 0] == pytest.approx(2e200)
    assert result["disagreement"][0, 0] == pytest.approx(1e200)


def test_unrepresentable_sample_spread_is_rejected_not_silently_infinite():
    maximum = np.finfo(np.float64).max
    members = np.array([[[-maximum, .01, 0.]], [[-maximum, .01, 0.]], [[maximum, .01, 0.]]])
    with pytest.raises(ValueError, match="remain finite"):
        aggregate_predictions(members)


def test_log_error_spearman_and_prediction_only_bins_keep_all_rows():
    frame = diagnostic_frame()
    result = diagnostic_result(frame)
    before = deepcopy(result)
    report = disagreement_report(frame, result)
    assert report["n_rows"] == 10 and report["n_geometries"] == 5
    assert report["spearman_abs_log_Cd_error_vs_log_Cd_std"] == pytest.approx(1.)
    assert report["spread_thresholds"] == {"p50": pytest.approx(.45), "p90": pytest.approx(.81), "source": "prediction_only"}
    assert [item["name"] for item in report["bins"]] == ["low", "mid", "high"]
    assert [item["n_rows"] for item in report["bins"]] == [5, 4, 1]
    assert [item["n_geometries"] for item in report["bins"]] == [3, 3, 1]
    assert sum(item["n_rows"] for item in report["bins"]) == len(frame)
    error = result["prediction"][:5, 1] - frame.Cd.to_numpy()[:5]
    assert report["bins"][0]["median_relative_error_Cd"] == pytest.approx(np.median(error / .02))
    assert report["bins"][0]["Cd_rmse"] == pytest.approx(np.sqrt(np.mean(error ** 2)))
    assert report["bins"][0]["Cd_log_mae"] == pytest.approx(.1)
    assert report["definitions"]["relative_error_units"] == "fraction"
    assert report["definitions"]["prediction_units"] == "physical_coefficients"
    assert report["definitions"]["disagreement_units"] == ["Cl", "log(Cd)", "Cm"]
    assert report["definitions"]["disagreement_ddof"] == 1
    for flag in ("calibrated_uncertainty", "confidence_interval", "physics_guarantee", "certified_abstention", "row_filtering"):
        assert report[flag] is False
    json.dumps(report, allow_nan=False)
    for key in ("prediction", "disagreement"):
        np.testing.assert_array_equal(result[key], before[key])


def test_bin_thresholds_do_not_depend_on_observed_targets():
    frame = diagnostic_frame()
    result = diagnostic_result(frame)
    first = disagreement_report(frame, result)
    frame.Cd *= 30
    frame.Cl *= -10
    second = disagreement_report(frame, result)
    assert first["spread_thresholds"] == second["spread_thresholds"]
    assert [item["n_rows"] for item in first["bins"]] == [item["n_rows"] for item in second["bins"]]
    assert first["bins"][0]["Cd_log_mae"] != second["bins"][0]["Cd_log_mae"]


@pytest.mark.parametrize("spread", [np.zeros(10), np.ones(10), np.array([0.] * 9 + [1.])])
def test_ties_empty_bins_have_zero_counts_null_metrics_and_exact_totals(spread):
    frame = diagnostic_frame()
    report = disagreement_report(frame, diagnostic_result(frame, spreads=spread))
    assert sum(item["n_rows"] for item in report["bins"]) == 10
    if np.ptp(spread) == 0:
        assert report["spearman_abs_log_Cd_error_vs_log_Cd_std"] is None
    for item in report["bins"]:
        if item["n_rows"] == 0:
            assert item["n_geometries"] == 0
            assert item["median_relative_error_Cd"] is item["Cd_rmse"] is item["Cd_log_mae"] is None
    json.dumps(report, allow_nan=False)


def test_constant_error_has_null_spearman_even_when_spread_varies():
    frame = diagnostic_frame()
    result = diagnostic_result(frame, log_errors=np.zeros(10))
    assert disagreement_report(frame, result)["spearman_abs_log_Cd_error_vs_log_Cd_std"] is None


def test_spearman_uses_average_ranks_for_ties_without_scipy(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            pytest.fail("Ensemble diagnostics must not require SciPy.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.import_module("src.models.ensemble")
    importlib.reload(module)
    frame = diagnostic_frame(5)
    spreads = np.array([0., 0., .2, .3, .3])
    errors = np.array([.3, .2, .2, .1, .1])
    expected = np.corrcoef(pd.Series(spreads).rank(), pd.Series(errors).rank())[0, 1]
    report = module.disagreement_report(frame, diagnostic_result(frame, spreads, errors))
    assert report["spearman_abs_log_Cd_error_vs_log_Cd_std"] == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["shape", "spread_shape", "nan_label", "inf_label", "zero_cd", "negative_cd", "nan_prediction", "inf_spread", "negative_spread", "zero_prediction", "seeds", "method", "missing_target", "missing_geometry", "duplicate_index", "empty", "frame_prediction"])
def test_diagnostics_reject_invalid_evidence_without_dropping_rows(bad):
    frame = diagnostic_frame()
    result = diagnostic_result(frame)
    if bad == "shape":
        result["prediction"] = result["prediction"][:-1]
    elif bad == "spread_shape":
        result["disagreement"] = result["disagreement"][:-1]
    elif bad in ("nan_label", "inf_label", "zero_cd", "negative_cd"):
        frame.loc[frame.index[0], "Cd"] = {"nan_label": np.nan, "inf_label": np.inf, "zero_cd": 0., "negative_cd": -.01}[bad]
    elif bad in ("nan_prediction", "zero_prediction"):
        result["prediction"][0, 1] = np.nan if bad == "nan_prediction" else 0.
    elif bad in ("inf_spread", "negative_spread"):
        result["disagreement"][0, 1] = np.inf if bad == "inf_spread" else -.1
    elif bad == "seeds":
        result["member_seeds"] = [0, 1, 2]
    elif bad == "method":
        result["method"] = "arithmetic_mean_Cd"
    elif bad == "missing_target":
        frame = frame.drop(columns="Cm")
    elif bad == "missing_geometry":
        frame.loc[frame.index[0], "geometry_id"] = None
    elif bad == "duplicate_index":
        frame.index = [0] * len(frame)
    elif bad == "frame_prediction":
        result["prediction"] = pd.DataFrame(result["prediction"], index=frame.index[::-1])
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        disagreement_report(frame, result)


def test_diagnostic_rmse_avoids_overflow_for_finite_large_errors():
    frame = diagnostic_frame(3)
    frame.Cd = 1e199
    result = diagnostic_result(frame, log_errors=np.full(3, np.log(10.)))
    report = disagreement_report(frame, result)
    assert all(np.isfinite(item["Cd_rmse"]) for item in report["bins"] if item["n_rows"])


@pytest.mark.parametrize("target,value", [("Cl", np.nan), ("Cm", np.inf)])
def test_diagnostics_require_finite_observed_non_drag_targets_too(target, value):
    frame = diagnostic_frame()
    result = diagnostic_result(frame)
    frame.loc[frame.index[0], target] = value
    with pytest.raises(ValueError, match="finite"):
        disagreement_report(frame, result)


def test_all_zero_member_arrays_are_not_valid_physical_drag():
    with pytest.raises(ValueError, match="strictly positive"):
        aggregate_predictions(np.zeros((3, 10, 3)))


def test_disagreement_report_one_row_is_retained_with_null_rank_correlation():
    frame = diagnostic_frame(1)
    report = disagreement_report(frame, diagnostic_result(frame))
    assert [item["n_rows"] for item in report["bins"]] == [1, 0, 0]
    assert report["spearman_abs_log_Cd_error_vs_log_Cd_std"] is None


def test_unrepresentable_relative_error_is_rejected_not_filtered():
    frame = diagnostic_frame(3)
    frame.Cd = np.nextafter(0., 1.)
    result = diagnostic_result(frame)
    result["prediction"][:, 1] = np.finfo(float).max
    with pytest.raises(ValueError, match="row errors must remain finite"):
        disagreement_report(frame, result)


def test_even_count_large_finite_relative_median_does_not_overflow():
    frame = diagnostic_frame(4)
    frame.Cd = 1.
    result = diagnostic_result(frame, spreads=np.zeros(4))
    result["prediction"][:, 1] = 1.2e308
    report = disagreement_report(frame, result)
    assert report["bins"][0]["median_relative_error_Cd"] == pytest.approx(1.2e308)
    assert report["bins"][0]["Cd_rmse"] == pytest.approx(1.2e308)
    json.dumps(report, allow_nan=False)
