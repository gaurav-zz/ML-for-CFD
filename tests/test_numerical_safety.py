"""Fit-free numerical diagnostics: no models, fitting, or Torch."""

import builtins
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import aggregate_predictions
from src.training.numerical_safety import numerical_report
from src.training.tail_risk import _identity
from src.training.targets import DEFAULT_TARGET_TRANSFORMS, INVERSE_LIMITS, inverse_targets


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Numerical diagnostics must never import Torch.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def frame():
    return pd.DataFrame({
        "geometry_id": ["g1", "g0", "g1", "g0"],
        "Alpha": [0., 0., 1., 1.], "Re": [1e5] * 4, "Ncrit": [9.] * 4,
        "max_thickness": [.2, .1, .2, .1], "Cl": [-.2, .3, .4, .5],
        "Cd": [.01, .02, .03, .04], "Cm": [-.02] * 4,
    }, index=[30, 10, 70, 50])


@pytest.fixture
def members(frame):
    result = np.broadcast_to(frame[["Cl", "Cd", "Cm"]].to_numpy(), (3, len(frame), 3)).copy()
    result[:, :, 1] = np.log(result[:, :, 1])
    result += np.array([-1., 0., 1.])[:, None, None] * .1
    return result


def test_exact_contract_nonsaturation_sample_spread_and_inverse(frame, members):
    report, arrays = numerical_report(frame, members)
    assert set(report) == {
        "schema_version", "target_names", "member_seeds", "n_rows", "n_geometries",
        "row_sha256", "inverse_limits", "status", "n_nonfinite", "n_rows_any_clipped",
        "n_member_values_clipped", "per_member", "per_geometry",
        "zero_clipped_spread_nonzero_raw_rows", "raw_log_Cd_std_summary",
        "clipped_log_Cd_std_summary", "safe_no_clipping", "calibrated_uncertainty",
    }
    assert report["schema_version"] == 1
    assert report["target_names"] == ["Cl", "Cd", "Cm"]
    assert report["member_seeds"] == [42, 43, 44]
    assert report["n_rows"] == 4 and report["n_geometries"] == 2
    assert report["row_sha256"] == _identity(frame)
    assert report["inverse_limits"] == list(INVERSE_LIMITS)
    assert report["status"] == "complete" and report["n_nonfinite"] == 0
    assert report["n_rows_any_clipped"] == report["n_member_values_clipped"] == 0
    assert report["zero_clipped_spread_nonzero_raw_rows"] == 0
    assert report["safe_no_clipping"] is True
    assert report["calibrated_uncertainty"] is False
    assert report["per_member"] == [
        {"seed": seed, "n_lower_clipped": 0, "n_upper_clipped": 0} for seed in (42, 43, 44)
    ]
    assert report["per_geometry"] == [
        {"geometry_id": name, "n_rows": 2, "n_rows_any_clipped": 0,
         "zero_clipped_spread_nonzero_raw_rows": 0} for name in ("g0", "g1")
    ]
    assert set(arrays) == {"fitting_predictions", "physical_predictions", "lower_clipped",
                           "upper_clipped", "raw_log_Cd_std", "clipped_log_Cd_std"}
    np.testing.assert_array_equal(arrays["fitting_predictions"], members)
    np.testing.assert_array_equal(arrays["physical_predictions"], inverse_targets(
        members.reshape(-1, 3), ["Cl", "Cd", "Cm"], DEFAULT_TARGET_TRANSFORMS,
    ).reshape(3, 4, 3))
    for key in ("lower_clipped", "upper_clipped"):
        assert arrays[key].shape == (3, 4) and arrays[key].dtype == bool
        assert not arrays[key].any()
    for key in ("raw_log_Cd_std", "clipped_log_Cd_std"):
        assert arrays[key].shape == (4,)
        np.testing.assert_allclose(arrays[key], members[:, :, 1].std(axis=0, ddof=1))
        assert report[key + "_summary"] == {
            "min": pytest.approx(.1), "max": pytest.approx(.1), "mean": pytest.approx(.1),
        }
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_strict_below_equal_above_log_bounds_not_exp_equality(frame, members):
    lower, upper = np.log(INVERSE_LIMITS)
    members[:, :, 1] = [np.nextafter(lower, -np.inf), lower, upper, np.nextafter(upper, np.inf)]
    report, arrays = numerical_report(frame, members)
    np.testing.assert_array_equal(arrays["lower_clipped"], [[True, False, False, False]] * 3)
    np.testing.assert_array_equal(arrays["upper_clipped"], [[False, False, False, True]] * 3)
    assert report["n_rows_any_clipped"] == 2 and report["n_member_values_clipped"] == 6
    assert report["safe_no_clipping"] is False
    assert report["per_member"] == [
        {"seed": seed, "n_lower_clipped": 1, "n_upper_clipped": 1} for seed in (42, 43, 44)
    ]
    np.testing.assert_array_equal(arrays["physical_predictions"][:, 0, 1], arrays["physical_predictions"][:, 1, 1])
    np.testing.assert_array_equal(arrays["physical_predictions"][:, 2, 1], arrays["physical_predictions"][:, 3, 1])
    members[:, :, 1] = [lower, upper, lower, upper]
    exact_report, _ = numerical_report(frame, members)
    assert exact_report["safe_no_clipping"] is True
    assert exact_report["n_member_values_clipped"] == 0


@pytest.mark.parametrize("sign", [1., -1.])
def test_huge_unequal_logs_clip_identically_hide_spread_and_mean_does_not_overflow(frame, members, sign):
    members[:, :, 1] = sign * np.array([1e308, 1e307, 1e306])[:, None]
    with np.errstate(over="raise", invalid="raise"):
        report, arrays = numerical_report(frame, members)
    expected = np.std([1., .1, .01], ddof=1) * 1e308
    np.testing.assert_allclose(arrays["raw_log_Cd_std"], expected)
    np.testing.assert_array_equal(arrays["clipped_log_Cd_std"], np.zeros(4))
    np.testing.assert_array_equal(arrays["physical_predictions"][:, :, 1], np.full((3, 4), np.exp(np.log(INVERSE_LIMITS[1 if sign > 0 else 0]))))
    assert report["zero_clipped_spread_nonzero_raw_rows"] == 4
    assert report["n_member_values_clipped"] == 12
    assert report["safe_no_clipping"] is False
    assert report["raw_log_Cd_std_summary"]["mean"] == pytest.approx(expected)
    assert report["clipped_log_Cd_std_summary"] == {"min": 0., "max": 0., "mean": 0.}
    assert all(item["zero_clipped_spread_nonzero_raw_rows"] == 2 for item in report["per_geometry"])
    json.dumps(report, allow_nan=False)


def test_opposite_saturation_can_give_plausible_ensemble_but_members_are_unsafe(frame, members):
    members[:, :, 1] = np.array([-1000., 1000., np.log(.01)])[:, None]
    report, arrays = numerical_report(frame, members)
    ensemble = aggregate_predictions(arrays["physical_predictions"])
    np.testing.assert_allclose(ensemble["prediction"][:, 1], (1e-6 * 1e3 * .01) ** (1 / 3))
    assert report["n_rows_any_clipped"] == 4 and report["n_member_values_clipped"] == 8
    assert report["safe_no_clipping"] is False
    assert report["zero_clipped_spread_nonzero_raw_rows"] == 0
    assert (arrays["raw_log_Cd_std"] > arrays["clipped_log_Cd_std"]).all()
    np.testing.assert_allclose(arrays["clipped_log_Cd_std"], ensemble["disagreement"][:, 1])


def test_per_geometry_counts_and_order_independent_identity_with_positional_alignment(frame, members):
    lower, upper = np.log(INVERSE_LIMITS)
    members[:, 0, 1] = [upper + 1, upper + 2, upper + 3]
    members[0, 1, 1] = lower - 1
    members[:, 3, 1] = [lower - 1, lower - 2, lower - 3]
    report, arrays = numerical_report(frame, members)
    assert report["n_rows_any_clipped"] == 3 and report["n_member_values_clipped"] == 7
    assert report["zero_clipped_spread_nonzero_raw_rows"] == 2
    assert report["per_geometry"] == [
        {"geometry_id": "g0", "n_rows": 2, "n_rows_any_clipped": 2,
         "zero_clipped_spread_nonzero_raw_rows": 1},
        {"geometry_id": "g1", "n_rows": 2, "n_rows_any_clipped": 1,
         "zero_clipped_spread_nonzero_raw_rows": 1},
    ]
    order = [3, 0, 2, 1]
    reordered_report, reordered_arrays = numerical_report(frame.iloc[order], members[:, order])
    assert reordered_report == report
    for key in arrays:
        expected = arrays[key][:, order] if arrays[key].ndim > 1 else arrays[key][order]
        np.testing.assert_array_equal(reordered_arrays[key], expected)
    changed = frame.copy()
    changed.loc[30, "Alpha"] = 2.
    assert numerical_report(changed, members)[0]["row_sha256"] != report["row_sha256"]


def test_zero_logs_and_finite_nonpositive_observed_cd_are_not_accuracy_checks(frame, members):
    members[:, :, 1] = 0.
    frame.Cd = [-1., 0., 1., 2.]
    report, arrays = numerical_report(frame, members)
    assert report["safe_no_clipping"] is True
    np.testing.assert_array_equal(arrays["physical_predictions"][:, :, 1], np.ones((3, 4)))
    np.testing.assert_array_equal(arrays["raw_log_Cd_std"], np.zeros(4))
    assert report["raw_log_Cd_std_summary"] == {"min": 0., "max": 0., "mean": 0.}


def test_unrepresentable_raw_spread_is_rejected_not_marked_safe(frame, members):
    maximum = np.finfo(float).max
    members[:, :, 1] = np.array([-maximum, -maximum, maximum])[:, None]
    with pytest.raises(ValueError, match="spread must remain finite and representable"):
        numerical_report(frame, members)


def test_varied_spread_summaries_and_one_row_sample_spread(frame, members):
    spread = np.array([.1, .2, .4, .8])
    members[:, :, 1] = np.array([-1., 0., 1.])[:, None] * spread
    report, _ = numerical_report(frame, members)
    for key in ("raw_log_Cd_std_summary", "clipped_log_Cd_std_summary"):
        assert report[key] == {"min": pytest.approx(.1), "max": pytest.approx(.8),
                               "mean": pytest.approx(.375)}
    single_report, arrays = numerical_report(frame.iloc[:1], members[:, :1])
    assert single_report["n_rows"] == single_report["n_geometries"] == 1
    np.testing.assert_allclose(arrays["raw_log_Cd_std"], [.1])


def test_identical_maximum_finite_logs_have_exact_zero_spread(frame, members):
    members[:, :, 1] = np.finfo(float).max
    with np.errstate(over="raise", invalid="raise"):
        report, arrays = numerical_report(frame, members)
    np.testing.assert_array_equal(arrays["raw_log_Cd_std"], np.zeros(4))
    np.testing.assert_array_equal(arrays["clipped_log_Cd_std"], np.zeros(4))
    assert report["zero_clipped_spread_nonzero_raw_rows"] == 0
    assert report["safe_no_clipping"] is False


@pytest.mark.parametrize("column", ["geometry_id", "Alpha", "Re", "Ncrit", "max_thickness", "Cl", "Cd", "Cm"])
def test_missing_required_columns_rejected(frame, members, column):
    with pytest.raises(ValueError, match="required columns"):
        numerical_report(frame.drop(columns=column), members)


@pytest.mark.parametrize("bad", ["empty", "not_frame", "duplicate_index", "duplicate_columns",
                                  "duplicate_identity", "missing_geometry", "empty_geometry",
                                  "numeric_geometry", "zero_re", "negative_thickness", "varying_thickness"])
def test_bad_rows_and_identities_rejected_without_dropping_rows(frame, members, bad):
    if bad == "empty":
        frame = frame.iloc[:0]
    elif bad == "not_frame":
        frame = frame.to_numpy()
    elif bad == "duplicate_index":
        frame.index = [0] * 4
    elif bad == "duplicate_columns":
        frame = pd.concat([frame, frame[["Cl"]]], axis=1)
    elif bad == "duplicate_identity":
        frame.loc[70, "Alpha"] = 0.
    elif bad in ("missing_geometry", "empty_geometry", "numeric_geometry"):
        frame["geometry_id"] = frame["geometry_id"].astype(object)
        frame.loc[30, "geometry_id"] = {"missing_geometry": None, "empty_geometry": " ", "numeric_geometry": 1}[bad]
    elif bad == "zero_re":
        frame.loc[30, "Re"] = 0.
    elif bad == "negative_thickness":
        frame.loc[30, "max_thickness"] = -.1
    else:
        frame.loc[30, "max_thickness"] = .3
    with pytest.raises(ValueError):
        numerical_report(frame, members)


@pytest.mark.parametrize("column", ["Alpha", "Re", "Ncrit", "max_thickness", "Cl", "Cd", "Cm"])
@pytest.mark.parametrize("bad", ["nan", "inf", "negative_inf", "bool", "string", "complex", "object"])
def test_strict_numeric_finite_coordinates_and_labels(frame, members, column, bad):
    if bad in ("nan", "inf", "negative_inf"):
        frame.loc[30, column] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[bad]
    else:
        frame[column] = frame[column].astype({"bool": bool, "string": str, "complex": complex, "object": object}[bad])
    with pytest.raises(ValueError):
        numerical_report(frame, members)


@pytest.mark.parametrize("bad", ["two", "four", "empty", "wrong_rows", "axes", "targets", "two_dims",
                                  "list", "object", "string", "complex", "bool", "nan", "inf", "negative_inf"])
def test_strict_three_member_shape_dtype_and_finiteness(frame, members, bad):
    if bad == "two":
        members = members[:2]
    elif bad == "four":
        members = np.concatenate([members, members[:1]])
    elif bad == "empty":
        members = members[:, :0]
    elif bad == "wrong_rows":
        members = members[:, :-1]
    elif bad == "axes":
        members = members.transpose(1, 0, 2)
    elif bad == "targets":
        members = members[:, :, :2]
    elif bad == "two_dims":
        members = members[0]
    elif bad == "list":
        members = members.tolist()
    elif bad in ("object", "string", "complex", "bool"):
        members = members.astype({"object": object, "string": str, "complex": complex, "bool": bool}[bad])
    else:
        members[0, 0, 1] = {"nan": np.nan, "inf": np.inf, "negative_inf": -np.inf}[bad]
    with pytest.raises(ValueError):
        numerical_report(frame, members)


@pytest.mark.parametrize("target", [0, 2])
def test_nonfinite_non_drag_member_values_rejected(frame, members, target):
    members[0, 0, target] = np.inf
    with pytest.raises(ValueError, match="finite"):
        numerical_report(frame, members)


def test_read_only_input_nonmutation_json_and_pickle_free_npz_roundtrip(frame, members, tmp_path):
    before_frame, before_members = frame.copy(deep=True), members.copy()
    members.setflags(write=False)
    report, arrays = numerical_report(frame, members)
    pd.testing.assert_frame_equal(frame, before_frame)
    np.testing.assert_array_equal(members, before_members)
    assert not members.flags.writeable
    assert not np.shares_memory(arrays["fitting_predictions"], members)
    assert not np.shares_memory(arrays["physical_predictions"], members)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    path = tmp_path / "numerical_predictions.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
            assert saved[key].dtype == value.dtype


def test_module_import_is_torch_free():
    module = importlib.import_module("src.training.numerical_safety")
    importlib.reload(module)
