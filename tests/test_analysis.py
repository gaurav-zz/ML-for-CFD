import json

import numpy as np
import pandas as pd
import pytest

from src.training.analysis import (
    STALL_ALPHA,
    breakdown,
    lift_to_drag,
    per_geometry,
    prediction_frame,
    stall_regime,
    summarize,
    worst_geometries,
)


@pytest.fixture
def rows():
    return pd.DataFrame({
        "geometry_id": ["foil-a", "foil-a", "foil-b", "foil-b"],
        "Alpha": [2.0, 18.0, 2.0, -20.0],
        "Re": [1e6, 1e6, 5e6, 5e6],
        "Cl": [0.5, 1.2, 0.4, -0.9],
        "Cd": [0.008, 0.090, 0.010, 0.120],
    })


@pytest.fixture
def scored(rows):
    predictions = np.column_stack([
        rows["Cl"] + [0.01, 0.20, 0.01, 0.30],
        rows["Cd"] + [0.001, 0.030, 0.001, 0.040],
    ])
    return prediction_frame(rows, predictions, ["Cl", "Cd"])


def test_prediction_frame_keeps_actuals_and_adds_signed_errors(rows, scored):
    assert scored["Cl_predicted"].tolist() == pytest.approx([0.51, 1.40, 0.41, -0.60])
    np.testing.assert_allclose(scored["Cl_error"], [0.01, 0.20, 0.01, 0.30])
    np.testing.assert_allclose(scored["Cd_error"], [0.001, 0.030, 0.001, 0.040])
    # The original rows are untouched.
    pd.testing.assert_frame_equal(scored[rows.columns], rows)


@pytest.mark.parametrize("predictions,error", [
    (np.zeros((3, 2)), "do not match"),
    (np.zeros((4, 3)), "do not match"),
])
def test_prediction_frame_rejects_shape_mismatches(rows, predictions, error):
    with pytest.raises(ValueError, match=error):
        prediction_frame(rows, predictions, ["Cl", "Cd"])


def test_prediction_frame_requires_the_actual_columns(rows):
    with pytest.raises(ValueError, match="missing actual target columns"):
        prediction_frame(rows, np.zeros((4, 1)), ["Cm"])


def test_stall_regime_splits_on_angle_magnitude(scored):
    labels = stall_regime(scored)
    assert labels.tolist() == ["pre-stall", "post-stall", "pre-stall", "post-stall"]
    assert STALL_ALPHA == 12.0
    # Both signs of a large angle are separated flow.
    assert stall_regime(scored, threshold=19.0).tolist() == [
        "pre-stall", "pre-stall", "pre-stall", "post-stall",
    ]


def test_stall_regime_needs_alpha_and_a_positive_threshold(scored):
    with pytest.raises(ValueError, match="must contain Alpha"):
        stall_regime(scored.drop(columns=["Alpha"]))
    with pytest.raises(ValueError, match="positive angle"):
        stall_regime(scored, threshold=0)


def test_post_stall_drag_error_is_far_worse_than_pre_stall(scored):
    table = breakdown(scored, "Cd", stall_regime(scored), "regime").set_index("regime")
    assert table.loc["pre-stall", "n"] == 2
    assert table.loc["post-stall", "mae"] > table.loc["pre-stall", "mae"] * 10
    # A single pooled number would have hidden that gap entirely.
    assert table.loc["pre-stall", "median_relative_error"] < 0.15


def test_breakdown_groups_by_a_column_name_too(scored):
    table = breakdown(scored, "Cl", "Re")
    assert table["Re"].tolist() == [1e6, 5e6]
    assert table["n"].tolist() == [2, 2]
    assert set(table.columns) == {
        "Re", "n", "mae", "rmse", "r2", "mean_relative_error", "median_relative_error",
    }


def test_breakdown_requires_predictions(rows):
    with pytest.raises(ValueError, match="call prediction_frame first"):
        breakdown(rows, "Cl", "Re")


def test_per_geometry_is_the_unit_the_generalization_claim_is_about(scored):
    table = per_geometry(scored, "Cl")
    assert table["geometry_id"].tolist() == ["foil-a", "foil-b"]
    assert table["n"].tolist() == [2, 2]


def test_worst_geometries_ranks_the_nameable_failures(scored):
    worst = worst_geometries(scored, "Cl", count=1)
    assert worst["geometry_id"].tolist() == ["foil-b"]
    assert len(worst_geometries(scored, "Cl")) == 2
    with pytest.raises(ValueError, match="at least one"):
        worst_geometries(scored, "Cl", count=0)
    with pytest.raises(ValueError, match="Unknown ranking column"):
        worst_geometries(scored, "Cl", column="nope")


def test_lift_to_drag_amplifies_drag_error(scored):
    ratios = lift_to_drag(scored)
    assert len(ratios) == len(scored)
    np.testing.assert_allclose(ratios["lift_to_drag"], scored["Cl"] / scored["Cd"])
    np.testing.assert_allclose(
        ratios["lift_to_drag_predicted"], scored["Cl_predicted"] / scored["Cd_predicted"],
    )
    # 12.5% over on drag at the cruise point moves L/D by more than 5 counts.
    assert abs(ratios["lift_to_drag_error"].iloc[0]) > 5


def test_lift_to_drag_drops_rows_it_cannot_form_a_ratio_for(scored):
    scored.loc[0, "Cd_predicted"] = 0.0
    scored.loc[1, "Cd"] = -0.1
    ratios = lift_to_drag(scored)
    assert len(ratios) == 2
    assert np.isfinite(ratios["lift_to_drag_error"]).all()


def test_lift_to_drag_requires_both_coefficients(scored):
    with pytest.raises(ValueError, match="missing columns needed for L/D"):
        lift_to_drag(scored.drop(columns=["Cd_predicted"]))


def test_summary_is_json_ready_and_covers_every_requested_breakdown(scored):
    report = summarize(scored, ["Cl", "Cd"])
    assert report["n_rows"] == 4
    assert report["n_geometries"] == 2
    assert report["stall_threshold_degrees"] == STALL_ALPHA
    for target in ("Cl", "Cd"):
        entry = report["targets"][target]
        assert set(entry) == {
            "overall", "regime", "reynolds", "worst_geometries",
            "error_concentration", "per_geometry_distribution",
        }
        assert {record["regime"] for record in entry["regime"]} == {"pre-stall", "post-stall"}
        assert len(entry["reynolds"]) == 2
        assert entry["overall"]["mae"] > 0
    assert report["lift_to_drag"]["n_rows"] == 4
    assert report["lift_to_drag"]["n_dropped_rows"] == 0
    # run.json is written with allow_nan=False.
    json.dumps(report, allow_nan=False)


def test_summary_omits_lift_to_drag_when_drag_is_not_a_target(rows):
    scored = prediction_frame(rows, (rows["Cl"] + 0.01).to_numpy(), ["Cl"])
    report = summarize(scored, ["Cl"])
    assert "lift_to_drag" not in report
    json.dumps(report, allow_nan=False)


def test_error_concentration_exposes_tail_sse_and_airfoil_quantiles():
    rows = pd.DataFrame({
        "geometry_id": ["ordinary"] * 4 + ["fx79w660a"] * 2,
        "Alpha": [2.0] * 6, "Re": [2e5] * 6, "Ncrit": [9] * 6,
        "Cd": [1.0] * 6,
    })
    scored = prediction_frame(rows, [2, 2, 2, 2, 3, 5], ["Cd"])
    entry = summarize(scored, ["Cd"])["targets"]["Cd"]
    concentration = entry["error_concentration"]
    assert concentration["total_sse"] == pytest.approx(24)
    tail, ordinary = concentration["top_geometries"]
    assert tail["geometry_id"] == "fx79w660a"
    assert tail["n"] == 2
    assert tail["sse"] == pytest.approx(20)
    assert tail["sse"] == pytest.approx(tail["n"] * tail["rmse"] ** 2)
    assert tail["sse_share"] == pytest.approx(20 / 24)
    assert ordinary["sse_share"] == pytest.approx(4 / 24)
    assert concentration["top_10_sse_share"] == pytest.approx(1)
    distribution = entry["per_geometry_distribution"]
    assert distribution["n_geometries"] == 2
    for metric, values in {
        "mae": [1, 3], "rmse": [1, np.sqrt(10)], "median_relative_error": [1, 3],
    }.items():
        quantiles = distribution["metrics"][metric]
        assert quantiles["n_defined"] == 2
        for name, q in (("p50", .5), ("p90", .9), ("p95", .95)):
            assert quantiles[name] == pytest.approx(np.quantile(values, q))


def test_error_concentration_ranks_sse_not_rmse_and_limits_to_ten():
    rows = pd.DataFrame({
        "geometry_id": ["many"] * 10 + [f"rare-{i:02}" for i in range(11)],
        "Alpha": [2.0] * 21, "Re": [2e5] * 21, "Cl": [1.0] * 21,
    })
    scored = prediction_frame(rows, [3.0] * 10 + [4.0] * 11, ["Cl"])
    entry = summarize(scored, ["Cl"])["targets"]["Cl"]
    top = entry["error_concentration"]["top_geometries"]
    assert len(top) == 10
    assert top[0]["geometry_id"] == "many"
    assert top[0]["sse"] == pytest.approx(40)
    assert entry["worst_geometries"][0]["geometry_id"] == "rare-00"
    assert entry["error_concentration"]["top_10_sse_share"] < 1


def test_zero_sse_and_undefined_relative_errors_are_json_safe(rows):
    rows["Cl"] = 0.0
    scored = prediction_frame(rows, rows["Cl"], ["Cl"])
    report = summarize(scored, ["Cl"])
    entry = report["targets"]["Cl"]
    concentration = entry["error_concentration"]
    assert concentration["total_sse"] == 0
    assert concentration["top_10_sse_share"] == 0
    assert all(record["sse_share"] == 0 for record in concentration["top_geometries"])
    assert entry["per_geometry_distribution"]["metrics"]["median_relative_error"] == {
        "n_defined": 0, "p50": None, "p90": None, "p95": None,
    }
    json.dumps(report, allow_nan=False)


def test_training_ranges_use_only_deduplicated_training_and_keep_all_targets(scored):
    scored["max_thickness"] = [.2, .2, .664, .664]
    scored["trailing_edge_angle"] = [.8, .8, 2.759, 2.759]
    scored["profile_thickness_00"] = [100.0] * 4
    training = pd.DataFrame({
        "geometry_id": ["train-a", "train-a", "train-b"],
        # Repeated polar rows do not redefine a geometry's descriptors.
        "max_thickness": [.1, 999, .478],
        "trailing_edge_angle": [.5, 999, 1.332],
        "max_camber": [0, 0, .1],  # Not present in evaluation: excluded.
        "profile_thickness_00": [0, 0, 1],
        "Alpha": [0, 0, 0], "Re": [1e6, 1e6, 1e6], "Cl": [0, 0, 0],
    })
    original = scored.copy(deep=True)
    pooled = summarize(scored, ["Cl", "Cd"])
    report = summarize(scored, ["Cl", "Cd"], training_frame=training)
    domain = report["training_domain"]
    assert domain["available"] is True
    assert domain["n_training_rows"] == 3
    assert domain["n_training_geometries"] == 2
    assert domain["descriptor_ranges"] == {
        "max_thickness": {"min": .1, "max": .478, "n_geometries": 2},
        "trailing_edge_angle": {"min": .5, "max": 1.332, "n_geometries": 2},
    }
    inside, outside = domain["domains"]
    assert inside["geometry_ids"] == ["foil-a"]
    assert outside["geometry_ids"] == ["foil-b"]
    assert inside["row_mask"] == [True, True, False, False]
    assert outside["row_mask"] == [False, False, True, True]
    assert sum(group["n_rows"] for group in domain["domains"]) == report["n_rows"] == 4
    assert sum(group["n_geometries"] for group in domain["domains"]) == 2
    for target in ("Cl", "Cd"):
        assert report["targets"][target]["overall"] == pooled["targets"][target]["overall"]
        for group, subset in ((inside, scored.iloc[:2]), (outside, scored.iloc[2:])):
            assert set(group["targets"]) == {"Cl", "Cd"}
            assert group["targets"][target]["mae"] == pytest.approx(subset[target + "_error"].abs().mean())
    assert domain["geometries"][1]["domain"] == "outside-training-range"
    assert domain["geometries"][1]["violations"] == [
        {"descriptor": "max_thickness", "value": .664, "min": .1, "max": .478},
        {"descriptor": "trailing_edge_angle", "value": 2.759, "min": .5, "max": 1.332},
    ]
    assert "Univariate" in domain["methodology"]
    assert "not uncertainty or a certified domain" in domain["methodology"]
    # Neither target errors nor the evaluated extremes change range assignments.
    changed = scored.copy()
    changed[["Cl_predicted", "Cd_predicted"]] = 1000
    changed_report = summarize(changed, ["Cl", "Cd"], training_frame=training)
    assert changed_report["training_domain"]["geometries"] == domain["geometries"]
    pd.testing.assert_frame_equal(scored, original)
    json.dumps(report, allow_nan=False)


def test_training_range_boundaries_are_inclusive_and_empty_domain_is_explicit(rows):
    rows["max_thickness"] = [.1, .1, .478, .478]
    training = rows[["geometry_id", "max_thickness"]].copy()
    scored = prediction_frame(rows, rows["Cl"], ["Cl"])
    report = summarize(scored, ["Cl"], training_frame=training)
    inside, outside = report["training_domain"]["domains"]
    assert inside["n_rows"] == 4
    assert outside["n_rows"] == outside["n_geometries"] == 0
    assert outside["geometry_ids"] == []
    assert outside["row_mask"] == [False] * 4
    assert all(value is None for value in outside["targets"]["Cl"].values())
    assert "training_domain" not in summarize(scored, ["Cl"])
    json.dumps(report, allow_nan=False)


def test_empty_training_ranges_are_explicit_and_json_safe(scored):
    scored["max_thickness"] = .2
    training = pd.DataFrame(columns=["geometry_id", "max_thickness"])
    report = summarize(scored, ["Cl", "Cd"], training_frame=training)
    domain = report["training_domain"]
    assert domain["available"] is False
    assert domain["descriptor_ranges"] == {}
    assert domain["n_training_geometries"] == 0
    assert domain["domains"][0]["n_rows"] == 4
    json.dumps(report, allow_nan=False)


def test_easy_subset_uses_fixed_ncrit_and_observed_reynolds_with_boundary_angles():
    rows = pd.DataFrame({
        "geometry_id": ["a", "a", "b", "b", "c", "c"],
        "Alpha": [12, -12, 5, 13, 1, 1],
        "Ncrit": [9, 9, 9, 9, 5, 5],
        "Re": [2e5, 2e5, 3e5, 3e5, 1e6, 1e6],
        "Cl": [1, 2, 3, 4, 5, 6], "Cd": [.01] * 6,
    }, index=[8, 8, 3, 7, 2, 9])
    scored = prediction_frame(rows, rows[["Cl", "Cd"]].to_numpy() + [.1, .001], ["Cl", "Cd"])
    report = summarize(scored, ["Cl", "Cd"])
    easy = report["matched_easy_subset"]
    assert easy["criteria"] == {"max_abs_alpha_degrees": 12, "Ncrit": 9}
    assert easy["missing_columns"] == []
    assert easy["row_mask"] == [True, True, True, False, False, False]
    assert easy["n_rows"] == 3
    assert easy["n_geometries"] == 2
    for target in ("Cl", "Cd"):
        groups = easy["targets"][target]["reynolds"]
        assert [group["Re"] for group in groups] == [2e5, 3e5]
        assert [group["n"] for group in groups] == [2, 1]
        assert [group["n_geometries"] for group in groups] == [1, 1]
        assert groups[1]["r2"] is None
        assert easy["targets"][target]["overall"]["mae"] == pytest.approx(.1 if target == "Cl" else .001)
    narrower = summarize(scored, ["Cl", "Cd"], threshold=5)["matched_easy_subset"]
    assert narrower["row_mask"] == [False, False, True, False, False, False]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("missing_ncrit", [False, True])
def test_empty_easy_subset_is_explicit_and_json_safe(scored, missing_ncrit):
    if not missing_ncrit:
        scored["Ncrit"] = 5
    report = summarize(scored, ["Cl", "Cd"])
    easy = report["matched_easy_subset"]
    assert easy["missing_columns"] == (["Ncrit"] if missing_ncrit else [])
    assert easy["row_mask"] == [False] * 4
    assert easy["n_rows"] == easy["n_geometries"] == 0
    for target in ("Cl", "Cd"):
        assert easy["targets"][target]["reynolds"] == []
        assert all(value is None for value in easy["targets"][target]["overall"].values())
    json.dumps(report, allow_nan=False)


def test_empty_evaluation_and_unusable_lift_to_drag_are_json_safe(scored):
    report = summarize(scored.iloc[:0], ["Cl", "Cd"], training_frame=scored)
    assert report["n_rows"] == report["n_geometries"] == 0
    for target in ("Cl", "Cd"):
        entry = report["targets"][target]
        assert entry["worst_geometries"] == entry["regime"] == entry["reynolds"] == []
        assert entry["error_concentration"]["total_sse"] == 0
        assert entry["error_concentration"]["top_geometries"] == []
        assert all(value is None for value in entry["overall"].values())
    assert report["lift_to_drag"]["n_rows"] == 0
    assert report["lift_to_drag"]["rmse"] is None
    json.dumps(report, allow_nan=False)
    scored["Cd_predicted"] = 0
    report = summarize(scored, ["Cl", "Cd"])
    assert report["lift_to_drag"]["n_dropped_rows"] == 4
    assert report["lift_to_drag"]["mae"] is None
    json.dumps(report, allow_nan=False)
