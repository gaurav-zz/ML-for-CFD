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
        assert set(entry) == {"overall", "regime", "reynolds", "worst_geometries"}
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
