import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    add_flow_features,
    assemble_features,
    feature_names,
    join_geometry,
    profile_columns,
)
from src.geometry.descriptors import DESCRIPTOR_NAMES


@pytest.fixture
def tables():
    geometry = pd.DataFrame({
        "geometry_id": ["foil-a", "foil-b"],
        "airfoil_name": ["A", "B"],
        "thickness_000": [0.0, 0.0],
        "thickness_001": [0.10, 0.20],
        "camber_000": [0.0, 0.0],
        "camber_001": [0.01, 0.02],
        **{name: [float(index), index + 1.0] for index, name in enumerate(DESCRIPTOR_NAMES)},
        "clipped_stations": [0, 2],
    })
    polars = pd.DataFrame({
        "geometry_id": ["foil-a", "foil-a", "foil-b"],
        "Alpha": [0.0, 10.0, -5.0],
        "Re": [1e5, 1e6, 5e6],
        "Ncrit": [9.0, 9.0, 5.0],
        "Cl": [0.0, 1.1, -0.4],
        "Cd": [0.006, 0.02, 0.008],
        "Cm": [0.0, -0.05, 0.03],
    })
    return geometry, polars


def test_flow_features_rescale_reynolds_and_linearize_alpha(tables):
    _, polars = tables
    enriched = add_flow_features(polars)
    np.testing.assert_allclose(enriched["log10_re"], [5.0, 6.0, np.log10(5e6)])
    radians = np.radians([0.0, 10.0, -5.0])
    np.testing.assert_allclose(enriched["alpha_sin"], np.sin(radians))
    np.testing.assert_allclose(enriched["alpha_cos"], np.cos(radians))
    # The original columns survive untouched.
    pd.testing.assert_frame_equal(enriched[polars.columns], polars)


def test_alpha_sine_tracks_thin_aerofoil_lift():
    """Cl ~ 2 pi sin(alpha) in the linear regime, which degrees cannot express."""
    frame = add_flow_features(pd.DataFrame({"Alpha": [-8.0, 0.0, 8.0], "Re": [1e6] * 3}))
    lift = 2 * np.pi * frame["alpha_sin"]
    assert lift.iloc[0] < 0 < lift.iloc[2]
    assert lift.iloc[1] == pytest.approx(0.0)


@pytest.mark.parametrize("column,value,error", [
    ("Re", 0.0, "finite and positive"),
    ("Re", -1.0, "finite and positive"),
    ("Re", np.inf, "finite and positive"),
    ("Alpha", np.nan, "Alpha must be finite"),
])
def test_flow_features_reject_unusable_conditions(tables, column, value, error):
    _, polars = tables
    polars.loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        add_flow_features(polars)


def test_flow_features_require_their_inputs():
    with pytest.raises(ValueError, match="missing required columns"):
        add_flow_features(pd.DataFrame({"Alpha": [1.0]}))


def test_profile_columns_are_thickness_then_camber_in_grid_order(tables):
    geometry, _ = tables
    assert profile_columns(geometry) == [
        "thickness_000", "thickness_001", "camber_000", "camber_001",
    ]


def test_feature_order_and_no_geometry_id_leak(tables):
    geometry, polars = tables
    names = feature_names(geometry, polars)
    assert names == [
        "thickness_000", "thickness_001", "camber_000", "camber_001",
        *DESCRIPTOR_NAMES,
        "Alpha", "alpha_sin", "alpha_cos", "log10_re", "Ncrit",
    ]
    # geometry_id defines the split groups, so it must never become a feature.
    for excluded in ("geometry_id", "airfoil_name", "clipped_stations", "Cl", "Cd", "Cm", "Re"):
        assert excluded not in names


def test_ncrit_is_optional(tables):
    geometry, polars = tables
    assert "Ncrit" not in feature_names(geometry, polars.drop(columns=["Ncrit"]))


def test_join_attaches_geometry_without_changing_the_row_count(tables):
    geometry, polars = tables
    joined = join_geometry(polars, geometry)
    assert len(joined) == len(polars)
    assert joined.loc[joined.geometry_id == "foil-b", "thickness_001"].tolist() == [0.20]
    # Geometry is constant within an airfoil after the join.
    assert joined.groupby("geometry_id")["camber_001"].nunique().eq(1).all()


@pytest.mark.parametrize("mutate,error", [
    (lambda g, p: (pd.concat([g, g.head(1)]), p), "one row per geometry_id"),
    (lambda g, p: (g.head(1), p), "have no geometry row"),
    (lambda g, p: (g.assign(Alpha=0.0), p), "both define"),
    (lambda g, p: (g.drop(columns=["geometry_id"]), p), "must contain a geometry_id"),
    (lambda g, p: (g, p.drop(columns=["geometry_id"])), "must contain a geometry_id"),
    (lambda g, p: (g.head(0), p), "table is empty"),
])
def test_join_refuses_silent_row_loss_and_duplication(tables, mutate, error):
    geometry, polars = mutate(*tables)
    with pytest.raises(ValueError, match=error):
        join_geometry(polars, geometry)


def test_assemble_returns_a_finite_numeric_design_matrix(tables):
    geometry, polars = tables
    joined, names = assemble_features(polars, geometry)
    assert len(joined) == len(polars)
    matrix = joined[names].to_numpy(dtype=float)
    assert matrix.shape == (3, len(names))
    assert np.isfinite(matrix).all()
    # Targets stay available on the same frame for scoring, just not as features.
    assert {"Cl", "Cd", "Cm"}.issubset(joined.columns)


def test_assemble_rejects_a_non_numeric_or_missing_feature(tables):
    geometry, polars = tables
    with pytest.raises(ValueError, match="must be numeric"):
        assemble_features(polars, geometry.assign(max_thickness=["a", "b"]))
    with pytest.raises(ValueError, match="NaN or infinite"):
        assemble_features(polars, geometry.assign(area=[1.0, np.nan]))
