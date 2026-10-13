import numpy as np
import pytest

from src.geometry.descriptors import describe_profile, describe_surfaces
from src.geometry.naca import generate_naca4, parse_naca4
from src.geometry.representation import (
    cosine_spacing,
    flatten_geometry,
    geometry_columns,
    normalize_coordinates,
    normalize_surfaces,
    sample_profile,
)


def naca(code, n_points=400):
    return generate_naca4(code, n_points)


def test_parse_naca4():
    assert parse_naca4("NACA2412") == (.02, .4, .12)


def test_invalid_naca_code():
    with pytest.raises(ValueError):
        parse_naca4("241")


def test_symmetric_airfoil():
    upper, lower = generate_naca4("0012", 32)
    assert np.allclose(upper[:, 1], -lower[:, 1])


def test_cambered_airfoil_and_normalization():
    upper, lower = generate_naca4("2412", 32)
    coordinates = normalize_coordinates(np.vstack((upper, lower)))
    assert coordinates[:, 0].min() == pytest.approx(0)
    assert coordinates[:, 0].max() == pytest.approx(1)
    assert coordinates[:, 1].max() > 0


def test_normalization_scales_y_by_the_same_chord_factor():
    """v1 divided y by chord and then removed its mean, destroying thickness."""
    upper, lower = naca("2412", 64)
    points = np.vstack((upper, lower))
    reference = normalize_coordinates(points)
    # Uniform scaling and an x shift are undone exactly.
    np.testing.assert_allclose(
        normalize_coordinates(points * 7.5 + np.array([3.0, 0.0])), reference, atol=1e-12,
    )
    # A y offset survives as a constant shift instead of being mean-centered,
    # which is precisely what v1 destroyed.
    difference = normalize_coordinates(points + np.array([0.0, 0.25]))[:, 1] - reference[:, 1]
    assert difference.min() > 0
    np.testing.assert_allclose(difference, difference[0], atol=1e-12)


@pytest.mark.parametrize("code", ["0012", "2412", "4415", "6409", "0006"])
def test_thickness_is_never_negative_and_edges_close(code):
    """The v1 bug put 43% of stations below zero thickness and opened both edges."""
    profile = sample_profile(*naca(code), 60)
    assert profile.clipped_stations == 0
    assert (profile.thickness >= 0).all()
    assert profile.thickness.max() > 0
    # A NACA four-digit section closes to a small, strictly positive wedge.
    assert profile.thickness[0] == pytest.approx(0, abs=1e-6)
    assert 0 < profile.thickness[-1] < 0.01
    assert profile.camber[0] == pytest.approx(0, abs=1e-6)


def test_shared_cosine_grid_resolves_the_nose():
    x = cosine_spacing(50)
    assert x[0] == pytest.approx(0) and x[-1] == pytest.approx(1)
    assert (np.diff(x) > 0).all()
    # Clustered at both edges: the first gap is far smaller than the mid-chord gap.
    assert np.diff(x)[0] < np.diff(x)[len(x) // 2] / 10
    profile = sample_profile(*naca("2412"), 50)
    np.testing.assert_allclose(profile.x, x)


@pytest.mark.parametrize("n_samples", [1, 0, -3])
def test_cosine_spacing_rejects_degenerate_grids(n_samples):
    with pytest.raises(ValueError, match="at least 2"):
        cosine_spacing(n_samples)


def test_surfaces_are_normalized_with_one_shared_transform():
    upper, lower = normalize_surfaces(*naca("4415"))
    # Chord line runs from (0, 0) to (1, 0) for the pair, not per surface.
    assert (upper[0] + lower[0]) / 2 == pytest.approx([0.0, 0.0], abs=1e-12)
    assert (upper[-1] + lower[-1]) / 2 == pytest.approx([1.0, 0.0], abs=1e-12)
    assert upper[:, 1].mean() > 0 > lower[:, 1].mean()


@pytest.mark.parametrize("degrees", [-45.0, -20.0, 0.0, 20.0, 45.0])
def test_normalization_is_invariant_to_rotation_scale_and_translation(degrees):
    upper, lower = naca("2412")
    reference = describe_profile(sample_profile(upper, lower, 50))
    radians = np.radians(degrees)
    rotation = np.array([
        [np.cos(radians), np.sin(radians)],
        [-np.sin(radians), np.cos(radians)],
    ])
    offset = np.array([9.0, -4.0])
    moved = describe_profile(sample_profile(
        (upper * 3.25) @ rotation + offset, (lower * 3.25) @ rotation + offset, 50,
    ))
    for name, value in reference.items():
        assert moved[name] == pytest.approx(value, abs=1e-9, rel=1e-9), name


def test_point_order_does_not_matter():
    upper, lower = naca("4415")
    reference = describe_profile(sample_profile(upper, lower, 50))
    reversed_order = describe_profile(sample_profile(upper[::-1], lower[::-1], 50))
    assert reversed_order == pytest.approx(reference, abs=1e-12)


def test_thickness_camber_basis_round_trips_to_the_surfaces():
    profile = sample_profile(*naca("2412"), 50)
    upper, lower = profile.surfaces()
    np.testing.assert_allclose(upper[:, 1] - lower[:, 1], profile.thickness, atol=1e-15)
    np.testing.assert_allclose((upper[:, 1] + lower[:, 1]) / 2, profile.camber, atol=1e-15)
    np.testing.assert_allclose(upper[:, 0], profile.x)


def test_crossed_coordinates_are_clipped_and_counted_not_raised():
    """Some Drela sections in the archive cross by ~1e-4 chord; count, do not reject."""
    upper, lower = naca("0006", 80)
    crossed = lower.copy()
    crossed[30:50, 1] = upper[30:50, 1] + 0.002
    profile = sample_profile(upper, crossed, 50)
    assert profile.clipped_stations > 0
    assert (profile.thickness >= 0).all()


def test_flatten_and_column_names_agree():
    profile = sample_profile(*naca("2412"), 12)
    flat = flatten_geometry(*naca("2412"), 12)
    names = geometry_columns(12)
    assert len(flat) == len(names) == 24
    assert names[0] == "thickness_000" and names[12] == "camber_000"
    np.testing.assert_allclose(flat[:12], profile.thickness)
    np.testing.assert_allclose(flat[12:], profile.camber)


@pytest.mark.parametrize("points,message", [
    (np.array([[0.0, 0.0]]), "at least two points"),
    (np.array([0.0, 1.0]), "at least two points"),
    (np.array([[0.0, np.nan], [1.0, 0.0]]), "NaN or infinite"),
])
def test_degenerate_surfaces_are_rejected(points, message):
    with pytest.raises(ValueError, match=message):
        normalize_surfaces(points, points)


def test_zero_chord_is_rejected():
    vertical = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    with pytest.raises(ValueError, match="zero chord"):
        normalize_coordinates(vertical)
    # normalize_surfaces measures leading-to-trailing distance, so zero chord
    # means the two edges coincide rather than that x is constant.
    coincident = np.array([[1.0, 2.0], [1.5, 2.5], [1.0, 2.0]])
    with pytest.raises(ValueError, match="zero chord"):
        normalize_surfaces(coincident, coincident)


# Analytic ground truth: a NACA four-digit section's thickness, camber, and nose
# radius are known in closed form, which makes it the right check for descriptors.
@pytest.mark.parametrize("code", ["0012", "2412", "4415", "6409", "0021"])
def test_descriptors_match_analytic_naca_geometry(code):
    camber, position, thickness = parse_naca4(code)
    described = describe_surfaces(*naca(code), 80)

    # Thickness here is the vertical gap between surfaces; NACA applies yt
    # normal to the camber line, so a strongly cambered section reads a few
    # tenths of a percent high. That is geometry, not error.
    assert described["max_thickness"] == pytest.approx(thickness, rel=5e-3)
    assert described["max_thickness_x"] == pytest.approx(0.30, abs=0.01)
    # The standard thickness polynomial leaves yt(1) = 0.0105 t per surface.
    assert described["trailing_edge_thickness"] == pytest.approx(2 * 0.0105 * thickness, rel=2e-2)
    assert described["area"] == pytest.approx(0.685 * thickness, rel=1e-2)
    assert described["trailing_edge_angle"] > 0

    # For a symmetric section the nose fit reproduces the textbook radius. Camber
    # tilts the nose off the chord line, so a vertical-thickness fit then runs
    # above the radius measured normal to the camber line.
    radius = 1.1019 * thickness**2
    assert described["leading_edge_radius"] == pytest.approx(
        radius, rel=2e-2 if camber == 0 else 0.25,
    )

    if camber == 0:
        assert described["max_camber"] == pytest.approx(0, abs=1e-10)
    else:
        assert described["max_camber"] == pytest.approx(camber, rel=5e-3)
        assert described["max_camber_x"] == pytest.approx(position, abs=0.03)


def test_thicker_sections_have_larger_nose_radius_and_area():
    thin = describe_surfaces(*naca("0006"), 80)
    thick = describe_surfaces(*naca("0021"), 80)
    assert thick["leading_edge_radius"] > thin["leading_edge_radius"]
    assert thick["area"] > thin["area"]
    assert thick["trailing_edge_angle"] > thin["trailing_edge_angle"]


def test_reflexed_camber_keeps_its_sign():
    """A section cambered downward must report negative camber, not its magnitude."""
    upper, lower = naca("2412")
    flipped = describe_surfaces(lower * np.array([1.0, -1.0]), upper * np.array([1.0, -1.0]), 80)
    assert flipped["max_camber"] < 0
    assert flipped["max_thickness"] == pytest.approx(describe_surfaces(upper, lower, 80)["max_thickness"])


def test_descriptors_need_at_least_two_stations():
    profile = sample_profile(*naca("0012"), 2)
    assert describe_profile(profile)["max_thickness"] >= 0
    with pytest.raises(ValueError, match="at least two stations"):
        describe_profile(profile.__class__(
            x=np.array([0.0]), thickness=np.array([0.0]), camber=np.array([0.0]),
            clipped_stations=0,
        ))
