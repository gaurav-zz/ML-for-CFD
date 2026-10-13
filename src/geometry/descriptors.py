"""Physical scalar descriptors of an airfoil section.

These are the quantities an aerodynamicist would name when describing a
section, and they are what v1's broken representation could not express at
all. They are cheap, interpretable, and — unlike a 100-number coordinate
fingerprint — they transfer to geometries the model has never seen.
"""

from __future__ import annotations

import numpy as np

from src.geometry.representation import GeometryProfile, sample_profile

DESCRIPTOR_NAMES = (
    "max_thickness",
    "max_thickness_x",
    "max_camber",
    "max_camber_x",
    "leading_edge_radius",
    "trailing_edge_angle",
    "trailing_edge_thickness",
    "area",
)

# Stations within this fraction of chord define the nose for the radius fit.
NOSE_FRACTION = 0.05


def _leading_edge_radius(x: np.ndarray, thickness: np.ndarray) -> float:
    """Fit the nose radius from the square of the thickness near the nose.

    A rounded leading edge behaves like ``y = +/- sqrt(2 r x)``, so
    ``thickness**2 = 8 r x`` to leading order. The higher terms matter over any
    usable fitting range: for a NACA four-digit section
    ``thickness**2 = 8 r x - c x**1.5 + ...``, and fitting the linear term alone
    returns a radius about 18% low. Fitting three terms and reading off the
    linear one recovers the textbook ``r = 1.1019 t**2`` for symmetric sections.

    Camber tilts the nose away from the chord line, so for a strongly cambered
    section this vertical-thickness estimate runs above the radius measured
    normal to the camber line. It stays a monotonic, physical feature either way.
    """
    nose = (x > 0) & (x <= NOSE_FRACTION)
    near_x = x[nose]
    if len(near_x) < 3:
        # A coarse grid may not resolve the nose; fall back to one station.
        index = int(np.argmin(np.where(x > 0, x, np.inf)))
        return max(float(thickness[index] ** 2 / (8 * x[index])), 0.0)
    design = np.column_stack((near_x, near_x**1.5, np.square(near_x)))
    (slope, *_), *_ = np.linalg.lstsq(design, np.square(thickness[nose]), rcond=None)
    return max(float(slope) / 8, 0.0)


def describe_profile(profile: GeometryProfile) -> dict[str, float]:
    """Reduce one thickness/camber profile to named physical scalars."""
    x, thickness, camber = profile.x, profile.thickness, profile.camber
    if len(x) < 2:
        raise ValueError("A profile needs at least two stations to describe.")
    # Camber is signed: a reflexed section's dominant camber is negative, so
    # take the extreme by magnitude and keep its sign.
    camber_index = int(np.abs(camber).argmax())
    thickness_index = int(thickness.argmax())
    upper, lower = profile.surfaces()
    final_run = x[-1] - x[-2]
    upper_slope = (upper[-1, 1] - upper[-2, 1]) / final_run
    lower_slope = (lower[-1, 1] - lower[-2, 1]) / final_run
    return {
        "max_thickness": float(thickness[thickness_index]),
        "max_thickness_x": float(x[thickness_index]),
        "max_camber": float(camber[camber_index]),
        "max_camber_x": float(x[camber_index]),
        "leading_edge_radius": _leading_edge_radius(x, thickness),
        # Included wedge angle between the surfaces, positive for a normal section.
        "trailing_edge_angle": float(np.arctan(lower_slope) - np.arctan(upper_slope)),
        "trailing_edge_thickness": float(thickness[-1]),
        # Enclosed area is the integral of thickness; trapezoid by hand keeps
        # this independent of the numpy trapz/trapezoid rename.
        "area": float(np.sum(np.diff(x) * (thickness[:-1] + thickness[1:]) / 2)),
    }


def describe_surfaces(upper, lower, n_samples: int = 50) -> dict[str, float]:
    """Sample one airfoil and describe it in a single step."""
    return describe_profile(sample_profile(upper, lower, n_samples))
