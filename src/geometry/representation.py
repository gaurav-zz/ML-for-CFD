"""Physical, fixed-length geometry representations for classical ML.

v1 normalized each surface independently and subtracted that surface's own mean
``y``. Two offsets per airfoil were discarded, and the difference between them
is exactly what sets thickness, so thickness and camber became unrecoverable
from the stored features: 43% of sampled stations ended up with the upper
surface below the lower one, and every airfoil had negative thickness
somewhere. Everything here derives a single transform from the combined point
set and never mean-centers ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_spacing(n_samples: int) -> np.ndarray:
    """Return unit-chord stations clustered at the leading and trailing edges.

    Uniform spacing under-resolves the leading edge, where curvature is highest
    and where thickness and camber change fastest.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    return (1 - np.cos(np.linspace(0, np.pi, n_samples))) / 2


def _as_points(points, label: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"{label} must be an (n, 2) array with at least two points.")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return points


def normalize_coordinates(coordinates) -> np.ndarray:
    """Scale a combined point set to unit chord without distorting ``y``.

    ``x`` and ``y`` are divided by the same chord length, so thickness and
    camber keep their physical meaning, and ``y`` is never mean-centered. This
    helper deliberately does not rotate: a combined point set does not identify
    the two trailing-edge points unambiguously. Use :func:`normalize_surfaces`
    when the surfaces are known separately, which is the case for every NASA
    record.
    """
    coordinates = _as_points(coordinates, "Geometry").copy()
    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    chord = x_max - x_min
    if chord <= 0:
        raise ValueError("Geometry has zero chord length.")
    coordinates[:, 0] = (coordinates[:, 0] - x_min) / chord
    coordinates[:, 1] = coordinates[:, 1] / chord
    return coordinates


def _oriented(surface: np.ndarray) -> np.ndarray:
    """Return one surface running nose to tail, reversing a tail-to-nose file."""
    return surface[::-1] if surface[0, 0] > surface[-1, 0] else surface


def normalize_surfaces(upper, lower) -> tuple[np.ndarray, np.ndarray]:
    """Align one airfoil to the unit chord with a single rigid transform.

    The leading edge is where the two surfaces meet and the trailing edge is
    the midpoint of their far ends; both surfaces are then rotated, translated,
    and scaled *together* so the chord line runs from ``(0, 0)`` to ``(1, 0)``.
    Applying one transform to both surfaces is what keeps
    ``thickness = y_upper - y_lower`` physically meaningful.

    Taking the leading edge from the shared surface endpoint matters: picking
    the combined point farthest from the trailing edge instead selects a point
    just around a rounded nose, which tilts the chord line by ~0.002 rad and
    biases camber several percent low.

    Coordinates are expected in the usual airfoil convention, with ``x``
    increasing from nose to tail as in the NASA archive; either point order is
    accepted. Within that convention the result is invariant to rotation,
    uniform scaling, and translation of the input.
    """
    upper = _oriented(_as_points(upper, "Upper surface"))
    lower = _oriented(_as_points(lower, "Lower surface"))
    leading = (upper[0] + lower[0]) / 2
    trailing = (upper[-1] + lower[-1]) / 2
    chord_vector = trailing - leading
    chord = float(np.hypot(*chord_vector))
    if chord <= 0:
        raise ValueError("Geometry has zero chord length.")
    cosine, sine = chord_vector / chord
    # Rotate by -angle(chord_vector) and scale, so the chord maps to (0,0)-(1,0).
    transform = np.array([[cosine, -sine], [sine, cosine]]) / chord
    return (upper - leading) @ transform, (lower - leading) @ transform


@dataclass(frozen=True)
class GeometryProfile:
    """Thickness and camber sampled on shared, cosine-spaced unit-chord stations.

    ``thickness`` and ``camber`` span the same information as upper/lower
    surface samples (``y_upper = camber + thickness / 2``) but are far better
    conditioned and directly physical. ``clipped_stations`` counts stations
    where the source coordinates crossed and thickness was clipped to zero.
    """

    x: np.ndarray
    thickness: np.ndarray
    camber: np.ndarray
    clipped_stations: int

    def surfaces(self) -> tuple[np.ndarray, np.ndarray]:
        """Invert the reparametrization back to upper and lower surfaces."""
        return (
            np.column_stack((self.x, self.camber + self.thickness / 2)),
            np.column_stack((self.x, self.camber - self.thickness / 2)),
        )


def _interpolate(surface: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Interpolate one surface onto shared stations, tolerating a wrapped nose."""
    order = np.argsort(surface[:, 0], kind="stable")
    return np.interp(x, surface[order, 0], surface[order, 1])


def sample_profile(upper, lower, n_samples: int = 50) -> GeometryProfile:
    """Sample thickness and camber for one airfoil on a single shared grid.

    Negative thickness is clipped to zero and counted rather than raising: the
    raw NASA archive holds a few coordinate artifacts of order 1e-4 chord
    (mostly Drela AG-series sections), which are not worth rejecting an airfoil
    over but must not be absorbed silently either.
    """
    upper, lower = normalize_surfaces(upper, lower)
    x = cosine_spacing(n_samples)
    y_upper, y_lower = _interpolate(upper, x), _interpolate(lower, x)
    thickness = y_upper - y_lower
    return GeometryProfile(
        x=x,
        thickness=np.clip(thickness, 0.0, None),
        camber=(y_upper + y_lower) / 2,
        clipped_stations=int((thickness < 0).sum()),
    )


def geometry_columns(n_samples: int = 50) -> list[str]:
    """Name the flattened profile features, thickness first and then camber.

    The names deliberately differ from v1's ``geom_000`` so a v1 feature list
    can never be mistaken for a v2 one.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    return (
        [f"thickness_{index:03d}" for index in range(n_samples)]
        + [f"camber_{index:03d}" for index in range(n_samples)]
    )


def flatten_geometry(upper, lower, n_samples: int = 50) -> np.ndarray:
    """Return the thickness/camber profile as one fixed-length vector."""
    profile = sample_profile(upper, lower, n_samples)
    return np.concatenate((profile.thickness, profile.camber))
