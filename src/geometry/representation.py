"""Fixed-length geometry representations for classical ML."""

from __future__ import annotations

import numpy as np


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float).copy()
    x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    if x_max <= x_min:
        raise ValueError("Geometry has zero chord length.")
    coordinates[:, 0] = (coordinates[:, 0] - x_min) / (x_max - x_min)
    coordinates[:, 1] /= x_max - x_min
    coordinates[:, 1] -= coordinates[:, 1].mean()
    return coordinates


def sample_surface(surface: np.ndarray, n_samples: int) -> np.ndarray:
    surface = normalize_coordinates(surface)
    order = np.argsort(surface[:, 0])
    x = np.linspace(0, 1, n_samples)
    return np.column_stack((x, np.interp(x, surface[order, 0], surface[order, 1])))


def flatten_geometry(upper: np.ndarray, lower: np.ndarray, n_samples: int = 50) -> np.ndarray:
    return np.concatenate((sample_surface(upper, n_samples)[:, 1], sample_surface(lower, n_samples)[:, 1]))
