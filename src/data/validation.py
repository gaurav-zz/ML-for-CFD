"""Validation checks for aerodynamic tabular data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_dataframe(frame: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    """Raise ValueError with actionable diagnostics when data violate basic assumptions."""
    if frame.empty:
        raise ValueError("Dataset is empty.")
    required_columns = required_columns or ["Alpha", "Re", "Cl", "Cd"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(frame.columns)}")
    if frame.duplicated().any():
        raise ValueError("Dataset contains duplicate records.")
    numeric = frame.select_dtypes(include=np.number)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Dataset contains NaN or infinite numeric values.")
    if (frame["Re"] <= 0).any():
        raise ValueError("Reynolds number must be positive.")
    if (frame["Cd"] < 0).any():
        raise ValueError("Drag coefficient Cd cannot be negative.")


def validate_geometry(coordinates: np.ndarray, tolerance: float = 1e-6) -> None:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 4:
        raise ValueError("Geometry must be an (n, 2) array with at least four points.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Geometry contains NaN or infinite values.")
    chord = coordinates[:, 0].max() - coordinates[:, 0].min()
    if chord <= 0 or not np.isclose(chord, 1.0, atol=tolerance):
        raise ValueError("Geometry must be normalized to a unit chord.")
