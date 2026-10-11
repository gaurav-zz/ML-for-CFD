"""Validation checks for aerodynamic tabular data."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main() -> None:
    """Validate a processed CSV from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/airfoil_learning.csv"),
        help="Path to the processed CSV file.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {args.data}. "
            "Run the dataset loader first."
        )

    frame = pd.read_csv(args.data)
    validate_dataframe(frame)

    print("Dataset validation passed.")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {len(frame.columns):,}")
    if "geometry_id" in frame.columns:
        print(f"Unique geometries: {frame['geometry_id'].nunique():,}")


if __name__ == "__main__":
    main()
