"""Flow features and design-matrix assembly for the v2 experiment.

The loader now stores geometry once per airfoil and polars once per operating
condition. This module joins the two and appends the flow features, so the
40 MB pair of tables only expands to a wide design matrix in memory, at the
point of use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geometry.descriptors import DESCRIPTOR_NAMES

GEOMETRY_KEY = "geometry_id"
ENGINEERED_FLOW_COLUMNS = ("alpha_sin", "alpha_cos", "log10_re")
# Alpha stays alongside its sine and cosine: the trigonometric pair carries the
# thin-aerofoil relationship, while raw degrees let tree models split cleanly
# on the post-stall boundary, which about 17% of rows fall beyond.
FLOW_COLUMNS = ("Alpha", *ENGINEERED_FLOW_COLUMNS)
OPTIONAL_FLOW_COLUMNS = ("Ncrit",)


def add_flow_features(polars: pd.DataFrame) -> pd.DataFrame:
    """Append scale-corrected flow features to a copy of the polars table.

    ``log10(Re)`` because the eight distinct Reynolds numbers span two decades,
    and raw values standard-scale terribly. ``sin``/``cos`` of ``Alpha`` because
    thin-aerofoil theory gives ``Cl ~ 2 pi sin(alpha)``, a relationship a linear
    model can use directly but cannot express from degrees alone.
    """
    missing = [column for column in ("Alpha", "Re") if column not in polars.columns]
    if missing:
        raise ValueError(f"Polars table is missing required columns: {missing}.")
    reynolds = polars["Re"].to_numpy(dtype=float)
    if not np.isfinite(reynolds).all() or (reynolds <= 0).any():
        raise ValueError("Reynolds number must be finite and positive to take its logarithm.")
    radians = np.radians(polars["Alpha"].to_numpy(dtype=float))
    if not np.isfinite(radians).all():
        raise ValueError("Alpha must be finite.")
    # Concatenate rather than assign: the joined frame already has ~110 columns
    # and 768k rows, where repeated inserts fragment it badly.
    engineered = pd.DataFrame(
        {
            "alpha_sin": np.sin(radians),
            "alpha_cos": np.cos(radians),
            "log10_re": np.log10(reynolds),
        },
        index=polars.index,
    )
    return pd.concat([polars, engineered], axis=1)


def profile_columns(geometry: pd.DataFrame) -> list[str]:
    """Return the station columns present in a geometry table, in grid order."""
    stations = [
        column for column in geometry.columns
        if column.startswith(("thickness_", "camber_")) and column.split("_")[-1].isdigit()
    ]
    return sorted(stations, key=lambda column: (column.startswith("camber_"), column))


def feature_names(geometry: pd.DataFrame, polars: pd.DataFrame) -> list[str]:
    """Order the v2 design-matrix columns: profile, descriptors, then flow.

    ``geometry_id`` is deliberately absent. It identifies the group the split is
    built on, so including it would leak the very thing the unseen-geometry
    claim rests on.
    """
    descriptors = [name for name in DESCRIPTOR_NAMES if name in geometry.columns]
    optional = [name for name in OPTIONAL_FLOW_COLUMNS if name in polars.columns]
    names = [*profile_columns(geometry), *descriptors, *FLOW_COLUMNS, *optional]
    if len(set(names)) != len(names):
        raise ValueError("Geometry and polars tables share a feature column name.")
    return names


def join_geometry(polars: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Attach each polar row's geometry, refusing silent row loss or duplication."""
    for name, frame in (("Polars", polars), ("Geometry", geometry)):
        if GEOMETRY_KEY not in frame.columns:
            raise ValueError(f"{name} table must contain a {GEOMETRY_KEY} column.")
        if frame.empty:
            raise ValueError(f"{name} table is empty.")
    if geometry[GEOMETRY_KEY].duplicated().any():
        raise ValueError(f"Geometry table must hold one row per {GEOMETRY_KEY}.")
    unmatched = set(polars[GEOMETRY_KEY]).difference(geometry[GEOMETRY_KEY])
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} geometry IDs in the polars table have no geometry row, "
            f"for example {sorted(unmatched)[:3]}. Regenerate both tables together."
        )
    shared = set(polars.columns).intersection(geometry.columns).difference({GEOMETRY_KEY})
    if shared:
        raise ValueError(f"Polars and geometry tables both define {sorted(shared)}.")
    joined = polars.merge(geometry, on=GEOMETRY_KEY, how="left", validate="many_to_one")
    if len(joined) != len(polars):
        raise ValueError("Joining geometry changed the row count.")
    return joined


def assemble_features(
    polars: pd.DataFrame, geometry: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the joined design matrix and its ordered feature-column names."""
    names = feature_names(geometry, polars)
    joined = add_flow_features(join_geometry(polars, geometry))
    missing = [name for name in names if name not in joined.columns]
    if missing:
        raise ValueError(f"Assembled frame is missing expected features: {missing}.")
    if not all(pd.api.types.is_numeric_dtype(joined[name]) for name in names):
        raise ValueError("All design-matrix features must be numeric.")
    if not np.isfinite(joined[names].to_numpy(dtype=float)).all():
        raise ValueError("Design matrix contains NaN or infinite values.")
    return joined, names
