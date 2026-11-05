"""Fit-free inverse-clipping diagnostics, not calibrated uncertainty or accuracy gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.tail_risk import ROW_KEYS, SEEDS, TARGETS, _identity
from src.training.targets import DEFAULT_TARGET_TRANSFORMS, INVERSE_LIMITS, inverse_targets


def _real_array(value, shape):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype.kind not in "iuf"):
        raise ValueError(f"Expected a real numeric ndarray with shape {shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = value.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("All coordinates, labels and predictions must be finite; no rows may be dropped.")
    return array


def _sample_spread(values):
    # Normalize before centering/squaring, as in ensemble.aggregate_predictions.
    scale = np.max(np.abs(values), axis=0)
    scale = np.where(scale == 0, 1., scale)
    normalized = values / scale
    center = np.mean(normalized, axis=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        spread = np.sqrt(np.sum((normalized - center) ** 2, axis=0) / 2) * scale
    if not np.isfinite(spread).all():
        raise ValueError("Sample log-Cd spread must remain finite and representable.")
    return spread


def _summary(values):
    maximum = float(values.max())
    return {"min": float(values.min()), "max": maximum,
            "mean": float(np.mean(values / maximum) * maximum) if maximum else 0.}


def numerical_report(frame, fitting_predictions):
    """Return a JSON-safe report and separate, positionally aligned NumPy arrays.

    Predictions have shape (3, n, 3), seeds 42/43/44 and targets Cl/log(Cd)/Cm,
    already decoded from model target standardization, but not inverse-transformed.
    Inputs may be read-only and are never mutated. Labels are validated, not scored;
    no observed-Cd positivity or accuracy threshold is imposed here. Only the explicit
    ``safe_no_clipping`` flag screens clipping; spreads are uncalibrated diagnostics.
    Invalid/nonfinite evidence or unrepresentable sample spread raises ValueError.
    """
    if (not isinstance(frame, pd.DataFrame) or frame.empty
            or not frame.columns.is_unique or not frame.index.is_unique):
        raise ValueError("A nonempty frame with unique columns and row indices is required.")
    numeric_columns = (*ROW_KEYS[1:], "max_thickness", *TARGETS)
    missing = set(("geometry_id", *numeric_columns)).difference(frame.columns)
    if missing:
        raise ValueError(f"Numerical report lacks required columns: {sorted(missing)}.")
    for column in numeric_columns:
        _real_array(frame[column].to_numpy(), (len(frame),))
    digest = _identity(frame)
    members = _real_array(fitting_predictions, (3, len(frame), 3)).copy()
    physical = inverse_targets(
        members.reshape(-1, 3), TARGETS, DEFAULT_TARGET_TRANSFORMS,
    ).reshape(members.shape)
    if not np.isfinite(physical).all():
        raise ValueError("Inverse-transformed member predictions must remain finite.")

    raw_log_cd = members[:, :, 1]
    log_lower, log_upper = np.log(INVERSE_LIMITS)
    lower = raw_log_cd < log_lower
    upper = raw_log_cd > log_upper
    clipped = lower | upper
    rows_clipped = clipped.any(axis=0)
    raw_spread = _sample_spread(raw_log_cd)
    clipped_spread = _sample_spread(np.clip(raw_log_cd, log_lower, log_upper))
    hidden_spread = (clipped_spread == 0) & (raw_spread > 0)
    per_geometry = [
        {"geometry_id": name, "n_rows": int(len(positions)),
         "n_rows_any_clipped": int(rows_clipped[positions].sum()),
         "zero_clipped_spread_nonzero_raw_rows": int(hidden_spread[positions].sum())}
        for name, positions in frame.groupby("geometry_id", sort=True).indices.items()
    ]
    report = {
        "schema_version": 1, "target_names": list(TARGETS), "member_seeds": list(SEEDS),
        "n_rows": len(frame), "n_geometries": len(per_geometry), "row_sha256": digest,
        "inverse_limits": list(INVERSE_LIMITS), "status": "complete", "n_nonfinite": 0,
        "n_rows_any_clipped": int(rows_clipped.sum()),
        "n_member_values_clipped": int(clipped.sum()),
        "per_member": [
            {"seed": seed, "n_lower_clipped": int(lower[index].sum()),
             "n_upper_clipped": int(upper[index].sum())}
            for index, seed in enumerate(SEEDS)
        ],
        "per_geometry": per_geometry,
        "zero_clipped_spread_nonzero_raw_rows": int(hidden_spread.sum()),
        "raw_log_Cd_std_summary": _summary(raw_spread),
        "clipped_log_Cd_std_summary": _summary(clipped_spread),
        "safe_no_clipping": not bool(clipped.any()), "calibrated_uncertainty": False,
    }
    arrays = {
        "fitting_predictions": members, "physical_predictions": physical,
        "lower_clipped": lower, "upper_clipped": upper,
        "raw_log_Cd_std": raw_spread, "clipped_log_Cd_std": clipped_spread,
    }
    return report, arrays
