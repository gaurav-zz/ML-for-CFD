"""Per-target transforms, so each coefficient is fitted on a sensible scale.

``Cd`` spans a 223x range (0.00265 to 0.59) and its base-10 logarithm spans
-2.58 to -0.23. A squared loss on raw ``Cd`` is therefore dominated by stalled,
high-drag cases and barely constrains the low-drag cruise points the surrogate
is meant to be useful for: v1's headline ``Cd`` MAE of 0.0049 is roughly an 80%
relative error against a 1st-percentile ``Cd`` of 0.006.

Fitting ``log(Cd)`` fixes the scale. Every metric is still computed after
inverting back to physical units, so reported numbers stay comparable with v1
and with anything a reader would measure themselves.
"""

from __future__ import annotations

import numpy as np

# An extrapolating model can predict a log-drag of +400; exponentiating that
# overflows to inf and poisons every downstream metric. Clamp the inverse to a
# range that is already far outside any physical drag coefficient.
INVERSE_LIMITS = (1e-6, 1e3)


class IdentityTransform:
    """Fit a target directly, for coefficients that are already well scaled."""

    name = "identity"

    def forward(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)


class LogTransform:
    """Fit the natural logarithm of a strictly positive target."""

    name = "log"

    def forward(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Cannot log-transform non-finite target values.")
        if (values <= 0).any():
            raise ValueError(
                "Log-transformed targets must be strictly positive; "
                f"found a minimum of {values.min():.6g}."
            )
        return np.log(values)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.exp(np.clip(values, *np.log(INVERSE_LIMITS)))


TRANSFORMS = {"identity": IdentityTransform(), "log": LogTransform()}

# The mapping to pass for the v2 experiment. Cm, like Cl, is an order-one signed
# coefficient and needs no rescaling. Nothing applies this implicitly: a caller
# that passes no mapping fits its targets exactly as given, so enabling the log
# is always a visible decision recorded in run.json.
DEFAULT_TARGET_TRANSFORMS = {"Cl": "identity", "Cd": "log", "Cm": "identity"}


def resolve_transforms(target_names, mapping=None) -> list[object]:
    """Pick one transform per target from an explicit mapping.

    Targets the mapping does not name are fitted as given. Naming a target that
    is neither in use nor a known project target is an error rather than a
    silent no-op, because that is almost always a typo that would leave a target
    untransformed. This is why :data:`DEFAULT_TARGET_TRANSFORMS` can be passed
    wholesale even when only a subset of its targets is being fitted.
    """
    if isinstance(target_names, str) or not len(target_names):
        raise ValueError("Provide a nonempty sequence of target names.")
    mapping = mapping or {}
    unknown = set(mapping).difference(target_names).difference(DEFAULT_TARGET_TRANSFORMS)
    if unknown:
        raise ValueError(f"Transform mapping names targets that are not in use: {sorted(unknown)}.")
    resolved = []
    for name in target_names:
        choice = mapping.get(name, "identity")
        if choice not in TRANSFORMS:
            raise ValueError(
                f"Unknown transform {choice!r} for target {name!r}; choose from {sorted(TRANSFORMS)}."
            )
        resolved.append(TRANSFORMS[choice])
    return resolved


def transform_names(target_names, mapping=None) -> dict[str, str]:
    """Report the transform chosen per target, for run.json provenance."""
    return {
        name: transform.name
        for name, transform in zip(target_names, resolve_transforms(target_names, mapping))
    }


def _as_columns(values, target_names) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != len(target_names):
        raise ValueError(
            f"Expected one column per target {list(target_names)}; got shape {np.shape(values)}."
        )
    return values


def forward_targets(y, target_names, mapping=None) -> np.ndarray:
    """Map physical targets into fitting space, column by column."""
    values = _as_columns(y, target_names)
    transforms = resolve_transforms(target_names, mapping)
    return np.column_stack([
        transform.forward(values[:, index]) for index, transform in enumerate(transforms)
    ])


def inverse_targets(z, target_names, mapping=None) -> np.ndarray:
    """Map fitting-space values back to physical units, column by column."""
    values = _as_columns(z, target_names)
    transforms = resolve_transforms(target_names, mapping)
    return np.column_stack([
        transform.inverse(values[:, index]) for index, transform in enumerate(transforms)
    ])


class PhysicalPredictor:
    """Wrap a model fitted in transformed space so predictions are physical.

    This is what lets the existing evaluation and plotting paths stay unchanged:
    hand them this wrapper and the untransformed targets, and every metric comes
    out in physical ``Cl``/``Cd``/``Cm`` units.
    """

    def __init__(self, model, target_names, mapping=None):
        self.model = model
        self.target_names = list(target_names)
        # Resolve now, so a bad mapping fails here rather than at predict time.
        self.transforms = resolve_transforms(self.target_names, mapping)
        self.mapping = mapping

    def predict(self, x) -> np.ndarray:
        prediction = inverse_targets(self.model.predict(x), self.target_names, self.mapping)
        return prediction[:, 0] if len(self.target_names) == 1 else prediction

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{name}={transform.name}" for name, transform in zip(self.target_names, self.transforms)
        )
        return f"PhysicalPredictor({self.model.__class__.__name__}, {pairs})"
