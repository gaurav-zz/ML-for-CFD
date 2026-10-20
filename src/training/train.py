"""Train configured models from a prepared feature matrix."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.models.gradient_boosting import build_model as build_gb
from src.models.linear import build_model as build_linear
from src.models.mlp import build_model as build_mlp
from src.models.polynomial import build_model as build_polynomial
from src.models.random_forest import build_model as build_rf


def build_torch_mlp(**kwargs):
    # CPU workflows and portable prediction artifacts do not require PyTorch.
    from src.models.torch_mlp import build_model
    return build_model(**kwargs)


MODEL_BUILDERS = {
    "linear": build_linear, "polynomial": build_polynomial, "random_forest": build_rf,
    "gradient_boosting": build_gb, "mlp": build_mlp,
    "torch_mlp_mse": build_torch_mlp, "torch_mlp_huber": build_torch_mlp,
}


def train_model(
    name: str,
    x_train,
    y_train,
    output_dir: str | Path = "results/models",
    *,
    scale_targets: bool = False,
    validation_data=None,
    auxiliary_targets=None,
    validation_auxiliary_targets=None,
    **kwargs,
) -> tuple[object, float]:
    """Fit and save a model, optionally standardizing training targets only."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_BUILDERS)}")
    if (auxiliary_targets is not None or validation_auxiliary_targets is not None) and not name.startswith("torch_mlp_"):
        raise ValueError("Auxiliary labels are supported only by Torch MLP training.")
    model = MODEL_BUILDERS[name](**kwargs)
    if name.startswith("torch_mlp_") and scale_targets:
        raise ValueError("Torch MLP normalizes its own targets; use scale_targets=False.")
    if scale_targets:
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    if name == "gradient_boosting" and np.ndim(y_train) == 2:
        # Each target wrapper sees 1D y, including when the input has just one column.
        model = MultiOutputRegressor(model)
    started = time.perf_counter()
    if name.startswith("torch_mlp_"):
        fit_parameters = {"validation_data": validation_data}
        if auxiliary_targets is not None or validation_auxiliary_targets is not None:
            fit_parameters.update(auxiliary_targets=auxiliary_targets,
                                  validation_auxiliary_targets=validation_auxiliary_targets)
        model.fit(x_train, y_train, **fit_parameters)
    else:
        model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / f"{name}.joblib")
    return model, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--target", default="Cl")
    parser.add_argument("--model", default="linear", choices=sorted(MODEL_BUILDERS))
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    x = frame.drop(columns=[args.target])
    model, elapsed = train_model(args.model, x, frame[args.target])
    print(f"Trained {args.model} in {elapsed:.3f}s; saved under results/models ({model.__class__.__name__}).")


if __name__ == "__main__":
    main()
