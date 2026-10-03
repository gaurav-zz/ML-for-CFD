"""Train configured models from a prepared feature matrix."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from src.models.gradient_boosting import build_model as build_gb
from src.models.linear import build_model as build_linear
from src.models.mlp import build_model as build_mlp
from src.models.polynomial import build_model as build_polynomial
from src.models.random_forest import build_model as build_rf


MODEL_BUILDERS = {"linear": build_linear, "polynomial": build_polynomial, "random_forest": build_rf, "gradient_boosting": build_gb, "mlp": build_mlp}


def train_model(name: str, x_train, y_train, output_dir: str | Path = "results/models", **kwargs) -> tuple[object, float]:
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_BUILDERS)}")
    model = MODEL_BUILDERS[name](**kwargs)
    if name == "gradient_boosting" and np.ndim(y_train) == 2:
        model = MultiOutputRegressor(model)
    started = time.perf_counter()
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
