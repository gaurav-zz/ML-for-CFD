from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json
import sys

import joblib
import nbformat
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import geometry_aware_split
from src.training.evaluate import evaluate_model
from src.training.train import train_model


@pytest.fixture
def baseline_cells():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "02_baseline_model.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}


def test_baseline_notebook_cells_compile(baseline_cells):
    for cell_id, source in baseline_cells.items():
        compile(source, f"baseline:{cell_id}", "exec")


@pytest.mark.parametrize("n_geometries,rows_per_geometry", [(5, 1), (10, 4)])
def test_baseline_persists_and_reloads_a_synthetic_run(
    baseline_cells, tmp_path, monkeypatch, n_geometries, rows_per_geometry,
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    geometry = np.repeat(np.arange(n_geometries), rows_per_geometry)
    alpha = np.tile(np.linspace(-4.0, 5.0, rows_per_geometry), n_geometries)
    frame = pd.DataFrame({
        "geometry_id": [f"foil-{index}" for index in geometry],
        "geom_000": geometry / 100,
        "geom_001": geometry / 200,
        "Alpha": alpha,
        "Re": 1_000_000.0,
        "Ncrit": 9.0,
        "Cl": 0.1 * alpha + geometry / 100,
        "Cd": 0.01 + 0.001 * alpha**2 + geometry / 1000,
    })
    processed_path = tmp_path / "synthetic.csv"
    frame.to_csv(processed_path, index=False)
    dataset_digest = hashlib.sha256(processed_path.read_bytes())
    namespace = {
        "df": frame,
        "pd": pd,
        "datetime": datetime,
        "timezone": timezone,
        "version": version,
        "json": json,
        "sys": sys,
        "subprocess": SimpleNamespace(check_output=Mock(side_effect=["a" * 40, ""] * 2)),
        "DRIVE_ROOT": tmp_path / "drive",
        "PROJECT_ROOT": tmp_path / "checkout",
        "PROCESSED_PATH": processed_path,
        "dataset_digest": dataset_digest,
        "geometry_aware_split": geometry_aware_split,
        "train_model": train_model,
        "evaluate_model": evaluate_model,
        "display": lambda *args: None,
    }

    # Exercise the real workflow cells without cloning, mounting Drive, or loading NASA data.
    for cell_id in ("c64b2d6e", "9158ae64", "07c20e1c", "08424ef2"):
        exec(compile(baseline_cells[cell_id], f"baseline:{cell_id}", "exec"), namespace)

    output = namespace["RUN_OUTPUT"]
    assert output.is_relative_to(tmp_path / "drive" / "experiments" / "baseline")
    assert {path.name for path in output.iterdir()} == {
        "linear.joblib", "metrics.csv", "run.json", "predicted_vs_actual.png",
    }
    assert (output / "predicted_vs_actual.png").stat().st_size > 0
    restored = joblib.load(output / "linear.joblib")
    np.testing.assert_allclose(
        restored.predict(namespace["X_test"]), namespace["model"].predict(namespace["X_test"]),
    )
    np.testing.assert_allclose(
        restored.named_steps["scaler"].mean_, namespace["X_train"].mean().to_numpy(),
    )
    assert restored.feature_names_in_.tolist() == namespace["feature_columns"]

    metrics = pd.read_csv(output / "metrics.csv")
    assert set(zip(metrics["split"], metrics["target"])) == {
        (partition, target) for partition in ("validation", "test") for target in ("Cl", "Cd")
    }
    assert np.isfinite(metrics[["mae", "rmse"]].to_numpy()).all()
    metadata = json.loads((output / "run.json").read_text())
    for partition in ("validation", "test"):
        r2_values = metrics.loc[metrics["split"] == partition, "r2"]
        if metadata["split_rows"][partition] == 1:
            assert r2_values.isna().all()
            assert all(scores["r2"] is None for scores in metadata["evaluation"][partition]["targets"].values())
        else:
            assert np.isfinite(r2_values).all()
    assert metadata["dataset"]["sha256"] == dataset_digest.hexdigest()
    assert metadata["dataset"]["rows"] == len(frame)
    assert metadata["target_columns"] == ["Cl", "Cd"]
    assert metadata["feature_columns"] == namespace["feature_columns"]
    assert metadata["git_commit"] == "a" * 40
    assert sum(metadata["split_rows"].values()) == len(frame)
    groups = [set(values) for values in metadata["split_geometries"].values()]
    assert set.union(*groups) == set(frame["geometry_id"])
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert set(metadata["evaluation"]["test"]["targets"]) == {"Cl", "Cd"}

    exec(compile(baseline_cells["9158ae64"], "baseline:rerun", "exec"), namespace)
    assert namespace["RUN_OUTPUT"] != output
    assert (output / "run.json").is_file()
    assert (namespace["RUN_OUTPUT"] / "linear.joblib").is_file()
