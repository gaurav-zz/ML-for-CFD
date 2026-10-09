import hashlib
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import geometry_aware_split


@pytest.fixture
def comparison_cells():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "03_model_comparison.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}


def test_comparison_notebook_cells_compile(comparison_cells):
    for cell_id, source in comparison_cells.items():
        compile(source, f"comparison:{cell_id}", "exec")


def test_comparison_notebook_runs_the_complete_queue_on_synthetic_data(comparison_cells, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    geometry = np.repeat(np.arange(10), 4)
    alpha = np.tile([-4.0, -1.0, 2.0, 5.0], 10)
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
    processed = tmp_path / "synthetic.csv"
    frame.to_csv(processed, index=False)
    split = geometry_aware_split(frame, frame["geometry_id"])
    baseline = {
        "git_commit": "a" * 40,
        "dataset": {"sha256": hashlib.sha256(processed.read_bytes()).hexdigest(), "rows": len(frame)},
        "feature_columns": ["geom_000", "geom_001", "Alpha", "Re", "Ncrit"],
        "target_columns": ["Cl", "Cd"],
        "split_geometries": {
            name: sorted(getattr(split, name).geometry_id.unique())
            for name in ("train", "validation", "test")
        },
        "split_rows": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
    }
    drive_root = tmp_path / "drive"
    baseline_path = drive_root / "experiments" / "baseline" / "20261003T120000Z" / "run.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    namespace = {
        "Path": Path, "sys": sys, "SOURCE_REVISION": "b" * 40,
        "DRIVE_ROOT": drive_root, "PROCESSED_PATH": processed, "BASELINE_RUN": None,
        "display": lambda *args: None,
    }

    # Only cloud setup is skipped. Loading, splitting, fitting, selection, and saving are real.
    for cell_id in ("comparison-data", "comparison-queue"):
        exec(compile(comparison_cells[cell_id], f"comparison:{cell_id}", "exec"), namespace)
    assert namespace["BASELINE_RUN"] == baseline_path
    configs = namespace["MODEL_CONFIGS"]
    assert [config["name"] for config in configs] == ["linear", "gradient_boosting", "random_forest", "mlp"]
    configs[1]["parameters"].update(max_iter=3, max_leaf_nodes=4)
    configs[2]["parameters"].update(n_estimators=3, max_depth=3, max_leaf_nodes=8, n_jobs=1)
    configs[3]["parameters"].update(hidden_layers=(8,), max_iter=3, batch_size=8)
    for cell_id in ("comparison-train", "validation-results", "test-results", "test-plots"):
        exec(compile(comparison_cells[cell_id], f"comparison:{cell_id}", "exec"), namespace)

    output = namespace["RUN_OUTPUT"]
    assert output.is_relative_to(drive_root / "experiments" / "model_comparison")
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["run_metadata"]["dataset"]["sha256"] == baseline["dataset"]["sha256"]
    assert manifest["run_metadata"]["split_geometries"] == baseline["split_geometries"]
    assert manifest["run_metadata"]["feature_columns"] == baseline["feature_columns"]
    assert manifest["run_metadata"]["baseline_run"] == str(baseline_path)
    assert manifest["run_metadata"]["git_commit"] == "b" * 40
    validation = pd.read_csv(output / "comparison.csv")
    assert len(validation) == 8
    assert set(validation.model) == {config["name"] for config in configs}
    assert (validation.n_predictions == len(split.validation)).all()
    assert np.isfinite(validation[["mae", "rmse", "r2"]].to_numpy()).all()
    for config in configs:
        name = config["name"]
        assert manifest["results"][name]["status"] == "success"
        assert (output / name / f"{name}.joblib").is_file()
        assert json.loads((output / name / "result.json").read_text()) == manifest["results"][name]
    assert manifest["results"]["mlp"]["warnings"]
    test = pd.read_csv(output / "test_metrics.csv")
    assert len(test) == 2
    assert (test.n_predictions == len(split.test)).all()
    for target, model_name in manifest["selected_models"].items():
        scores = validation[validation.target == target]
        assert scores.loc[scores.model == model_name, "rmse"].iloc[0] == scores.rmse.min()
        assert test.loc[test.target == target, "model"].iloc[0] == model_name
    assert json.loads((output / "test_evaluation.json").read_text())["selected_models"] == manifest["selected_models"]
    for name in ("validation_rmse.png", "selected_predictions.png"):
        assert (output / name).stat().st_size > 0


def test_comparison_notebook_reports_missing_baseline(comparison_cells, tmp_path):
    namespace = {"Path": Path, "DRIVE_ROOT": tmp_path, "BASELINE_RUN": None}
    with pytest.raises(FileNotFoundError, match="Complete notebook 02 first"):
        exec(compile(comparison_cells["comparison-data"], "comparison:missing-baseline", "exec"), namespace)
