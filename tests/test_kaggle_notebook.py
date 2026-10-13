import json
from pathlib import Path
import sys
import zipfile

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.data.loaders import load_dataset_tables, save_dataset_tables
from src.geometry.naca import generate_naca4

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "v2_experiment_kaggle.ipynb"
# Enough sections for a three-way geometry-disjoint split with two held out.
CODES = ["0006", "0009", "0012", "0015", "0018", "0021", "1408", "2412", "4415", "6409"]


@pytest.fixture
def cells():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}


@pytest.fixture
def kaggle_inputs(tmp_path):
    """Build real v2 tables from synthetic NACA sections, plus a frozen split."""
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for code in CODES:
            upper, lower = generate_naca4(code, 60)
            polars = [
                {
                    "alpha": float(alpha), "Re": float(reynolds), "Ncrit": 9.0,
                    # Smooth, learnable physics so the queue has signal to fit.
                    "Cl": 0.1 * alpha + int(code[0]) / 50,
                    "Cd": 0.006 + 0.0004 * alpha**2 + int(code[2:]) / 2000,
                    "Cm": -0.02 - int(code[0]) / 200,
                }
                for alpha in (-16.0, -8.0, -2.0, 0.0, 2.0, 8.0, 16.0)
                for reynolds in (5e4, 1e6, 5e6)
            ]
            bundle.writestr(f"json/naca{code}-il.json", json.dumps({
                "name": f"NACA {code}",
                "xss": upper[:, 0].tolist(), "yss": upper[:, 1].tolist(),
                "xps": lower[:, 0].tolist(), "yps": lower[:, 1].tolist(),
                "polars": polars,
            }))

    dataset_dir = tmp_path / "input" / "airfoil-tables"
    dataset_dir.mkdir(parents=True)
    save_dataset_tables(load_dataset_tables(archive, geometry_samples=12), dataset_dir)

    identifiers = [f"naca{code}-il" for code in CODES]
    project_root = tmp_path / "repo"
    (project_root / "configs").mkdir(parents=True)
    (project_root / "configs" / "frozen_split.json").write_text(json.dumps({
        "git_commit": "a" * 40,
        "split_geometries": {
            "train": identifiers[:6], "validation": identifiers[6:8], "test": identifiers[8:],
        },
        "split_rows": {"train": 126, "validation": 42, "test": 42},
    }))
    return dataset_dir, project_root


def test_every_code_cell_compiles(cells):
    for cell_id, source in cells.items():
        compile(source, f"kaggle:{cell_id}", "exec")


def test_setup_cell_targets_kaggle_paths_not_drive(cells):
    """Kaggle has no Drive mount; a leftover mount call would fail at runtime."""
    setup = cells["kaggle-setup"]
    assert "google.colab" not in setup and "drive.mount" not in setup
    assert "/kaggle/temp" in setup
    # The checkout must not live in /kaggle/working, which becomes kernel output.
    statements = [line for line in setup.splitlines() if not line.strip().startswith("#")]
    assert not any("/kaggle/working" in line for line in statements)
    assert "/kaggle/working" in cells["v2-config"]
    assert "/kaggle/input" in cells["v2-resolve"]


def test_notebook_runs_the_whole_v2_experiment_on_synthetic_tables(
    cells, kaggle_inputs, tmp_path, monkeypatch,
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    dataset_dir, project_root = kaggle_inputs
    namespace = {
        "Path": Path, "sys": sys, "SOURCE_REVISION": "b" * 40,
        "PROJECT_ROOT": project_root, "DATASET_DIR": dataset_dir,
        "OUTPUT_ROOT": tmp_path / "working", "display": lambda *args: None,
    }

    # Only the clone/pip cell is skipped; loading, fitting, selection, the frozen
    # test evaluation, and the analysis all run for real.
    exec(compile(cells["v2-resolve"], "kaggle:v2-resolve", "exec"), namespace)
    assert namespace["GEOMETRY_PATH"].is_file() and namespace["POLARS_PATH"].is_file()

    exec(compile(cells["v2-data"], "kaggle:v2-data", "exec"), namespace)
    assert namespace["target_columns"] == ["Cl", "Cd", "Cm"]
    features = namespace["feature_columns"]
    assert "geometry_id" not in features
    assert features[-5:] == ["Alpha", "alpha_sin", "alpha_cos", "log10_re", "Ncrit"]
    split = namespace["split"]
    assert split.test.geometry_id.nunique() == 2

    exec(compile(cells["v2-queue"], "kaggle:v2-queue", "exec"), namespace)
    configs = namespace["MODEL_CONFIGS"]
    assert [config["name"] for config in configs] == [
        "linear", "gradient_boosting", "random_forest", "mlp",
    ]
    assert namespace["TARGET_TRANSFORMS"] == {"Cl": "identity", "Cd": "log", "Cm": "identity"}
    # Shrink the queue so the test stays fast; the code paths are unchanged.
    configs[1]["parameters"].update(max_iter=3, max_leaf_nodes=4)
    configs[2]["parameters"].update(n_estimators=3, max_depth=3, max_leaf_nodes=8, n_jobs=1)
    configs[3]["parameters"].update(hidden_layers=(8,), max_iter=3, batch_size=16)

    for cell_id in ("v2-train", "validation-results", "test-results", "v2-analysis", "analysis-plots"):
        exec(compile(cells[cell_id], f"kaggle:{cell_id}", "exec"), namespace)

    output = namespace["RUN_OUTPUT"]
    assert output.is_relative_to(tmp_path / "working")
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["target_transforms"] == {"Cl": "identity", "Cd": "log", "Cm": "identity"}
    assert manifest["run_metadata"]["platform"] == "kaggle"
    assert manifest["run_metadata"]["git_commit"] == "b" * 40
    assert manifest["run_metadata"]["dataset"]["polar_rows"] == 210
    assert set(manifest["selected_models"]) == {"Cl", "Cd", "Cm"}

    validation = pd.read_csv(output / "comparison.csv")
    assert len(validation) == 12
    assert np.isfinite(validation[["mae", "rmse"]].to_numpy()).all()
    # Drag is fitted in log space but every reported number is physical, so the
    # error sits on the scale of Cd itself rather than of log(Cd).
    drag = validation[validation.target == "Cd"]
    assert (drag.rmse < 0.1).all()

    test = pd.read_csv(output / "test_metrics.csv")
    assert test.target.tolist() == ["Cl", "Cd", "Cm"]
    assert (test.n_predictions == len(split.test)).all()

    report = json.loads((output / "analysis.json").read_text())
    assert report["n_geometries"] == 2
    assert report["n_rows"] == len(split.test)
    for target in ("Cl", "Cd", "Cm"):
        regimes = {record["regime"] for record in report["targets"][target]["regime"]}
        assert regimes == {"pre-stall", "post-stall"}
        assert len(report["targets"][target]["reynolds"]) == 3
        assert report["targets"][target]["worst_geometries"]
    assert report["lift_to_drag"]["n_rows"] > 0

    for name in ("validation_rmse.png", "regime_and_reynolds.png", "lift_to_drag.png"):
        assert (output / name).stat().st_size > 0


def test_notebook_reports_a_missing_dataset_clearly(cells, tmp_path):
    namespace = {
        "Path": Path, "PROJECT_ROOT": tmp_path, "DATASET_DIR": tmp_path / "absent",
        "OUTPUT_ROOT": tmp_path,
    }
    with pytest.raises(FileNotFoundError, match="Required input not found"):
        exec(compile(cells["v2-resolve"], "kaggle:missing-input", "exec"), namespace)
