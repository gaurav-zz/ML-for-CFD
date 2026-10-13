import hashlib
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import geometry_aware_split
from src.training.validation_reports import write_validation_reports
from src.utils.sync_results import REPORT_FILES, sync_results


@pytest.fixture
def cells():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "05_mlp_validation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}


def test_validation_notebook_compiles_with_test_opt_in(cells):
    for name, source in cells.items():
        compile(source, f"validation:{name}", "exec")
    assert "RUN_FINAL_TEST = False" in cells["validation-config"]


def test_validation_notebook_runs_syncs_and_reopens_without_fitting(cells, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    geometry = np.repeat(np.arange(10), 4)
    alpha = np.tile([-4., -1., 2., 5.], 10)
    frame = pd.DataFrame({
        "geometry_id": [f"foil-{index}" for index in geometry],
        "geom_000": geometry / 100, "geom_001": geometry / 200,
        "Alpha": alpha, "Re": 1_000_000., "Ncrit": 9.,
        "Cl": .1 * alpha + geometry / 100,
        "Cd": .01 + .001 * alpha**2 + geometry / 1000,
    })
    data = tmp_path / "synthetic.csv"
    frame.to_csv(data, index=False)
    split = geometry_aware_split(frame, frame.geometry_id)
    baseline = {
        "git_commit": "a" * 40,
        "dataset": {"sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "rows": len(frame)},
        "feature_columns": ["geom_000", "geom_001", "Alpha", "Re", "Ncrit"],
        "target_columns": ["Cl", "Cd"],
        "split_geometries": {
            name: sorted(getattr(split, name).geometry_id.unique()) for name in ("train", "validation", "test")
        },
        "split_rows": {name: len(getattr(split, name)) for name in ("train", "validation", "test")},
    }
    drive = tmp_path / "drive"
    baseline_path = drive / "experiments" / "baseline" / "20261003T120000000000Z" / "run.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps(baseline))
    namespace = {
        "Path": Path, "sys": sys, "SOURCE_REVISION": "b" * 40,
        "DRIVE_ROOT": drive, "PROCESSED_PATH": data, "BASELINE_RUN": None,
        "EXISTING_RUN": None, "RUN_FINAL_TEST": False, "display": lambda *args: None,
    }

    def execute(name):
        exec(compile(cells[name], f"validation:{name}", "exec"), namespace)

    execute("validation-data")
    execute("validation-queue")
    assert [config["id"] for config in namespace["MODEL_CONFIGS"]] == ["mlp_val_control", "mlp_val_low_lr"]
    for config in namespace["MODEL_CONFIGS"]:
        config["parameters"].update(hidden_layers=(4,), batch_size=8)
        config["validation_training"].update(max_epochs=4, min_epochs=1, check_every=1, patience=None)
    execute("validation-train")
    execute("validation-reports")
    namespace["evaluate_selected_models"] = lambda *args, **kwargs: pytest.fail("Test evaluation must remain opt-in.")
    execute("validation-final-test")
    output = namespace["RUN_OUTPUT"]
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "completed"
    assert report["run_metadata"]["split_geometries"] == baseline["split_geometries"]
    assert not (output / "test_metrics.csv").exists()
    history = pd.read_csv(output / "validation_history.csv")
    assert len(history) == 16
    assert history.epoch.max() == 4
    for result in report["results"].values():
        assert result["training"]["epochs_completed"] == 4
        assert result["training"]["stop_reason"] == "max_epochs"
        assert 1 <= result["training"]["best_epoch"] <= 4
    for filename in REPORT_FILES["mlp_validation"]:
        assert (output / filename).stat().st_size > 0
    local = tmp_path / "local_reports"
    snapshot = local / "mlp_validation" / output.name
    assert snapshot in sync_results(drive / "experiments", local)
    assert {path.name for path in snapshot.iterdir()} == set(REPORT_FILES["mlp_validation"])
    with pytest.raises(ValueError, match="validation geometries"):
        write_validation_reports(output, split.test, baseline["feature_columns"], baseline["target_columns"])

    namespace.update(EXISTING_RUN=output, BASELINE_RUN=None, RUN_FINAL_TEST=True)
    execute("validation-data")
    execute("validation-queue")
    namespace["compare_models"] = lambda *args, **kwargs: pytest.fail("Existing experiments must not retrain.")
    execute("validation-train")
    execute("validation-reports")
    for filename in REPORT_FILES["mlp_validation"]:
        assert (snapshot / filename).read_bytes() == (output / filename).read_bytes()
    execute("validation-final-test")
    test = pd.read_csv(output / "test_metrics.csv")
    assert (test.n_predictions == len(split.test)).all()
    assert dict(zip(test.target, test.model)) == report["selected_models"]
    final_snapshot = local / "mlp_validation" / f"{output.name}_with_test"
    assert final_snapshot in sync_results(drive / "experiments", local)
    assert not (snapshot / "test_metrics.csv").exists()

    report["run_metadata"]["feature_columns"] = ["wrong"]
    (output / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="feature_columns"):
        execute("validation-data")
