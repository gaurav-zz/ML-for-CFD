from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import geometry_aware_split
from src.utils.sync_results import REPORT_FILES, sync_results


@pytest.fixture
def cells():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "04_mlp_improvement.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}


def test_improvement_notebook_compiles_with_test_disabled_by_default(cells):
    for name, source in cells.items():
        compile(source, f"improvement:{name}", "exec")
    assert "RUN_FINAL_TEST = False" in cells["improvement-config"]


def test_improvement_notebook_validation_reports_reopening_and_opt_in_test(cells, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
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
        exec(compile(cells[name], f"improvement:{name}", "exec"), namespace)

    execute("improvement-data")
    execute("improvement-queue")
    assert [config["id"] for config in namespace["MODEL_CONFIGS"]] == ["mlp_100", "mlp_200", "mlp_400"]
    for config, budget in zip(namespace["MODEL_CONFIGS"], (2, 3, 4)):
        config["id"] = f"mlp_{budget}"
        config["parameters"].update(max_iter=budget, hidden_layers=(4,), batch_size=8)
    execute("improvement-train")
    execute("improvement-summary")
    execute("improvement-plots")
    namespace["evaluate_selected_models"] = lambda *args, **kwargs: pytest.fail("Test evaluation must be opt-in.")
    execute("improvement-final-test")

    output = namespace["RUN_OUTPUT"]
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "completed"
    assert report["run_metadata"]["split_geometries"] == baseline["split_geometries"]
    assert report["run_metadata"]["dataset"] == baseline["dataset"]
    assert set(report["results"]) == {"mlp_2", "mlp_3", "mlp_4"}
    assert not (output / "test_metrics.csv").exists()
    assert not (output / "test_evaluation.json").exists()
    summary = json.loads((output / "validation_summary.json").read_text())
    assert summary["evaluation_scope"] == "validation"
    assert summary["test_used_for_selection"] is False
    assert summary["validation_geometry_count"] == split.validation.geometry_id.nunique()
    assert len(summary["per_geometry_summary"]) == 6
    by_geometry = pd.read_csv(output / "validation_by_geometry.csv")
    assert set(by_geometry.geometry_id) == set(split.validation.geometry_id)
    assert set(by_geometry.geometry_id).isdisjoint(split.train.geometry_id)
    assert set(by_geometry.geometry_id).isdisjoint(split.test.geometry_id)
    for name in REPORT_FILES["mlp_improvement"]:
        assert (output / name).stat().st_size > 0
    local = tmp_path / "local_reports"
    validation_snapshot = local / "mlp_improvement" / output.name
    assert validation_snapshot in sync_results(drive / "experiments", local)
    assert {path.name for path in validation_snapshot.iterdir()} == set(REPORT_FILES["mlp_improvement"])

    # Restore an existing experiment and run final evaluation without fitting again.
    namespace.update(EXISTING_RUN=output, BASELINE_RUN=None, RUN_FINAL_TEST=True)
    execute("improvement-data")
    execute("improvement-queue")
    namespace["compare_models"] = lambda *args, **kwargs: pytest.fail("An existing experiment must not retrain.")
    before = (output / "run.json").read_bytes()
    execute("improvement-train")
    execute("improvement-summary")
    execute("improvement-plots")
    for name in REPORT_FILES["mlp_improvement"]:
        assert (output / name).read_bytes() == (validation_snapshot / name).read_bytes()
    assert validation_snapshot in sync_results(drive / "experiments", local)
    execute("improvement-final-test")
    assert (output / "run.json").read_bytes() == before
    final = pd.read_csv(output / "test_metrics.csv")
    assert (final.n_predictions == len(split.test)).all()
    assert dict(zip(final.target, final.model)) == report["selected_models"]
    final_snapshot = local / "mlp_improvement" / f"{output.name}_with_test"
    assert final_snapshot in sync_results(drive / "experiments", local)
    assert not (validation_snapshot / "test_metrics.csv").exists()
    assert (final_snapshot / "test_metrics.csv").is_file()

    legacy = deepcopy(report)
    for result in legacy["results"].values():
        for key in ("tol", "n_iter_no_change", "early_stopping"):
            del result["training"][key]
    (output / "run.json").write_text(json.dumps(legacy))
    displayed = []
    namespace["display"] = displayed.append
    execute("improvement-data")
    execute("improvement-train")
    displayed.clear()
    execute("improvement-summary")
    assert displayed[1]["tolerance"].tolist() == ["not recorded"] * 3
    assert displayed[1]["patience"].tolist() == ["not recorded"] * 3

    altered = deepcopy(report)
    altered["run_metadata"]["feature_columns"] = ["wrong"]
    (output / "run.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="feature_columns"):
        execute("improvement-data")
    altered = deepcopy(report)
    altered["run_metadata"]["packages"]["scikit-learn"] = "incompatible"
    (output / "run.json").write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="scikit-learn version"):
        execute("improvement-data")
