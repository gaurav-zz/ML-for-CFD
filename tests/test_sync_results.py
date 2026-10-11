import json
from pathlib import Path

import pytest

from src.utils import sync_results as reports


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "drive" / "experiments"
    for experiment in ("baseline", "model_comparison"):
        filenames = reports.REPORT_FILES[experiment]
        run = root / experiment / "20261009T111211403902Z"
        run.mkdir(parents=True)
        for name in filenames:
            if name.endswith(".png"):
                content = b"\x89PNG\r\n\x1a\n" + b"image bytes" + b"IEND\xaeB`\x82"
            elif name.endswith(".json"):
                content = json.dumps({"status": "completed", "selected_models": {"Cl": "mlp", "Cd": "mlp"}}).encode()
            else:
                content = b"target,rmse\nCl,0.116122\nCd,0.008957\n"
            (run / name).write_bytes(content)
        (run / "mlp").mkdir()
        (run / "mlp" / "mlp.joblib").write_bytes(b"must not be read")
        (run / "raw_dataset.csv").write_bytes(b"must not be read")
    return root


def test_sync_copies_only_final_reports_and_is_idempotent(source, tmp_path, monkeypatch):
    original_open = Path.open

    def open_only_reports(path, mode="r", *args, **kwargs):
        if "r" in mode:
            assert path.name in {name for names in reports.REPORT_FILES.values() for name in names}
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_only_reports)
    destination = tmp_path / "workspace" / "results"
    imported = reports.sync_results(source, destination)
    assert len(imported) == 2
    for output in imported:
        expected = reports.REPORT_FILES[output.parent.name]
        assert {path.name for path in output.iterdir()} == set(expected)
        for name in expected:
            assert (output / name).read_bytes() == (source / output.parent.name / output.name / name).read_bytes()
    mtimes = {path: path.stat().st_mtime_ns for output in imported for path in output.iterdir()}
    assert reports.sync_results(source, destination) == imported
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes
    assert not list(destination.glob("**/.report-*"))


@pytest.mark.parametrize("incomplete", ["missing_file", "running", "partial_png", "partial_json", "wrong_json_type"])
def test_sync_skips_incomplete_comparisons(source, tmp_path, incomplete):
    run = source / "model_comparison" / "20261009T111211403902Z"
    if incomplete == "missing_file":
        (run / "test_metrics.csv").unlink()
    elif incomplete == "running":
        (run / "run.json").write_text(json.dumps({"status": "running"}))
    elif incomplete == "partial_png":
        (run / "selected_predictions.png").write_bytes(b"\x89PNG\r\n\x1a\npartial")
    elif incomplete == "partial_json":
        (run / "run.json").write_text('{"status":')
    else:
        (run / "run.json").write_text("[]")
    imported = reports.sync_results(source, tmp_path / "reports")
    assert [path.parent.name for path in imported] == ["baseline"]


def test_sync_refuses_changed_local_reports(source, tmp_path):
    destination = tmp_path / "reports"
    imported = reports.sync_results(source, destination)
    changed = imported[0] / "metrics.csv"
    changed.write_text("user edits\n")
    with pytest.raises(FileExistsError, match="different local report"):
        reports.sync_results(source, destination)
    assert changed.read_text() == "user edits\n"


def test_sync_rejects_mismatched_selection(source, tmp_path):
    path = source / "model_comparison" / "20261009T111211403902Z" / "test_evaluation.json"
    path.write_text(json.dumps({"selected_models": {"Cl": "linear", "Cd": "linear"}}))
    with pytest.raises(ValueError, match="selections disagree"):
        reports.sync_results(source, tmp_path / "reports")


def test_sync_does_not_follow_symlinks_or_read_oversized_files(source, tmp_path, monkeypatch):
    run = source / "baseline" / "20261009T111211403902Z"
    metrics = run / "metrics.csv"
    original = metrics.read_bytes()
    metrics.unlink()
    metrics.symlink_to(run / "raw_dataset.csv")
    with pytest.raises(ValueError, match="symlinked"):
        reports.sync_results(source, tmp_path / "reports")
    metrics.unlink()
    metrics.write_bytes(original)
    monkeypatch.setattr(reports, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="limit"):
        reports.sync_results(source, tmp_path / "reports")


def test_sync_rejects_overlapping_or_missing_paths(source, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        reports.sync_results(source, source / "reports")
    with pytest.raises(FileNotFoundError, match="unavailable"):
        reports.sync_results(tmp_path / "missing", tmp_path / "reports")


def test_source_discovery_is_limited_to_project_experiments(tmp_path):
    expected = tmp_path / "GoogleDrive-test" / "My Drive" / "airfoil-ml-surrogate" / "experiments"
    expected.mkdir(parents=True)
    (tmp_path / "GoogleDrive-test" / "My Drive" / "other-project").mkdir()
    assert reports.discover_sources(tmp_path) == [expected]


def test_sync_refuses_symlinked_experiment_folders(source, tmp_path):
    linked_source = tmp_path / "linked-drive"
    linked_source.mkdir()
    (linked_source / "baseline").symlink_to(source / "baseline", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked experiment"):
        reports.sync_results(linked_source, tmp_path / "reports")


@pytest.mark.parametrize("link_parent", [False, True])
def test_sync_refuses_destination_symlinks(source, tmp_path, link_parent):
    destination = tmp_path / "reports"
    destination.mkdir()
    if link_parent:
        (destination / "baseline").symlink_to(tmp_path / "outside", target_is_directory=True)
    else:
        (destination / "baseline").mkdir()
        (destination / "baseline" / "20261009T111211403902Z").symlink_to(tmp_path / "outside")
    with pytest.raises((ValueError, FileExistsError), match="symlinked destination|different local report"):
        reports.sync_results(source, destination)
    assert not (tmp_path / "outside").exists()


def test_interrupted_copy_does_not_publish_partial_report(source, tmp_path, monkeypatch):
    original_write = Path.write_bytes

    def fail_mid_copy(path, data):
        if path.parent.name.startswith(".report-") and path.name == "metrics.csv":
            raise OSError("simulated copy failure")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_mid_copy)
    destination = tmp_path / "reports"
    with pytest.raises(OSError, match="simulated copy failure"):
        reports.sync_results(source, destination)
    assert not (destination / "baseline" / "20261009T111211403902Z").exists()
    assert not list(destination.glob("**/.report-*"))


def test_validation_only_improvements_sync_before_optional_final_test(source, tmp_path):
    run = source / "mlp_improvement" / "20261011T120000000000Z"
    run.mkdir(parents=True)
    selected = {"Cl": "mlp_200", "Cd": "mlp_400"}
    for name in reports.REPORT_FILES["mlp_improvement"]:
        if name.endswith(".json"):
            (run / name).write_text(json.dumps({
                "status": "completed", "selected_models": selected, "evaluation_scope": "validation",
            }))
        elif name.endswith(".png"):
            (run / name).write_bytes(b"\x89PNG\r\n\x1a\nimage bytesIEND\xaeB`\x82")
        else:
            (run / name).write_text("target,rmse\nCl,0.1\n")
    destination = tmp_path / "reports"
    imported = reports.sync_results(source, destination)
    validation_copy = destination / "mlp_improvement" / run.name
    assert validation_copy in imported
    assert not (validation_copy / "test_metrics.csv").exists()
    before = {path.name: path.read_bytes() for path in validation_copy.iterdir()}

    (run / "test_evaluation.json").write_text(json.dumps({"selected_models": selected}))
    (run / "test_metrics.csv").write_text("target,rmse\nCl,0.12\n")
    imported = reports.sync_results(source, destination)
    final_copy = destination / "mlp_improvement" / f"{run.name}_with_test"
    assert final_copy in imported
    assert (final_copy / "test_metrics.csv").read_bytes() == (run / "test_metrics.csv").read_bytes()
    assert {path.name: path.read_bytes() for path in validation_copy.iterdir()} == before
