import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from src.utils import kaggle_sync
from src.utils.kaggle_sync import (
    PUSH_FILES,
    kernel_status,
    pull_run,
    push_kernel,
    push_tables,
    stage_dataset,
    stage_kernel,
)


class FakeRunner:
    """Records Kaggle CLI invocations instead of making network calls."""

    def __init__(self, stdout="done", error=None):
        self.commands = []
        self.stdout = stdout
        self.error = error

    def __call__(self, command, *, check, capture_output, text):
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(stdout=self.stdout, returncode=0)


@pytest.fixture
def processed(tmp_path):
    directory = tmp_path / "processed"
    directory.mkdir()
    for name in PUSH_FILES:
        (directory / name).write_text(f"contents of {name}\n")
    # The real directory also holds large intermediates that must not be pushed.
    (directory / "airfoil_learning_v1.csv").write_text("x" * 5000)
    (directory / "scratch").mkdir()
    return directory


def test_staging_pushes_only_the_two_tables_and_the_report(processed, tmp_path):
    staging = stage_dataset("me/airfoil-v2", tmp_path / "stage", processed)
    assert {path.name for path in staging.iterdir()} == {*PUSH_FILES, "dataset-metadata.json"}
    metadata = json.loads((staging / "dataset-metadata.json").read_text())
    assert metadata["id"] == "me/airfoil-v2"
    assert len(metadata["title"]) >= 6
    assert metadata["licenses"] == [{"name": "CC0-1.0"}]


def test_staging_reports_missing_tables_actionably(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="src.data.loaders"):
        stage_dataset("me/airfoil-v2", tmp_path / "stage", empty)


@pytest.mark.parametrize("reference", ["noslash", "me/", "/slug", "me/slug/extra", "", None])
def test_malformed_references_are_rejected(reference, tmp_path):
    with pytest.raises(ValueError, match="owner/slug"):
        stage_dataset(reference, tmp_path / "stage", tmp_path)
    with pytest.raises(ValueError, match="owner/slug"):
        pull_run(reference, tmp_path / "out")


def test_push_versions_an_existing_dataset_and_cleans_up_staging(processed):
    runner = FakeRunner()
    push_tables("me/airfoil-v2", processed, message="fresh tables", runner=runner)
    assert len(runner.commands) == 1
    command = runner.commands[0]
    assert command[:3] == ["kaggle", "datasets", "version"]
    assert "fresh tables" in command
    assert "--dir-mode" in command and "skip" in command
    assert not (processed / ".kaggle-upload").exists()


def test_push_can_create_the_dataset_the_first_time(processed):
    runner = FakeRunner()
    push_tables("me/airfoil-v2", processed, create=True, runner=runner)
    assert runner.commands[0][:3] == ["kaggle", "datasets", "create"]


def test_push_removes_staging_even_when_the_cli_fails(processed):
    runner = FakeRunner(error=subprocess.CalledProcessError(1, "kaggle", stderr="403 Forbidden"))
    with pytest.raises(RuntimeError, match="403 Forbidden"):
        push_tables("me/airfoil-v2", processed, runner=runner)
    assert not (processed / ".kaggle-upload").exists()


def test_a_missing_cli_explains_how_to_install_it(processed):
    runner = FakeRunner(error=FileNotFoundError())
    with pytest.raises(RuntimeError, match="pip install kaggle"):
        push_tables("me/airfoil-v2", processed, runner=runner)


def test_pull_lands_each_run_in_its_own_directory(tmp_path):
    destination_root = tmp_path / "kaggle"

    def runner(command, *, check, capture_output, text):
        target = tmp_path / command[command.index("-p") + 1]
        (target / "run.json").write_text("{}")
        return SimpleNamespace(stdout="", returncode=0)

    destination = pull_run("me/airfoil-v2-experiment", destination_root, "run-001", runner=runner)
    assert destination == destination_root / "run-001"
    assert (destination / "run.json").is_file()
    # A finished run is never clobbered by a later pull.
    with pytest.raises(FileExistsError, match="already exists"):
        pull_run("me/airfoil-v2-experiment", destination_root, "run-001", runner=runner)


def test_pull_leaves_nothing_behind_when_the_kernel_has_no_output(tmp_path):
    runner = FakeRunner()
    with pytest.raises(RuntimeError, match="no output files"):
        pull_run("me/airfoil-v2-experiment", tmp_path / "kaggle", "run-002", runner=runner)
    assert not (tmp_path / "kaggle" / "run-002").exists()


def test_pull_leaves_nothing_behind_when_the_cli_fails(tmp_path):
    runner = FakeRunner(error=subprocess.CalledProcessError(1, "kaggle", stderr="404"))
    with pytest.raises(RuntimeError, match="404"):
        pull_run("me/airfoil-v2-experiment", tmp_path / "kaggle", "run-003", runner=runner)
    assert not (tmp_path / "kaggle" / "run-003").exists()


@pytest.mark.parametrize("run_id", ["../escape", "with/slash", "a" * 65, "sp ace"])
def test_pull_rejects_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError, match="short alphanumeric"):
        pull_run("me/airfoil-v2-experiment", tmp_path / "kaggle", run_id, runner=FakeRunner())


def test_default_run_id_is_a_utc_timestamp(tmp_path):
    def runner(command, *, check, capture_output, text):
        (tmp_path / command[command.index("-p") + 1] / "metrics.csv").write_text("a,b\n")
        return SimpleNamespace(stdout="", returncode=0)

    destination = pull_run("me/airfoil-v2-experiment", tmp_path / "kaggle", runner=runner)
    assert destination.name.endswith("Z")
    assert destination.name[:8].isdigit()


def test_push_files_track_the_loader_table_names():
    """Keeps the pushed set from drifting away from what the loader writes."""
    from src.data.loaders import GEOMETRY_TABLE, LOAD_REPORT, POLARS_TABLE

    assert set(kaggle_sync.PUSH_FILES) == {GEOMETRY_TABLE, POLARS_TABLE, LOAD_REPORT}


def test_kernel_metadata_enables_internet_and_attaches_the_dataset(tmp_path):
    """The notebook clones the source checkout, so it cannot run without egress."""
    staging = stage_kernel(
        "me/airfoil-v2-experiment", "me/airfoil-v2", tmp_path / "stage",
    )
    metadata = json.loads((staging / "kernel-metadata.json").read_text())
    assert metadata["id"] == "me/airfoil-v2-experiment"
    assert metadata["code_file"] == "v2_experiment_kaggle.ipynb"
    assert metadata["kernel_type"] == "notebook" and metadata["language"] == "python"
    assert metadata["enable_internet"] is True
    # Every model in the queue is CPU scikit-learn.
    assert metadata["enable_gpu"] is False and metadata["enable_tpu"] is False
    assert metadata["dataset_sources"] == ["me/airfoil-v2"]
    assert metadata["is_private"] is True
    assert len(metadata["title"]) >= 5
    # The notebook itself is uploaded alongside the metadata.
    assert (staging / "v2_experiment_kaggle.ipynb").is_file()


def test_kernel_can_be_published_explicitly(tmp_path):
    staging = stage_kernel(
        "me/airfoil-v2-experiment", "me/airfoil-v2", tmp_path / "stage", private=False,
    )
    assert json.loads((staging / "kernel-metadata.json").read_text())["is_private"] is False


def test_kernel_staging_defaults_to_the_repo_notebook():
    assert kaggle_sync.DEFAULT_NOTEBOOK.is_file()
    assert kaggle_sync.DEFAULT_NOTEBOOK.name == "v2_experiment_kaggle.ipynb"


def test_kernel_staging_reports_a_missing_notebook(tmp_path):
    with pytest.raises(FileNotFoundError, match="Notebook not found"):
        stage_kernel(
            "me/airfoil-v2-experiment", "me/airfoil-v2", tmp_path / "stage",
            notebook=tmp_path / "absent.ipynb",
        )


@pytest.mark.parametrize("kernel,dataset", [
    ("bad", "me/airfoil-v2"),
    ("me/airfoil-v2-experiment", "bad"),
])
def test_kernel_staging_rejects_malformed_references(tmp_path, kernel, dataset):
    with pytest.raises(ValueError, match="owner/slug"):
        stage_kernel(kernel, dataset, tmp_path / "stage")


def test_pushing_a_kernel_queues_a_run_and_cleans_up_staging(tmp_path):
    runner = FakeRunner(stdout="ready")
    output = push_kernel("me/airfoil-v2-experiment", "me/airfoil-v2", runner=runner)
    # Readiness is checked before anything is uploaded.
    assert runner.commands[0] == ["kaggle", "datasets", "status", "me/airfoil-v2"]
    command = runner.commands[1]
    assert command[:3] == ["kaggle", "kernels", "push"]
    staging = Path(command[command.index("-p") + 1])
    assert output.strip() == "ready"
    # Pushing is what starts execution; nothing is left on disk afterwards.
    assert not staging.exists()


@pytest.mark.parametrize("status", ["processing", "error", ""])
def test_pushing_a_kernel_refuses_a_dataset_that_is_not_ready(status):
    """Kaggle accepts the attach, mounts nothing, and the run fails much later."""
    runner = FakeRunner(stdout=status)
    with pytest.raises(RuntimeError, match="not ready"):
        push_kernel("me/airfoil-v2-experiment", "me/airfoil-v2", runner=runner)
    # Nothing is uploaded when the input is not usable yet.
    assert [command[:3] for command in runner.commands] == [
        ["kaggle", "datasets", "status", "me/airfoil-v2"][:3],
    ]


def test_dataset_status_asks_kaggle_and_validates_the_reference():
    runner = FakeRunner(stdout="ready\n")
    assert kaggle_sync.dataset_status("me/airfoil-v2", runner=runner) == "ready"
    assert runner.commands[0] == ["kaggle", "datasets", "status", "me/airfoil-v2"]
    with pytest.raises(ValueError, match="owner/slug"):
        kaggle_sync.dataset_status("nope", runner=FakeRunner())


def test_kernel_status_asks_kaggle_for_the_run_state():
    runner = FakeRunner(stdout='has status "complete"')
    assert "complete" in kernel_status("me/airfoil-v2-experiment", runner=runner)
    assert runner.commands[0] == [
        "kaggle", "kernels", "status", "me/airfoil-v2-experiment",
    ]


def test_kernel_status_rejects_a_malformed_reference():
    with pytest.raises(ValueError, match="owner/slug"):
        kernel_status("nope", runner=FakeRunner())
