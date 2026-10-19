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

    def __call__(self, command, *, check, capture_output, text, env):
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

    def runner(command, *, check, capture_output, text, env):
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
    def runner(command, *, check, capture_output, text, env):
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
    assert metadata["enable_gpu"] is True and metadata["enable_tpu"] is False
    assert metadata["dataset_sources"] == ["me/airfoil-v2"]
    assert metadata["is_private"] is True
    assert len(metadata["title"]) >= 5
    # The notebook itself is uploaded alongside the metadata.
    assert (staging / "v2_experiment_kaggle.ipynb").is_file()


def test_kernel_cpu_override_and_explicit_gpu_shape(tmp_path):
    stage = stage_kernel("me/run", "me/data", tmp_path / "cpu", accelerator="cpu")
    assert json.loads((stage / "kernel-metadata.json").read_text())["enable_gpu"] is False
    stage = stage_kernel("me/run", "me/data", tmp_path / "gpu", machine_shape="NvidiaTeslaT4")
    assert json.loads((stage / "kernel-metadata.json").read_text())["machine_shape"] == "NvidiaTeslaT4"
    source = "".join(json.loads((stage / "v2_experiment_kaggle.ipynb").read_text())["cells"][0]["source"])
    assert "KAGGLE_MACHINE_SHAPE = 'NvidiaTeslaT4'" in source
    with pytest.raises(ValueError, match="requires the GPU"):
        stage_kernel("me/run", "me/data", tmp_path / "bad", accelerator="cpu", machine_shape="NvidiaL4")


def test_study_notebook_reuses_preflight_without_standard_test_cells(tmp_path):
    staged = stage_kernel("me/study", "me/data", tmp_path / "study", study=True)
    document = json.loads((staged / "v2_experiment_kaggle.ipynb").read_text())
    ids = [cell.get("id") for cell in document["cells"]]
    assert "kaggle-gpu-tests" in ids and "v2-data" in ids
    assert ids[-1] == "v2-train"
    assert "test-results" not in ids and "v2-analysis" not in ids
    training = "".join(document["cells"][-1]["source"])
    assert "run_study(split, feature_columns, target_columns, RUN_OUTPUT" in training
    assert "compare_models(" not in training and "split.test" not in training
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")


def test_cli_zero_exit_push_error_is_not_reported_as_success():
    runner = FakeRunner(stdout="Kernel push error: Maximum batch GPU session count of 2 reached.")
    with pytest.raises(RuntimeError, match="Maximum batch GPU"):
        kaggle_sync._run(["kaggle", "kernels", "push", "-p", "/unused"], runner)


def test_source_bundle_is_deterministic_and_excludes_private_files(tmp_path):
    import ast
    import base64
    import hashlib
    import io
    import zipfile

    first = kaggle_sync._source_cell(kaggle_sync.DEFAULT_SOURCE_ROOT, "gpu")
    second = kaggle_sync._source_cell(kaggle_sync.DEFAULT_SOURCE_ROOT, "gpu")
    assert first == second
    syntax = ast.parse("".join(first["source"]))
    assignments = {node.targets[0].id: node.value for node in syntax.body if isinstance(node, ast.Assign)}
    payload = base64.b64decode(ast.literal_eval(assignments["_source_archive"].args[0]))
    assert hashlib.sha256(payload).hexdigest() == ast.literal_eval(assignments["BUNDLED_SOURCE_SHA256"])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        paths = archive.namelist()
        assert "src/training/comparison.py" in paths
        assert "configs/frozen_split.json" in paths
        assert "configs/accuracy_baseline/test_metrics.csv" in paths
        assert "configs/accuracy_baseline/run.json" in paths
        assert "configs/accuracy_baseline/test_evaluation.json" in paths
        assert "requirements-kaggle.txt" in paths
        assert "notebooks/v2_experiment_kaggle.ipynb" in paths
        assert all(path.startswith(("src/", "tests/", "configs/", "notebooks/", "requirements", "pyproject")) for path in paths)
        assert not any(path.startswith(("data/", "results/", ".")) for path in paths)


def test_extracted_bundle_supports_notebook_and_kernel_checks_without_git(tmp_path):
    import ast
    import base64
    import io
    import os
    import sys
    import zipfile

    cell = kaggle_sync._source_cell(kaggle_sync.DEFAULT_SOURCE_ROOT, "cpu")
    syntax = ast.parse("".join(cell["source"]))
    assignment = next(node for node in syntax.body if isinstance(node, ast.Assign)
                      and node.targets[0].id == "_source_archive")
    payload = base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    extracted = tmp_path / "unpacked"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(extracted)
    assert not (extracted / ".git").exists()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_kaggle_sync.py",
         "tests/test_kaggle_notebook.py", "-k",
         "kernel_metadata_enables_internet or kernel_staging_defaults or every_code_cell_compiles"],
        cwd=extracted, env={**os.environ, "PYTHONPATH": str(extracted)},
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "3 passed" in completed.stdout


def test_cli_token_file_workaround_does_not_expose_or_replace_credentials(monkeypatch):
    seen = {}
    def runner(command, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(stdout="ready")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    kaggle_sync.dataset_status("me/data", runner=runner)
    assert seen["KAGGLE_API_TOKEN"] == ""
    monkeypatch.setenv("KAGGLE_API_TOKEN", "test-token")
    kaggle_sync.dataset_status("me/data", runner=runner)
    assert seen["KAGGLE_API_TOKEN"] == "test-token"


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
