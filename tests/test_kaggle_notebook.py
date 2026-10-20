import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
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
    save_dataset_tables(load_dataset_tables(archive, geometry_samples=50), dataset_dir)

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
    assignments = [node for node in ast.walk(ast.parse(setup)) if isinstance(node, ast.Assign)]
    project_assignments = [
        node for node in assignments
        if any(isinstance(target, ast.Name) and target.id == "PROJECT_ROOT" for target in node.targets)
    ]
    assert project_assignments
    assert all("/kaggle/working" not in ast.unparse(node.value) for node in project_assignments)
    assert "/kaggle/working" in cells["v2-config"]
    assert "/kaggle/input" in cells["v2-config"]


def test_auxiliary_data_cell_preserves_rows_and_defers_holdout_matrices(cells, kaggle_inputs, monkeypatch):
    from src.data import auxiliary
    from src.training import comparison

    data_dir, root = kaggle_inputs
    polars_path = data_dir / "airfoil_polars.csv"
    original_bytes = polars_path.read_bytes()
    keys = ["geometry_id", "Alpha", "Re", "Ncrit"]
    sidecar = pd.read_csv(polars_path)[keys].assign(Cdp=np.arange(210) * .001, Cdp_observed=True)
    # Deliberately reorder labels to require an identity-based, not positional, join.
    monkeypatch.setattr(auxiliary, "load_auxiliary", lambda *args: sidecar.iloc[::-1])
    (data_dir / "auxiliary_report.json").write_text(json.dumps({"sidecar_sha256": "d" * 64}))
    original_loader = comparison.load_frozen_split

    class Protected(pd.DataFrame):
        def __getitem__(self, key):
            if isinstance(key, list) and any(name not in keys for name in key):
                pytest.fail("Production holdout feature/target matrices must wait for the development decision")
            return super().__getitem__(key)

    def loader(*args, **kwargs):
        split, features, targets, metadata = original_loader(*args, **kwargs)
        split.validation, split.test = Protected(split.validation), Protected(split.test)
        return split, features, targets, metadata
    monkeypatch.setattr(comparison, "load_frozen_split", loader)
    namespace = {
        "Path": Path, "PROJECT_ROOT": root,
        "KAGGLE_STUDY_MODE": "auxiliary",
    }
    # Run the real setup chain: injecting dataset variables here concealed a
    # DATA_DIR typo that the notebook's actual resolver never defines.
    exec(compile(cells["v2-config"], "kaggle:config", "exec"), namespace)
    namespace["INPUT_ROOT"] = data_dir.parent
    exec(compile(cells["v2-resolve"], "kaggle:resolve", "exec"), namespace)
    assert "DATA_DIR" not in namespace
    exec(compile(cells["v2-data"], "kaggle:auxiliary-data", "exec"), namespace)
    assert not {"X_train", "X_validation", "y_train", "y_validation"}.intersection(namespace)
    assert len(namespace["feature_columns"]) == 113
    assert not {"Cdp", "Cdp_observed"}.intersection(namespace["feature_columns"])
    assert namespace["dataset_metadata"]["auxiliary"] == {"sidecar_sha256": "d" * 64}
    assert "auxiliary" not in namespace["dataset_metadata"]["dataset"]
    for partition in ("train", "validation", "test"):
        frame = getattr(namespace["split"], partition)
        np.testing.assert_array_equal(frame.Cdp, sidecar.set_index(keys).loc[
            pd.MultiIndex.from_frame(frame[keys]), "Cdp"].to_numpy())
        assert frame.Cdp_observed.all()
    assert polars_path.read_bytes() == original_bytes


def test_staged_auxiliary_notebook_reaches_study_entry_without_fitting(kaggle_inputs, tmp_path, monkeypatch):
    from importlib import metadata as package_metadata

    from src.data import auxiliary
    from src.training import auxiliary_study
    from src.utils.kaggle_sync import stage_kernel

    data_dir, root = kaggle_inputs
    keys = ["geometry_id", "Alpha", "Re", "Ncrit"]
    labels = pd.read_csv(data_dir / "airfoil_polars.csv")[keys].assign(Cdp=.01, Cdp_observed=True)
    (data_dir / "auxiliary_report.json").write_text(json.dumps({"sidecar_sha256": "d" * 64}))
    seen = []

    def load_labels(sidecar, polars, report):
        assert sidecar == data_dir / "airfoil_auxiliary.csv"
        assert polars == data_dir / "airfoil_polars.csv"
        assert report == data_dir / "auxiliary_report.json"
        return labels

    def record_study(split, features, targets, output, **kwargs):
        seen.append((split, features, targets, output, kwargs))
        return {"status": "mocked: no fitting"}

    monkeypatch.setattr(auxiliary, "load_auxiliary", load_labels)
    monkeypatch.setattr(auxiliary_study, "run_auxiliary_study", record_study)
    installed_version = package_metadata.version
    # Version logging must not require the cloud-only training/plot dependencies
    # in the minimal fit-free CI environment.
    monkeypatch.setattr(package_metadata, "version",
                        lambda name: "mock" if name in {"torch", "matplotlib"} else installed_version(name))
    staged = stage_kernel("me/study", "me/data", tmp_path / "stage", auxiliary_study=True)
    document = nbformat.read(staged / NOTEBOOK.name, as_version=4)
    sources = {cell.id: cell.source for cell in document.cells if cell.cell_type == "code"}
    # These are the real bootstrap's outputs, not guessed dataset globals.
    namespace = {
        "Path": Path, "sys": sys, "PROJECT_ROOT": root,
        "SOURCE_REVISION": "a" * 40, "SOURCE_SHA256": "b" * 64, "SOURCE_MODE": "bundle",
        "ACCELERATOR": "gpu", "GPU_TESTS_PASSED": True, "CPU_TESTS_PASSED": False,
        "GPU_METADATA": {"device_name": "mock GPU"},
        "torch": SimpleNamespace(__version__="mock", cuda=SimpleNamespace(is_available=lambda: True)),
    }
    for statement in ast.parse(sources["kaggle-source-bundle"]).body:
        if isinstance(statement, ast.Assign) and statement.targets[0].id.startswith("KAGGLE_"):
            namespace[statement.targets[0].id] = ast.literal_eval(statement.value)
    exec(compile(sources["v2-config"], "kaggle:config", "exec"), namespace)
    namespace.update(INPUT_ROOT=data_dir.parent, OUTPUT_ROOT=tmp_path / "working")
    for name in ("v2-resolve", "v2-data", "v2-queue", "v2-train"):
        exec(compile(sources[name], f"kaggle:{name}", "exec"), namespace)
    assert "DATA_DIR" not in namespace
    assert len(seen) == 1
    split, features, targets, output, options = seen[0]
    assert targets == ["Cl", "Cd", "Cm"] and len(features) == 113
    assert "Cdp" not in features and split.train.Cdp_observed.all()
    assert options["device"] == "cuda"
    assert options["run_metadata"]["study_mode"] == "auxiliary"
    assert options["run_metadata"]["auxiliary"]["sidecar_sha256"] == "d" * 64
    assert not output.exists(), "The study entrypoint is mocked; no fit/artifact work may occur"


@pytest.mark.parametrize("accelerator,cuda_available", [("gpu", False), ("gpu", True), ("cpu", False)])
def test_bundled_setup_skips_git_and_records_actual_gpu(
    cells, tmp_path, monkeypatch, accelerator, cuda_available,
):
    """Mock external bootstrap and torch; no dependency install or local fitting."""
    project_root = tmp_path / "bundle"
    (project_root / "src").mkdir(parents=True)
    is_dir = Path.is_dir
    monkeypatch.setattr(
        Path, "is_dir",
        lambda path: True if str(path) in {"/kaggle/input", "/kaggle/working"} else is_dir(path),
    )
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(
        subprocess, "check_output",
        lambda *args, **kwargs: pytest.fail("Bundled setup must not query Git"),
    )
    fake_torch = SimpleNamespace(
        __version__="2.test", version=SimpleNamespace(cuda="12.test"),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available, current_device=lambda: 0,
            get_device_name=lambda index: "Actual Kaggle GPU", device_count=lambda: 2,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {
        "BUNDLED_SOURCE_DIR": project_root, "BUNDLED_SOURCE_SHA256": "c" * 64,
        "BUNDLED_GIT_REVISION": "b" * 40,
    }
    # Omitting the setting must enforce GPU, while CPU is an explicit override.
    if accelerator == "cpu":
        namespace["KAGGLE_ACCELERATOR"] = "cpu"
    if cuda_available or accelerator == "cpu":
        exec(compile(cells["kaggle-setup"], "kaggle:setup", "exec"), namespace)
        assert namespace["ACCELERATOR"] == accelerator
        assert namespace["PROJECT_ROOT"] == project_root
        assert namespace["SOURCE_REVISION"] == "b" * 40
        assert namespace["SOURCE_SHA256"] == "c" * 64
        assert namespace["SOURCE_MODE"] == "bundle"
        assert namespace["GPU_METADATA"] == {
            "torch_version": "2.test", "cuda_version": "12.test",
            "device_name": "Actual Kaggle GPU" if accelerator == "gpu" else None,
            "device_count": 2 if accelerator == "gpu" else 0,
        }
    else:
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            exec(compile(cells["kaggle-setup"], "kaggle:setup", "exec"), namespace)
        assert "GPU_METADATA" not in namespace
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:5] == [sys.executable, "-m", "pip", "install", "-q"]
    assert str(project_root / "requirements.txt") in command
    assert str(project_root / "requirements-kaggle.txt") in command
    assert kwargs["check"] is True


def test_setup_refuses_local_execution_before_installing(cells, monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda path: False)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("Local setup must not install anything"),
    )
    with pytest.raises(RuntimeError, match="not locally"):
        exec(compile(cells["kaggle-setup"], "kaggle:local-setup", "exec"), {})


def test_browser_fallback_preserves_checkout_guards(cells):
    tree = ast.parse(cells["kaggle-setup"])
    bundled = next(
        node for node in tree.body if isinstance(node, ast.If)
        and "BUNDLED_SOURCE_DIR" in ast.unparse(node.test)
    )
    fallback = "\n".join(ast.unparse(node) for node in bundled.orelse)
    assert "clone" in fallback and "pull" in fallback and "--ff-only" in fallback
    assert "--porcelain" in fallback and "Restart and run all" in fallback
    assert "hashlib.sha256" in fallback and "archive" in fallback
    assert "SOURCE_REVISION = BUNDLED_GIT_REVISION" in cells["kaggle-setup"]


@pytest.mark.parametrize("accelerator", ["gpu", "cpu"])
def test_accuracy_queue_is_copied_and_selects_drag_by_validation_median(cells, accelerator):
    from src.training.comparison import ACCURACY_MODEL_CONFIGS
    from src.training.targets import DEFAULT_TARGET_TRANSFORMS

    namespace = {
        "deepcopy": deepcopy, "pd": pd, "display": lambda *args: None,
        "ACCURACY_MODEL_CONFIGS": ACCURACY_MODEL_CONFIGS,
        "DEFAULT_TARGET_TRANSFORMS": DEFAULT_TARGET_TRANSFORMS,
        "ACCELERATOR": accelerator,
    }
    exec(compile(cells["v2-queue"], "kaggle:queue", "exec"), namespace)
    configs = namespace["MODEL_CONFIGS"]
    expected = deepcopy(list(ACCURACY_MODEL_CONFIGS))
    if accelerator == "cpu":
        for entry in expected[1:]:
            entry["parameters"]["device"] = "cpu"
    assert list(configs) == expected
    assert configs is not ACCURACY_MODEL_CONFIGS
    assert [config["name"] for config in configs] == ["mlp", "torch_mlp_mse", "torch_mlp_huber"]
    assert all(entry["parameters"]["device"] == ("cuda" if accelerator == "gpu" else "cpu") for entry in configs[1:])
    assert all(config["scale_targets"] is False for config in configs[1:])
    configs[0]["parameters"]["max_iter"] = -1
    assert ACCURACY_MODEL_CONFIGS[0]["parameters"]["max_iter"] != -1
    assert namespace["TARGET_TRANSFORMS"] == {"Cl": "identity", "Cd": "log", "Cm": "identity"}
    assert namespace["SELECTION_METRICS"] == {"Cd": "median_relative_error"}


def test_gpu_preflight_is_kaggle_only_and_precedes_experiment(cells, monkeypatch):
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    namespace = {
        "Path": Path, "sys": sys, "subprocess": subprocess, "PROJECT_ROOT": NOTEBOOK.parents[1],
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        "ACCELERATOR": "gpu",
    }
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    exec(compile(cells["kaggle-gpu-tests"], "kaggle:gpu-tests", "exec"), namespace)
    assert namespace["GPU_TESTS_PASSED"] is True
    command, kwargs = commands[0]
    assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert "tests" in command
    assert "not test_notebook_runs_the_whole_v2_experiment_on_synthetic_tables" in command
    assert kwargs["env"]["KAGGLE_GPU_TESTS"] == "1" and kwargs["check"] is True
    assert kwargs["cwd"] == NOTEBOOK.parents[1]
    namespace["torch"].cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="must run on Kaggle"):
        exec(compile(cells["kaggle-gpu-tests"], "kaggle:gpu-tests", "exec"), namespace)
    assert namespace["GPU_TESTS_PASSED"] is False
    assert len(commands) == 1
    order = list(cells)
    assert order.index("kaggle-setup") < order.index("kaggle-gpu-tests") < order.index("v2-train")
    train_tree = ast.parse(cells["v2-train"])
    assert isinstance(train_tree.body[0], ast.If)
    assert "GPU_TESTS_PASSED" in ast.unparse(train_tree.body[0].test)
    assert "torch.cuda.is_available()" in ast.unparse(train_tree.body[0].test)
    assert "ACCELERATOR == 'gpu'" in ast.unparse(train_tree.body[0].test)
    namespace["ACCELERATOR"] = "cpu"
    exec(compile(cells["kaggle-gpu-tests"], "kaggle:gpu-tests", "exec"), namespace)
    assert namespace["GPU_TESTS_PASSED"] is False
    assert namespace["CPU_TESTS_PASSED"] is True
    assert len(commands) == 2
    assert commands[1][1]["env"]["KAGGLE_CPU_TESTS"] == "1"
    assert commands[1][1]["env"]["KAGGLE_GPU_TESTS"] == "0"


def test_training_uses_only_train_validation_and_records_provenance(cells):
    tree = ast.parse(cells["v2-train"])
    comparison = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "compare_models"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in comparison.keywords}
    assert keywords["x_train"] == "X_train" and keywords["y_train"] == "y_train"
    assert keywords["x_validation"] == "X_validation" and keywords["y_validation"] == "y_validation"
    assert keywords["selection_metrics"] == "SELECTION_METRICS"
    assert keywords["target_transforms"] == "TARGET_TRANSFORMS"
    assert "split.test" not in cells["v2-train"]
    metadata = next(
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "run_metadata" for target in node.targets)
    )
    values = {ast.literal_eval(key): ast.unparse(value) for key, value in zip(metadata.keys, metadata.values) if key}
    assert values["git_commit"] == "SOURCE_REVISION"
    assert values["source_sha256"] == "SOURCE_SHA256" and values["source_mode"] == "SOURCE_MODE"
    assert values["gpu"] == "GPU_METADATA" and values["gpu_tests_passed"] == "GPU_TESTS_PASSED"
    assert values["accelerator"] == "ACCELERATOR"
    assert values["machine_shape"] == "globals().get('KAGGLE_MACHINE_SHAPE')"
    assert values["device"] == "'cuda' if ACCELERATOR == 'gpu' else 'cpu'"
    assert values["inference_device"] == "'cpu-numpy'"
    assert "'torch'" in values["packages"] and "'pytest'" in values["packages"]


def test_frozen_test_report_keeps_metrics_analysis_and_saved_figures(cells):
    assert "evaluate_selected_models" in cells["test-results"]
    assert "compare_models" not in cells["test-results"]
    assert "test median relative error" in cells["test-results"]
    assert "row.rmse" in cells["test-results"] and "row.r2" in cells["test-results"]
    assert "mean_relative_error" not in cells["test-results"]
    assert "training_frame=split.train" in cells["v2-analysis"]
    assert "analysis.json" in cells["v2-analysis"]
    assert "training_metadata" in cells["validation-results"]
    assert "set_xlabel('RMSE (physical units)')" in cells["validation-results"]
    for name in ("validation_rmse.png", "regime_and_reynolds.png", "lift_to_drag.png"):
        assert any(name in source for source in cells.values())


@pytest.mark.parametrize("layout", ["flat", "nested", "deeply-nested"])
def test_dataset_discovery_does_not_assume_a_mount_depth(cells, kaggle_inputs, tmp_path, layout):
    """Kaggle has mounted inputs both flat and under /kaggle/input/datasets/."""
    dataset_dir, project_root = kaggle_inputs
    relative = {
        "flat": "airfoil-tables",
        "nested": "datasets/airfoil-tables",
        "deeply-nested": "datasets/purplray/airfoil-tables/1",
    }[layout]
    input_root = tmp_path / "mount"
    target = input_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_dir, target)

    namespace = {
        "Path": Path, "PROJECT_ROOT": project_root, "DATASET_DIR": None,
        "INPUT_ROOT": input_root, "OUTPUT_ROOT": tmp_path / "working",
    }
    exec(compile(cells["v2-resolve"], "kaggle:v2-resolve", "exec"), namespace)
    assert namespace["DATASET_DIR"] == target
    assert namespace["GEOMETRY_PATH"].is_file() and namespace["POLARS_PATH"].is_file()


def test_discovery_error_lists_what_is_actually_mounted(cells, tmp_path):
    """The first two real runs failed here; the message has to be self-diagnosing."""
    input_root = tmp_path / "mount"
    (input_root / "datasets" / "something-else").mkdir(parents=True)
    (input_root / "datasets" / "something-else" / "unrelated.csv").write_text("a\n")
    namespace = {
        "Path": Path, "PROJECT_ROOT": tmp_path, "DATASET_DIR": None,
        "INPUT_ROOT": input_root, "OUTPUT_ROOT": tmp_path,
    }
    with pytest.raises(FileNotFoundError, match="unrelated.csv"):
        exec(compile(cells["v2-resolve"], "kaggle:v2-resolve", "exec"), namespace)


def test_discovery_refuses_an_ambiguous_mount(cells, kaggle_inputs, tmp_path):
    dataset_dir, project_root = kaggle_inputs
    input_root = tmp_path / "mount"
    for name in ("copy-a", "copy-b"):
        shutil.copytree(dataset_dir, input_root / name)
    namespace = {
        "Path": Path, "PROJECT_ROOT": project_root, "DATASET_DIR": None,
        "INPUT_ROOT": input_root, "OUTPUT_ROOT": tmp_path,
    }
    with pytest.raises(RuntimeError, match="Several datasets"):
        exec(compile(cells["v2-resolve"], "kaggle:v2-resolve", "exec"), namespace)


@pytest.mark.skipif(
    os.environ.get("KAGGLE_GPU_TESTS") != "1" or not Path("/kaggle/working").is_dir(),
    reason="Notebook fitting tests run only on Kaggle via API with KAGGLE_GPU_TESTS=1",
)
def test_notebook_runs_the_whole_v2_experiment_on_synthetic_tables(
    cells, kaggle_inputs, tmp_path, monkeypatch,
):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib.pyplot as plt
    import torch

    assert torch.cuda.is_available(), "The Kaggle notebook fitting test requires a GPU"

    monkeypatch.setattr(plt, "show", lambda: None)
    dataset_dir, project_root = kaggle_inputs
    namespace = {
        "Path": Path, "sys": sys, "SOURCE_REVISION": "b" * 40,
        "SOURCE_SHA256": "c" * 64, "SOURCE_MODE": "bundle",
        "torch": torch, "subprocess": subprocess,
        "ACCELERATOR": "gpu",
        "GPU_METADATA": {
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "device_count": torch.cuda.device_count(),
        },
        "PROJECT_ROOT": NOTEBOOK.parents[1], "DATASET_DIR": dataset_dir,
        "OUTPUT_ROOT": tmp_path / "working", "display": lambda *args: None,
    }

    # Clone/pip is skipped. GPU preflight, loading, fitting, frozen selection,
    # test evaluation and analysis run for real, only inside Kaggle.
    exec(compile(cells["kaggle-gpu-tests"], "kaggle:gpu-tests", "exec"), namespace)
    namespace["PROJECT_ROOT"] = project_root
    exec(compile(cells["v2-resolve"], "kaggle:v2-resolve", "exec"), namespace)
    assert namespace["GEOMETRY_PATH"].is_file() and namespace["POLARS_PATH"].is_file()

    exec(compile(cells["v2-data"], "kaggle:v2-data", "exec"), namespace)
    assert namespace["target_columns"] == ["Cl", "Cd", "Cm"]
    features = namespace["feature_columns"]
    assert len(features) == 113
    assert "geometry_id" not in features
    assert features[-5:] == ["Alpha", "alpha_sin", "alpha_cos", "log10_re", "Ncrit"]
    split = namespace["split"]
    assert split.test.geometry_id.nunique() == 2

    exec(compile(cells["v2-queue"], "kaggle:v2-queue", "exec"), namespace)
    configs = namespace["MODEL_CONFIGS"]
    assert [config["name"] for config in configs] == [
        "mlp", "torch_mlp_mse", "torch_mlp_huber",
    ]
    assert namespace["TARGET_TRANSFORMS"] == {"Cl": "identity", "Cd": "log", "Cm": "identity"}
    # Shrink the queue so the test stays fast; the code paths are unchanged.
    configs[0]["parameters"].update(hidden_layers=(8,), max_iter=3, batch_size=16)
    for config in configs[1:]:
        config["parameters"].update(
            hidden_layers=(8,), max_iter=3, batch_size=32, patience=2, lr_patience=1, device="cuda",
        )

    for cell_id in ("v2-train", "validation-results", "test-results", "v2-analysis", "analysis-plots"):
        exec(compile(cells[cell_id], f"kaggle:{cell_id}", "exec"), namespace)

    output = namespace["RUN_OUTPUT"]
    assert output.is_relative_to(tmp_path / "working")
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["target_transforms"] == {"Cl": "identity", "Cd": "log", "Cm": "identity"}
    assert manifest["run_metadata"]["platform"] == "kaggle"
    assert manifest["run_metadata"]["git_commit"] == "b" * 40
    assert manifest["run_metadata"]["source_sha256"] == "c" * 64
    assert manifest["run_metadata"]["source_mode"] == "bundle"
    assert manifest["run_metadata"]["gpu"]["device_name"] == torch.cuda.get_device_name(0)
    assert manifest["run_metadata"]["gpu_tests_passed"] is True
    assert manifest["run_metadata"]["dataset"]["polar_rows"] == 210
    assert set(manifest["selected_models"]) == {"Cl", "Cd", "Cm"}
    assert manifest["selection_metrics"] == {
        "Cl": "rmse", "Cd": "median_relative_error", "Cm": "rmse",
    }
    for result in manifest["results"].values():
        assert result["training_metadata"]

    validation = pd.read_csv(output / "comparison.csv")
    assert len(validation) == 9
    assert np.isfinite(validation[["mae", "rmse"]].to_numpy()).all()
    for target, metric in manifest["selection_metrics"].items():
        scores = validation[validation.target == target]
        winner = scores.loc[scores[metric].idxmin(), "model"]
        assert manifest["selected_models"][target] == winner
    # Drag is fitted in log space but every reported number is physical, so the
    # error sits on the scale of Cd itself rather than of log(Cd).
    drag = validation[validation.target == "Cd"]
    assert (drag.rmse < 0.1).all()

    test = pd.read_csv(output / "test_metrics.csv")
    assert test.target.tolist() == ["Cl", "Cd", "Cm"]
    assert (test.n_predictions == len(split.test)).all()
    evaluation = json.loads((output / "test_evaluation.json").read_text())
    assert evaluation["selected_models"] == manifest["selected_models"]
    assert evaluation["selection_metrics"] == manifest["selection_metrics"]

    report = json.loads((output / "analysis.json").read_text())
    assert report["n_geometries"] == 2
    assert report["n_rows"] == len(split.test)
    for target in ("Cl", "Cd", "Cm"):
        regimes = {record["regime"] for record in report["targets"][target]["regime"]}
        assert regimes == {"pre-stall", "post-stall"}
        assert len(report["targets"][target]["reynolds"]) == 3
        assert report["targets"][target]["worst_geometries"]
    assert report["lift_to_drag"]["n_rows"] > 0
    assert "training_domain" in report and "matched_easy_subset" in report
    for target in ("Cl", "Cd", "Cm"):
        assert "error_concentration" in report["targets"][target]

    for name in ("validation_rmse.png", "regime_and_reynolds.png", "lift_to_drag.png"):
        assert (output / name).stat().st_size > 0


def test_notebook_reports_a_missing_dataset_clearly(cells, tmp_path):
    namespace = {
        "Path": Path, "PROJECT_ROOT": tmp_path, "DATASET_DIR": tmp_path / "absent",
        "OUTPUT_ROOT": tmp_path,
    }
    with pytest.raises(FileNotFoundError, match="Required input not found"):
        exec(compile(cells["v2-resolve"], "kaggle:missing-input", "exec"), namespace)
