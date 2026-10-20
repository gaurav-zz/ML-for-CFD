"""Report-only tests: synthetic candidate edits, never a model fit."""

import ast
import csv
from decimal import Decimal
import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from src.utils.accuracy_guard import (
    COLUMNS, DEFAULT_BASELINE, REPORT_FILES, TARGETS,
    compare_reports, find_candidate_directory, load_report, main,
)


@pytest.fixture
def candidate(tmp_path):
    directory = tmp_path / "candidate"
    directory.mkdir()
    for name in REPORT_FILES:
        shutil.copyfile(DEFAULT_BASELINE / name, directory / name)
    return directory


def edit_json(directory, filename, path, value):
    target = directory / filename
    document = json.loads(target.read_text())
    node = document
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    target.write_text(json.dumps(document, allow_nan=True))


def shift_median(directory, target, delta):
    # Decimal text editing tests the exact inclusive boundary without float rounding.
    records = json.loads((directory / "test_evaluation.json").read_text(), parse_float=Decimal)["results"]
    old = next(row["median_relative_error"] for row in records if row["target"] == target)
    new = old + Decimal(delta)
    for name in ("test_evaluation.json", "test_metrics.csv"):
        path = directory / name
        text = path.read_text()
        assert text.count(str(old)) == 1
        path.write_text(text.replace(str(old), str(new)))


def test_pin_has_exact_original_csv_and_expected_baseline():
    provenance = json.loads((DEFAULT_BASELINE / "provenance.json").read_text())
    assert provenance["source_run_sha256"] == "ca707a25508b069ad09f0bfc5ac99e85a524ced0400d0a33eb0e0aa45cf72628"
    assert hashlib.sha256((DEFAULT_BASELINE / "test_metrics.csv").read_bytes()).hexdigest() == (
        "4b0d59ded8e222b049de3551ece1853e07207e91aa8002e99fe7c29a0ed8f207"
    )
    run, records = load_report(DEFAULT_BASELINE)
    assert [records[target]["median_relative_error"] for target in TARGETS] == [
        Decimal("0.03813765374607342"), Decimal("0.05572903434415842"), Decimal("0.09649760948968417"),
    ]
    assert run["selected_models"] == dict.fromkeys(TARGETS, "torch_mlp_huber")
    assert run["run_metadata"]["split_rows"] == {"train": 538380, "validation": 113678, "test": 116484}
    assert run["results"]["torch_mlp_huber"]["training_metadata"]["best_epoch"] == 184


def test_projection_matches_original_artifacts_when_available():
    provenance = json.loads((DEFAULT_BASELINE / "provenance.json").read_text())
    original = DEFAULT_BASELINE.parents[1] / provenance["source_directory"]
    if not original.is_dir():
        pytest.skip("Original ignored Kaggle output is not distributed; pin hashes are checked separately")
    assert hashlib.sha256((original / "run.json").read_bytes()).hexdigest() == provenance["source_run_sha256"]
    for name in ("test_metrics.csv", "test_evaluation.json"):
        assert (original / name).read_bytes() == (DEFAULT_BASELINE / name).read_bytes()
    expected = json.loads((original / "run.json").read_text())
    del expected["run_metadata"]["feature_columns"]
    for result in expected["results"].values():
        result["training_metadata"].pop("history", None)
    actual = json.loads((DEFAULT_BASELINE / "run.json").read_text())
    for split, names in expected["run_metadata"]["split_geometries"].items():
        pinned = actual["run_metadata"]["split_geometries"][split]
        assert len(pinned) == len(names)
        assert [(a, b) for a, b in zip(pinned, names) if a != b] == []
    assert actual == expected


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("delta,passed", [("0", True), ("0.00499999", True), ("0.005", True),
                                          ("0.00500001", False), ("-0.02", True)])
def test_fraction_units_boundary_and_improvements(candidate, target, delta, passed):
    shift_median(candidate, target, delta)
    rows = compare_reports(candidate)
    row = next(row for row in rows if row["target"] == target)
    assert row["delta_pp"] == Decimal(delta) * 100
    assert row["passed"] is passed
    assert all(row["passed"] for row in rows if row["target"] != target)


def test_custom_tolerance_and_zero_tolerance(candidate):
    shift_median(candidate, "Cd", "0.001")
    assert not compare_reports(candidate, tolerance_pp=0)[1]["passed"]
    assert compare_reports(candidate, tolerance_pp="0.1")[1]["passed"]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "-0.1", "nonsense", True, None])
def test_bad_tolerance(candidate, value):
    with pytest.raises(ValueError, match="tolerance_pp"):
        compare_reports(candidate, tolerance_pp=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, None, True, "bad"])
def test_invalid_json_metric(candidate, value):
    edit_json(candidate, "test_evaluation.json", ["results", 0, "median_relative_error"], value)
    with pytest.raises(ValueError):
        compare_reports(candidate)


@pytest.mark.parametrize("value", ["NaN", "Inf", "-0.01", "", "not-a-number"])
def test_invalid_csv_metric(candidate, value):
    path = candidate / "test_metrics.csv"
    path.write_text(path.read_text().replace("0.03813765374607342", value))
    with pytest.raises(ValueError):
        compare_reports(candidate)


@pytest.mark.parametrize("path,value,match", [
    (["status"], "running", "completed"),
    (["results", "mlp", "status"], "failed", "success"),
    (["run_metadata", "platform"], "local", "kaggle"),
    (["run_metadata", "source_mode"], "unknown", "source_mode"),
    (["run_metadata", "source_sha256"], "unknown", "source_sha256"),
    (["run_metadata", "git_commit"], "unknown", "git_commit"),
    (["run_metadata", "cpu_tests_passed"], False, "preflight"),
    (["run_metadata", "accelerator"], "unknown", "accelerator"),
    (["run_metadata", "test_used_for_tuning"], True, "must not be used for tuning"),
    (["run_metadata", "dataset", "geometry_sha256"], "a" * 64, "dataset"),
    (["run_metadata", "dataset", "polar_rows"], 768541, "row counts"),
    (["run_metadata", "split_rows", "test"], 116483, "row counts"),
    (["run_metadata", "split_rows", "train"], -1, "nonnegative"),
    (["run_metadata", "split_geometries", "test", 0], "wrong-foil", "split_geometries"),
    (["run_metadata", "target_columns"], ["Cl", "Cd"], "targets"),
    (["target_names"], ["Cl", "Cd"], "targets"),
    (["selection_metric"], "test", "validation"),
    (["selection_metric"], "cross_validation", "validation"),
    (["selection_metrics", "Cd"], "test_median", "selection metric"),
    (["selected_models", "Cd"], "mlp", "validation winner"),
    (["selected_models", "Cd"], "missing-model", "Unknown selected model"),
    (["results", "torch_mlp_huber", "n_predictions"], 116484, "Validation prediction count"),
    (["results", "torch_mlp_huber", "parameters", "loss"], "mse", "parameters"),
    (["results", "torch_mlp_huber", "training_metadata", "device"], "unknown", "device"),
    (["results", "torch_mlp_huber", "training_metadata", "device_name"], "unknown", "hardware"),
    (["results", "torch_mlp_huber", "training_metadata", "cuda_available"], True, "CUDA provenance"),
    (["results", "torch_mlp_huber", "training_metadata", "cuda_version"], "12.1", "CUDA provenance"),
    (["results", "torch_mlp_huber", "training_metadata", "n_validation"], 116484, "stopping counts"),
    (["results", "torch_mlp_huber", "training_metadata", "stop_reason"], "test_patience", "stopping policy"),
    (["results", "torch_mlp_huber", "training_metadata", "restored_best_checkpoint"], False, "checkpoint"),
    (["results", "torch_mlp_huber", "training_metadata", "best_epoch"], 500, "best validation epoch"),
    (["results", "torch_mlp_huber", "training_metadata", "n_iter"], 500, "iteration counts"),
])
def test_invalid_or_incomparable_manifest(candidate, path, value, match):
    edit_json(candidate, "run.json", path, value)
    with pytest.raises(ValueError, match=match):
        compare_reports(candidate)


@pytest.mark.parametrize("field,value", [("target_transforms", {"Cl": "identity", "Cd": "identity", "Cm": "identity"}),
                                         ("selected_models", {"Cl": "mlp", "Cd": "mlp", "Cm": "mlp"})])
def test_evaluation_manifest_mismatch(candidate, field, value):
    edit_json(candidate, "test_evaluation.json", [field], value)
    with pytest.raises(ValueError, match="differs from run manifest"):
        compare_reports(candidate)


def test_consistent_but_changed_transforms_rejected(candidate):
    for name in ("run.json", "test_evaluation.json"):
        edit_json(candidate, name, ["target_transforms", "Cd"], "identity")
    with pytest.raises(ValueError, match="target_transforms differs"):
        compare_reports(candidate)


@pytest.mark.parametrize("count", [113678, 0, -1, 116483, 116484.5, True])
def test_test_count_cannot_be_validation_or_partial_count(candidate, count):
    edit_json(candidate, "test_evaluation.json", ["results", 0, "n_predictions"], count)
    with pytest.raises(ValueError, match="n_predictions"):
        compare_reports(candidate)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown"])
@pytest.mark.parametrize("filename", ["test_metrics.csv", "test_evaluation.json"])
def test_each_target_exactly_once(candidate, filename, kind):
    path = candidate / filename
    if filename.endswith(".csv"):
        with path.open(newline="") as stream:
            records = list(csv.DictReader(stream))
    else:
        document = json.loads(path.read_text())
        records = document["results"]
    if kind == "missing":
        records.pop()
    elif kind == "duplicate":
        records.append(records[0])
    else:
        records[0]["target"] = "unknown"
    if filename.endswith(".csv"):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(records)
    else:
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="target"):
        compare_reports(candidate)


def test_csv_duplicate_columns_rejected(candidate):
    path = candidate / "test_metrics.csv"
    path.write_text(path.read_text().replace("mae,rmse", "mae,mae"))
    with pytest.raises(ValueError, match="columns"):
        compare_reports(candidate)


@pytest.mark.parametrize("column,value", [("rmse", 1), ("batch_inference_time_seconds", 9)])
def test_full_csv_json_consistency_not_only_headline(candidate, column, value):
    edit_json(candidate, "test_evaluation.json", ["results", 0, column], value)
    with pytest.raises(ValueError, match="inconsistent"):
        compare_reports(candidate)


def test_csv_selected_model_must_match_manifest(candidate):
    path = candidate / "test_metrics.csv"
    path.write_text(path.read_text().replace("torch_mlp_huber", "mlp", 1))
    with pytest.raises(ValueError, match="selected model mismatch"):
        compare_reports(candidate)


def test_split_overlap_and_missing_metadata_rejected(candidate):
    run = json.loads((candidate / "run.json").read_text())
    edit_json(candidate, "run.json", ["run_metadata", "split_geometries", "test", 0],
              run["run_metadata"]["split_geometries"]["train"][0])
    with pytest.raises(ValueError, match="overlapping"):
        compare_reports(candidate)
    edit_json(candidate, "run.json", ["run_metadata"], {})
    with pytest.raises(ValueError, match="missing report field"):
        compare_reports(candidate)


def test_json_duplicate_keys_rejected(candidate):
    path = candidate / "run.json"
    path.write_text(path.read_text().replace('"status": "completed"', '"status": "running", "status": "completed"'))
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        compare_reports(candidate)


def test_malformed_nested_mapping_rejected(candidate):
    edit_json(candidate, "run.json", ["selection_metrics"], list(TARGETS))
    with pytest.raises(ValueError, match="Malformed or missing report field"):
        compare_reports(candidate)


def test_new_validation_winner_is_allowed(candidate):
    run = json.loads((candidate / "run.json").read_text())
    # A new candidate may select MSE; only within-run selection consistency matters.
    for target in TARGETS:
        metric = run["selection_metrics"][target]
        edit_json(candidate, "run.json", ["results", "torch_mlp_mse", "targets", target, metric], 0.001)
    for name in REPORT_FILES:
        path = candidate / name
        if name == "run.json":
            edit_json(candidate, name, ["selected_models"], dict.fromkeys(TARGETS, "torch_mlp_mse"))
        else:
            path.write_text(path.read_text().replace("torch_mlp_huber", "torch_mlp_mse"))
    assert all(row["passed"] for row in compare_reports(candidate))


def test_gpu_candidate_with_actual_device_and_stopping_provenance(candidate):
    path = candidate / "run.json"
    run = json.loads(path.read_text())
    metadata = run["run_metadata"]
    metadata["accelerator"] = "gpu"
    metadata["gpu_tests_passed"] = True
    metadata["gpu"] = {"device_name": "Tesla P100", "device_count": 1,
                       "torch_version": "2.2.0+cu121", "cuda_version": "12.1"}
    for config in run["model_configs"]:
        if config["name"] == "mlp":
            continue
        config["parameters"]["device"] = "cuda"
        result = run["results"][config["name"]]
        result["parameters"]["device"] = "cuda"
        result["training_metadata"].update(
            accelerator="cuda", device="cuda", device_name="Tesla P100", cuda_available=True,
            cuda_device_index=0, torch_version="2.2.0+cu121", cuda_version="12.1",
        )
    path.write_text(json.dumps(run))
    assert all(row["passed"] for row in compare_reports(candidate))
    edit_json(candidate, "run.json", ["run_metadata", "gpu", "device_count"], 0)
    with pytest.raises(ValueError, match="GPU count"):
        compare_reports(candidate)


def test_negative_r2_is_valid_not_negative_error(candidate):
    for name in ("test_metrics.csv", "test_evaluation.json"):
        path = candidate / name
        path.write_text(path.read_text().replace("0.9871140431090842", "-0.5"))
    assert all(row["passed"] for row in compare_reports(candidate))


def test_pinned_baseline_detects_edit(candidate, tmp_path):
    baseline = tmp_path / "baseline"
    shutil.copytree(DEFAULT_BASELINE, baseline)
    shift_median(baseline, "Cd", "0.01")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        compare_reports(candidate, baseline)


def test_expected_source_pins_are_enforced(candidate):
    metadata = json.loads((candidate / "run.json").read_text())["run_metadata"]
    assert all(row["passed"] for row in compare_reports(
        candidate, expected_source_commit=metadata["git_commit"],
        expected_source_sha256=metadata["source_sha256"],
    ))
    for kwargs in ({"expected_source_commit": "a" * 40}, {"expected_source_sha256": "a" * 64},
                   {"expected_source_commit": "$(touch attack)"}):
        with pytest.raises(ValueError, match="expected source|hexadecimal"):
            compare_reports(candidate, **kwargs)


def test_cli_pass_failure_and_malformed_report(candidate, capsys):
    assert main(["--candidate-dir", str(candidate)]) == 0
    output = capsys.readouterr().out
    assert all(word in output for word in ("Baseline", "Candidate", "Delta (pp)", "PASS", "Improvements"))
    shift_median(candidate, "Cm", "0.006")
    assert main(["--candidate-dir", str(candidate)]) == 1
    assert "FAIL" in capsys.readouterr().out
    (candidate / "run.json").unlink()
    with pytest.raises(SystemExit) as error:
        main(["--candidate-dir", str(candidate)])
    assert error.value.code == 2
    assert "Accuracy guard FAIL" in capsys.readouterr().err


def test_cli_works_without_site_packages_or_training_imports(candidate):
    result = subprocess.run([sys.executable, "-S", "-m", "src.utils.accuracy_guard",
                             "--candidate-dir", str(candidate)], text=True, capture_output=True,
                            cwd=DEFAULT_BASELINE.parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No training performed" in result.stdout


def test_discovery_requires_explicit_unique_report_run(candidate, tmp_path):
    run_id = "20261019T120000Z"
    destination = tmp_path / "pulled" / "experiments" / "v2" / run_id
    shutil.copytree(candidate, destination)
    assert find_candidate_directory(tmp_path / "pulled", run_id) == destination
    # Another study/trial with the same run ID is ambiguous, never choose the first.
    shutil.copytree(candidate, tmp_path / "pulled" / "study" / run_id)
    with pytest.raises(ValueError, match="exactly one"):
        find_candidate_directory(tmp_path / "pulled", run_id)
    for unsafe in ("../escape", "$(echo x)", "", "x\ny"):
        with pytest.raises(ValueError, match="run ID"):
            find_candidate_directory(tmp_path, unsafe)


def test_discovery_rejects_missing_reports_and_symlinks(candidate, tmp_path):
    (candidate / "test_metrics.csv").unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        find_candidate_directory(tmp_path, "candidate")
    (candidate / "test_metrics.csv").symlink_to(DEFAULT_BASELINE / "test_metrics.csv")
    with pytest.raises(ValueError, match="Missing or unsafe"):
        find_candidate_directory(tmp_path, "candidate")


def test_workflow_has_explicit_fit_free_allowlist_and_safe_dispatch():
    workflow_path = DEFAULT_BASELINE.parents[1] / ".github" / "workflows" / "accuracy.yml"
    if not workflow_path.is_file():
        pytest.skip("Workflow files are intentionally not shipped in the Kaggle source bundle")
    yaml = pytest.importorskip("yaml")
    # BaseLoader avoids YAML 1.1 interpreting GitHub's `on` key as a boolean.
    workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    portable = workflow["jobs"]["portable-checks"]
    command = next(step["run"] for step in portable["steps"] if "pytest -q" in step.get("run", ""))
    assert command.split()[4:] == ["tests/test_accuracy_guard.py", "tests/test_metrics.py",
                                    "tests/test_evaluation.py", "tests/test_kaggle_sync.py",
                                    "tests/test_analysis.py", "tests/test_study.py", "tests/test_auxiliary.py",
                                    "tests/test_features.py", "tests/test_tail_risk.py", "tests/test_auxiliary_study.py"]
    cloud = workflow["jobs"]["cloud-accuracy"]
    assert cloud["if"] == "github.event_name == 'workflow_dispatch'"
    script = next(step["run"] for step in cloud["steps"] if "pull_run(" in step.get("run", ""))
    assert "${{" not in script
    assert "purplray/" in script
    assert "os.environ['GITHUB_SHA']" in script
    assert "--expected-source-commit" in script and "--expected-source-sha256" in script
    assert "find_candidate_directory(pulled, run_id)" in script
    assert "kernel_status(kernel)" in script
    assert "push_kernel" not in script and "exec(" not in script and "eval(" not in script
    ast.parse(script.split("\n", 1)[1].rsplit("\nPY", 1)[0])
