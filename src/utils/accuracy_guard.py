"""Compare completed Kaggle reports without importing models or fitting anything.

Relative errors are fractions. A 0.5 percentage-point tolerance is 0.005,
not a 0.5% relative change. Only degradation fails; improvements are allowed.
Reports are evidence, not a proof that their producer never consulted test data.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


DEFAULT_BASELINE = Path(__file__).resolve().parents[2] / "configs" / "accuracy_baseline"
TARGETS = ("Cl", "Cd", "Cm")
SPLITS = ("train", "validation", "test")
METRICS = ("mae", "rmse", "r2", "mean_relative_error", "median_relative_error")
COLUMNS = ("model", "target", *METRICS, "batch_inference_time_seconds", "n_predictions")
REPORT_FILES = ("run.json", "test_evaluation.json", "test_metrics.csv")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_float=Decimal, parse_constant=_invalid_constant,
                         object_pairs_hook=_unique_object)


def _number(value, label, *, nonnegative=True):
    _require(not isinstance(value, bool) and isinstance(value, (str, int, float, Decimal)),
             f"{label} must be numeric")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{label} must be numeric") from error
    _require(number.is_finite() and (not nonnegative or number >= 0),
             f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return number


def _count(value, label):
    number = _number(value, label)
    _require(number > 0 and number == number.to_integral_value(),
             f"{label} must be a positive integer")
    return int(number)


def _records(records, selected, count, label):
    _require(isinstance(records, list), f"{label} must contain records")
    result = {}
    for record in records:
        _require(isinstance(record, dict) and set(record) == set(COLUMNS),
                 f"{label} requires exactly the full test metric columns")
        target = record["target"]
        _require(target in TARGETS and target not in result,
                 f"{label} duplicate or unknown target: {target}")
        _require(record["model"] == selected[target], f"{label} selected model mismatch for {target}")
        _require(_count(record["n_predictions"], f"{label} n_predictions") == count,
                 f"{label} n_predictions must match held-out test row count ({count})")
        result[target] = {
            key: _number(record[key], f"{label} {target} {key}", nonnegative=key != "r2")
            for key in (*METRICS, "batch_inference_time_seconds")
        }
        _require(result[target]["r2"] <= 1, f"{label} r2 cannot exceed 1")
    _require(set(result) == set(TARGETS), f"{label} missing targets")
    return result


def load_report(directory):
    """Validate the three report files, never load weights or execute artifacts."""
    directory = Path(directory)
    try:
        run = _json(directory / "run.json")
        evaluation = _json(directory / "test_evaluation.json")
        _require(run["status"] == "completed", "run status must be completed")
        metadata = run["run_metadata"]
        _require(metadata["platform"] == "kaggle", "run platform must be kaggle")
        _require(metadata["source_mode"] == "bundle", "source_mode must be bundle")
        if "test_used_for_tuning" in metadata:
            _require(metadata["test_used_for_tuning"] is False, "Test data must not be used for tuning")
        for key, length in (("git_commit", 40), ("source_sha256", 64)):
            _require(isinstance(metadata[key], str) and
                     re.fullmatch(rf"[0-9a-f]{{{length}}}", metadata[key]), f"Invalid {key}")
        _require(isinstance(metadata["created_at_utc"], str) and metadata["created_at_utc"],
                 "Missing run creation time")
        accelerator = metadata["accelerator"]
        _require(accelerator in ("cpu", "gpu"), "Unknown run accelerator")
        _require(metadata[f"{accelerator}_tests_passed"] is True, "Cloud fitting preflight not passed")
        gpu = metadata["gpu"]
        _require(isinstance(gpu["torch_version"], str) and gpu["torch_version"]
                 and gpu["torch_version"].lower() != "unknown",
                 "Missing torch runtime version")
        if accelerator == "gpu":
            _require(_count(gpu["device_count"], "GPU count") > 0 and gpu["device_name"]
                     and gpu["cuda_version"], "Incomplete actual GPU provenance")
        else:
            _require(gpu["device_count"] == 0 and gpu["device_name"] is None,
                     "CPU run has inconsistent GPU provenance")

        dataset = metadata["dataset"]
        for key in ("geometry_sha256", "polars_sha256"):
            _require(isinstance(dataset[key], str) and re.fullmatch(r"[0-9a-f]{64}", dataset[key]),
                     f"Invalid dataset {key}")
        rows, groups = metadata["split_rows"], metadata["split_geometries"]
        _require(set(rows) == set(groups) == set(SPLITS), "Missing or extra splits")
        seen = set()
        for split in SPLITS:
            _count(rows[split], f"{split} rows")
            names = groups[split]
            _require(isinstance(names, list) and names and
                     all(isinstance(name, str) and name for name in names), "Invalid split lists")
            _require(len(set(names)) == len(names) and not seen.intersection(names),
                     "Duplicate or overlapping split geometries")
            seen.update(names)
        _require(sum(rows.values()) == _count(dataset["polar_rows"], "polar_rows"),
                 "Split row counts do not sum to dataset polar_rows")
        _require(len(seen) == _count(dataset["geometry_rows"], "geometry_rows"),
                 "Split geometry counts do not match dataset")
        _require(metadata["target_columns"] == list(TARGETS), "Unexpected metadata targets")
        _require(run["target_names"] == list(TARGETS), "Unexpected run targets")
        _require(set(run["target_transforms"]) == set(TARGETS), "Missing target transforms")
        _require(run["selection_metric"] == "validation_per_target",
                 "Model selection must use validation, never test or CV reports")
        _require(set(run["selection_metrics"]) == set(TARGETS), "Missing selection metrics")
        _require(all(metric in METRICS for metric in run["selection_metrics"].values()),
                 "Unknown validation selection metric")
        selected = run["selected_models"]
        _require(set(selected) == set(TARGETS), "Missing selected targets")
        configs = run["model_configs"]
        _require(isinstance(configs, list) and configs, "Missing model configs")
        names = [config["name"] for config in configs]
        _require(len(set(names)) == len(names) and set(names) == set(run["results"]),
                 "Duplicate or inconsistent model configs/results")
        for config in configs:
            name = config["name"]
            result = run["results"][name]
            _require(result["status"] == "success", f"Model {name} status must be success")
            _require(result["parameters"] == config["parameters"] and
                     result["scale_targets"] == config["scale_targets"],
                     f"Model {name} parameters differ from manifest")
            _require(_count(result["n_predictions"], "validation n_predictions") == rows["validation"],
                     "Validation prediction count differs from split")
            _require(set(result["targets"]) == set(TARGETS), "Missing validation targets")
            for target in TARGETS:
                for metric in METRICS:
                    _number(result["targets"][target][metric], f"validation {name} {target} {metric}",
                            nonnegative=metric != "r2")
            training = result["training_metadata"]
            _require(_count(training["n_iter"], "n_iter") <= _count(training["max_iter"], "max_iter"),
                     "Invalid training iteration counts")
            if name == "mlp":
                # This shipped sklearn estimator records loss-based stopping, not a device field.
                _number(training["tol"], "mlp stopping tolerance")
                _count(training["n_iter_no_change"], "mlp stopping patience")
                _require(config["parameters"].get("early_stopping") is False,
                         "MLP must not use an unrecorded row-random stopping split")
            else:
                device = training["device"]
                _require(training["accelerator"] == ("cuda" if accelerator == "gpu" else "cpu"),
                         f"Model {name} accelerator differs from actual run")
                _require(isinstance(device, str) and
                         (re.fullmatch(r"cuda(?::[0-9]+)?", device) if accelerator == "gpu"
                          else device == "cpu"), "Unknown or inconsistent actual training device")
                _require(isinstance(training["device_name"], str) and training["device_name"]
                         and training["device_name"].lower() != "unknown"
                         and training["torch_version"] == gpu["torch_version"],
                         "Missing or inconsistent training hardware provenance")
                _require(training["cuda_version"] == gpu["cuda_version"] and
                         training["cuda_available"] is (accelerator == "gpu"),
                         "Training CUDA provenance differs from actual run")
                if accelerator == "gpu":
                    _require(training["device_name"] == gpu["device_name"], "GPU device name mismatch")
                requested = config["parameters"].get("device", "auto")
                _require(requested in ("auto", device) or
                         (requested == "cuda" and accelerator == "gpu"),
                         "Requested device differs from actual training device")
                _require(training["n_train"] == rows["train"] and
                         training["n_validation"] == rows["validation"],
                         "Training/stopping counts differ from frozen splits")
                _require(training["target_names"] == list(TARGETS), "Training target order differs")
                _require(training["stop_reason"] in ("validation_patience", "max_iter") and
                         training["restored_best_checkpoint"] is True,
                         "Unknown stopping policy or best validation checkpoint not restored")
                _require(_count(training["best_epoch"], "best_epoch") <= training["n_iter"],
                         "Invalid best validation epoch")
                _number(training["best_loss"], "best validation loss")
                _require(training["loss"] == config["parameters"]["loss"], "Training loss mismatch")
        for target, name in selected.items():
            _require(name in run["results"], f"Unknown selected model for {target}")
            metric = run["selection_metrics"][target]
            scores = {model: _number(result["targets"][target][metric], "selection score",
                                      nonnegative=metric != "r2")
                      for model, result in run["results"].items()}
            best = max(scores.values()) if metric == "r2" else min(scores.values())
            _require(scores[name] == best, f"{target} selected model is not validation winner")
        for key in ("target_names", "target_transforms", "selected_models",
                    "selection_metric", "selection_metrics"):
            _require(evaluation[key] == run[key], f"test_evaluation {key} differs from run manifest")
        json_records = _records(evaluation["results"], selected, rows["test"], "test JSON")
        with (directory / "test_metrics.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames == list(COLUMNS), "CSV requires full test metric columns")
            csv_records = _records(list(reader), selected, rows["test"], "test CSV")
        _require(csv_records == json_records, "CSV and JSON test metrics are inconsistent")
        return run, csv_records
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed or missing report field in {directory}: {error}") from error


def compare_reports(candidate_dir, baseline_dir=DEFAULT_BASELINE, tolerance_pp=0.5, *,
                    expected_source_commit=None, expected_source_sha256=None):
    """Return target rows with signed pp deltas; a strictly positive excess fails."""
    tolerance = _number(tolerance_pp, "tolerance_pp")
    baseline_dir = Path(baseline_dir)
    provenance = _json(baseline_dir / "provenance.json")
    _require(set(provenance["files_sha256"]) == set(REPORT_FILES), "Incomplete baseline fingerprints")
    for name, digest in provenance["files_sha256"].items():
        _require(hashlib.sha256((baseline_dir / name).read_bytes()).hexdigest() == digest,
                 f"Pinned baseline fingerprint mismatch: {name}")
    baseline, old = load_report(baseline_dir)
    candidate, new = load_report(candidate_dir)
    for key in ("dataset", "split_rows", "split_geometries", "target_columns"):
        _require(candidate["run_metadata"][key] == baseline["run_metadata"][key],
                 f"Candidate {key} differs from pinned baseline")
    for key in ("target_names", "target_transforms"):
        _require(candidate[key] == baseline[key], f"Candidate {key} differs from pinned baseline")
    for key, expected, length in (("git_commit", expected_source_commit, 40),
                                  ("source_sha256", expected_source_sha256, 64)):
        if expected is not None:
            _require(isinstance(expected, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", expected),
                     f"Expected {key} must be a full lowercase hexadecimal digest")
            _require(candidate["run_metadata"][key] == expected, f"Candidate {key} does not match expected source")
    return [{"target": target, "baseline": old[target]["median_relative_error"],
             "candidate": new[target]["median_relative_error"],
             "delta_pp": (new[target]["median_relative_error"] - old[target]["median_relative_error"]) * 100,
             "passed": new[target]["median_relative_error"] - old[target]["median_relative_error"]
             <= tolerance / 100} for target in TARGETS]


def find_candidate_directory(root, run_id):
    """Locate exactly one explicit run ID in pulled reports, never import it."""
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id),
             "candidate run ID must be a short alphanumeric name")
    root = Path(root).resolve()
    matches = [path.parent for path in root.rglob("run.json") if path.parent.name == run_id]
    _require(len(matches) == 1, f"Expected exactly one run.json for {run_id}; found {len(matches)}")
    candidate = matches[0]
    for name in REPORT_FILES:
        path = candidate / name
        _require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                 f"Missing or unsafe candidate report: {name}")
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tolerance-pp", default="0.5")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    try:
        rows = compare_reports(args.candidate_dir, args.baseline_dir, args.tolerance_pp,
                               expected_source_commit=args.expected_source_commit,
                               expected_source_sha256=args.expected_source_sha256)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, f"Accuracy guard FAIL: {error}\n")
    print(f"Held-out test median relative error (fractions); degradation tolerance {args.tolerance_pp} pp")
    print("Target       Baseline      Candidate       Delta (pp)  Result")
    for row in rows:
        print(f"{row['target']:<8} {row['baseline']:>13.10f} {row['candidate']:>14.10f} "
              f"{row['delta_pp']:>+16.10f}  {'PASS' if row['passed'] else 'FAIL'}")
    print("Improvements are allowed; positive deltas mean worse accuracy. No training performed.")
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
