"""Copy only completed Colab reports from Drive for Desktop into the workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


REPORT_FILES = {
    "baseline": ("run.json", "metrics.csv", "predicted_vs_actual.png"),
    "model_comparison": (
        "run.json", "comparison.csv", "test_evaluation.json", "test_metrics.csv",
        "validation_rmse.png", "selected_predictions.png",
    ),
    "mlp_improvement": (
        "run.json", "comparison.csv", "validation_by_geometry.csv", "validation_summary.json",
        "validation_rmse.png", "training_loss.png", "validation_predictions.png", "worst_airfoils.png",
    ),
}
REPORT_FILES["mlp_validation"] = REPORT_FILES["mlp_improvement"] + ("validation_history.csv", "validation_curves.png")
VALIDATION_EXPERIMENTS = ("mlp_improvement", "mlp_validation")
FINAL_TEST_FILES = ("test_evaluation.json", "test_metrics.csv")
MAX_FILE_BYTES = 10 * 1024 * 1024
DEFAULT_DESTINATION = Path(__file__).resolve().parents[2] / "results" / "colab"


def discover_sources(cloud_storage: Path | None = None) -> list[Path]:
    """Discover only this project's experiment folders, not other Drive contents."""
    root = cloud_storage or Path.home() / "Library" / "CloudStorage"
    return sorted(root.glob("GoogleDrive-*/My Drive/airfoil-ml-surrogate/experiments"))


def sync_results(source: str | Path, destination: str | Path = DEFAULT_DESTINATION) -> list[Path]:
    """Import complete reports without loading models or copying datasets.

    Existing identical snapshots are reused. Conflicting snapshots are never
    overwritten. Incomplete runs or still-syncing images are left for the next call.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Experiment directory is unavailable: {source}")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Report destination must not overlap the source experiment directory.")

    imported = []
    for experiment, filenames in REPORT_FILES.items():
        folder = source / experiment
        if folder.is_symlink():
            raise ValueError(f"Refusing a symlinked experiment directory: {folder}")
        if not folder.is_dir():
            continue
        for run in sorted(folder.iterdir()):
            if run.is_symlink() or not run.is_dir():
                continue
            has_final_test = experiment in VALIDATION_EXPERIMENTS and all((run / name).is_file() for name in FINAL_TEST_FILES)
            selected_files = filenames + FINAL_TEST_FILES if has_final_test else filenames
            paths = [run / name for name in selected_files]
            if any(path.is_symlink() for path in paths):
                raise ValueError(f"Refusing a symlinked report file in {run}")
            if not all(path.is_file() for path in paths):
                continue
            if any(path.stat().st_size > MAX_FILE_BYTES for path in paths):
                raise ValueError(f"Report file exceeds the {MAX_FILE_BYTES // 1024**2} MiB limit in {run}")

            payloads = {}
            for path in paths:
                with path.open("rb") as handle:
                    payload = handle.read(MAX_FILE_BYTES + 1)
                if len(payload) > MAX_FILE_BYTES:
                    raise ValueError(f"Report file exceeds the size limit: {path}")
                payloads[path.name] = payload
            # A visible PNG may still be syncing; require its final IEND chunk.
            if any(
                not payload or (
                    name.endswith(".png")
                    and not (payload.startswith(b"\x89PNG\r\n\x1a\n") and payload.endswith(b"IEND\xaeB`\x82"))
                )
                for name, payload in payloads.items()
            ):
                continue
            try:
                metadata = {name: json.loads(data) for name, data in payloads.items() if name.endswith(".json")}
            except ValueError:
                continue
            if not all(isinstance(value, dict) for value in metadata.values()):
                continue
            if experiment == "model_comparison" or experiment in VALIDATION_EXPERIMENTS:
                manifest = metadata["run.json"]
                if manifest.get("status") not in ("completed", "completed_with_failures"):
                    continue
                selected = manifest.get("selected_models")
                summary = metadata["validation_summary.json"] if experiment in VALIDATION_EXPERIMENTS else metadata["test_evaluation.json"]
                if not selected or selected != summary.get("selected_models"):
                    raise ValueError(f"Validation and report selections disagree in {run}")
                if experiment in VALIDATION_EXPERIMENTS and summary.get("evaluation_scope") != "validation":
                    raise ValueError(f"Expected validation-only improvement diagnostics in {run}")
                if has_final_test and selected != metadata["test_evaluation.json"].get("selected_models"):
                    raise ValueError(f"Validation and test selections disagree in {run}")

            # A later final-test evaluation is a new snapshot, not a rewrite of validation reports.
            output = destination / experiment / (f"{run.name}_with_test" if has_final_test else run.name)
            if output.parent.is_symlink():
                raise ValueError(f"Refusing a symlinked destination directory: {output.parent}")
            if output.exists() or output.is_symlink():
                if output.is_symlink() or not output.is_dir() or not all(
                    (output / name).is_file() and not (output / name).is_symlink()
                    and (output / name).stat().st_size == len(data)
                    and (output / name).read_bytes() == data
                    for name, data in payloads.items()
                ):
                    raise FileExistsError(f"A different local report already exists: {output}")
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".report-", dir=output.parent) as staging:
                    for name, data in payloads.items():
                        (Path(staging) / name).write_bytes(data)
                    Path(staging).rename(output)
            imported.append(output)
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="The synced project's experiments directory; auto-detected on macOS.")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    try:
        sources = [args.source] if args.source else discover_sources()
        if not sources:
            raise FileNotFoundError("No project Drive folder found. Start Drive for Desktop or provide --source.")
        reports = [report for source in sources for report in sync_results(source, args.destination)]
    except (OSError, ValueError) as error:
        parser.exit(1, f"Report sync failed: {error}\nCheck Drive for Desktop connectivity and retry.\n")
    if not reports:
        print("No complete reports are available yet. Finish the Colab run and let Drive sync, then retry.")
    for report in reports:
        print(f"Local report: {report}")
    print(f"{len(reports)} report folders ready; no datasets or model weights copied.")


if __name__ == "__main__":
    main()
