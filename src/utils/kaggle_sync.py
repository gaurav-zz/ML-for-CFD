"""Move data to and from Kaggle with the official CLI.

Kaggle has no Drive-style live mount. A notebook reads its inputs from
``/kaggle/input/<dataset>`` and writes to ``/kaggle/working``, which Kaggle
keeps as the kernel's output. The round trip is therefore explicit: push the
processed tables as a dataset version, run the notebook, then pull the kernel
output back into ``results/kaggle/<run-id>/``.

Only the two processed tables and their load report are ever pushed -- about
57 MB. The 1.5 GB raw archive stays local, and model weights stay on Kaggle
unless you ask for them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from src.data.loaders import GEOMETRY_TABLE, LOAD_REPORT, POLARS_TABLE

PUSH_FILES = (GEOMETRY_TABLE, POLARS_TABLE, LOAD_REPORT)
DEFAULT_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "kaggle"
DEFAULT_NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "notebooks" / "v2_experiment_kaggle.ipynb"
)
# Kaggle refs are "owner/slug"; both parts allow letters, digits, and dashes.
REFERENCE = re.compile(r"^[A-Za-z0-9][\w-]{0,59}/[A-Za-z0-9][\w-]{0,59}$", re.ASCII)


def _reference(value: str, label: str) -> str:
    if not isinstance(value, str) or not REFERENCE.match(value):
        raise ValueError(f"{label} must look like 'owner/slug', got {value!r}.")
    return value


def _title(reference: str, minimum: int) -> str:
    """Derive a display title from a slug, padded to Kaggle's minimum length."""
    slug = reference.split("/", 1)[1].replace("-", " ").replace("_", " ").strip()
    return (slug.title() or "Airfoil tables").ljust(minimum, ".")


def _run(command: list[str], runner=subprocess.run) -> str:
    """Invoke the Kaggle CLI, turning its failures into actionable messages."""
    try:
        completed = runner(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "The kaggle CLI is not installed. Run `pip install kaggle`, then save an API "
            "token to ~/.kaggle/kaggle.json from your Kaggle account settings."
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"`{' '.join(command)}` failed: {detail}") from error
    return completed.stdout or ""


def stage_dataset(dataset: str, staging_dir: Path, processed_dir: Path = DEFAULT_PROCESSED) -> Path:
    """Copy just the pushable files into a clean folder with Kaggle metadata.

    Staging matters: ``kaggle datasets version`` uploads a whole directory, and
    ``data/processed`` also holds the multi-gigabyte intermediates we keep local.
    """
    _reference(dataset, "dataset")
    processed_dir, staging_dir = Path(processed_dir), Path(staging_dir)
    missing = [name for name in PUSH_FILES if not (processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {processed_dir}. Run `python -m src.data.loaders` first."
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name in PUSH_FILES:
        shutil.copy2(processed_dir / name, staging_dir / name)
    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": _title(dataset, 6),
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return staging_dir


def push_tables(
    dataset: str,
    processed_dir: Path = DEFAULT_PROCESSED,
    message: str | None = None,
    create: bool = False,
    runner=subprocess.run,
) -> str:
    """Upload the processed tables as a new dataset version.

    Pass ``create=True`` the first time, when the dataset does not exist yet.
    """
    staging = Path(processed_dir) / ".kaggle-upload"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        stage_dataset(dataset, staging, processed_dir)
        if create:
            command = ["kaggle", "datasets", "create", "-p", str(staging), "--dir-mode", "skip"]
        else:
            note = message or f"v2 tables {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
            command = ["kaggle", "datasets", "version", "-p", str(staging), "-m", note,
                       "--dir-mode", "skip"]
        return _run(command, runner)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def stage_kernel(
    kernel: str,
    dataset: str,
    staging_dir: Path,
    notebook: Path = DEFAULT_NOTEBOOK,
    private: bool = True,
) -> Path:
    """Write the notebook and its ``kernel-metadata.json`` into a clean folder.

    Internet is enabled because the notebook clones the source checkout inside
    the session, and the tables dataset is attached as its only input. No GPU:
    every model in the queue is a CPU scikit-learn estimator.
    """
    _reference(kernel, "kernel")
    _reference(dataset, "dataset")
    notebook, staging_dir = Path(notebook), Path(staging_dir)
    if not notebook.is_file():
        raise FileNotFoundError(f"Notebook not found: {notebook}")
    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook, staging_dir / notebook.name)
    (staging_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel,
                "title": _title(kernel, 5),
                "code_file": notebook.name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": bool(private),
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return staging_dir


def push_kernel(
    kernel: str,
    dataset: str,
    notebook: Path = DEFAULT_NOTEBOOK,
    private: bool = True,
    runner=subprocess.run,
) -> str:
    """Upload the notebook and queue a run.

    Pushing *is* what starts execution on Kaggle: there is no separate run verb.
    Pushing an existing slug replaces that notebook and runs it again.
    """
    staging = Path(tempfile.mkdtemp(prefix="kaggle-kernel-"))
    try:
        stage_kernel(kernel, dataset, staging, notebook, private)
        return _run(["kaggle", "kernels", "push", "-p", str(staging)], runner)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def kernel_status(kernel: str, runner=subprocess.run) -> str:
    """Report whether the queued run is still running, complete, or errored."""
    _reference(kernel, "kernel")
    return _run(["kaggle", "kernels", "status", kernel], runner)


def pull_run(
    kernel: str,
    destination_root: Path = DEFAULT_RESULTS,
    run_id: str | None = None,
    runner=subprocess.run,
) -> Path:
    """Download a kernel's output into a fresh ``results/kaggle/<run-id>/``.

    Each pull lands in its own directory and an existing one is never
    overwritten, so a finished run cannot be clobbered by a later partial pull.
    """
    _reference(kernel, "kernel")
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    if not re.fullmatch(r"[\w-]{1,64}", run_id, re.ASCII):
        raise ValueError(f"run_id must be a short alphanumeric name, got {run_id!r}.")
    destination = Path(destination_root) / run_id
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; pass a different run_id.")
    destination.mkdir(parents=True)
    try:
        _run(["kaggle", "kernels", "output", kernel, "-p", str(destination)], runner)
    except Exception:
        # Leave no empty or half-filled directory behind on failure.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    if not any(destination.iterdir()):
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(
            f"{kernel} produced no output files. Check that the notebook finished successfully."
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    push = subcommands.add_parser("push", help="Upload the processed tables as a dataset version.")
    push.add_argument("dataset", help="Kaggle dataset ref, e.g. yourname/airfoil-ml-surrogate-v2.")
    push.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    push.add_argument("-m", "--message", help="Version note recorded on Kaggle.")
    push.add_argument("--create", action="store_true", help="Create the dataset for the first time.")

    kernel = subcommands.add_parser("kernel", help="Upload the notebook and queue a run.")
    kernel.add_argument("kernel", help="Kaggle kernel ref, e.g. yourname/airfoil-v2-experiment.")
    kernel.add_argument("--dataset", required=True, help="Tables dataset to attach as input.")
    kernel.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    kernel.add_argument(
        "--public", action="store_true", help="Publish the notebook instead of keeping it private.",
    )

    status = subcommands.add_parser("status", help="Check a queued or finished run.")
    status.add_argument("kernel", help="Kaggle kernel ref.")

    pull = subcommands.add_parser("pull", help="Download a kernel's output files.")
    pull.add_argument("kernel", help="Kaggle kernel ref, e.g. yourname/airfoil-v2-experiment.")
    pull.add_argument("--destination", type=Path, default=DEFAULT_RESULTS)
    pull.add_argument("--run-id", help="Subdirectory name; defaults to a UTC timestamp.")

    args = parser.parse_args()
    try:
        if args.command == "push":
            output = push_tables(args.dataset, args.processed_dir, args.message, args.create)
            print(output.strip() or f"Pushed {list(PUSH_FILES)} to {args.dataset}.")
        elif args.command == "kernel":
            output = push_kernel(
                args.kernel, args.dataset, args.notebook, private=not args.public,
            )
            print(output.strip() or f"Pushed {args.notebook.name} to {args.kernel}.")
            print(f"Track it with: python -m src.utils.kaggle_sync status {args.kernel}")
        elif args.command == "status":
            print(kernel_status(args.kernel).strip())
        else:
            destination = pull_run(args.kernel, args.destination, args.run_id)
            for path in sorted(destination.iterdir()):
                print(f"Local report: {path}")
            print(f"Kernel output ready in {destination}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Kaggle transfer failed: {error}\n")


if __name__ == "__main__":
    main()
