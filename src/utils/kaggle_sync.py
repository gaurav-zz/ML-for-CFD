"""Move data to and from Kaggle with the official CLI.

Kaggle has no Drive-style live mount. A notebook reads its inputs from
``/kaggle/input/<dataset>`` and writes to ``/kaggle/working``, which Kaggle
keeps as the kernel's output. The round trip is therefore explicit: push the
processed tables as a dataset version, run the notebook, then pull the kernel
output back into ``results/kaggle/<run-id>/``.

Only the two processed tables and their load report are ever pushed -- about
41 MB. The 1.5 GB raw archive stays local, and model weights stay on Kaggle
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

from src.data.loaders import GEOMETRY_TABLE, LOAD_REPORT, POLARS_TABLE

PUSH_FILES = (GEOMETRY_TABLE, POLARS_TABLE, LOAD_REPORT)
DEFAULT_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "kaggle"
# Kaggle refs are "owner/slug"; both parts allow letters, digits, and dashes.
REFERENCE = re.compile(r"^[A-Za-z0-9][\w-]{0,59}/[A-Za-z0-9][\w-]{0,59}$", re.ASCII)


def _reference(value: str, label: str) -> str:
    if not isinstance(value, str) or not REFERENCE.match(value):
        raise ValueError(f"{label} must look like 'owner/slug', got {value!r}.")
    return value


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
    # Kaggle requires a title of at least six characters.
    title = dataset.split("/", 1)[1].replace("-", " ").strip() or "airfoil tables"
    (staging_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {"id": dataset, "title": title.title().ljust(6, "."), "licenses": [{"name": "CC0-1.0"}]},
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

    pull = subcommands.add_parser("pull", help="Download a kernel's output files.")
    pull.add_argument("kernel", help="Kaggle kernel ref, e.g. yourname/airfoil-v2-experiment.")
    pull.add_argument("--destination", type=Path, default=DEFAULT_RESULTS)
    pull.add_argument("--run-id", help="Subdirectory name; defaults to a UTC timestamp.")

    args = parser.parse_args()
    try:
        if args.command == "push":
            output = push_tables(
                args.dataset, args.processed_dir, args.message, args.create,
            )
            print(output.strip() or f"Pushed {list(PUSH_FILES)} to {args.dataset}.")
        else:
            destination = pull_run(args.kernel, args.destination, args.run_id)
            for path in sorted(destination.iterdir()):
                print(f"Local report: {path}")
            print(f"Kernel output ready in {destination}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Kaggle transfer failed: {error}\n")


if __name__ == "__main__":
    main()
