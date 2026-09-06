"""Download, discover, and load the NASA Airfoil Learning Dataset."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

from src.geometry.representation import flatten_geometry

DEFAULT_URL = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip"


def download_dataset(url: str = DEFAULT_URL, raw_dir: str | Path = "data/raw") -> Path:
    """Download the archive reproducibly, without overwriting an existing file."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "airfoil-learning-dataset.zip"
    if not archive.exists():
        with urlopen(url) as response:  # noqa: S310 - URL is an explicit project input
            archive.write_bytes(response.read())
    return archive


def discover_archive(archive_path: str | Path) -> list[str]:
    """Return archive members so the real NASA layout can be inspected before parsing."""
    with zipfile.ZipFile(archive_path) as archive:
        return archive.namelist()


def _read_tabular_bytes(name: str, content: bytes) -> pd.DataFrame | None:
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(content))
    if suffix in {".txt", ".dat", ".tsv"}:
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", comment="#")
    return None


def load_raw_data(archive_path: str | Path) -> pd.DataFrame:
    """Load the NASA JSON collection or a supported tabular archive member.

    The NASA archive currently contains one JSON document per airfoil. JSON records
    are converted to one row per operating condition, with geometry repeated as
    fixed-length features so they can be used by classical ML models.
    """
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        json_members = [info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".json")]
        if json_members:
            return _load_json_members(archive, json_members)
        candidates = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            frame = _read_tabular_bytes(info.filename, archive.read(info))
            if frame is not None and not frame.empty:
                candidates.append((info.filename, frame))
    if not candidates:
        members = discover_archive(archive_path)
        raise ValueError(f"No non-empty CSV/TXT/DAT/TSV member found. Archive members: {members}")
    if len(candidates) > 1:
        # Prefer a filename that describes aerodynamic data, otherwise require explicit review.
        named = [item for item in candidates if any(k in item[0].lower() for k in ("airfoil", "data", "polar"))]
        if len(named) == 1:
            return named[0][1]
        raise ValueError(f"Multiple tabular members found; select the intended file explicitly: {[n for n, _ in candidates]}")
    return candidates[0][1]


def _load_json_members(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], geometry_samples: int = 50) -> pd.DataFrame:
    """Convert NASA per-airfoil JSON files into a row-wise aerodynamic table."""
    rows: list[dict] = []
    for info in members:
        document = json.loads(archive.read(info))
        required_geometry = ("xps", "xss", "yps", "yss")
        if not all(key in document for key in required_geometry) or "polars" not in document:
            raise ValueError(f"JSON member {info.filename!r} lacks required geometry or polars fields.")
        upper = list(zip(document["xps"], document["yps"]))
        lower = list(zip(document["xss"], document["yss"]))
        geometry = flatten_geometry(upper, lower, n_samples=geometry_samples)
        geometry_id = Path(info.filename).stem
        geometry_features = {f"geom_{index:03d}": float(value) for index, value in enumerate(geometry)}
        for polar in document["polars"]:
            if not all(key in polar for key in ("alpha", "Re", "Cl", "Cd")):
                continue
            row = {"geometry_id": geometry_id, "airfoil_name": document.get("name", geometry_id), "Alpha": polar["alpha"], "Re": polar["Re"], "Cl": polar["Cl"], "Cd": polar["Cd"]}
            if "Ncrit" in polar:
                row["Ncrit"] = polar["Ncrit"]
            if "Cm" in polar:
                row["Cm"] = polar["Cm"]
            row.update(geometry_features)
            rows.append(row)
    if not rows:
        raise ValueError("No valid aerodynamic polar records were found in the JSON archive.")
    return pd.DataFrame(rows)


def save_processed_data(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-path", default="data/processed/airfoil_learning.csv")
    args = parser.parse_args()
    archive = download_dataset(args.url, args.raw_dir)
    print("Archive members:", discover_archive(archive))
    save_processed_data(load_raw_data(archive), args.processed_path)


if __name__ == "__main__":
    main()
