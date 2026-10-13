"""Download the NASA Airfoil Learning Dataset and convert it to two tables.

v1 wrote a single 1.7 GB CSV that repeated 100 geometry columns across all
768,542 polar rows. Geometry is constant per airfoil, so it is now stored once:
an 818-row geometry table of 1.7 MB, plus a 768k-row polars table of 55 MB.
That shrinks the upload by about thirty times, which matters every time a
Kaggle session starts, and :mod:`src.data.features` joins the two back
together in memory at the point of use.

The archive layout is verified rather than guessed: 825 JSON members under
``json/``, one airfoil per document, each with ``name``, ``xss``/``yss``
(suction, i.e. upper), ``xps``/``yps`` (pressure, i.e. lower), and ``polars``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import zipfile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from src.geometry.descriptors import describe_profile
from src.geometry.representation import geometry_columns, sample_profile

DEFAULT_URL = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip"

GEOMETRY_TABLE = "airfoil_geometry.csv"
POLARS_TABLE = "airfoil_polars.csv"
LOAD_REPORT = "load_report.json"

REQUIRED_GEOMETRY_KEYS = ("xps", "xss", "yps", "yss")
# Field names as they appear in the archive; Alpha is capitalized in our tables.
REQUIRED_POLAR_FIELDS = ("alpha", "Re", "Cl", "Cd")
OPTIONAL_POLAR_FIELDS = ("Ncrit", "Cm")


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
    """Return archive members so the real NASA layout can be inspected."""
    with zipfile.ZipFile(archive_path) as archive:
        return archive.namelist()


@dataclass
class DatasetTables:
    """The two joined-on-``geometry_id`` tables plus the conversion provenance."""

    geometry: pd.DataFrame
    polars: pd.DataFrame
    report: dict = field(default_factory=dict)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _oriented_surfaces(document: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(upper, lower)``, repairing a swapped suction/pressure pair.

    ``yss`` is documented as the suction (upper) surface, but an earlier commit
    had to fix exactly this swap once already, and a swap makes every thickness
    negative rather than failing loudly. The mean ``y`` of the upper surface
    always exceeds the lower's, so check per airfoil instead of trusting names.
    """
    suction = np.column_stack((document["xss"], document["yss"]))
    pressure = np.column_stack((document["xps"], document["yps"]))
    if suction[:, 1].mean() < pressure[:, 1].mean():
        return pressure, suction, True
    return suction, pressure, False


def _polar_rows(document: dict, geometry_id: str) -> tuple[list[dict], int, int]:
    """Convert one document's polars, counting incomplete and non-numeric drops."""
    rows, incomplete, nonfinite = [], 0, 0
    for polar in document["polars"]:
        if not all(key in polar for key in REQUIRED_POLAR_FIELDS):
            incomplete += 1
            continue
        if not all(_is_number(polar[key]) for key in REQUIRED_POLAR_FIELDS):
            nonfinite += 1
            continue
        row = {
            "geometry_id": geometry_id,
            "Alpha": float(polar["alpha"]),
            "Re": float(polar["Re"]),
            "Cl": float(polar["Cl"]),
            "Cd": float(polar["Cd"]),
        }
        for name in OPTIONAL_POLAR_FIELDS:
            if name in polar and _is_number(polar[name]):
                row[name] = float(polar[name])
        rows.append(row)
    return rows, incomplete, nonfinite


def load_dataset_tables(archive_path: str | Path, geometry_samples: int = 50) -> DatasetTables:
    """Convert the archive into a geometry table and a polars table.

    Airfoils whose polars all lack the required fields contribute no rows and
    get no geometry row either; their names are recorded in the report rather
    than dropped silently, which is what v1 did.
    """
    archive_path = Path(archive_path)
    geometry_rows: list[dict] = []
    polar_rows: list[dict] = []
    without_polars: list[str] = []
    swapped: list[str] = []
    clipped_by_airfoil: dict[str, int] = {}
    incomplete = nonfinite = 0
    columns = geometry_columns(geometry_samples)

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".json")
        ]
        if not members:
            raise ValueError(
                f"No JSON members found in {archive_path}. "
                f"Archive members: {discover_archive(archive_path)}"
            )
        for info in sorted(members, key=lambda item: item.filename):
            document = json.loads(archive.read(info))
            if not all(key in document for key in REQUIRED_GEOMETRY_KEYS) or "polars" not in document:
                raise ValueError(
                    f"JSON member {info.filename!r} lacks required geometry or polars fields."
                )
            geometry_id = Path(info.filename).stem
            rows, member_incomplete, member_nonfinite = _polar_rows(document, geometry_id)
            incomplete += member_incomplete
            nonfinite += member_nonfinite
            if not rows:
                without_polars.append(geometry_id)
                continue
            polar_rows.extend(rows)

            upper, lower, was_swapped = _oriented_surfaces(document)
            if was_swapped:
                swapped.append(geometry_id)
            profile = sample_profile(upper, lower, geometry_samples)
            if profile.clipped_stations:
                clipped_by_airfoil[geometry_id] = profile.clipped_stations
            geometry_rows.append({
                "geometry_id": geometry_id,
                "airfoil_name": document.get("name", geometry_id),
                **dict(zip(columns, np.concatenate((profile.thickness, profile.camber)))),
                **describe_profile(profile),
                "clipped_stations": profile.clipped_stations,
            })

    if not polar_rows:
        raise ValueError("No valid aerodynamic polar records were found in the archive.")
    geometry = pd.DataFrame(geometry_rows)
    polars = pd.DataFrame(polar_rows)
    report = {
        "archive": archive_path.name,
        "geometry_samples": geometry_samples,
        "json_members": len(members),
        "airfoils_with_polars": len(geometry),
        "airfoils_without_polars": without_polars,
        "polar_rows": len(polars),
        "skipped_incomplete_polars": incomplete,
        "skipped_nonfinite_polars": nonfinite,
        "duplicate_polar_rows": int(polars.duplicated().sum()),
        "surfaces_swapped": swapped,
        "airfoils_with_clipped_thickness": len(clipped_by_airfoil),
        "clipped_thickness_stations": int(sum(clipped_by_airfoil.values())),
        "worst_clipped_airfoils": dict(
            sorted(clipped_by_airfoil.items(), key=lambda item: (-item[1], item[0]))[:10]
        ),
        # A partially populated optional column must be dropped, not imputed:
        # a NaN would otherwise reach the design matrix.
        "optional_column_coverage": {
            name: int(polars[name].notna().sum()) for name in OPTIONAL_POLAR_FIELDS
            if name in polars.columns
        },
    }
    return DatasetTables(geometry=geometry, polars=polars, report=report)


def save_dataset_tables(
    tables: DatasetTables, processed_dir: str | Path = "data/processed",
) -> dict[str, Path]:
    """Write both tables and the load report, returning the written paths."""
    directory = Path(processed_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "geometry": directory / GEOMETRY_TABLE,
        "polars": directory / POLARS_TABLE,
        "report": directory / LOAD_REPORT,
    }
    tables.geometry.to_csv(paths["geometry"], index=False)
    tables.polars.to_csv(paths["polars"], index=False)
    paths["report"].write_text(json.dumps(tables.report, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--geometry-samples", type=int, default=50)
    args = parser.parse_args()

    archive = download_dataset(args.url, args.raw_dir)
    tables = load_dataset_tables(archive, args.geometry_samples)
    paths = save_dataset_tables(tables, args.processed_dir)
    report = tables.report

    print(f"Geometry: {len(tables.geometry):,} airfoils -> {paths['geometry']}")
    print(f"Polars:   {len(tables.polars):,} rows -> {paths['polars']}")
    print(f"JSON members: {report['json_members']:,}")
    if report["airfoils_without_polars"]:
        names = report["airfoils_without_polars"]
        print(f"Dropped {len(names)} airfoils with no usable polars: {names}")
    print(
        f"Skipped polar records: {report['skipped_incomplete_polars']:,} incomplete, "
        f"{report['skipped_nonfinite_polars']:,} non-numeric"
    )
    print(
        f"Clipped negative thickness at {report['clipped_thickness_stations']:,} stations "
        f"across {report['airfoils_with_clipped_thickness']:,} airfoils"
    )
    if report["surfaces_swapped"]:
        print(f"Corrected swapped surfaces for {len(report['surfaces_swapped'])} airfoils")
    print(f"Report: {paths['report']}")


if __name__ == "__main__":
    main()
