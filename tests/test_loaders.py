import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import (
    GEOMETRY_TABLE,
    LOAD_REPORT,
    POLARS_TABLE,
    load_dataset_tables,
    save_dataset_tables,
)
from src.geometry.descriptors import DESCRIPTOR_NAMES
from src.geometry.naca import generate_naca4


def document(code, polars, swap=False):
    upper, lower = generate_naca4(code, 60)
    if swap:
        upper, lower = lower, upper
    return {
        "name": f"NACA {code}",
        "xss": upper[:, 0].tolist(), "yss": upper[:, 1].tolist(),
        "xps": lower[:, 0].tolist(), "yps": lower[:, 1].tolist(),
        "polars": polars,
    }


def polar(alpha, **extra):
    return {"alpha": alpha, "Re": 1e6, "Cl": 0.4, "Cd": 0.008, "Ncrit": 9.0, "Cm": -0.05, **extra}


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "dataset.zip"
    members = {
        "json/good-a.json": document("2412", [polar(0.0), polar(5.0)]),
        "json/good-b.json": document("0012", [polar(2.0, Ncrit=5.0, Cm=0.0)]),
        "json/swapped.json": document("4415", [polar(1.0)], swap=True),
        "json/no-polars.json": document("0006", []),
        # Lacks Cl and Cd, so it contributes no rows at all.
        "json/incomplete.json": document("0021", [{"alpha": 1.0, "Re": 1e6}]),
    }
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, json.dumps(content))
    return path


def test_geometry_is_stored_once_per_airfoil_not_once_per_row(archive):
    """v1 repeated 100 geometry columns on all 768,542 rows for a 1.7 GB CSV."""
    tables = load_dataset_tables(archive, geometry_samples=10)
    assert tables.geometry["geometry_id"].tolist() == ["good-a", "good-b", "swapped"]
    assert len(tables.polars) == 4
    assert tables.polars["geometry_id"].tolist() == ["good-a", "good-a", "good-b", "swapped"]
    # The polars table carries no geometry columns at all.
    assert not [column for column in tables.polars.columns if column.startswith(("thickness_", "camber_"))]
    assert set(tables.polars.columns) == {"geometry_id", "Alpha", "Re", "Cl", "Cd", "Ncrit", "Cm"}


def test_geometry_table_carries_the_profile_and_the_descriptors(archive):
    geometry = load_dataset_tables(archive, geometry_samples=10).geometry
    profile = [f"thickness_{index:03d}" for index in range(10)]
    profile += [f"camber_{index:03d}" for index in range(10)]
    assert set(profile).issubset(geometry.columns)
    assert set(DESCRIPTOR_NAMES).issubset(geometry.columns)
    assert {"geometry_id", "airfoil_name", "clipped_stations"}.issubset(geometry.columns)
    assert geometry["airfoil_name"].tolist() == ["NACA 2412", "NACA 0012", "NACA 4415"]
    # The headline v1 bug: thickness must never be negative anywhere.
    assert (geometry[profile[:10]].to_numpy() >= 0).all()
    assert geometry[profile[:10]].to_numpy().max() > 0


def test_airfoils_without_usable_polars_are_logged_not_dropped_silently(archive):
    """v1 skipped these inside the loader with no record that it had done so."""
    report = load_dataset_tables(archive, geometry_samples=10).report
    assert report["airfoils_without_polars"] == ["incomplete", "no-polars"]
    assert report["json_members"] == 5
    assert report["airfoils_with_polars"] == 3
    assert report["polar_rows"] == 4
    assert report["skipped_incomplete_polars"] == 1
    assert report["skipped_nonfinite_polars"] == 0


def test_a_swapped_suction_pressure_pair_is_corrected_and_recorded(archive):
    """A prior commit had to fix this swap once; a swap inverts every thickness."""
    tables = load_dataset_tables(archive, geometry_samples=10)
    assert tables.report["surfaces_swapped"] == ["swapped"]
    swapped = tables.geometry.set_index("geometry_id").loc["swapped"]
    # A ten-station grid straddles rather than hits the peak at x = 0.30.
    assert swapped["max_thickness"] == pytest.approx(0.15, rel=2e-2)
    # Correcting the swap restores positive camber for a cambered section.
    assert swapped["max_camber"] > 0
    assert (tables.geometry.filter(regex=r"^thickness_\d+$").to_numpy() >= 0).all()


def test_non_numeric_polar_records_are_counted_separately(tmp_path):
    path = tmp_path / "odd.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("json/a.json", json.dumps(document("0012", [
            polar(1.0),
            polar(2.0, Cd=None),
            polar(3.0, Cl="high"),
            polar(4.0, Re=float("inf")),
        ])))
    report = load_dataset_tables(path, geometry_samples=8).report
    assert report["polar_rows"] == 1
    assert report["skipped_nonfinite_polars"] == 3
    assert report["skipped_incomplete_polars"] == 0


def test_crossed_coordinates_are_clipped_and_counted(tmp_path):
    upper, lower = generate_naca4("0006", 60)
    crossed = lower.copy()
    crossed[20:40, 1] = upper[20:40, 1] + 0.002
    path = tmp_path / "crossed.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("json/x.json", json.dumps({
            "name": "crossed",
            "xss": upper[:, 0].tolist(), "yss": upper[:, 1].tolist(),
            "xps": crossed[:, 0].tolist(), "yps": crossed[:, 1].tolist(),
            "polars": [polar(0.0)],
        }))
    tables = load_dataset_tables(path, geometry_samples=20)
    assert tables.report["clipped_thickness_stations"] > 0
    assert tables.report["airfoils_with_clipped_thickness"] == 1
    assert tables.report["worst_clipped_airfoils"]["x"] > 0
    assert int(tables.geometry["clipped_stations"].iloc[0]) > 0
    assert (tables.geometry.filter(regex=r"^thickness_\d+$").to_numpy() >= 0).all()


def test_optional_column_coverage_is_reported(archive):
    report = load_dataset_tables(archive, geometry_samples=10).report
    assert report["optional_column_coverage"] == {"Ncrit": 4, "Cm": 4}
    assert report["duplicate_polar_rows"] == 0


def test_saving_writes_both_tables_and_the_report(archive, tmp_path):
    tables = load_dataset_tables(archive, geometry_samples=10)
    paths = save_dataset_tables(tables, tmp_path / "processed")
    assert {path.name for path in paths.values()} == {GEOMETRY_TABLE, POLARS_TABLE, LOAD_REPORT}
    pd.testing.assert_frame_equal(pd.read_csv(paths["geometry"]), tables.geometry)
    pd.testing.assert_frame_equal(pd.read_csv(paths["polars"]), tables.polars)
    assert json.loads(paths["report"].read_text()) == tables.report
    # The geometry table is the small one; that is the whole point of the split.
    assert paths["geometry"].stat().st_size < paths["polars"].stat().st_size * 10


def test_an_archive_without_json_members_is_rejected(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("readme.txt", "no airfoils here")
    with pytest.raises(ValueError, match="No JSON members"):
        load_dataset_tables(path)


def test_a_document_missing_geometry_is_rejected_loudly(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("json/a.json", json.dumps({"name": "a", "polars": [polar(0.0)]}))
    with pytest.raises(ValueError, match="lacks required geometry"):
        load_dataset_tables(path)


def test_an_archive_with_no_usable_polars_at_all_is_rejected(tmp_path):
    path = tmp_path / "none.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("json/a.json", json.dumps(document("0012", [])))
    with pytest.raises(ValueError, match="No valid aerodynamic polar records"):
        load_dataset_tables(path)
