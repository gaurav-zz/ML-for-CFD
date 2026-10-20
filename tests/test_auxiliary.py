import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.auxiliary import COLUMNS, build_auxiliary, load_auxiliary, main


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, rows=None, members=None):
    rows = rows if rows is not None else [
        {"geometry_id": "b", "Alpha": 2.0, "Re": 1e6, "Ncrit": 9.0, "Cl": 0.3, "Cd": 0.01, "Cm": -0.1},
        {"geometry_id": "a", "Alpha": 0.0, "Re": 1e6, "Ncrit": 9.0, "Cl": 0.3, "Cd": 0.01, "Cm": -0.1},
        {"geometry_id": "c", "Alpha": 1.0, "Re": 1e6, "Ncrit": 9.0, "Cl": 0.3, "Cd": 0.01, "Cm": -0.1},
    ]
    polars = tmp_path / "airfoil_polars.csv"
    pd.DataFrame(rows).to_csv(polars, index=False)
    if members is None:
        members = {
            "json/b.json": {"name": "Not the geometry ID", "polars": [_polar(2, Cdp=-0.002)]},
            "json/a.json": {"polars": [_polar(0, Cdp=0)]},
            "json/c.json": {"polars": [_polar(1)]},
            "json/orphan.json": {"polars": [_polar(5, Cdp=0.02)]},
            "json/empty.json": {"polars": []},
        }
    raw = tmp_path / "source.zip"
    with zipfile.ZipFile(raw, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, json.dumps(content))
    return raw, polars, tmp_path / "output"


def _polar(alpha, **kwargs):
    return {"alpha": alpha, "Re": 1e6, "Ncrit": 9.0, "Cl": 0.3, "Cd": 0.01, "Cm": -0.1, **kwargs}


def _metadata(paths):
    return json.loads(paths["report"].read_text())


def test_exact_left_preservation_missing_negative_zero_orphans_and_privacy(tmp_path):
    inputs = _inputs(tmp_path)
    geometry = tmp_path / "airfoil_geometry.csv"
    pd.DataFrame({"geometry_id": ["a", "b", "c"], "clipped_stations": [0, 2, 0]}).to_csv(geometry, index=False)
    before = {path: path.read_bytes() for path in [inputs[0], inputs[1], geometry]}
    paths = build_auxiliary(*inputs)
    result = load_auxiliary(paths["sidecar"], inputs[1], paths["report"])
    report = _metadata(paths)
    assert list(result.columns) == COLUMNS
    assert result.geometry_id.tolist() == ["b", "a", "c"]
    assert result.Cdp.iloc[:2].tolist() == [-0.002, 0.0]
    assert np.isnan(result.Cdp.iloc[2])
    assert result.Cdp_observed.tolist() == [True, True, False]
    assert report["coverage"] == {"rows": 3, "observed": 2, "missing": 1, "fraction_observed": 2 / 3}
    assert report["source"]["nonbase_geometries"] == 2
    assert report["source"]["nonbase_unique_rows"] == 1
    assert report["source"]["empty_geometry_ids"] == ["empty"]
    assert report["source"]["base_rows_without_source_key"] == 0
    assert report["audit"]["clipping_defect_flags"]["flagged_rows"]["observed"] == 1
    assert report["audit"]["overall"]["Cdp"]["min"] == -0.002
    assert report["audit"]["overall"]["Cdp"]["zero"] == 1
    assert len(report["audit"]["by_geometry_Re"]) == 3
    assert report["schema"]["feature_columns"] == []
    assert report["schema"]["target_transform"] == "identity"
    assert report["identity"]["rows"] == 3
    assert report["source_zip_sha256"] == _sha(inputs[0])
    assert report["baseline_polars_sha256"] == _sha(inputs[1])
    assert report["sidecar_sha256"] == _sha(paths["sidecar"])
    assert all(path.read_bytes() == data for path, data in before.items())
    assert {p.name for p in inputs[2].iterdir()} == {"airfoil_auxiliary.csv", "auxiliary_report.json"}
    assert all(path.stat().st_mode & 0o077 == 0 for path in paths.values())


@pytest.mark.parametrize("values", [[0.01, 0.01], [None, None]])
def test_same_value_duplicates_are_explicitly_counted(tmp_path, values):
    inputs = _inputs(tmp_path, members={"a.json": {"polars": [_polar(0, Cdp=v) for v in values]}})
    paths = build_auxiliary(*inputs)
    report = _metadata(paths)
    assert report["source"]["duplicate_same_value_rows"] == 1
    assert report["source"]["unique_condition_keys"] == 1
    assert report["source"]["base_rows_without_source_key"] == 2
    assert len(load_auxiliary(paths["sidecar"], inputs[1], report)) == 3


@pytest.mark.parametrize("values", [[0.01, 0.02], [None, 0.01], [float("nan"), 0.01]])
def test_inconsistent_source_duplicates_fail_before_writing(tmp_path, values):
    inputs = _inputs(tmp_path, members={"a.json": {"polars": [_polar(0, Cdp=v) for v in values]}})
    with pytest.raises(ValueError, match="Inconsistent Cdp.*duplicate source condition keys"):
        build_auxiliary(*inputs)
    assert not inputs[2].exists()


@pytest.mark.parametrize("field", ["alpha", "Re", "Ncrit"])
@pytest.mark.parametrize("value", [True, None, "9", float("nan"), float("inf")])
def test_invalid_source_condition_keys_fail(tmp_path, field, value):
    polar = _polar(0, Cdp=0.01)
    polar[field] = value
    inputs = _inputs(tmp_path, members={"a.json": {"polars": [polar]}})
    with pytest.raises(ValueError, match=f"condition key {field}"):
        build_auxiliary(*inputs)


@pytest.mark.parametrize("value", [True, "0.01", [], {}, float("inf"), -float("inf")])
def test_invalid_cdp_fails_even_for_nonbase_source_rows(tmp_path, value):
    inputs = _inputs(tmp_path, members={"orphan.json": {"polars": [_polar(0, Cdp=value)]}})
    with pytest.raises(ValueError, match="Cdp must be a finite numeric non-boolean scalar"):
        build_auxiliary(*inputs)


def test_absent_null_nan_labels_and_absent_keys_remain_missing(tmp_path):
    inputs = _inputs(tmp_path, members={
        "a.json": {"polars": [_polar(0, Cdp=None)]},
        "b.json": {"polars": [_polar(2, Cdp=float("nan"))]},
    })
    paths = build_auxiliary(*inputs)
    result = load_auxiliary(paths["sidecar"], inputs[1], paths["report"])
    assert result.Cdp.isna().all() and not result.Cdp_observed.any()
    assert _metadata(paths)["source"]["base_rows_without_source_key"] == 1
    assert _metadata(paths)["audit"]["overall"]["Cdp"]["min"] is None


def test_all_empty_source_geometries_keep_baseline_rows(tmp_path):
    inputs = _inputs(tmp_path, members={"empty.json": {"polars": []}})
    paths = build_auxiliary(*inputs)
    assert _metadata(paths)["coverage"]["missing"] == 3


@pytest.mark.parametrize("defect", ["duplicate", "missing_key", "bool_key", "nonfinite", "empty_id"])
def test_invalid_baseline_identity_fails(tmp_path, defect):
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    if defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif defect == "missing_key":
        frame = frame.drop(columns="Ncrit")
    elif defect == "bool_key":
        frame["Alpha"] = True
    elif defect == "nonfinite":
        frame.loc[0, "Alpha"] = np.inf
    else:
        frame.loc[0, "geometry_id"] = ""
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(ValueError, match="Baseline polars"):
        build_auxiliary(*inputs)


def test_frozen_partition_counts_distributions_and_signs_are_audit_only(tmp_path):
    inputs = _inputs(tmp_path, members={
        "a.json": {"polars": [_polar(0, Cdp=0.1)]},
        "b.json": {"polars": [_polar(2, Cdp=-0.02)]},
        "c.json": {"polars": [_polar(1, Cdp=0.01)]},
    })
    frozen = tmp_path / "split.json"
    frozen.write_text(json.dumps({"split_geometries": {"train": ["a"], "validation": ["b"], "test": ["c"]},
                                  "v1_split_rows": {"train": 1, "validation": 1, "test": 1}}))
    paths = build_auxiliary(*inputs, frozen_split=frozen)
    report = _metadata(paths)
    assert report["coverage"]["observed"] == 3
    assert report["audit"]["overall"]["Cd_minus_Cdp"]["negative"] == 1
    assert report["audit"]["overall"]["Cd_minus_Cdp"]["zero"] == 1
    assert report["audit"]["canonical_partitions"]["validation"]["Cdp"]["min"] == -0.02
    assert report["frozen_split_sha256"] == _sha(frozen)


@pytest.mark.parametrize("groups", [
    {"train": ["a"], "validation": ["a"], "test": ["c"]},
    {"train": ["a"], "validation": ["b"], "test": ["extra"]},
    {"train": ["a"], "validation": [], "test": ["c"]},
])
def test_invalid_frozen_partition_fails(tmp_path, groups):
    inputs = _inputs(tmp_path)
    frozen = tmp_path / "split.json"
    frozen.write_text(json.dumps(groups))
    with pytest.raises(ValueError, match="Frozen"):
        build_auxiliary(*inputs, frozen_split=frozen)


def test_deterministic_idempotence_refuses_changed_output_and_supports_explicit_overwrite(tmp_path):
    inputs = _inputs(tmp_path)
    paths = build_auxiliary(*inputs)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths.values()}
    assert build_auxiliary(*inputs) == paths
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == saved for path, saved in before.items())
    paths["sidecar"].write_text(paths["sidecar"].read_text().replace("-0.002", "-0.003"))
    with pytest.raises(FileExistsError, match="overwrite"):
        build_auxiliary(*inputs)
    build_auxiliary(*inputs, overwrite=True)
    assert all(path.read_bytes() == saved[0] for path, saved in before.items())


@pytest.mark.parametrize("target", ["baseline", "sidecar"])
def test_cross_hash_mismatch_rejected(tmp_path, target):
    inputs = _inputs(tmp_path)
    paths = build_auxiliary(*inputs)
    path = inputs[1] if target == "baseline" else paths["sidecar"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_auxiliary(paths["sidecar"], inputs[1], paths["report"])


@pytest.mark.parametrize("defect", ["extra_column", "reorder_columns", "row_order", "drop_row", "extra_row",
                                   "duplicate", "key", "bool_key", "infinite", "bool_cdp", "mask", "numeric_mask"])
def test_loader_validates_content_even_with_recomputed_sidecar_digest(tmp_path, defect):
    inputs = _inputs(tmp_path)
    paths = build_auxiliary(*inputs)
    frame = pd.read_csv(paths["sidecar"])
    report = _metadata(paths)
    if defect == "extra_column":
        frame["feature"] = 0
    elif defect == "reorder_columns":
        frame = frame[list(reversed(COLUMNS))]
    elif defect == "row_order":
        frame = frame.iloc[::-1]
    elif defect == "drop_row":
        frame = frame.iloc[:-1]
    elif defect in ("extra_row", "duplicate"):
        extra = frame.iloc[:1].copy()
        if defect == "extra_row":
            extra["geometry_id"] = "extra"
        frame = pd.concat([frame, extra])
    elif defect == "key":
        frame.loc[0, "Alpha"] = 3
    elif defect == "bool_key":
        frame["Re"] = True
    elif defect == "infinite":
        frame.loc[0, "Cdp"] = np.inf
    elif defect == "bool_cdp":
        frame["Cdp"] = True
    elif defect == "mask":
        frame.loc[2, "Cdp_observed"] = True
    else:
        frame["Cdp_observed"] = frame.Cdp_observed.astype(int)
    frame.to_csv(paths["sidecar"], index=False)
    report["sidecar_sha256"] = _sha(paths["sidecar"])
    with pytest.raises(ValueError):
        load_auxiliary(paths["sidecar"], inputs[1], report)


@pytest.mark.parametrize("defect", ["schema", "identity", "coverage", "source_hash"])
def test_loader_rejects_inconsistent_metadata(tmp_path, defect):
    inputs = _inputs(tmp_path)
    paths = build_auxiliary(*inputs)
    report = _metadata(paths)
    if defect == "source_hash":
        report["source_zip_sha256"] = "bad"
    else:
        report[defect] = {}
    with pytest.raises(ValueError):
        load_auxiliary(paths["sidecar"], inputs[1], report)


def test_csv_round_trip_preserves_exact_float_keys_and_string_ids(tmp_path):
    alpha = 0.30000000000000004
    rows = [{"geometry_id": "0012", "Alpha": alpha, "Re": 1e6, "Ncrit": 9.0, "Cd": 0.01},
            {"geometry_id": "NA", "Alpha": 0, "Re": 1e6, "Ncrit": 9.0, "Cd": 0.01}]
    inputs = _inputs(tmp_path, rows=rows, members={
        "json/0012.json": {"polars": [_polar(alpha, Cdp=0.001)]},
        "json/NA.json": {"polars": [_polar(0, Cdp=0.002)]},
    })
    paths = build_auxiliary(*inputs)
    result = load_auxiliary(paths["sidecar"], inputs[1], paths["report"])
    assert result.Alpha.iloc[0] == alpha
    assert result.geometry_id.tolist() == ["0012", "NA"]
    assert result.Cdp_observed.all()


def test_symlink_output_cannot_overwrite_baseline(tmp_path):
    inputs = _inputs(tmp_path)
    inputs[2].mkdir()
    (inputs[2] / "airfoil_auxiliary.csv").symlink_to(inputs[1])
    before = inputs[1].read_bytes()
    with pytest.raises(ValueError, match="aliases"):
        build_auxiliary(*inputs, overwrite=True)
    assert inputs[1].read_bytes() == before


def test_cli_only_generates_requested_sidecar_and_report(tmp_path, capsys):
    inputs = _inputs(tmp_path)
    main(["--raw-path", str(inputs[0]), "--polars-path", str(inputs[1]),
          "--output-dir", str(inputs[2]), "--no-frozen-split"])
    output = json.loads(capsys.readouterr().out)
    assert output["coverage"]["rows"] == 3
    assert Path(output["paths"]["sidecar"]).exists()


def test_import_has_no_torch_or_training_dependency():
    script = (
        "import sys; import src.data.auxiliary; "
        "assert 'torch' not in sys.modules; "
        "assert not any(name.startswith('src.training') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("document", [[], {}, {"polars": {}}, {"polars": [None]}])
def test_malformed_source_documents_fail_actionably(tmp_path, document):
    inputs = _inputs(tmp_path, members={"a.json": document})
    with pytest.raises(ValueError, match="polars list|must be an object"):
        build_auxiliary(*inputs)


def test_archive_without_json_members_is_rejected(tmp_path):
    inputs = _inputs(tmp_path, members={})
    with pytest.raises(ValueError, match="No JSON members"):
        build_auxiliary(*inputs)


@pytest.mark.parametrize("metadata", [[], None, {"schema_version": True}])
def test_nonobject_or_boolean_schema_metadata_rejected(tmp_path, metadata):
    inputs = _inputs(tmp_path)
    paths = build_auxiliary(*inputs)
    paths["report"].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="report schema"):
        load_auxiliary(paths["sidecar"], inputs[1], paths["report"])


@pytest.mark.parametrize("counts", [None, {"train": True, "validation": 1, "test": 1},
                                     {"train": 2, "validation": 1, "test": 1}])
def test_invalid_frozen_row_counts_fail(tmp_path, counts):
    inputs = _inputs(tmp_path)
    frozen = tmp_path / "split.json"
    frozen.write_text(json.dumps({"split_geometries": {"train": ["a"], "validation": ["b"], "test": ["c"]},
                                  "v1_split_rows": counts}))
    with pytest.raises(ValueError, match="Frozen.*row count"):
        build_auxiliary(*inputs, frozen_split=frozen)


def test_source_key_conflicts_across_members_with_same_stem_fail(tmp_path):
    inputs = _inputs(tmp_path, members={
        "first/a.json": {"polars": [_polar(0, Cdp=0.01)]},
        "second/a.json": {"polars": [_polar(0, Cdp=0.02)]},
    })
    with pytest.raises(ValueError, match="Inconsistent Cdp"):
        build_auxiliary(*inputs)
