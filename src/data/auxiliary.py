"""Extract audited, target-only Cdp labels without rewriting either baseline table.

Identity uses geometry_id as a string and Alpha/Re/Ncrit as finite float64.
CSV reads use round-trip float parsing; no rounding or approximate joins are used.
The row hash matches the study's sorted pandas identity-hash convention, with
these dtypes made explicit, without importing any model or training module.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd

KEYS = ["geometry_id", "Alpha", "Re", "Ncrit"]
COLUMNS = [*KEYS, "Cdp", "Cdp_observed"]
AUXILIARY_TABLE = "airfoil_auxiliary.csv"
AUXILIARY_REPORT = "auxiliary_report.json"
SCHEMA = {
    "columns": COLUMNS,
    "dtypes": {"geometry_id": "string", "Alpha": "float64", "Re": "float64",
               "Ncrit": "float64", "Cdp": "float64", "Cdp_observed": "bool"},
    "nullable": ["Cdp"],
    "target_columns": ["Cdp"],
    "feature_columns": [],
    "target_transform": "identity",
}
HASH_ALGORITHM = (
    "sha256(pandas.hash_pandas_object(keys, index=False) as little-endian uint64); "
    "row_sha256 sorts hashes, ordered_row_sha256 preserves baseline order; "
    "geometry_id=object strings, Alpha/Re/Ncrit=float64"
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, dtype={"geometry_id": str}, float_precision="round_trip",
        keep_default_na=False,
        na_values={name: [""] for name in [*KEYS[1:], "Cl", "Cd", "Cm", "Cdp"]},
    )


def _numeric(series: pd.Series, context: str, *, missing: bool = False) -> pd.Series:
    if (not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)):
        raise ValueError(f"{context} must be numeric, non-boolean scalars.")
    result = series.astype("float64")
    bad = np.isinf(result) if missing else ~np.isfinite(result)
    if bad.any():
        raise ValueError(f"{context} contains invalid nonfinite values at rows {list(series.index[bad][:5])}.")
    return result


def _keys(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    absent = set(KEYS) - set(frame.columns)
    if absent:
        raise ValueError(f"{context} lacks required condition keys: {sorted(absent)}.")
    identity = frame[KEYS].copy()
    ids = identity.geometry_id
    if ids.isna().any() or not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"{context} geometry_id must contain nonempty strings.")
    identity["geometry_id"] = ids.astype(object)
    for name in KEYS[1:]:
        identity[name] = _numeric(identity[name], f"{context} {name}")
    duplicates = identity.duplicated(keep=False)
    if duplicates.any():
        raise ValueError(
            f"{context} has duplicate condition keys; resolve before extraction/loading: "
            f"{identity.loc[duplicates].head(5).to_dict('records')}"
        )
    return identity


def _identity(identity: pd.DataFrame) -> dict:
    hashes = pd.util.hash_pandas_object(identity[KEYS], index=False).to_numpy()
    return {
        "rows": len(identity), "geometries": int(identity.geometry_id.nunique()),
        "row_sha256": hashlib.sha256(np.sort(hashes).astype("<u8").tobytes()).hexdigest(),
        "ordered_row_sha256": hashlib.sha256(hashes.astype("<u8").tobytes()).hexdigest(),
        "key_columns": KEYS, "hash_algorithm": HASH_ALGORITHM,
    }


def _coverage(frame: pd.DataFrame) -> dict:
    observed = int(frame.Cdp_observed.sum())
    return {"rows": len(frame), "observed": observed, "missing": len(frame) - observed,
            "fraction_observed": observed / len(frame) if len(frame) else None}


def _distribution(values: pd.Series) -> dict:
    values = values.dropna()
    quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]
    return {
        "count": len(values), "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "quantiles": {str(q): float(values.quantile(q)) if len(values) else None for q in quantiles},
        "negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
        "positive": int((values > 0).sum()),
    }


def _summary(frame: pd.DataFrame) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = frame.loc[frame.Cdp_observed, "Cd"] - frame.loc[frame.Cdp_observed, "Cdp"]
    finite = np.isfinite(difference)
    return {
        **_coverage(frame), "Cdp": _distribution(frame.Cdp),
        "Cd_minus_Cdp": _distribution(difference[finite]),
        "nonfinite_Cd_minus_Cdp": int((~finite).sum()),
    }


def _partitions(frame: pd.DataFrame, frozen_split: Path | None) -> dict | None:
    if frozen_split is None:
        return None
    frozen = json.loads(frozen_split.read_text(encoding="utf-8"))
    if not isinstance(frozen, dict):
        raise ValueError("Frozen split must be a JSON object.")
    groups = frozen.get("split_geometries", frozen)
    if not isinstance(groups, dict) or set(groups) != {"train", "validation", "test"}:
        raise ValueError("Frozen split must name exactly train, validation, and test.")
    if "v1_split_rows" in frozen and (
        not isinstance(frozen["v1_split_rows"], dict)
        or set(frozen["v1_split_rows"]) != set(groups)
        or any(type(value) is not int for value in frozen["v1_split_rows"].values())
    ):
        raise ValueError("Frozen split row counts must be integers for all three partitions.")
    seen, result = set(), {}
    for name in ("train", "validation", "test"):
        ids = groups[name]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError(f"Frozen {name} geometry IDs must be nonempty, unique, and disjoint.")
        seen.update(ids)
        subset = frame.loc[frame.geometry_id.isin(ids)]
        if "v1_split_rows" in frozen and frozen["v1_split_rows"].get(name) != len(subset):
            raise ValueError(f"Frozen {name} row count does not match the baseline.")
        result[name] = {**_summary(subset), "geometries": len(ids)}
    if seen != set(frame.geometry_id):
        raise ValueError("Frozen geometry IDs must exactly cover the baseline, with no extra IDs.")
    return result


def _clipping(frame: pd.DataFrame, geometry_path: Path | None) -> dict:
    if geometry_path is None:
        return {"available": False, "reason": "No baseline geometry table supplied or found."}
    geometry = _read_csv(geometry_path)
    if "geometry_id" not in geometry or "clipped_stations" not in geometry:
        raise ValueError("Geometry audit requires geometry_id and clipped_stations columns.")
    if geometry.geometry_id.duplicated().any() or set(geometry.geometry_id) != set(frame.geometry_id):
        raise ValueError("Geometry audit IDs must uniquely and exactly cover baseline geometries.")
    counts = _numeric(geometry.clipped_stations, "Geometry clipped_stations")
    if (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("Geometry clipped_stations must be nonnegative integer counts.")
    flagged = geometry.loc[counts > 0, ["geometry_id", "clipped_stations"]]
    return {
        "available": True, "meaning": "Existing geometry thickness-clipping flag; not a Cdp filter.",
        "geometries": len(flagged), "stations": int(counts.sum()),
        "by_geometry": dict(zip(flagged.geometry_id, flagged.clipped_stations.astype(int))),
        "flagged_rows": _summary(frame.loc[frame.geometry_id.isin(flagged.geometry_id)]),
        "unflagged_rows": _summary(frame.loc[~frame.geometry_id.isin(flagged.geometry_id)]),
    }


def load_auxiliary(
    sidecar: str | Path, polars: str | Path, report: str | Path | Mapping,
) -> pd.DataFrame:
    """Return target-only sidecar rows after digest, schema, mask and exact-key checks.

    ``polars`` must be the original CSV, not an assembled feature DataFrame.
    ``report`` is a trusted build report path or metadata mapping. Checksums bind
    labels to that report; they are integrity checks, not a cryptographic signature.
    No labels are merged into features or transformed by this function.
    """
    sidecar, polars = Path(sidecar), Path(polars)
    metadata = dict(report) if isinstance(report, Mapping) else json.loads(Path(report).read_text(encoding="utf-8"))
    if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 1 or metadata.get("schema") != SCHEMA):
        raise ValueError("Auxiliary report schema is missing or incompatible.")
    for name in ("source_zip_sha256", "baseline_polars_sha256", "sidecar_sha256"):
        value = metadata.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Auxiliary report requires a valid {name}.")
    if _digest(polars) != metadata["baseline_polars_sha256"]:
        raise ValueError("Baseline polars SHA256 mismatch; use the exact CSV bound to this sidecar.")
    if _digest(sidecar) != metadata["sidecar_sha256"]:
        raise ValueError("Auxiliary sidecar SHA256 mismatch; regenerate rather than editing labels.")
    baseline = _keys(_read_csv(polars), "Baseline polars")
    auxiliary = _read_csv(sidecar)
    if list(auxiliary.columns) != COLUMNS:
        raise ValueError(f"Auxiliary columns must be exactly {COLUMNS}, in that order (targets only).")
    identity = _keys(auxiliary, "Auxiliary sidecar")
    if not identity.equals(baseline):
        raise ValueError("Auxiliary keys must exactly cover all baseline rows in original order; no approximate joins.")
    if metadata.get("identity") != _identity(baseline):
        raise ValueError("Auxiliary row count/key hash metadata does not match the baseline identity.")
    auxiliary[KEYS] = identity
    auxiliary["Cdp"] = _numeric(auxiliary.Cdp, "Auxiliary Cdp", missing=True)
    if auxiliary.Cdp_observed.dtype != bool or not auxiliary.Cdp_observed.equals(auxiliary.Cdp.notna()):
        raise ValueError("Cdp_observed must be boolean and exactly equal to nonmissing Cdp; no imputation.")
    if metadata.get("coverage") != _coverage(auxiliary):
        raise ValueError("Auxiliary coverage metadata does not match observed labels.")
    if _digest(polars) != metadata["baseline_polars_sha256"] or _digest(sidecar) != metadata["sidecar_sha256"]:
        raise ValueError("Baseline or sidecar changed during validation; retry with immutable input files.")
    return auxiliary


def build_auxiliary(
    raw_path: str | Path, polars_path: str | Path, output_dir: str | Path, *,
    frozen_split: str | Path | None = None, geometry_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Build private CSV/report outputs, preserving every baseline row in order.

    Missing labels (absent/null/NaN, or absent source keys) remain NaN/False.
    Equal source-key duplicates are deduplicated and counted; inconsistent Cdp
    duplicates fail, including observed/missing conflicts. Non-base source rows
    and empty/non-base source geometries are counted, never added to the baseline.
    Existing byte-identical outputs are retained; different outputs require
    explicit overwrite. Optional split audits never select or remove labels.
    """
    raw_path, polars_path, output_dir = Path(raw_path), Path(polars_path), Path(output_dir)
    frozen_split = Path(frozen_split) if frozen_split is not None else None
    if geometry_path is None:
        sibling = polars_path.with_name("airfoil_geometry.csv")
        geometry_path = sibling if sibling.exists() else None
    else:
        geometry_path = Path(geometry_path)
    inputs = [path for path in (raw_path, polars_path, frozen_split, geometry_path) if path is not None]
    paths = {"sidecar": output_dir / AUXILIARY_TABLE, "report": output_dir / AUXILIARY_REPORT}
    for path in paths.values():
        if path.is_symlink() or path.resolve() in {item.resolve() for item in inputs}:
            raise ValueError(f"Refusing output that aliases an input or symbolic link: {path}")
    input_hashes = {path: _digest(path) for path in inputs}
    baseline = _read_csv(polars_path)
    identity = _keys(baseline, "Baseline polars")
    if baseline.empty or "Cd" not in baseline:
        raise ValueError("Baseline polars must be nonempty and include Cd for coefficient auditing.")
    cd = _numeric(baseline.Cd, "Baseline Cd")
    records, source_ids, empty_ids = [], set(), set()
    missing_fields = null_labels = nan_labels = 0
    with zipfile.ZipFile(raw_path) as archive:
        members = sorted(
            (info for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".json")),
            key=lambda info: info.filename,
        )
        if not members:
            raise ValueError(f"No JSON members in source archive {raw_path}.")
        for member in members:
            document = json.loads(archive.read(member))
            if not isinstance(document, dict) or not isinstance(document.get("polars"), list):
                raise ValueError(f"Source member {member.filename!r} requires a polars list.")
            geometry_id = Path(member.filename).stem
            source_ids.add(geometry_id)
            if not document["polars"]:
                empty_ids.add(geometry_id)
            for index, polar in enumerate(document["polars"]):
                context = f"{member.filename} polar {index}"
                if not isinstance(polar, dict):
                    raise ValueError(f"{context} must be an object.")
                values = []
                for name in ("alpha", "Re", "Ncrit"):
                    value = polar.get(name)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"{context}: condition key {name} must be a finite numeric non-boolean scalar.")
                    values.append(float(value))
                value = polar.get("Cdp")
                if "Cdp" not in polar:
                    missing_fields += 1
                elif value is None:
                    null_labels += 1
                elif isinstance(value, bool) or not isinstance(value, (int, float)) or math.isinf(value):
                    raise ValueError(f"{context}: Cdp must be a finite numeric non-boolean scalar, or missing.")
                elif math.isnan(value):
                    nan_labels += 1
                records.append((geometry_id, *values, float(value) if value is not None else np.nan))
    source = pd.DataFrame.from_records(records, columns=[*KEYS, "Cdp"])
    for name in [*KEYS[1:], "Cdp"]:
        source[name] = source[name].astype("float64")
    source["geometry_id"] = source.geometry_id.astype(object)
    duplicated = source.duplicated(KEYS, keep=False)
    if duplicated.any():
        conflicts = source.loc[duplicated].groupby(KEYS, sort=False).Cdp.nunique(dropna=False)
        if (conflicts > 1).any():
            raise ValueError(
                "Inconsistent Cdp for duplicate source condition keys; resolve source labels: "
                f"{list(conflicts[conflicts > 1].index[:5])}"
            )
    duplicate_count = int(source.duplicated(KEYS).sum())
    source = source.drop_duplicates(KEYS, keep="first")
    auxiliary = identity.merge(source, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True)
    absent_keys = int(auxiliary._merge.eq("left_only").sum())
    matched_keys = len(auxiliary) - absent_keys
    auxiliary = auxiliary.drop(columns="_merge")
    auxiliary["Cdp_observed"] = auxiliary.Cdp.notna()
    if not _keys(auxiliary, "Generated auxiliary").equals(identity):
        raise ValueError("Extraction changed baseline row order or identity.")
    audited = auxiliary.assign(Cd=cd)
    report = {
        "schema_version": 1, "schema": SCHEMA, "identity": _identity(identity),
        "source_zip_sha256": input_hashes[raw_path],
        "baseline_polars_sha256": input_hashes[polars_path],
        "geometry_sha256": input_hashes.get(geometry_path),
        "frozen_split_sha256": input_hashes.get(frozen_split),
        "coverage": _coverage(auxiliary),
        "source": {
            "json_members": len(members), "geometries": len(source_ids),
            "polar_rows": len(records), "unique_condition_keys": len(source),
            "duplicate_same_value_rows": duplicate_count, "inconsistent_duplicate_keys": 0,
            "Cdp_absent_fields": missing_fields, "Cdp_null_labels": null_labels, "Cdp_nan_labels": nan_labels,
            "empty_geometry_ids": sorted(empty_ids),
            "nonbase_geometry_ids": sorted(source_ids - set(identity.geometry_id)),
            "nonbase_geometries": len(source_ids - set(identity.geometry_id)),
            "nonbase_unique_rows": len(source) - matched_keys,
            "matched_base_rows": matched_keys, "base_rows_without_source_key": absent_keys,
        },
        "audit": {
            "overall": _summary(audited),
            "by_geometry": [{"geometry_id": key, **_coverage(group)} for key, group in audited.groupby("geometry_id", sort=True)],
            "by_Re": [{"Re": float(key), **_coverage(group)} for key, group in audited.groupby("Re", sort=True)],
            "by_geometry_Re": [{"geometry_id": key[0], "Re": float(key[1]), **_coverage(group)}
                               for key, group in audited.groupby(["geometry_id", "Re"], sort=True)],
            "canonical_partitions": _partitions(audited, frozen_split),
            "clipping_defect_flags": _clipping(audited, geometry_path),
        },
        "policy": {
            "target_only": True, "private_output": True, "model_weights_included": False,
            "baseline_rows_dropped": 0, "labels_imputed": 0,
            "negative_or_zero_Cdp_allowed": True, "Cd_minus_Cdp_sign_is_filter": False,
            "row_order": "baseline original order; source members sorted by archive filename",
            "partition_audit": "Aggregate coverage/label diagnostics only; not model selection or filtering.",
        },
    }
    for path, digest in input_hashes.items():
        if _digest(path) != digest:
            raise ValueError(f"Input changed during extraction: {path}. Retry with immutable inputs.")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".csv", delete=False) as handle:
            csv_path = Path(handle.name)
            temporary.append(csv_path)
        auxiliary.to_csv(csv_path, index=False, lineterminator="\n")
        report["sidecar_sha256"] = _digest(csv_path)
        load_auxiliary(csv_path, polars_path, report)
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".json", mode="w", encoding="utf-8", delete=False) as handle:
            report_path = Path(handle.name)
            temporary.append(report_path)
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        replacements = {paths["sidecar"]: csv_path, paths["report"]: report_path}
        if all(path.is_file() and _digest(path) == _digest(temp) for path, temp in replacements.items()):
            return paths
        if not overwrite and any(path.exists() for path in paths.values()):
            raise FileExistsError("Auxiliary outputs already exist and differ or are incomplete; use --overwrite explicitly.")
        for path, temp in replacements.items():
            temp.replace(path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw/airfoil-learning-dataset.zip"))
    parser.add_argument("--polars-path", type=Path, default=Path("data/processed/airfoil_polars.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--frozen-split", type=Path, default=Path("configs/frozen_split.json"))
    parser.add_argument("--no-frozen-split", action="store_true")
    parser.add_argument("--geometry-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    paths = build_auxiliary(
        args.raw_path, args.polars_path, args.output_dir,
        frozen_split=None if args.no_frozen_split else args.frozen_split,
        geometry_path=args.geometry_path, overwrite=args.overwrite,
    )
    metadata = json.loads(paths["report"].read_text(encoding="utf-8"))
    print(json.dumps({"paths": {name: str(path) for name, path in paths.items()},
                      "coverage": metadata["coverage"], "source": metadata["source"],
                      "overall": metadata["audit"]["overall"],
                      "sha256": {key: value for key, value in metadata.items() if key.endswith("sha256")}},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
