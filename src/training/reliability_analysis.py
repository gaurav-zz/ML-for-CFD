"""Offline, descriptive OOF error/referral analysis. Never a confidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["geometry_id", "Alpha", "Re", "Ncrit"]
TARGETS = ["Cl", "Cd", "Cm"]
COVERAGES = (1., .95, .9, .75, .5)
BUDGETS = (.05, .10, .20)
ARMS = ("cdp_column_scaling", "cdp_profile_scaling")
DISTANCE = "nearest_floor_scaled_rms_distance"
OUTSIDE = "outside_fit_geometry_features"
SOURCE = "canonical_train_geometry_fold_evaluation"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(values, shape, name):
    values = np.asarray(values)
    _require(values.shape == shape and values.dtype.kind in "iuf" and np.isfinite(values).all(),
             f"{name}: expected finite real numeric array of shape {shape}")
    return values


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(frame):
    values = pd.util.hash_pandas_object(frame[KEYS], index=False).to_numpy()
    return hashlib.sha256(np.sort(values).astype("<u8").tobytes()).hexdigest()


def load_train(polars, split):
    """Read narrow chunks, retain only frozen train rows in original CSV order."""
    frozen = json.loads(Path(split).read_text())
    groups = frozen["split_geometries"]
    _require(set(groups) == {"train", "validation", "test"}, "Invalid frozen partitions")
    flat = [item for group in groups.values() for item in group]
    _require(len(flat) == len(set(flat)), "Duplicate/overlapping frozen geometry IDs")
    chunks = [chunk.loc[chunk.geometry_id.isin(groups["train"])] for chunk in
              pd.read_csv(polars, dtype={"geometry_id": str}, usecols=KEYS + TARGETS, chunksize=100_000)]
    train = pd.concat(chunks, ignore_index=True)
    _require(len(train) == frozen["v1_split_rows"]["train"] > 0 and
             train.geometry_id.nunique() == frozen["geometry_counts"]["train"] and
             set(train.geometry_id) == set(groups["train"]) and not train[KEYS].duplicated().any(),
             "Canonical train coverage/identity mismatch")
    _finite(train[KEYS[1:] + TARGETS].to_numpy(), (len(train), 6), "Train coordinates/labels")
    _require((train.Cd > 0).all() and (train.Re > 0).all(), "Actual Cd and Re must be positive; no clamp")
    return train


def validate_alignment(train, positions, expected):
    positions = np.asarray(positions)
    _require(positions.ndim == 1 and positions.dtype.kind in "iu" and len(positions) > 0 and
             (positions >= 0).all() and (positions < len(train)).all() and
             len(np.unique(positions)) == len(positions) and np.array_equal(positions, expected),
             "Prediction row alignment requires exact ordered, unique canonical-train positions")


def load_run(directory, train, polars_sha256, mode):
    """Validate saved-array consistency and align each configuration across folds."""
    directory = Path(directory)
    manifest = json.loads((directory / "fold_study.json").read_text())
    plan, metadata = manifest["plan"], manifest["run_metadata"]
    candidate = {"input_scaling": "profile_rms_floor", "selective_scaling": "profile_rms_floor_1pct"}[mode]
    modes = dict(zip(ARMS, ("standard", candidate)))
    canonical = manifest["canonical_split"]["train"]
    _require(manifest.get("schema_version") == 1 and
             [manifest.get(key) for key in ("completed_fits", "completed_member_evaluations", "completed_ensemble_evaluations")] == [18, 18, 6] and
             len(manifest["trials"]) == 18 and manifest["status"] == manifest["phase"] == "completed" and
             manifest["source"] == SOURCE and metadata["study_mode"] == mode and
             manifest["study_kind"] == "paired_geometry_input_scaling" and
             manifest["target_names"] == TARGETS and plan["input_scaling_modes"] == modes and
             plan["seeds"] == [42, 43, 44] and plan["n_folds"] == 3 and plan["arms"] == list(ARMS) and
             plan["aggregation"] == "mean_Cl_Cm_geometric_mean_Cd" and
             manifest["canonical_validation_test_used_for_model_apis"] is False,
             "Run mode/completion/ensemble contract mismatch")
    _require(metadata["dataset"]["polars_sha256"] == polars_sha256 and
             canonical["rows"] == len(train) and canonical["geometry_ids"] == sorted(train.geometry_id.unique()) and
             canonical["row_sha256"] == _row_hash(train), "Canonical data/hash mismatch")
    folds = plan["fold_plan"]["folds"]
    ledger = manifest["prediction_artifacts"]
    _require(set(folds) == set(ledger) == {"0", "1", "2"}, "Missing/extra folds")
    identity = train[KEYS].sort_values(KEYS)
    covered, fold_ids, configs = np.zeros(len(train), int), np.empty(len(train), int), {}
    for fold, spec in folds.items():
        partition_ids = [part["geometry_ids"] for part in spec["partitions"].values()]
        flat = [item for group in partition_ids for item in group]
        _require(set(spec["partitions"]) == {"fit", "stopping", "evaluation"} and
                 len(flat) == len(set(flat)) and set(flat) == set(train.geometry_id), "Fold partition mismatch")
        for part in spec["partitions"].values():
            rows = train.loc[train.geometry_id.isin(part["geometry_ids"])]
            _require(part["n_rows"] == len(rows) and part["row_sha256"] == _row_hash(rows), "Fold identity/hash mismatch")
        ids = spec["partitions"]["evaluation"]["geometry_ids"]
        expected = identity.index[identity.geometry_id.isin(ids)].to_numpy()
        covered[expected] += 1
        fold_ids[expected] = int(fold)
        _require(set(ledger[fold]) == set(ARMS), "Missing/extra configuration")
        for arm in ARMS:
            item = ledger[fold][arm]
            path = (directory / item["path"]).resolve()
            _require(path.is_relative_to(directory.resolve()) and item["status"] == "complete" and
                     item["ensemble_present"] is True and item["n_members"] == 3 and
                     item["n_rows"] == len(expected), "Incomplete/invalid prediction artifact")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            validate_alignment(train, arrays["canonical_train_positions"], expected)
            _require(arrays["member_seeds"].dtype.kind in "iu" and
                     np.array_equal(arrays["member_seeds"], [42, 43, 44]), "Member seed/order mismatch")
            if mode == "selective_scaling":
                _require(manifest.get("study_variant") == "selective_profile_floor_1pct" and
                         arrays["input_scaling"].item() == modes[arm] and
                         arrays["study_variant"].item() == manifest["study_variant"], "NPZ mode/variant mismatch")
            n = len(expected)
            for name, shape in (("physical_predictions", (3, n, 3)), ("fitting_predictions", (3, n, 3)),
                                ("physical_ensemble_predictions", (n, 3)), ("raw_log_Cd_std", (n,)),
                                ("clipped_log_Cd_std", (n,))):
                _finite(arrays[name], shape, name)
            raw, physical = arrays["fitting_predictions"], arrays["physical_predictions"]
            low, high = np.log([1e-6, 1e3])
            for name, expected_mask in (("lower_clipped", raw[:, :, 1] < low), ("upper_clipped", raw[:, :, 1] > high)):
                _require(arrays[name].dtype.kind == "b" and np.array_equal(arrays[name], expected_mask), "Invalid clipping mask")
            decoded = raw.copy()
            decoded[:, :, 1] = np.exp(np.clip(raw[:, :, 1], low, high))
            ensemble = physical.mean(axis=0)
            ensemble[:, 1] = np.exp(np.log(physical[:, :, 1]).mean(axis=0))
            for actual, wanted in ((physical, decoded), (arrays["physical_ensemble_predictions"], ensemble),
                                   (arrays["raw_log_Cd_std"], raw[:, :, 1].std(axis=0, ddof=1)),
                                   (arrays["clipped_log_Cd_std"], np.clip(raw[:, :, 1], low, high).std(axis=0, ddof=1))):
                _require(np.allclose(actual, wanted, rtol=1e-12, atol=1e-14), "Inconsistent decoded predictions/spreads")
            record = configs.setdefault(modes[arm], {"prediction": np.empty((len(train), 3)),
                "raw": np.empty(len(train)), "clipped": np.empty(len(train)), "hard": np.empty(len(train), bool),
                "array_fingerprints": {}, "trial_configs": {}})
            for key, source in (("prediction", "physical_ensemble_predictions"), ("raw", "raw_log_Cd_std"), ("clipped", "clipped_log_Cd_std")):
                record[key][expected] = arrays[source]
            record["hard"][expected] = (arrays["lower_clipped"] | arrays["upper_clipped"]).any(axis=0)
            record["array_fingerprints"][fold] = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for key, value in arrays.items()
                if key not in {"input_scaling", "study_variant"}}
            for seed in (42, 43, 44):
                trial_id = f"fold_{fold}_seed_{seed}_{arm}"
                trial = manifest["trials"][trial_id]
                config = trial["config"]
                _require(trial["status"] == trial["evaluation_status"] == "success" and
                         config == plan["trial_configs"][trial_id] and config["parameters"]["input_scaling"] == modes[arm] and
                          config["parameters"]["random_state"] == seed and
                          config["parameters"]["auxiliary_weight"] == .1, "Trial config/seed/Cdp mismatch")
                record["trial_configs"][f"{fold}:{seed}"] = config
    _require((covered == 1).all(), "OOF coverage must include every train row exactly once")
    return configs, fold_ids, {"directory": str(directory), "manifest_sha256": _digest(directory / "fold_study.json"),
        "source_sha256": metadata["source_sha256"], "study_mode": mode,
        "partition_row_hashes": {fold: {key: value["row_sha256"] for key, value in spec["partitions"].items()}
                                 for fold, spec in folds.items()}}


def geometry_support(table, train, fold_ids):
    keys = pd.DataFrame({"fold_id": fold_ids, "geometry_id": train.geometry_id.to_numpy()})
    expected = pd.MultiIndex.from_frame(keys.drop_duplicates())
    _require(set(["fold_id", "geometry_id", DISTANCE, OUTSIDE]).issubset(table.columns), "Missing geometry coverage columns")
    supplied = pd.MultiIndex.from_frame(table[["fold_id", "geometry_id"]])
    _require(not table.geometry_id.duplicated().any() and supplied.is_unique and
             len(supplied) == len(expected) and set(supplied) == set(expected), "Duplicate/missing/extra geometry coverage")
    values = _finite(table[[DISTANCE, OUTSIDE]].to_numpy(), (len(table), 2), "Geometry coverage")
    _require((values >= 0).all() and (values[:, 1] == np.floor(values[:, 1])).all(), "Invalid geometry distance/count")
    return keys.merge(table[["fold_id", "geometry_id", DISTANCE, OUTSIDE]], how="left", sort=False, validate="many_to_one")


def prediction_scores(raw, clipped, distance=None):
    """Row-weighted empirical ranks use only saved predictions and fixed inputs."""
    raw = np.asarray(raw)
    _require(raw.ndim == 1 and len(raw) > 0, "Nonempty spread vector required")
    scores = {"raw_log_Cd_std": raw, "clipped_log_Cd_std": clipped}
    if distance is not None:
        scores["geometry_distance"] = distance
    for name, values in scores.items():
        scores[name] = _finite(values, raw.shape, name)
        _require((scores[name] >= 0).all(), "Scores must be nonnegative")
    if distance is not None:
        scores["max_raw_geometry_percentile"] = np.maximum(
            pd.Series(raw).rank(method="max", pct=True), pd.Series(distance).rank(method="max", pct=True)).to_numpy()
    return scores


def retained_mask(score, requested_coverage, hard):
    """Inclusive all-row order statistic, THEN hard referrals. Never break ties."""
    score = np.asarray(score)
    _finite(score, (len(score),), "Score")
    hard = np.asarray(hard)
    _require(len(score) > 0 and 0 < requested_coverage <= 1 and hard.dtype.kind == "b" and
             hard.shape == score.shape, "Invalid coverage/hard-referral mask")
    index = int(np.ceil(requested_coverage * len(score))) - 1
    cutoff = float(np.partition(score, index)[index])
    return (score <= cutoff) & ~hard, cutoff


def error_metrics(actual, predicted, retained):
    actual = _finite(actual, (len(actual), 3), "Actual")
    predicted = _finite(predicted, actual.shape, "Predictions")
    retained = np.asarray(retained)
    _require(retained.dtype.kind == "b" and retained.shape == (len(actual),) and (actual[:, 1] > 0).all(),
             "Boolean retention mask and positive actual Cd required; no zero/near-zero clamp")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        error = predicted - actual
        relative = np.abs(error[:, 1]) / actual[:, 1]
        squared = error ** 2
    _finite(squared, actual.shape, "Squared errors")
    _finite(relative, (len(actual),), "Relative errors")
    names = ["Cl_rmse", "Cd_rmse", "Cm_rmse", "Cd_median_relative_error", "Cd_p95_absolute_error"]
    values = [*np.sqrt(squared[retained].mean(axis=0)), np.median(relative[retained]),
              np.quantile(np.abs(error[retained, 1]), .95)] if retained.any() else [None] * 5
    return dict(zip(names, (float(value) if value is not None else None for value in values))), relative


def sensitivity(relative, retained, budget):
    relative = _finite(relative, (len(relative),), "Relative errors")
    retained = np.asarray(retained)
    _require(retained.dtype.kind == "b" and retained.shape == relative.shape and (relative >= 0).all() and
             np.isfinite(budget) and budget > 0, "Invalid sensitivity inputs")
    unsafe = relative > budget
    tp, fn = int((unsafe & ~retained).sum()), int((unsafe & retained).sum())
    fp, tn = int((~unsafe & ~retained).sum()), int((~unsafe & retained).sum())
    return {"illustrative_relative_error_budget": budget, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "FN_rate_among_unsafe": fn / (tp + fn) if tp + fn else None,
            "false_accept_fraction_among_retained": fn / (fn + tn) if fn + tn else None,
            "unnecessary_referrals": fp, "unnecessary_fraction_among_referrals": fp / (tp + fp) if tp + fp else None,
            "actual_coverage": float(retained.mean()) if len(retained) else None}


def analyze(train, configurations, support=None):
    actual, n = train[TARGETS].to_numpy(), len(train)
    groups, ids = pd.factorize(train.geometry_id, sort=True)
    totals = np.bincount(groups)
    summaries, geometries, curves, matrices = [], [], [], []
    for name, record in configurations.items():
        prediction, hard = record["prediction"], record["hard"]
        scores = prediction_scores(record["raw"], record["clipped"], None if support is None else support[DISTANCE].to_numpy())
        full, relative = error_metrics(actual, prediction, np.ones(n, bool))
        summaries.append({"configuration": name, "n_rows": n, "n_geometries": len(ids),
                          "hard_referral_rows": int(hard.sum()), **full})
        for group, geometry_id in enumerate(ids):
            selected = groups == group
            metrics, _ = error_metrics(actual[selected], prediction[selected], np.ones(totals[group], bool))
            geometries.append({"configuration": name, "geometry_id": geometry_id, "n_rows": int(totals[group]),
                               "hard_referral_rows": int(hard[selected].sum()), **metrics})
        for method, score in scores.items():
            for requested in COVERAGES:
                retained, cutoff = retained_mask(score, requested, hard)
                counts = np.bincount(groups[retained], minlength=len(ids))
                metrics, _ = error_metrics(actual, prediction, retained)
                identity = {"configuration": name, "method": method, "requested_coverage": requested}
                curves.append({**identity, "score_cutoff": cutoff, "actual_coverage": float(retained.mean()),
                    "retained_rows": int(retained.sum()), "hard_referral_rows": int(hard.sum()),
                    "fully_retained_geometries": int((counts == totals).sum()),
                    "partially_retained_geometries": int(((counts > 0) & (counts < totals)).sum()),
                    "fully_referred_geometries": int((counts == 0).sum()), **metrics})
                matrices.extend({**identity, **sensitivity(relative, retained, budget)} for budget in BUDGETS)
    return {"full_row_summary": pd.DataFrame(summaries), "per_geometry": pd.DataFrame(geometries),
            "risk_coverage": pd.DataFrame(curves), "illustrative_sensitivity": pd.DataFrame(matrices)}


def run_analysis(broad_run, selective_run, polars, split, output_dir, geometry_coverage=None):
    output = Path(output_dir)
    _require(not output.exists() and not output.is_symlink(), "Output directory must be fresh (nonexistent)")
    train, polars_hash = load_train(polars, split), _digest(polars)
    broad, folds, broad_info = load_run(broad_run, train, polars_hash, "input_scaling")
    selective, other_folds, selective_info = load_run(selective_run, train, polars_hash, "selective_scaling")
    _require(np.array_equal(folds, other_folds) and broad_info["partition_row_hashes"] == selective_info["partition_row_hashes"],
             "Cross-run fit/stopping/evaluation fold alignment mismatch")
    control, repeated = broad["standard"], selective["standard"]
    _require(control["trial_configs"] == repeated["trial_configs"] and
             control["array_fingerprints"] == repeated["array_fingerprints"], "Repeated control is not exactly numerically equivalent")
    configs = {"control": control, "broad": broad["profile_rms_floor"], "selective": selective["profile_rms_floor_1pct"]}
    for record in configs.values():
        for key, config in record["trial_configs"].items():
            normalized = {**config, "parameters": {**config["parameters"], "input_scaling": "standard"}}
            _require(normalized == control["trial_configs"][key], "Paired configurations differ beyond input scaling")
    support = None if geometry_coverage is None else geometry_support(pd.read_csv(geometry_coverage), train, folds)
    tables = analyze(train, configs, support)
    compared = tables["per_geometry"].pivot(index="geometry_id", columns="configuration",
        values=["Cd_rmse", "Cd_median_relative_error", "Cd_p95_absolute_error", "Cl_rmse", "Cm_rmse"])
    compared.columns = [f"{config}_{metric}" for metric, config in compared.columns]
    for metric in ("Cd_rmse", "Cd_median_relative_error", "Cl_rmse", "Cm_rmse"):
        compared[f"minimum_across_configs_{metric}"] = compared[[f"{name}_{metric}" for name in configs]].min(axis=1)
        for left, right in (("broad", "control"), ("selective", "control"), ("selective", "broad")):
            compared[f"{left}_minus_{right}_{metric}"] = compared[f"{left}_{metric}"] - compared[f"{right}_{metric}"]
    identities = pd.DataFrame({"geometry_id": train.geometry_id, "fold_id": folds}).drop_duplicates()
    if support is not None:
        identities = support.drop_duplicates("geometry_id")
    tables["geometry_comparisons"] = compared.reset_index().merge(identities, on="geometry_id", validate="one_to_one")
    report = {"status": "complete_offline_descriptive_oof_diagnostic_not_a_production_gate", "schema_version": 1,
        "production_policy_recommendation": None, "calibrated_confidence": False, "holdouts_scored": False,
        "limitations": ["Inspected OOF development data, not calibrated or blind validation.",
            "Illustrative 5%, 10%, 20% Cd relative-error budgets are NOT approved engineering requirements.",
            "Spread alone cannot guarantee accuracy; agreement can coexist with shared bias.",
            "Fixed geometry distance is input-only support evidence, not calibrated OOD probability.",
            "Repeated configurations/seeds are not independent replications; control counted once.",
            "Retained rows change the task distribution and may only partially cover geometries; no subset R2 gains claimed.",
            "No selector fitting, label-derived referral ranks/cutoffs, best-method selection, or deployment recommendation."],
        "definitions": {"cutoff": "ceil(requested_coverage*n)-th all-row smallest score, inclusive ties, THEN exclude any raw-member clipping",
            "percentile": "row-weighted empirical CDF (maximum rank for ties) across all OOF rows, prediction/input only",
            "positive": "Cd absolute relative error strictly exceeds illustrative budget: needs referral",
            "FN": "unsafe accepted", "FP": "acceptable referred", "TP": "unsafe referred", "TN": "acceptable accepted",
            "relative_error": "abs(predicted_Cd-actual_Cd)/actual_Cd; every actual Cd must be positive, no clamp",
            "p95_absolute_error": "linear 0.95 quantile of absolute row Cd errors",
            "persistent_errors": "Per-geometry minimum error across all three configs describes residual burden without a binary failure cutoff; never a referral score",
            "full_row_summary": "All rows, including hard referrals; distinct from requested coverage 1.0 after hard referrals"},
        "requested_coverages": COVERAGES, "illustrative_budgets": BUDGETS, "n_rows": len(train),
        "n_geometries": train.geometry_id.nunique(), "sources": [broad_info, selective_info],
        "polars_sha256": polars_hash, "split_sha256": _digest(split), "row_sha256": _row_hash(train),
        "geometry_coverage": None if support is None else {"path": str(geometry_coverage), "sha256": _digest(geometry_coverage)},
        "control_repeat": {"exact_numeric_equivalence": True, "verification": "identical shapes, dtypes and SHA256 of all nine numeric/bool arrays per fold",
                           "independent_replications": False, "array_fingerprints": control["array_fingerprints"]},
        "configuration_modes": dict(zip(configs, ("standard", "profile_rms_floor", "profile_rms_floor_1pct"))),
        "configuration_sha256": {name: hashlib.sha256(json.dumps(record["trial_configs"], sort_keys=True).encode()).hexdigest()
                                 for name, record in configs.items()}}
    serialized = json.dumps(report, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "analysis.json").write_text(serialized + "\n")
    return tables, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("broad-run", "selective-run", "polars", "split", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--geometry-coverage", type=Path)
    tables, _ = run_analysis(**vars(parser.parse_args()))
    print(tables["full_row_summary"].to_string(index=False))
    print("\nSelective candidate: descriptive retained risk, not validated confidence")
    print(tables["risk_coverage"].query("configuration == 'selective'")[["method", "requested_coverage", "actual_coverage",
        "Cd_rmse", "Cd_median_relative_error", "Cd_p95_absolute_error", "fully_retained_geometries",
        "partially_retained_geometries"]].to_string(index=False))
    print("\nIllustrative 10% error budget, selective candidate, requested 95% coverage; positive=needs referral")
    print(tables["illustrative_sensitivity"].query("configuration == 'selective' and requested_coverage == .95 and illustrative_relative_error_budget == .1")
          [["method", "actual_coverage", "TP", "FN", "FP", "TN", "FN_rate_among_unsafe", "false_accept_fraction_among_retained"]].to_string(index=False))


if __name__ == "__main__":
    main()
