"""Frozen, development-only referral calibration of saved selective-Cdp predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import reliability_analysis as ra

METHODS = ("raw_log_Cd_std", "geometry_distance", "max_raw_geometry_percentile")
BUDGETS = (.05, .10, .20)
RISK_CAPS = (.05, .10)
QUANTILES = tuple(float(q) for q in np.linspace(0., 1., 21))
MIN_SUPPORT = 20
PARTITION_SEED = 20261105
CONFIGURATION = "profile_rms_floor_1pct"
SOURCE_VARIANT = "selective_profile_floor_1pct"
SOURCE_CREATED_AT = "2026-10-29T13:44:26.839881+00:00"


def _json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def partition_geometries(train, fold_ids, fold_plan, frozen_groups):
    """Split whole evaluation geometries, retaining canonical row positions, never labels."""
    fold_ids = np.asarray(fold_ids)
    ra._require(fold_ids.shape == (len(train),) and fold_ids.dtype.kind in "iu" and
                set(fold_ids) == {0, 1, 2} and set(fold_plan) == {"0", "1", "2"}, "Invalid fold IDs")
    ra._require(set(frozen_groups) == {"train", "validation", "test"}, "Invalid canonical partitions")
    flat = [item for ids in frozen_groups.values() for item in ids]
    ra._require(len(flat) == len(set(flat)) and set(train.geometry_id) == set(frozen_groups["train"]),
                "Canonical geometry coverage/holdout overlap")
    holdouts = set(frozen_groups["validation"]) | set(frozen_groups["test"])
    rows = train[ra.KEYS].reset_index(drop=True).copy()
    rows.insert(0, "canonical_train_position", np.arange(len(train)))
    rows["fold_id"], rows["partition"] = fold_ids, ""
    identities = {}
    for fold in range(3):
        parts = fold_plan[str(fold)]["partitions"]
        ra._require(set(parts) == {"fit", "stopping", "evaluation"}, "Invalid source partitions")
        flat = [item for part in parts.values() for item in part["geometry_ids"]]
        ra._require(len(flat) == len(set(flat)) and set(flat) == set(train.geometry_id),
                    "Source fit/stopping/evaluation overlap or coverage mismatch")
        evaluation = sorted(parts["evaluation"]["geometry_ids"])
        expected = train.geometry_id.isin(evaluation).to_numpy()
        ra._require(len(evaluation) >= 2 and np.array_equal(expected, fold_ids == fold),
                    "Evaluation geometry/fold row alignment mismatch")
        shuffled = np.random.default_rng(PARTITION_SEED + fold).permutation(evaluation)
        calibration, audit = shuffled[:len(shuffled) // 2].tolist(), shuffled[len(shuffled) // 2:].tolist()
        forbidden = holdouts | set(parts["fit"]["geometry_ids"]) | set(parts["stopping"]["geometry_ids"])
        ra._require(not (set(calibration) & set(audit)) and not (set(calibration) & forbidden) and
                    not (set(audit) & forbidden) and set(calibration + audit) == set(evaluation),
                    "Calibration/audit must be disjoint from each other, fit, stopping and holdouts")
        identities[str(fold)] = {"seed": PARTITION_SEED + fold, "source_partitions": parts}
        for name, ids in (("calibration", calibration), ("audit", audit)):
            mask = train.geometry_id.isin(ids).to_numpy()
            rows.loc[mask, "partition"] = name
            identities[str(fold)][name] = {"geometry_ids": ids, "n_geometries": len(ids),
                "n_rows": int(mask.sum()), "row_sha256": ra._row_hash(train.loc[mask])}
    ra._require(rows.partition.isin(["calibration", "audit"]).all(), "Incomplete partition coverage")
    return rows, identities


def _inputs(raw, distance, hard):
    raw, distance, hard = np.asarray(raw), np.asarray(distance), np.asarray(hard)
    ra._require(raw.ndim == 1 and hard.shape == raw.shape and hard.dtype.kind == "b", "Invalid input/hard mask")
    for values, name in ((raw, "Raw spread"), (distance, "Geometry distance")):
        ra._finite(values, raw.shape, name)
        ra._require((values >= 0).all(), "Scores must be nonnegative")
    return raw, distance, hard


def fit_cdfs(raw, distance, hard):
    """Row-weighted, right-continuous ECDF references on eligible calibration inputs only."""
    raw, distance, hard = _inputs(raw, distance, hard)
    return {"raw": np.sort(raw[~hard]), "geometry": np.sort(distance[~hard])}


def transform_scores(raw, distance, hard, cdfs):
    """Apply frozen CDFs without audit reranking; constant/tied values have maximum rank."""
    raw, distance, hard = _inputs(raw, distance, hard)
    ranks, outside = [], np.zeros(len(raw), bool)
    for key, values in (("raw", raw), ("geometry", distance)):
        reference = np.asarray(cdfs[key])
        ra._finite(reference, (len(reference),), "CDF reference")
        ra._require((reference >= 0).all() and (np.diff(reference) >= 0).all(), "Invalid sorted CDF")
        if len(reference):
            ranks.append(np.searchsorted(reference, values, side="right") / len(reference))
            outside |= (values < reference[0]) | (values > reference[-1])
        else:
            ranks.append(np.ones(len(values)))
            outside[:] = True
    ra._require(len(cdfs["raw"]) == len(cdfs["geometry"]), "CDF eligible-row count mismatch")
    return dict(zip(METHODS, (raw, distance, np.maximum(*ranks)))), outside


def candidate_thresholds(score, hard):
    """Exactly 21 fixed lower quantiles when eligible; ties are deliberately not split."""
    score, _, hard = _inputs(score, score, hard)
    if not (~hard).any():
        return []
    return [{"quantile": q, "threshold": float(value)} for q, value in
            zip(QUANTILES, np.quantile(score[~hard], QUANTILES, method="lower"))]


def select_threshold(score, hard, relative, geometry_ids, budget, risk_cap, *, min_support=MIN_SUPPORT):
    """Inspect every candidate; optimize empirical ROW coverage, not geometry-macro risk."""
    score, _, hard = _inputs(score, score, hard)
    relative = ra._finite(relative, score.shape, "Calibration relative errors")
    ids = np.asarray(geometry_ids)
    ra._require(ids.shape == score.shape and not pd.isna(ids).any() and (relative >= 0).all() and
                np.isfinite(budget) and budget > 0 and np.isfinite(risk_cap) and 0 <= risk_cap <= 1 and
                isinstance(min_support, int) and not isinstance(min_support, bool) and min_support > 0,
                "Invalid selection budget/risk/support/geometry inputs")
    candidates = []
    for cutoff in candidate_thresholds(score, hard):
        accepted = (score <= cutoff["threshold"]) & ~hard
        n, support = int(accepted.sum()), len(np.unique(ids[accepted]))
        fn = int(((relative > budget) & accepted).sum())
        candidates.append({**cutoff, "accepted_rows": n, "accepted_geometries": support, "FN": fn,
            "conditional_risk": fn / n if n else None,
            "eligible": bool(n > 0 and support >= min_support and fn <= risk_cap * n)})
    eligible = [item for item in candidates if item["eligible"]]
    chosen = min(eligible, key=lambda item: (-item["accepted_rows"], item["threshold"], item["quantile"])) if eligible else None
    for item in candidates:
        item["selected"] = item is chosen
    policy = {"Cd_relative_error_budget": budget, "empirical_risk_cap": risk_cap, "min_support": min_support,
        "status": "selected" if chosen else "no_eligible_threshold", "threshold": None if chosen is None else chosen["threshold"],
        "quantile": None if chosen is None else chosen["quantile"],
        "calibration_accepted_rows": 0 if chosen is None else chosen["accepted_rows"],
        "calibration_accepted_geometries": 0 if chosen is None else chosen["accepted_geometries"],
        "calibration_conditional_risk": None if chosen is None else chosen["conditional_risk"]}
    policy["calibration_actual_coverage"] = policy["calibration_accepted_rows"] / len(score) if len(score) else None
    return policy, candidates


def calibrate_fold(raw, distance, hard, geometry_ids, actual, predicted, *, min_support=MIN_SUPPORT):
    """The complete selector sees only ONE ensemble's calibration partition."""
    cdfs = fit_cdfs(raw, distance, hard)
    scores, _ = transform_scores(raw, distance, hard, cdfs)
    _, relative = ra.error_metrics(actual, predicted, np.ones(len(raw), bool))
    policies, candidates = [], []
    for method, score in scores.items():
        for budget in BUDGETS:
            for cap in RISK_CAPS:
                policy, grid = select_threshold(score, hard, relative, geometry_ids, budget, cap, min_support=min_support)
                policy["method"] = method
                policies.append(policy)
                candidates.extend({"method": method, "Cd_relative_error_budget": budget,
                    "empirical_risk_cap": cap, **item} for item in grid)
    return cdfs, policies, candidates


def apply_policy(policy, raw, distance, hard, cdfs):
    """Label-free fixed-threshold application, including mandatory numerical referrals."""
    scores, outside = transform_scores(raw, distance, hard, cdfs)
    ra._require(policy["method"] in METHODS, "Unknown policy method")
    if policy["threshold"] is None:
        return np.zeros(len(raw), bool)
    threshold = policy["threshold"]
    ra._require(np.isfinite(threshold) and threshold >= 0, "Invalid policy threshold")
    accepted = (scores[policy["method"]] <= threshold) & ~np.asarray(hard)
    if policy["method"] == METHODS[2]:
        accepted &= ~outside
    return accepted


def diagnostics(actual, predicted, geometry_ids, retained, budget):
    retained = np.asarray(retained)
    metrics, relative = ra.error_metrics(actual, predicted, retained)
    matrix = ra.sensitivity(relative, retained, budget)
    groups, ids = pd.factorize(np.asarray(geometry_ids), sort=True)
    ra._require(len(groups) == len(relative) and (groups >= 0).all(), "Invalid diagnostic geometry IDs")
    total = np.bincount(groups, minlength=len(ids))
    accepted = np.bincount(groups[retained], minlength=len(ids))
    fn = np.bincount(groups[retained & (relative > budget)], minlength=len(ids))
    risks = fn[accepted > 0] / accepted[accepted > 0]
    return {**metrics, **matrix, "n_rows": len(relative), "n_geometries": len(ids),
        "retained_rows": int(retained.sum()), "retained_geometries": int((accepted > 0).sum()),
        "fully_retained_geometries": int(((accepted == total) & (total > 0)).sum()),
        "partially_retained_geometries": int(((accepted > 0) & (accepted < total)).sum()),
        "fully_referred_geometries": int((accepted == 0).sum()),
        "conditional_risk": matrix["false_accept_fraction_among_retained"],
        "missed_referral_rate": matrix["FN_rate_among_unsafe"],
        "FPR": matrix["FP"] / (matrix["FP"] + matrix["TN"]) if matrix["FP"] + matrix["TN"] else None,
        "geometry_macro_retained_risk": float(risks.mean()) if len(risks) else None,
        "geometry_p95_retained_risk": float(np.quantile(risks, .95)) if len(risks) else None,
        "geometry_max_retained_risk": float(risks.max()) if len(risks) else None,
        "retained_geometries_with_over_budget_rows": int((fn > 0).sum()),
        "Cd_median_absolute_error": float(np.median(np.abs(np.asarray(predicted)[retained, 1] -
                                                        np.asarray(actual)[retained, 1]))) if retained.any() else None}


def run_calibration(selective_run, polars, split, geometry_coverage, output_dir):
    """Frozen CLI study. Persist partitions and every policy before computing audit errors."""
    output, source = Path(output_dir), Path(selective_run)
    ra._require(not output.exists() and not output.is_symlink(), "Output directory must be fresh (nonexistent)")
    train, polars_hash = ra.load_train(polars, split), ra._digest(polars)
    configurations, fold_ids, provenance = ra.load_run(source, train, polars_hash, "selective_scaling")
    manifest = json.loads((source / "fold_study.json").read_text())
    ra._require(manifest.get("study_variant") == SOURCE_VARIANT and
                manifest["run_metadata"].get("created_at_utc") == SOURCE_CREATED_AT,
                "Only the frozen selective source study is permitted")
    record = configurations[CONFIGURATION]
    support = ra.geometry_support(pd.read_csv(geometry_coverage, dtype={"geometry_id": str}), train, fold_ids)
    distance = support[ra.DISTANCE].to_numpy()
    rows, identities = partition_geometries(train, fold_ids, manifest["plan"]["fold_plan"]["folds"],
                                           json.loads(Path(split).read_text())["split_geometries"])
    ra._require([identities[str(i)]["calibration"]["n_geometries"] for i in range(3)] == [95, 95, 95] and
                [identities[str(i)]["audit"]["n_geometries"] for i in range(3)] == [96, 96, 95],
                "Frozen evaluation geometry counts mismatch")
    provenance.update(study_variant=SOURCE_VARIANT, created_at_utc=SOURCE_CREATED_AT,
        array_fingerprints=record["array_fingerprints"],
        trial_config_sha256=hashlib.sha256(json.dumps(record["trial_configs"], sort_keys=True).encode()).hexdigest())
    plan = {"schema_version": 1, "status": "development_only", "qualified_confidence": False,
        "production_policy_recommendation": None, "source": provenance, "configuration": CONFIGURATION,
        "member_seeds": [42, 43, 44], "partition_seed": PARTITION_SEED, "methods": METHODS,
        "Cd_relative_error_budgets": BUDGETS, "empirical_risk_caps": RISK_CAPS,
        "quantiles": QUANTILES, "quantile_method": "lower", "minimum_calibration_geometries": MIN_SUPPORT,
        "selection": "maximum accepted calibration rows; tie lower threshold then lower quantile; no fallback",
        "cdf": "row-weighted right-continuous ECDF of nonclipped calibration rows only; ties use maximum rank",
        "support_rule": "combined score refers if either native component is outside its closed eligible calibration min/max interval",
        "hard_referral": "any raw-member clipping; nonfinite source evidence is rejected by loader",
        "weighting": "row-weighted risk and coverage; geometry summaries descriptive only",
        "polars": {"path": str(polars), "sha256": polars_hash},
        "split": {"path": str(split), "sha256": ra._digest(split)},
        "geometry_coverage": {"path": str(geometry_coverage), "sha256": ra._digest(geometry_coverage)},
        "implementation_sha256": {Path(path).name: ra._digest(path) for path in (__file__, ra.__file__)},
        "limitations": ["Previously inspected folds; not a fresh blind benchmark or statistical guarantee.",
            "Experimental error bands and empirical caps are not deployment limits or confidence levels.",
            "Twenty retained geometries is a support safeguard, not a guarantee or complete-polar qualification.",
            "No pooled cross-fold calibration, surrogate fitting, holdout scoring or audit method selection."]}
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / "partitions.csv", index=False)
    _json(output / "partitions.json", identities)
    plan["partition_sha256"] = {name: ra._digest(output / name) for name in ("partitions.csv", "partitions.json")}
    _json(output / "frozen_plan.json", plan)

    all_policies, all_candidates, references, fold_cdfs = [], [], {}, {}
    for fold in range(3):
        positions = rows.loc[(rows.fold_id == fold) & (rows.partition == "calibration"), "canonical_train_position"].to_numpy()
        cdfs, policies, candidates = calibrate_fold(record["raw"][positions], distance[positions], record["hard"][positions],
            train.geometry_id.to_numpy()[positions], train.iloc[positions][ra.TARGETS].to_numpy(), record["prediction"][positions])
        fold_cdfs[fold] = cdfs
        references.update({f"fold_{fold}_{key}": value for key, value in cdfs.items()})
        all_policies.extend({"fold_id": fold, **policy} for policy in policies)
        all_candidates.extend({"fold_id": fold, **candidate} for candidate in candidates)
    pd.DataFrame(all_candidates, columns=["fold_id", "method", "Cd_relative_error_budget", "empirical_risk_cap",
        "quantile", "threshold", "accepted_rows", "accepted_geometries", "FN", "conditional_risk", "eligible", "selected"]
        ).to_csv(output / "calibration_candidates.csv", index=False)
    np.savez_compressed(output / "score_cdfs.npz", **references)
    _json(output / "selected_policies.json", {"status": "development_only", "qualified_confidence": False, "policies": all_policies,
        "cdf_file": "score_cdfs.npz", "cdf_keys": "fold_{fold_id}_raw and fold_{fold_id}_geometry",
        "cdf_sha256": ra._digest(output / "score_cdfs.npz"), "frozen_plan_sha256": ra._digest(output / "frozen_plan.json")})

    # Every fold's selection is on disk before any audit or full-population error calculation.
    audit_summary, baselines, audit_rows, full = [], [], [], []
    for fold in range(3):
        positions = rows.loc[(rows.fold_id == fold) & (rows.partition == "audit"), "canonical_train_position"].to_numpy()
        actual, prediction = train.iloc[positions][ra.TARGETS].to_numpy(), record["prediction"][positions]
        raw, geo, hard = record["raw"][positions], distance[positions], record["hard"][positions]
        ids = train.geometry_id.to_numpy()[positions]
        common = {"fold_id": fold, "row_sha256": ra._row_hash(train.iloc[positions]), "hard_referral_rows": int(hard.sum())}
        for budget in BUDGETS:
            baselines.append({**common, "baseline": "no_referral_same_audit_rows",
                **diagnostics(actual, prediction, ids, np.ones(len(positions), bool), budget)})
        detail = rows.iloc[positions].copy()
        scores, outside = transform_scores(raw, geo, hard, fold_cdfs[fold])
        for key, value in scores.items():
            detail[key] = value
        detail["hard_referral"], detail["combined_outside_calibration_support"] = hard, outside
        for index, target in enumerate(ra.TARGETS):
            detail[f"actual_{target}"], detail[f"predicted_{target}"] = actual[:, index], prediction[:, index]
        for policy in (item for item in all_policies if item["fold_id"] == fold):
            accepted = apply_policy(policy, raw, geo, hard, fold_cdfs[fold])
            audit_summary.append({**common, **policy,
                **diagnostics(actual, prediction, ids, accepted, policy["Cd_relative_error_budget"])})
            key = f"accept_{policy['method']}_budget_{policy['Cd_relative_error_budget']:g}_cap_{policy['empirical_risk_cap']:g}"
            detail[key] = accepted
        audit_rows.append(detail)
    for fold in (-1, 0, 1, 2):
        positions = np.flatnonzero(np.ones(len(train), bool) if fold == -1 else fold_ids == fold)
        for budget in BUDGETS:
            full.append({"fold_id": fold, "population": "all_oof" if fold == -1 else "full_original_evaluation_fold",
                "row_sha256": ra._row_hash(train.iloc[positions]), "hard_referral_rows": int(record["hard"][positions].sum()),
                **diagnostics(train.iloc[positions][ra.TARGETS].to_numpy(), record["prediction"][positions],
                    train.geometry_id.to_numpy()[positions], np.ones(len(positions), bool), budget)})
    tables = {"audit_summary": pd.DataFrame(audit_summary), "audit_baseline": pd.DataFrame(baselines),
              "full_row_summary": pd.DataFrame(full), "audit_rows": pd.concat(audit_rows, ignore_index=True)}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    report = {"schema_version": 1, "status": "complete_development_only", "qualified_confidence": False,
        "production_policy_recommendation": None, "holdouts_scored": False, "surrogate_fitted": False,
        "n_policies": len(all_policies), "n_rows": len(train), "n_geometries": train.geometry_id.nunique(),
        "source": provenance, "limitations": plan["limitations"], "audit_summary": audit_summary,
        "audit_baseline": baselines, "full_row_summary": full,
        "definitions": {"positive": "Cd relative error strictly exceeds budget", "FN": "over-budget accepted",
            "FP": "within-budget referred", "conditional_risk": "FN/(FN+TN)", "missed_referral_rate": "FN/(TP+FN)",
            "FPR": "FP/(FP+TN)", "unnecessary_fraction_among_referrals": "FP/(TP+FP)",
            "geometry_macro_retained_risk": "equal-weight mean of within-geometry retained-row exceedance fractions",
            "undefined_denominators": None, "full_row_fold_id_minus_one": "descriptive all-row OOF, not pooled calibration"},
        "artifact_sha256": {path.name: ra._digest(path) for path in sorted(output.iterdir()) if path.is_file()}}
    _json(output / "summary.json", report)
    return tables, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selective-run", "polars", "split", "geometry-coverage", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    tables, _ = run_calibration(**vars(parser.parse_args()))
    print(tables["audit_summary"][["fold_id", "method", "Cd_relative_error_budget", "empirical_risk_cap",
          "status", "actual_coverage", "conditional_risk", "retained_geometries"]].to_string(index=False))
    print("Development-only empirical calibration; no qualified confidence or production recommendation.")


if __name__ == "__main__":
    main()
