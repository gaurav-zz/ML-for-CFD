# Airfoil ML Surrogate

**Rapid aerodynamic screening, with reference-data and physics-solver support where the surrogate is insufficient.**

## Goal

The project synopsis defines the aim as reducing the time and computational cost of
airfoil analysis through machine-learning predictions of aerodynamic characteristics.
The objective is not to replace every simulation or maximize one aggregate R² value.

The intended workflow is to batch-screen many geometries and operating points,
identify where predictions are sufficiently useful for the task, and refer uncertain
or unsupported cases to existing reference data or, when eligible, a physics solver.
Useful acceleration must be measured together with prediction error, referral quality,
and the computational cost of the remaining reference work.

This follows the local `synopsis minor project (1).pdf`: geometry-based modelling,
comparative regression models, computational efficiency, unseen-geometry validation,
and pressure prediction/visualization where feasible.

## Current Status

| Available | Not yet qualified or automated |
|---|---|
| Geometry-derived features and Cl/Cd/Cm regression models | A calibrated prediction-time acceptance/referral policy |
| Portable NumPy inference from saved neural models | Guaranteed error bounds or confidence percentages |
| Geometry-disjoint development folds and paired experiments | An end-to-end surrogate-plus-CFD service or measured solver speedup |
| Raw clipping, ensemble disagreement, geometry-support, and offline risk/coverage diagnostics | A validated surrogate pressure-distribution output |
| Reproducible source snapshots, saved predictions, and audit receipts | Automatic production promotion of the latest experimental models |

The models can provide research screening estimates. Low disagreement or proximity
to training shapes is not, by itself, evidence that a prediction is accurate.
Historical model decisions and the pinned reference remain unchanged.

## Approach

1. **Validate the request.** Check geometry, units, coefficient conventions, and
   operating-condition scope. Invalid geometry is not made valid by low uncertainty.
2. **Batch predictions.** Reuse geometry features across operating points and cache
   appropriate computations. Model inputs are shape and flow features, not airfoil IDs.
3. **Assess applicability.** Combine whole-profile similarity, camber/thickness and
   other descriptors, operating conditions, raw numerical diagnostics, and ensemble
   information. The acceptance rule must be calibrated and independently evaluated.
4. **Retain or refer.** Qualified estimates support fast screening. Uncertain cases
   are deferred for reference evaluation, not silently dropped or declared bad designs.
5. **Use existing data first.** Seek suitable published polars, measurements, or CFD
   for the exact geometry and required conditions before considering a new simulation.
6. **Validate decisions.** Important finalists need appropriate reference confirmation;
   an optimizer or ranking procedure can exploit surrogate errors.

The automatic routing part is a target workflow, not an already validated capability.
Known difficult airfoils remain in evaluation. We do not improve reported accuracy by
removing them, nor assume a name-based blacklist will generalize to new geometries.

## Current Results

The latest selective-floor Cdp ensemble was evaluated on **538,380 out-of-fold rows
covering 572 canonical-training geometries**. Every row was evaluated once per ensemble;
the original canonical validation/test partitions were not scored by this study.

| Target | Pooled OOF R² | RMSE | Median relative error |
|---|---:|---:|---:|
| Cl | 0.9873 | 0.08479 | 4.32% |
| Cd | 0.8983 | 0.009512 | 6.11% |
| Cm | 0.9245 | 0.01375 | 9.73% |

These are full-population development scores, not accuracy guarantees or percentages
of trustworthy predictions. The candidate has no observed inverse-limit clipping,
but fails a frozen worst-geometry drag regression check. It was not promoted.

The offline referral analysis illustrates both opportunity and limitations: retaining
95% of rows by a combined input-support/disagreement ranking reduces retained Cd RMSE
to **0.00723**. However, about **30.3% of those retained rows exceed an illustrative
10% Cd-error budget**. Retained coverage is not confidence, and these inspected data
do not validate an operational gate. See [validation and reliability](docs/validation.md).

## Data And Scope

- Source: the NASA `airfoil-learning` archive, with XFOIL-associated AirfoilTools polars.
  These labels are not automatically experimental or CFD ground truth.
- **825 source cases**, of which **818** contribute the **768,542** processed rows.
  Seven source cases without usable polars still count as belonging to the dataset.
- Geometry uses a shared chord transformation and sampled thickness/camber profiles,
  plus scalar descriptors. The current input has **113 features**, including angle
  features, log Reynolds number, and Ncrit.
- Geometry is nondimensionalized by chord. Dimensional chord is reference metadata
  and contributes through Reynolds number; it is not an independent learned feature.
  The model has no independent Mach input and does not establish compressible,
  three-dimensional, or unsteady-flow validity.
- Cdp is experimental auxiliary supervision, not an input or a pressure-field output.
  The public primary quantities are Cl, Cd, and Cm.
- Geometry-ID separation is not a guarantee that every related shape family or near
  duplicate is withheld. Current development folds have also been repeatedly inspected.

## Cloud-First Execution

Training, experiments, computational analysis/validation, and simulations should be
attempted on cloud resources first. Local compute is a fallback only after cloud
execution is found infeasible or unsuitable and the reason is documented. Repository
editing, inspection, orchestration, and cleanup remain local work.

Kaggle is the established ML execution path; cloud CPU is sufficient for fit-free
tests and offline analysis. Physics jobs may need a more suitable persistent cloud
runtime. OpenFOAM locally is a fallback, not the default next step.

Portable dependencies are in `requirements.txt`; cloud fitting additionally uses
`requirements-kaggle.txt`. Python 3.10+ is supported; 3.12 is used in the validated
workflow. In a prepared cloud runner:

```bash
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider -m 'not gpu and not cloud_fit'
python -m src.training.reliability_analysis --help
```

Real fitting tests have separate Kaggle opt-in guards. These guards do not make local
execution the preferred policy. Do not install or import local PyTorch merely to
inspect portable model outputs.

The local driver can inspect and retrieve an existing cloud job:

```bash
python -m src.utils.kaggle_sync status <owner>/<kernel-slug>
python -m src.utils.kaggle_sync pull <owner>/<kernel-slug>
```

Kernel push starts another run; it is not a status check. Source staging is private
by default and allowlisted, excluding credentials, personal reports, and raw archives.
Retain source/data hashes and check that scheduled monitoring remains active after
an interrupted session; a requested accelerator is not proof of utilization.

## Independent Solver Validation

No new simulation is approved by a low-confidence prediction alone. For this project:

1. The airfoil must be absent from the **entire source dataset**, including the seven
   unusable cases, aliases, transformed copies, and relevant near-duplicates. Being
   absent from a training fold is insufficient.
2. Search for suitable matching online data first and record its coverage/provenance.
3. Only if the required suitable data are unavailable may an eligible new geometry
   proceed to a separately planned, bounded physics run.

Known dataset airfoils should use their available references for error diagnosis;
they are not eligible for new project simulations under this policy. PyFR on cloud
and OpenFOAM on cloud or, if necessary, a local fallback are future options, not
automatic ground truth. See [the solver-validation plan](docs/solver-validation.md).

## Next Milestones

- Agree task-specific absolute/relative error budgets and design-screening criteria.
- Validate a referral rule on geometry-separated calibration/audit data, reporting
  false accepts, unnecessary referrals, coverage, and performance on new families.
- Build an explicit reference-data and full-dataset novelty registry before any CFD job.
- Benchmark batch inference, lookup, geometry setup, and fallback costs on matched tasks.
- Extend pressure visualization from qualified reference/solver fields first; add
  surrogate pressure prediction only after separate data-quality and accuracy checks.

## Repository Guide

- `src/data`, `src/geometry`: ingestion, validation, and geometry/flow representation.
- `src/models`, `src/training`: models, experiments, evaluation, and reliability analysis.
- `notebooks/v2_experiment_kaggle.ipynb`: cloud experiment entry point.
- `configs/`: frozen partitions and pinned reference evidence.
- `docs/validation.md`: current evidence, referral metrics, and calibration plan.
- `docs/solver-validation.md`: cloud-first, online-data-first independent validation.
- `docs/archive/`: historical notes, not current project instructions.
- `data/`, `results/`: preserve inputs, weights, predictions, and provenance; not clutter.

Detailed run narratives are preserved in the [documentation archive](docs/archive/README.md)
rather than appended to this project guide. The local synopsis and personal reports
are retained unchanged.
