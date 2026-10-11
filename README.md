# Airfoil ML Surrogate

Machine-learning surrogate modeling for aerodynamic prediction of airfoils.

## Status

The geometry-disjoint baseline and sequential model comparison have run in Google Colab. Notebook `04_mlp_improvement.ipynb` now implements the next controlled experiment: named 100-, 200-, and 400-iteration MLP variants with validation-only reports and opt-in final testing. Full-data experiments stay in Colab; local verification uses small synthetic tests. Final metrics, plots, and run metadata can be copied from Drive for Desktop into the Git-ignored `results/colab/` directory without copying datasets or trained models. Measured results below are from the earlier comparison, not claims about the new improvement variants.

## Research Question

Can machine-learning models learn the relationship between airfoil geometry, flow conditions, and aerodynamic coefficients accurately enough to act as a fast surrogate for conventional aerodynamic analysis?

## Objectives

- Predict `Cl` and `Cd` from geometry and operating conditions.
- Retain `Cm` only if it is supported cleanly by the source data.
- Establish interpretable linear and polynomial baselines before nonlinear models.
- Measure performance on airfoil geometries that are absent from training.
- Preserve reproducibility and prevent geometry leakage.

## Dataset

The primary source is the [NASA Airfoil Learning Dataset](https://github.com/nasa/airfoil-learning), an unprocessed collection of airfoil geometry and XFOIL aerodynamic results. The configured archive URL is:

`https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip`

The archive has been inspected and contains 825 JSON members under `json/`, with one airfoil per document. Each document includes `name`, `xps`, `xss`, `yps`, `yss`, and a `polars` list. The loader downloads the archive to `data/raw/`, discovers this structure, and converts each valid polar record into one tabular row. It preserves `geometry_id`, creates fixed-length `geom_000`-style features from the supplied coordinates, and retains `Alpha`, `Re`, `Ncrit`, `Cl`, `Cd`, and optional `Cm`. Large data files are ignored by Git.

## Methodology

Inputs are intended to include a fixed-length airfoil geometry representation plus angle of attack (`Alpha`), Reynolds number (`Re`), and `Ncrit` when available. Targets are `Cl` and `Cd`. NACA four-digit generation is implemented as a reusable utility, but dataset-provided geometry must be used for NASA records rather than assuming every airfoil is NACA.

The split uses `GroupShuffleSplit` twice. A geometry identifier must be supplied as the grouping series, so every operating condition for one geometry stays in exactly one of training, validation, or test. This is required because row-wise random splitting would leak geometry-specific information and overstate generalization.

Feature and target scalers are fitted on training data only. Metrics are reported separately for each target in original coefficient units: MAE, RMSE, and R². Notebook `03` verifies the processed CSV fingerprint and restores notebook `02`'s exact geometry partitions. Models are selected by validation RMSE per target; only those winners receive the final test evaluation. These scores measure agreement with the dataset's XFOIL labels, not independently validated experimental or CFD accuracy.

## Models

Implemented CPU-compatible model factories:

- Linear Regression
- Polynomial Features plus Ridge Regression
- Random Forest Regression
- Histogram Gradient Boosting Regression
- Multi-Layer Perceptron Regression

## Installation

Python 3.10 or newer is recommended. On an Apple Silicon Mac, the baseline uses CPU-compatible NumPy, pandas, scikit-learn, matplotlib, PyYAML, joblib, pytest, and Jupyter.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

Use a fresh Colab CPU runtime for `notebooks/02_baseline_model.ipynb`, followed by `notebooks/03_model_comparison.ipynb`. The notebooks clone the source repository, read the processed CSV from Drive, and save experiments under `MyDrive/airfoil-ml-surrogate/experiments/`. Notebook `03` runs linear regression, histogram gradient boosting, random forest, and MLP sequentially, saving progress after each model.

For the next accuracy experiment, use **[notebook 04 in Colab](https://colab.research.google.com/github/gaurav-zz/ML-for-CFD/blob/main/notebooks/04_mlp_improvement.ipynb)** after the new files are published. An existing saved baseline is required; do not rerun notebook `02` just to get another split.

Lightweight local commands, with the project's environment active:

```bash
python -m src.utils.sync_results
pytest
```

If data preparation is needed, run `python -m src.data.loaders` in Colab. It downloads if necessary and converts per-airfoil JSON records into `data/processed/airfoil_learning.csv`. An existing processed dataset need not be regenerated to run comparisons or sync reports.

The reusable `train_model` API expects explicitly selected feature and target matrices. Do not treat the raw processed table as a ready-to-fit matrix: it also contains identifiers and other aerodynamic outputs that must not leak into features.

## Repository Structure

```text
configs/config.yaml                 Reproducible parameters and paths
data/raw/                            Downloaded archive, ignored by Git
data/processed/                      Prepared tables, ignored by Git
src/data/                            Loading, validation, and splitting
src/geometry/                        NACA generation and fixed-length geometry
src/models/                          Classical model factories
src/training/                        Training and evaluation APIs
src/visualization/                   General and aerodynamic plots
src/utils/                           Metrics and report-only local sync
experiments/                         Baseline, comparison, and generalization outputs
notebooks/                           Exploration and analysis entry points
results/                             Ignored figures, metrics, and model artifacts
results/colab/                       Imported final Colab reports, ignored by Git
tests/                               Lightweight unit tests
```

## Experiments and Notebooks

- `experiments/baseline/`: interpretable linear baseline outputs.
- `experiments/model_comparison/`: common-split comparison across all model families.
- `experiments/generalization/`: unseen-geometry evaluation.
- `notebooks/01_data_exploration.ipynb`: dataset inspection and distributions.
- `notebooks/02_baseline_model.ipynb`: geometry-aware baseline workflow.
- `notebooks/03_model_comparison.ipynb`: model comparison and interpretation.
- `notebooks/04_mlp_improvement.ipynb`: sequential MLP budget variants, training loss curves, and held-out-airfoil validation diagnostics.

The notebooks call `src/` functions for training, splitting, and evaluation. Heavy computation is not part of the local report-sync workflow.

## Run the MLP Improvement Experiment

1. Open the newly published notebook `04` in a fresh Colab CPU runtime. An older Drive-saved notebook does not update when GitHub changes.
2. Leave `BASELINE_RUN = None` to use the latest saved baseline, or specify its `run.json`. The CSV hash, feature order, and original geometry partitions must match. Targets remain those of the baseline, currently Cl and Cd.
3. Leave `EXISTING_RUN = None` and `RUN_FINAL_TEST = False`, then select **Runtime > Run all**. The queue makes fresh fits named `mlp_100`, `mlp_200`, and `mlp_400`. The 100-iteration control is rerun for comparable runtime versions and diagnostics; the old completed model is not silently reused.
4. Compare full-row validation RMSE/R², actual training iteration counts, warnings, and per-airfoil errors. The optimizer can stop before its budget. The plots distinguish training loss from held-out performance; a lower training loss is not proof of improved generalization.
5. Refresh local reports with `python -m src.utils.sync_results`. Models remain in Drive under `experiments/mlp_improvement/<timestamp>/<variant>/mlp.joblib`.

Only the maximum iteration budget changes by default: architecture `(64, 32)`, batch size `1024`, initial learning rate `0.001`, seed `42`, and train-only feature/target scaling remain fixed. There is no internal random validation holdout. All three fits run sequentially; ordinary model failures are logged and later variants continue. More training may be slower or worse, so the result is an experiment rather than a promised R² increase.

The runner accepts an optional, unique `id` independently of the model-family `name`. For example, `id="mlp_200"` with `name="mlp"` saves a distinct artifact and score row. Existing family-named runs from notebook `03` remain readable. Unsafe or duplicate IDs are rejected.

### Reopen Without Retraining

Set `EXISTING_RUN` to the completed Drive directory, for example `/content/drive/MyDrive/airfoil-ml-surrogate/experiments/mlp_improvement/<timestamp>`. Leave `BASELINE_RUN = None` to restore that experiment's recorded baseline. Running notebook `04` then skips all fitting, preserves existing report files, and creates any missing diagnostic plots. Dataset, feature, split, target-order, and scikit-learn-version mismatches are rejected. Interrupted fits are not automatically resumed.

### Final Test Is Opt-In

Keep `RUN_FINAL_TEST = False` during development. Once decisions are frozen, set it to true and run only the notebook's final cell in the existing session. After reconnecting, use `EXISTING_RUN` as above to restore the completed experiment without fitting again. `evaluate_selected_models` loads the saved validation winners; test scores never select or rerank variants.

The default run creates no test metrics. Optional final-test evaluation adds `test_metrics.csv` and `test_evaluation.json` to the Drive run. The local importer creates a separate `<timestamp>_with_test` snapshot, leaving the earlier validation-only snapshot unchanged.

### What Unseen Airfoils Means

The existing benchmark already excludes all rows of the 123 test geometry IDs from training. Notebook `04` also reports every held-out validation airfoil separately, plus equal-airfoil mean/median and worst RMSE, rather than relying only on an aggregate dominated by densely sampled polars. Selection still uses row-weighted validation RMSE to keep this budget comparison controlled.

This is not yet proof of performance on arbitrary external airfoils. Duplicate or near-duplicate shapes under different IDs require an explicit similarity audit, and profiles/conditions outside the dataset need independent reference labels. Do not describe validation scores used for tuning as an untouched final test result.

## Local Results

Keep Google Drive for Desktop connected to the account used in Colab. From the repository root, refresh local report snapshots with:

```bash
.venv/bin/python -m src.utils.sync_results
```

On macOS the command discovers this project's `experiments` folder under Drive for Desktop. For another location, pass `--source "/path/to/airfoil-ml-surrogate/experiments"`. `--destination` overrides the default `results/colab/` destination.

- Baseline imports contain only `run.json`, `metrics.csv`, and `predicted_vs_actual.png`.
- Comparison imports contain only `run.json`, `comparison.csv`, `test_evaluation.json`, `test_metrics.csv`, `validation_rmse.png`, and `selected_predictions.png`.
- Improvement imports work **without test evaluation**: `run.json`, `comparison.csv`, `validation_by_geometry.csv`, `validation_summary.json`, and four plots (`validation_rmse.png`, `training_loss.png`, `validation_predictions.png`, `worst_airfoils.png`). If a final test is later run, its two report files are copied into a new `<run>_with_test` snapshot.
- Datasets, model weights, and per-model directories are not read or copied. Report files are limited to 10 MiB each.
- Incomplete reports are skipped. Identical local snapshots are reused; conflicting local files are never overwritten. New snapshots are published only after copying all report files successfully.
- Drive synchronizes its own folder automatically. The workspace copy is refreshed **on demand**, not by a background watcher. The assistant can run this command and inspect local reports when asked to check the latest results, with no pasted tables or screenshots needed.

The imported comparison from `20261009T111211403902Z` is available at `results/colab/model_comparison/20261009T111211403902Z/`. New improvement reports appear under `results/colab/mlp_improvement/`. The sync utility does not change the Colab notebooks or rerun training.

## Measured Results

The 2026-10-09 comparison used source commit `bd1848b`, with 538,380 training rows from 572 geometries, 113,678 validation rows from 123 geometries, and 116,484 test rows from 123 geometries. The MLP won validation RMSE for both targets. Values below come from the original imported CSVs, rounded for display.

| Target | MLP Validation RMSE | MLP Test RMSE | MLP Test R² | Approximate Test RMSE Needed for R² = 0.99 |
| --- | ---: | ---: | ---: | ---: |
| Cl | 0.084447 | 0.116122 | 0.976978 | 0.07653 |
| Cd | 0.008407 | 0.008957 | 0.925679 | 0.003286 |

The final column assumes the **same held-out target distribution**, using `RMSE_goal = RMSE_current * sqrt((1 - R2_goal) / (1 - R2_current))`. It implies about 34% lower Cl RMSE and 63% lower Cd RMSE, not a small final adjustment. R² = 0.99 means 99% of target variance explained relative to the constant-mean reference; it is not "99% of predictions correct" or a guarantee of physical accuracy.

Training took about 388 seconds for the MLP, 173 seconds for histogram gradient boosting, and 1,000 seconds for the random forest. The forest was slower and less accurate than the MLP on validation in this run. The real MLP run recorded a `ConvergenceWarning` at its 100-iteration limit; this motivates a training-budget experiment, but does not prove that more iterations will improve held-out performance.

## Testing

Tests cover geometry utilities, grouped splitting, train-only scaling, model serialization, per-target evaluation, sequential failure handling, validation-only winner selection, notebook workflows on synthetic data, and report-only copying. Run `pytest` after installing dependencies. Tests do not require downloading or training on the NASA dataset.

## Strategy Toward R² = 0.99

This is a prioritized plan, not a claim that these experiments have already run or that 0.99 is attainable for every target. Keep heavy work in Colab and make model decisions using training/grouped validation data, not repeated test-set tuning. Preserve the current test set for final confirmation; if continued inspection turns it into a development set, obtain an additional untouched geometry holdout for an unbiased final claim.

1. **Diagnose the remaining errors.** Export validation MAE/RMSE by geometry, angle-of-attack range, Reynolds number, and drag range, plus loss curves and convergence status. The saved test scatter plot shows a mostly aligned trend with large off-diagonal errors, but a sampled plot alone cannot identify their cause or quantify their contribution. Use validation diagnostics to decide whether the main issue is optimization, sparse regimes, geometry representation, or questionable labels.
2. **Increase budget before architecture.** Notebook `04` implements fresh 100-, 200-, and 400-iteration fits with the current `(64, 32)` architecture and other settings fixed, recording convergence and per-airfoil validation performance. Epoch-by-epoch validation early stopping is a separate future improvement; the notebook does not enable scikit-learn's internal random row holdout.
3. **Tune a small number of controlled alternatives.** At the selected training budget, vary learning rate (`0.001` versus `0.0005`), then batch size (`1024` versus `512`), then capacity (`(64, 32)` versus `(128, 64)`). Change one factor at a time before trying a deeper network or stronger regularization. Keep histogram gradient boosting as the faster nonlinear comparator rather than spending most of the budget on the slower forest. Named variant IDs now allow multiple configurations from the same family in one queue; these extra hyperparameter trials are not enabled by default.
4. **Improve physically meaningful inputs.** Audit common upper/lower-surface alignment, chord normalization, and leading/trailing-edge sampling. Compare thickness/camber descriptors, `log10(Re)`, and a few angle/shape interactions against the current coordinate representation. If using PCA or learned geometry encodings, fit them only on training geometries within each fold. Avoid expanding all 103 inputs into a dense quadratic matrix.
5. **Investigate drag-specific learning.** Compare joint standardized-target prediction against a dedicated Cd model. A positive/log-drag transform is an ablation, not an automatic improvement: choose any offset using training data, transform predictions back, and select using original-unit Cd RMSE/R². A log loss may improve low-drag relative errors while worsening high-drag squared errors. Do not drop difficult rows or clip predictions merely to inflate the reported score.
6. **Measure robustness, then ensemble.** Repeat promising configurations with at least three seeds and geometry-grouped cross-validation within the development data. Report per-geometry error summaries alongside row-weighted metrics. If seed-to-seed variation is material, compare an average of a few MLPs, or validation-weighted MLP/boosting predictions, on validation only. Ensembling may reduce variance but cannot reliably repair systematic feature or label problems.
7. **Check the data ceiling if improvement stalls.** Review solver validity and difficult flow regimes using documented, consistent criteria. More operating-condition rows from the same airfoils do not replace more independent training geometries. Additional clean geometry coverage may be more valuable than a larger network. Any narrower-domain claim should be labeled explicitly, not presented as unrestricted R² = 0.99.

## Additional Targets

- **Cm (pitching-moment coefficient):** the loader retains it and the processed CSV has a `Cm` column. Before adding it, check completeness, numeric validity, reference point, and sign convention. Compare a three-output model against separate/shared-head alternatives; fit target scalers on training labels only. Require acceptable Cl and Cd performance as well, since a new target can change shared-model tradeoffs.
- **Lift-to-drag ratio:** derive it from predicted Cl and Cd and validate the resulting ratio against reference values. Define how near-zero Cd is handled; small denominator errors can make this much less reliable than either coefficient alone.
- **Maximum lift and stall angle:** treat these as polar-level quantities grouped by geometry, Re, and Ncrit. They require sufficient angle coverage and reliable labels; a maximum at the sampled angle boundary does not establish a true stall point.
- **Pressure distributions or transition locations:** these are not in the current processed CSV. Inspect raw label availability and definitions before promising such outputs; geometry coordinates alone do not supply these labels.

New-target experiments should reference the existing geometry partitions in a **new** run manifest, never edit old baseline results. If missing labels change row eligibility, freeze and record a common mask for the models being compared, and keep that benchmark distinct from the existing full-row Cl/Cd benchmark. Do not feed true Cl, Cd, or Cm back as features unless the deployment problem explicitly provides those measurements.

Accuracy on unseen geometries and agreement with experimental aerodynamics remain different questions. The current results establish a useful XFOIL surrogate benchmark, not a replacement for independent aerodynamic validation.

## Citation

Cite the NASA Airfoil Learning Dataset and the original XFOIL-related source material according to the authoritative NASA project documentation: <https://github.com/nasa/airfoil-learning>.

## License

The project is intended to use the MIT License. A license file has not yet been added to this code-only milestone; dataset licensing and attribution should be checked independently before redistribution.
