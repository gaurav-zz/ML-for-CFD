# Airfoil ML Surrogate

Machine-learning surrogate modeling for aerodynamic prediction of airfoils.

## Status

This repository contains the code-only first milestone. The NASA archive has **not** been downloaded, the archive structure has **not** been inspected, and no training, evaluation, tests, or figures have been executed in this setup pass. Consequently, no numerical results or dataset statistics are claimed.

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

Scalers are fitted on training features only. Metrics are reported separately for each target: MAE, RMSE, and R². Training and inference timing are supported by the training/evaluation APIs. No claim about accuracy, speed, or superiority to CFD is made until experiments are actually run.

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

The commands below are intentionally documented but were not executed during the code-only implementation pass:

```bash
python -m src.data.loaders
python -m src.training.train --data data/processed/features.csv --target Cl --model linear
pytest
```

`python -m src.data.loaders` downloads if necessary, inspects the NASA archive, converts the per-airfoil JSON records into a tabular dataset, and saves it to `data/processed/airfoil_learning.csv`. The JSON schema has now been mapped, but the full conversion and validation have intentionally not yet been executed.

The training module expects a prepared feature matrix and currently provides the reusable `train_model` API. A production run should create target-specific or multi-output training matrices after the NASA schema mapping is finalized.

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
src/utils/                           Metrics
experiments/                         Baseline, comparison, and generalization outputs
notebooks/                           Exploration and analysis entry points
results/                             Ignored figures, metrics, and model artifacts
tests/                               Lightweight unit tests
```

## Experiments and Notebooks

- `experiments/baseline/`: interpretable linear baseline outputs.
- `experiments/model_comparison/`: common-split comparison across all model families.
- `experiments/generalization/`: unseen-geometry evaluation.
- `notebooks/01_data_exploration.ipynb`: dataset inspection and distributions.
- `notebooks/02_baseline_model.ipynb`: geometry-aware baseline workflow.
- `notebooks/03_model_comparison.ipynb`: model comparison and interpretation.

The notebooks are deliberately thin scaffolds and should call `src/` functions instead of duplicating ML logic.

## Testing

Tests cover NACA parsing and generation, normalization, grouped splitting, train-only scaling, model construction, prediction dimensions, and metric keys. They are designed to be lightweight and can be run with `pytest` after dependencies are installed. They have not been run in this setup pass.

## Limitations and Next Steps

- The loader has been mapped to the verified JSON schema, but the full archive conversion remains to be executed and checked for malformed or unusual records.
- Processed feature construction needs validation on all 825 airfoil documents.
- No model performance, generated figures, or generalization conclusions exist yet.
- Future work may add uncertainty estimates, cross-validation by geometry, richer descriptors, and optional `Cm` prediction after validating the core pipeline.

## Citation

Cite the NASA Airfoil Learning Dataset and the original XFOIL-related source material according to the authoritative NASA project documentation: <https://github.com/nasa/airfoil-learning>.

## License

The project is intended to use the MIT License. A license file has not yet been added to this code-only milestone; dataset licensing and attribution should be checked independently before redistribution.
