# Airfoil ML Surrogate

Machine-learning surrogate modeling for aerodynamic prediction of airfoils.

## Research Question

Can machine-learning models learn the relationship between airfoil geometry, flow
conditions, and aerodynamic coefficients accurately enough to act as a fast surrogate for
conventional aerodynamic analysis — **on airfoils they have never seen**?

That last clause is the whole claim, and it is why the geometry representation described
below matters more than the choice of model.

## Status

v2 is implemented and unit-tested. Heavy training runs on Kaggle; the local checkout
handles data preparation, the transfer to and from Kaggle, and the full test suite.

**No v2 model results are claimed yet.** The v1 numbers quoted below are real but are
reported with the caveat that v1's geometry representation was broken — see the next
section.

## The v1 geometry bug, and why v1's scores are optimistic

`src/geometry/representation.py` used to normalize the upper and lower surfaces
**independently**, and each normalization subtracted that surface's own mean `y`:

```python
coordinates[:, 1] -= coordinates[:, 1].mean()   # the bug, applied per surface
```

Two offsets per airfoil were discarded, and the difference between them is exactly what
sets thickness. Measured on the v1 processed dataset:

- **43% of sampled stations had the upper surface below the lower one** — negative
  thickness, which is physically impossible.
- **Every airfoil** had negative thickness somewhere.
- Both surfaces' mean `y` sat at ~0.003, the signature of independent centering.
- The leading and trailing edges failed to close, by up to 0.52 chord.

Thickness and camber were therefore **unrecoverable** from v1's `geom_000..geom_099`
features, and the offsets are not stored anywhere, so the old processed CSV could not be
repaired — only regenerated.

v1 still scored well (test `Cl` R² = 0.977) because 100 numbers still **fingerprint** each
of the 818 geometries, which lets a flexible model memorize per-geometry behavior. That
crutch does not exist for an unseen airfoil. **Do not compare v2 geometry results against
v1's numbers without stating that v1's representation was broken.**

### What v2 does instead

- Chord is taken from the **combined** point set, never per surface.
- Both surfaces are rotated, translated, and scaled by **one shared transform**, so the
  chord line runs from `(0, 0)` to `(1, 0)`.
- `y` is **never** mean-centered.
- Both surfaces are sampled on a **shared, cosine-spaced** `x` grid, which resolves the
  high-curvature leading edge that uniform spacing under-samples.
- Geometry is stored as a physical **thickness/camber** basis (`y_upper = camber + t/2`),
  a lossless reparametrization that is far better conditioned, plus named scalar
  descriptors: max thickness and its `x`, max camber and its `x`, leading-edge radius,
  trailing-edge angle and thickness, and enclosed area.
- Source coordinates that cross are clipped to zero thickness and **counted** in
  `load_report.json`, rather than silently absorbed. The raw archive has a few artifacts
  of order 1e-4 chord, mostly in the Drela AG series.

The descriptors are checked against analytic NACA four-digit sections, for which
thickness, camber, and leading-edge radius (`r = 1.1019 t²`) are known in closed form.

## Dataset

The source is the [NASA Airfoil Learning Dataset](https://github.com/nasa/airfoil-learning):
825 JSON members under `json/`, one airfoil per document, with `name`, `xss`/`yss`
(suction/upper), `xps`/`yps` (pressure/lower), and a `polars` list.

| field | observed |
|---|---|
| airfoils | 825 documents; **818** contribute rows (7 have no usable polars) |
| rows | 768,542 operating conditions |
| `Alpha` | −19.75 … 24.25°, 177 distinct; ~17% of rows are \|Alpha\| > 12 (post-stall) |
| `Re` | **8 distinct values**, 5e4 … 5e6 |
| `Ncrit` | **4 distinct values**: 0, 1, 5, 9 |
| `Cl` | −2.29 … 2.51 |
| `Cd` | 0.00265 … 0.59 — a **223×** range |
| `Cm` | fully populated, zero NaNs |

The loader emits **two tables** joined on `geometry_id` instead of one monolithic CSV,
because geometry is constant per airfoil and v1 repeated 100 columns across all 768k rows:

| file | rows | size |
|---|---|---|
| `data/processed/airfoil_geometry.csv` | 818 | 1.7 MB |
| `data/processed/airfoil_polars.csv` | 768,542 | 55 MB |
| `data/processed/load_report.json` | — | conversion provenance |

That is a 57 MB upload instead of 1.7 GB — about 30× smaller — which matters every time a
Kaggle session restarts. `src/data/features.py` joins them back into a design matrix in
memory (113 features; ~490 MB for the training partition in float64). Large data files are
ignored by Git.

## Methodology

**Features.** The thickness/camber profile and the scalar descriptors, plus flow features:
`Alpha`, `sin(Alpha)`, `cos(Alpha)`, `log10(Re)`, and `Ncrit`. `log10(Re)` is used because
the eight Reynolds numbers span two decades and raw values standard-scale poorly;
`sin(Alpha)` is used because thin-airfoil theory gives `Cl ≈ 2π sin α`, a relationship a
linear model cannot express from degrees alone. `Alpha` is kept in degrees alongside its
sine and cosine so tree models can split cleanly on the post-stall boundary.

`geometry_id` is never a feature — it defines the split groups, so including it would leak
precisely the thing the unseen-geometry claim rests on.

**Targets.** `Cl`, `Cd`, and `Cm`. `Cd` is fitted as `log(Cd)` and inverted for reporting,
so a squared loss stops being dominated by high-drag stalled cases. Every metric is
computed in **physical** coefficient units; `src/training/targets.py` enforces this
structurally rather than leaving it to notebook discipline.

**Split.** Geometry-disjoint, by `GroupShuffleSplit` on `geometry_id`, so every operating
condition for one airfoil stays in exactly one partition. Row-wise random splitting would
leak geometry-specific information and overstate generalization.

v2 **reuses v1's exact partitions**, committed to `configs/frozen_split.json`
(572 / 123 / 123 geometries), so the held-out airfoils are identical and the comparison is
honest. A new dataset fingerprint is recorded for the v2 tables; the leakage assertions
(disjoint, unique, exact coverage) are re-checked on every load.

**Scalers** are fitted on training features only.

**Metrics** are reported per target: MAE, RMSE, R², and mean/median absolute relative
error. Read the **median** relative error in preference to the mean, since rows whose
actual value is near zero — common for `Cl` near zero lift — dominate the mean.

**Evaluation depth.** `src/training/analysis.py` reports per-geometry error distributions,
a pre/post-stall split at |Alpha| > 12°, a per-Reynolds breakdown, the worst-case airfoils
by name, and `L/D` error, which amplifies drag error and is what a designer actually
compares sections on.

## Models

CPU-compatible scikit-learn factories: Linear Regression, Polynomial + Ridge, Random
Forest, Histogram Gradient Boosting, and an MLP. The comparison queue trains linear,
gradient boosting, random forest, and MLP **sequentially**, checkpointing each before the
next starts.

Full quadratic expansion is deliberately excluded from the queue: ~113 inputs expand to
thousands of features, tens of gigabytes for one float64 training matrix at 538k rows.

**Model capacity tuning is explicitly out of scope for v2.** Hyperparameters are held at
v1's values so the feature and target changes are the only difference. v1's MLP hit its
`max_iter` and logged a `ConvergenceWarning` while still winning, so headroom remains.

## v1 baseline, for reference only

Validation scores on the geometry-disjoint split (train 572 / validation 123 / test 123
geometries; 538,380 / 113,678 / 116,484 rows):

| model | Cl R² | Cd R² | train s |
|---|---|---|---|
| linear | 0.916 | 0.119 | 7.7 |
| gradient_boosting | 0.980 | 0.878 | 173 |
| random_forest | 0.969 | 0.875 | 1000 |
| **mlp** (selected) | **0.987** | **0.921** | 388 |

Frozen-winner test: `Cl` R² = 0.977, `Cd` R² = 0.926.

Two caveats. The representation was broken, as described above, so these are optimistic
for the wrong reason. And the `Cd` numbers are flattering in a second way: MAE 0.0049
against a 1st-percentile `Cd` of 0.006 is roughly an **80% relative error** on low-drag
cruise airfoils. Reporting relative error alongside absolute is why v2 adds it.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt
```

## Usage

### 1. Build the two tables locally

```bash
python -m src.data.loaders
```

Downloads the archive to `data/raw/` if absent, converts it, and writes both tables plus
`load_report.json`. The report records the airfoils with no usable polars, skipped polar
records, corrected surface swaps, and the clipped-thickness count. Validate a table with:

```bash
python -m src.data.validation --data data/processed/airfoil_polars.csv
```

### 2. Push the tables to Kaggle

Requires the `kaggle` CLI and an API token at `~/.kaggle/kaggle.json`.

```bash
python -m src.utils.kaggle_sync push <owner>/<dataset-slug> --create
```

Use `--create` only the first time; later runs omit it to add a new version. Only the two
tables and the report are uploaded — never the raw archive or model weights.

### 3. Run the experiment on Kaggle

The notebook clones this repo inside the Kaggle session, so **the branch it pins
(`REPO_BRANCH`, currently `v2-physical-geometry`) must be pushed to GitHub first** —
otherwise the setup cell fails on `git clone`.

Open `notebooks/v2_experiment_kaggle.ipynb` as a Kaggle notebook, then:

1. **Add Input → Datasets** → attach the dataset you just pushed.
2. **Settings → Internet: On**, so the source checkout can be cloned.
3. **Save Version → Save & Run All.**

The notebook clones this repo into Kaggle scratch (not `/kaggle/working`, which becomes
the kernel output), restores the frozen split, trains the queue, evaluates the frozen
winners on the test split, and writes metrics, `analysis.json`, and figures.

### 4. Pull the results back

```bash
python -m src.utils.kaggle_sync pull <owner>/<kernel-slug>
```

Each pull lands in its own `results/kaggle/<run-id>/` and never overwrites an existing run.

### Testing

```bash
pytest
```

Covers corrected normalization (non-negative thickness, edge closure, chord scaling,
invariance to rotation/scale/translation/point order), descriptors against analytic NACA
geometry, the two-table loader, feature assembly and leakage guards, the target-transform
round trip, the new metrics, the frozen-split guards, the Kaggle transfer tool, and an
end-to-end run of the Kaggle notebook against synthetic tables.

## Repository Structure

```text
configs/config.yaml                 Reproducible parameters and paths
configs/frozen_split.json           The geometry-disjoint partitions, reused by v2
data/raw/                           Downloaded archive, ignored by Git
data/processed/                     The two generated tables, ignored by Git
src/data/loaders.py                 Archive -> geometry + polars tables
src/data/features.py                Flow features and design-matrix assembly
src/data/preprocessing.py           Grouped splitting, frozen splits, train-only scaling
src/data/validation.py              Tabular and geometry validation
src/geometry/representation.py      Shared-transform normalization, thickness/camber
src/geometry/descriptors.py         Physical scalar descriptors
src/geometry/naca.py                NACA four-digit generation (analytic ground truth)
src/models/                         Model factories
src/training/comparison.py          Sequential queue on the frozen split
src/training/targets.py             Per-target transforms, physical-unit reporting
src/training/analysis.py            Regime, Reynolds, per-geometry and L/D breakdowns
src/training/{train,evaluate}.py    Fitting and scoring APIs
src/utils/metrics.py                Absolute and relative error
src/utils/kaggle_sync.py            Push tables to / pull results from Kaggle
src/visualization/                  Plot helpers
notebooks/                          The Kaggle experiment runner
results/kaggle/                     Pulled run outputs, ignored by Git
tests/                              Unit tests
```

## Limitations and Next Steps

- No v2 model results exist yet; the pipeline is implemented and tested, not yet run.
- `Cdp`, `Cp_ps`/`Cp_ss` (100-point pressure distributions), and the `alpha_min`/`max`/
  `step` fields are present in the archive and unused.
- The leading-edge radius is fitted from the vertical thickness distribution, so for a
  strongly cambered section it runs above the radius measured normal to the camber line.
  It remains a monotonic, physical feature; it is not a certified nose radius.
- Converging the MLP, cross-validation by geometry, and uncertainty estimates are all
  deferred.

## Citation

Cite the NASA Airfoil Learning Dataset and the original XFOIL-related source material per
the NASA project documentation: <https://github.com/nasa/airfoil-learning>.

## License

Intended to use the MIT License; no license file has been added yet. Dataset licensing and
attribution should be checked independently before redistribution.
