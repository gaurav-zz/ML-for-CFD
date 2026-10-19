# Airfoil ML Surrogate

Machine-learning surrogate modeling for aerodynamic prediction of airfoils.

## Research Question

Can machine-learning models learn the relationship between airfoil geometry, flow
conditions, and aerodynamic coefficients accurately enough to act as a fast surrogate for
conventional aerodynamic analysis — **on airfoils they have never seen**?

That last clause is the generalization claim. Valid geometry and a geometry-disjoint
evaluation protocol are prerequisites; accuracy improvements still require measurement.

## Status

**Latest:** the 12-fit controlled study completed on 19 October 2026. Huber improves
median error across all three seeds, but drag extrapolation is unstable. The
validation-selected no-weight-decay trial is **not an overall accuracy upgrade**:
its slightly better median conceals substantially worse RMSE. Defaults and the
pinned CPU reference have not been changed.

The completed Kaggle CPU run (`20261018T115938507650Z`) and automatic-GPU run
(`20261018T121854085520Z`) both selected **Torch Huber** on validation for all three
targets. Test results use the same **123 unseen airfoils / 116,484 operating points**:

| target | original MLP median relative error | CPU Huber median | GPU Huber median | CPU Huber RMSE | CPU Huber R² |
|---|---|---|---|---|---|
| `Cl` | 5.96% | **3.81%** | 3.81% | 0.08688 | 0.9871 |
| `Cd` | 8.11% | **5.57%** | 5.57% | 0.01154 | 0.8766 |
| `Cm` | 12.28% | **9.65%** | 9.65% | 0.01297 | 0.9369 |

Drag median error decreased by **2.54 percentage points**, about 31% relative to the
original MLP. On validation, the controlled GPU MSE/Huber pair gives **6.83% / 5.90%**
median drag error. This supports a loss-specific improvement in that experiment;
the larger baseline-to-Huber change also includes architecture, optimizer, features,
symmetry, and stopping changes. It is **not an isolated geometry-fix gain**.

The GPU runtime reported **Tesla T4**, PyTorch 2.11.0+cu128, CUDA 12.8, and two visible
GPUs; training used one CUDA device, not distributed training. GPU Huber fitted in
184.6 seconds, stopped after **214 epochs**, and restored **epoch 184**. The CPU Huber
run used the same stopping and best-epoch counts. GPU cloud preflight passed **325
tests** (two CPU-only cases skipped); inference is portable NumPy on CPU.

The original MLP had validation drag R² 0.9224 versus test 0.6393 while median error
stayed near 8.1%. Its 66%-thick FX 79-W-660A extrapolation case contributed 80.9% of
test drag squared error. This is why median relative error is the headline, with
absolute error, R², and tail diagnostics alongside it.

**All training runs on Kaggle through the API.** Local checks and saved-model analysis
do not require PyTorch. The completed CPU/GPU runs share seed 42; matching devices is
not a multi-seed study. The test set has been inspected diagnostically, so new folds
or new airfoils remain necessary for stronger confirmation.

### Completed controlled study

Run `20261019T115249707139Z`, from source `96bf0bc`, completed **12/12 fits** on a
Tesla T4 in about **33.4 minutes**, including preflight and evaluation. Cloud
preflight passed **469 tests**, with four skips and one deselected test. All fits
stopped on validation patience, restored their best checkpoints, and recorded their
device, seed, epoch count and hyperparameters. Training remained on Kaggle.

**Three paired seeds (42, 43, 44):** mean test median-relative error plus/minus sample
standard deviation, in percent. These are three-seed descriptive statistics, not
confidence intervals or geometry-fold uncertainty.

| target | Torch MSE | Torch Huber | mean reduction |
|---|---|---|---|
| `Cl` | 4.782 +/- 0.125% | **3.834 +/- 0.017%** | 0.948 pp |
| `Cd` | 6.633 +/- 0.061% | **5.612 +/- 0.065%** | 1.021 pp |
| `Cm` | 11.316 +/- 0.507% | **10.124 +/- 0.451%** | 1.192 pp |

Huber improved each target's median in every paired seed. However, its drag tail
performance is not seed-stable on the **same test rows**:

| Huber seed | test Cd median error | test Cd RMSE | test Cd R² |
|---|---|---|---|
| 42 | 5.574% | 0.01157 | 0.8760 |
| 43 | 5.575% | 0.01553 | 0.7766 |
| 44 | 5.687% | 0.04159 | **-0.6021** |

Saved-weight diagnostics attribute **96.08% of seed 44's drag squared error** to
FX 79-W-660A. This is genuine extrapolation instability, not a change of test-set
variance or evidence that a good pooled median protects every airfoil. No seed
was selected using its test performance.

**Seed-42 sweep, selected only on validation drag median error:**

| Huber delta | weight decay | validation Cd median error |
|---|---|---|
| 0.05 | 1e-5 (reference) | 5.898% |
| 0.02 | 1e-5 | 6.345% |
| 0.10 | 1e-5 | 5.846% |
| **0.05** | **0** | **5.660% (selected)** |
| 0.05 | 1e-4 | 7.762% |

Only the selected sweep alternative was evaluated on test, after the choice was
frozen. It produced median errors Cl **3.915%**, Cd **5.542%**, Cm **9.569%**. Against
the pinned CPU Huber reference, those changes are **+0.102, -0.031, -0.081 pp**,
respectively, so the requested 0.5-pp median guard **passes**.

That pass is limited: selected-model Cd RMSE rises from **0.01154 to 0.01935**, and
R² falls from **0.8766 to 0.6531**. FX 79-W-660A accounts for **81.87%** of its drag
squared error. Post-stall R² improves to 0.7691, while pre-stall R² is only 0.5023
and Re=50,000 R² is 0.0718. The 0.031-pp test-median improvement is smaller than the
observed three-seed variation of the reference. The median-only CI policy does not
guard these tails; it must not be interpreted as proof of overall improvement.

**Torch airfoil-exposure diagnostic:** both arms used 347,877 training rows, the
same 54,917 audit rows, and a common 80,612-row stopping set. The exposed arm replaced
54,974 training rows with other operating points from 114 audit airfoils. Actual
audit rows never entered training, and canonical validation/test rows were unused.

| audit median error | unseen-airfoil arm | exposed-airfoil arm |
|---|---|---|
| `Cl` | 4.277% | 3.238% |
| `Cd` | 5.938% | 4.154% |
| `Cm` | 10.151% | 7.745% |

Drag R² rises from 0.9350 to 0.9568 on identical audit rows/SST. Exposure makes the
audit easier: drag median error improves by **1.784 pp**, about 30%. This supports
geometry-disjoint evaluation for unseen-airfoil claims. It is not a production
gain, a universal leakage-effect estimate, or proof that geometry IDs are memorized.

Artifacts: `results/kaggle/controlled-study-v2/experiments/v2/20261019T115249707139Z/`.
The source archive SHA256 is
`e2512826802de6e7dfbfaa61221cb53a1cb471819225ebe10b01b18e44eb5fc5`.
`study_selection.json` preserves the validation choice, and the seed summaries,
paired deltas, leakage scores and selected-model diagnostics remain separate files.

### Remaining drag error

CPU Huber's post-stall angle proxy (`|Alpha| > 12`) has **Cd R² = 0.7138**. A more
detailed breakdown locates the remaining error:

| absolute Alpha | rows | airfoils | Cd RMSE | Cd R² | share of post-stall squared error |
|---|---|---|---|---|---|
| `(12,16]` | 14,960 | 115 | 0.01537 | 0.7659 | 45.39% |
| `(16,20]` | 5,566 | 103 | 0.02748 | 0.3735 | 53.98% |
| `>20` | 63 | 1 | 0.02806 | -0.7096 | 0.64% |

FX 79-W-660A contributes **68.71%** of the 16-20 degree bin's squared error and E863
another **11.30%**. Negative-alpha rows contribute **60.99%** of total post-stall
squared error. Only one airfoil has observations above 20 degrees; this cannot
establish general fully-separated-flow accuracy.

Using observed signed-lift maxima separately by airfoil/Re/Ncrit/angle sign, 2,076
half-polars have interior peaks, 376 are boundary-censored, and 11 are too sparse.
Among interior curves, post-stall Cd RMSE is **0.01088 within +/-2 degrees of the
peak**, versus **0.02529 farther beyond it**. Farther-post-peak rows account for
42.24% of post-stall squared error; boundary-censored curves account for another
35.89%. The evidence points beyond an onset-only explanation, toward post-peak
and difficult-geometry tails. Observed maxima and angle bins are diagnostics, not
measurements of physical stall or separation.

### Condition-matched reference

| CPU Huber test subset | rows | airfoils | Cd median relative error | Cd mean relative error |
|---|---|---|---|---|
| pooled | 116,484 | 123 | 5.57% | 9.02% |
| `|Alpha| <= 12`, `Ncrit=9`, all Re | 45,555 | 123 | 6.21% | 9.78% |
| same conditions, fixed `Re=1e6` | 9,360 | 123 | **6.44%** | **11.17%** |

The selected no-weight-decay trial scores **6.33% median / 11.51% mean** on that
same fixed-Re subset: the median improves slightly, but the mean worsens.

Re=1e6 was fixed in advance, not selected for the lowest test error. NeuralFoil's
0.37% is a **mean**, not a median, and its data, case split and averaging differ.
Matching these conditions improves reporting but does not make the experiments
directly equivalent. The geometry correction provides validity and interpretability;
the measured MLP gain must not be attributed to that correction alone.

The CPU-reference diagnostics were recomputed with saved NumPy weights, without
fitting or Torch. A local sklearn scaler-version warning was emitted; its pooled
drag metrics agree with the original CPU cloud report within 6e-8. Reproduce into
a new output file:

```bash
python -m src.training.analysis <completed-run-dir> --output <completed-run-dir>/diagnostics.json
```

## The v1 geometry caveat

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

v1 still scored well (test `Cl` R² = 0.977), so its distorted coordinates retained
predictive structure despite not being valid physical geometry. Its geometry-disjoint
test score cannot be explained by memorizing those held-out airfoils. Correcting the
representation establishes physical validity and interpretability, but does not by
itself establish an accuracy gain. **State the representation caveat when comparing
v1 and v2.**

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

CPU-compatible scikit-learn factories remain available: Linear Regression, Polynomial
+ Ridge, Random Forest, Histogram Gradient Boosting, and an MLP. The published v2 queue
is preserved in `DEFAULT_MODEL_CONFIGS`.

The Kaggle notebook now uses `ACCURACY_MODEL_CONFIGS`: the original MLP baseline plus
two otherwise-identical CUDA MLPs with MSE versus Huber loss. Inspired by NeuralFoil:

- SiLU layers and RAdam with small weight decay, without dropout or batch normalization.
- Internal `sin(2 Alpha)` and `sin²(Alpha)` features for separated-flow trends. The
  original baseline still receives exactly its published 113 features.
- Exact reflection symmetry: thickness unchanged, camber and angle negated; `Cl` and
  `Cm` odd, `log(Cd)` even. Symmetry is applied during training and inference, with
  target-scaling offsets handled in fitting-space units.
- Huber delta 0.05 **after** training-only standardization of `Cl`, `log(Cd)`, `Cm`.
  Our configurable target weights `(1, 2, 0.5)` prioritize drag; they are experiment
  choices, not numerical ratios quoted from the paper.
- Up to 300 epochs, learning-rate reduction, and restoration of the best
  **geometry-disjoint validation** checkpoint. Test targets never enter fitting,
  scheduling, stopping, or winner selection.

The baseline actually stopped at **58 epochs** with `tol=1e-4` and no convergence
warning. Raising `max_iter` alone would not change that stopping criterion.

Models train sequentially and checkpoint before the next starts. The notebook selects
drag on validation **median relative error**, and lift/moment on physical-unit RMSE.
All selection criteria are recorded; the generic comparison API retains RMSE defaults.
Saved neural weights and inference use NumPy, so pulling a model does not require
installing PyTorch locally.

Full quadratic expansion is deliberately excluded from the queue: ~113 inputs expand to
thousands of features, tens of gigabytes for one float64 training matrix at 538k rows.

The completed v2 benchmark held hyperparameters fixed. The initial accuracy experiment
changed physics features, architecture, optimizer, and stopping together versus that
baseline; only the MSE/Huber pair isolates a single intervention. The subsequent
three-seed study supports the median-error benefit of Huber, but also exposes drag
tail instability. It does not isolate the geometry correction from the other changes.

Every neural result records `training_metadata` in its trial `run.json`: `device`,
`device_name`, seed, `n_iter`, `max_iter`, `best_epoch`, stop reason, and restored-best
checkpoint status. Run-level `machine_shape` is the requested API shape (`null` means
automatic), not evidence of allocation; `gpu.device_name` records actual hardware.
Run-level `inference_device` is `cpu-numpy`. Selecting a GPU never changes the frozen
split or moves reported inference onto CUDA.

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

Two caveats. The representation was physically invalid, as described above; that alone
does not prove inflated geometry-disjoint scores. And MAE 0.0049
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

Requires the `kaggle` CLI and Kaggle credentials, such as a token file or
`~/.kaggle/kaggle.json`.

```bash
python -m src.utils.kaggle_sync push <owner>/<dataset-slug> --create
```

Use `--create` only the first time; later runs omit it to add a new version. Only the two
tables and the report are uploaded — never the raw archive or model weights.

### 3. Create and run the notebook on Kaggle

The API helper embeds an allowlisted snapshot of the current training source, tests,
requirements, and frozen split in the staged notebook. **No Git commit or GitHub push
is needed to run uncommitted changes.** Data, credentials, local state, and model
weights are excluded. A SHA256 of the exact source archive is recorded in `run.json`,
alongside the base Git revision. Browser-only execution retains the Git clone fallback.

Everything else is CLI-able. Pushing a kernel is what *starts* a run; there is no separate
run verb. Pushing an existing slug creates another version and **does not cancel an
older queued or running version**. Check existing jobs first to avoid duplicate GPU slots:

```bash
python -m src.utils.kaggle_sync kernel <owner>/<kernel-slug> --dataset <owner>/<dataset-slug>
python -m src.utils.kaggle_sync status <owner>/<kernel-slug>
```

This generates `kernel-metadata.json`: **GPU on by default**, internet on for Kaggle-only
dependencies, the tables dataset attached, and a private notebook unless `--public` is
passed. Optional `--machine-shape NvidiaTeslaT4` records a requested T4 shape; older
Kaggle CLI versions can ignore this metadata field. With no shape, Kaggle chooses the
GPU. The notebook checks CUDA and records the actual device separately from the request. It never
silently trains GPU candidates on CPU. `--accelerator cpu` explicitly runs the same
accuracy candidates on Kaggle CPU after cloud-only CPU fitting tests. It can be used
when GPU allocation is queued, but training will be slower. `requirements-kaggle.txt`
is installed on Kaggle, not locally.

Equivalently, in the browser: **Add Input → Datasets** → attach the dataset,
**Settings → Internet: On**, select an available **GPU accelerator**, then
**Save Version → Save & Run All**. Browser execution needs the source branch pushed.

The notebook extracts the snapshot into Kaggle scratch (not `/kaggle/working`, which
becomes the kernel output), restores the frozen split, trains the queue, evaluates the frozen
winners on the test split, and writes metrics, `analysis.json`, and figures.
The report also separates within/outside training descriptor ranges, matched pre-stall
`Ncrit=9` cases by Reynolds number, per-airfoil error distributions, and squared-error
concentration. Descriptor ranges are a diagnostic heuristic, not calibrated uncertainty.

### 4. Pull the results back

```bash
python -m src.utils.kaggle_sync pull <owner>/<kernel-slug>
```

Each pull lands in its own `results/kaggle/<run-id>/` and never overwrites an existing run.

### Controlled study

```bash
python -m src.utils.kaggle_sync kernel <owner>/<study-slug> --dataset <owner>/<dataset-slug> --accelerator gpu --study
```

Use one fresh study slug. The API reuses the tested notebook bootstrap and then hands
off to the cloud-only study runner, without executing the ordinary test-report cells.
The design is declared before fitting:

- Seeds 42, 43, 44 with paired MSE/Huber models and the same frozen geometry split.
  Report every seed and sample mean/standard deviation, never the best test seed.
- Huber deltas 0.02, 0.05, 0.1 at weight decay 1e-5, and weight decays 0, 1e-5,
  1e-4 at delta 0.05. This is a one-factor-at-a-time, seed-42 sweep, not a full grid.
  Reuse the central seed-42 Huber fit; select by validation median drag error.
- A separately labeled Torch leakage diagnostic, constructed only from canonical
  training airfoils. Both arms share fixed audit rows and a common disjoint stopping
  set. The exposed arm replaces training rows with other operating points from audit
  airfoils, preserving row count and excluding the actual audit observations.

Production choices are frozen before canonical test evaluation. The intentionally
exposed arm never influences production selection. A three-seed study measures
initialization variability on this split; it does not replace new geometry folds.

### Testing

```bash
python -m pytest tests/test_kaggle_sync.py tests/test_kaggle_notebook.py tests/test_analysis.py tests/test_torch_mlp.py -m 'not gpu'
python -m pytest tests/test_accuracy_guard.py tests/test_study.py
```

The full suite, including model fitting, runs in the Kaggle notebook preflight. The
local command above checks portable inference and integration without fitting.
GPU fitting and full notebook integration tests are opt-in on Kaggle; portable inference,
configuration, transfer, and numerical checks run without local PyTorch. Covers corrected
normalization (non-negative thickness, edge closure, chord scaling,
invariance to rotation/scale/translation/point order), descriptors against analytic NACA
geometry, the two-table loader, feature assembly and leakage guards, the target-transform
round trip, the new metrics, the frozen-split guards, the Kaggle transfer tool, and an
end-to-end run of the Kaggle notebook against synthetic tables.

### Accuracy regression guard

`configs/accuracy_baseline/` pins the CPU Huber test CSV and matching provenance.
The guard rejects inconsistent or incomplete reports, different data fingerprints,
different geometry splits, missing targets/counts, and unrecorded hardware/stopping.
It fails if **any target degrades by more than 0.5 percentage points** in median
relative error (`0.005` in the stored fractions). Improvements are allowed.

```bash
python -m src.utils.accuracy_guard --candidate-dir <completed-run-dir> --tolerance-pp 0.5
```

The existing automatic-GPU run passes: relative to CPU, median-error changes are
Cl **+0.000605 pp**, Cd **+0.000699 pp**, Cm **+0.001285 pp**.
The completed sweep winner also passes, despite the RMSE regression documented
above. This guard covers typical relative error only; tail checks remain necessary.

`.github/workflows/accuracy.yml` runs fit-free checks on push/PR. Its manual
`workflow_dispatch` accuracy job downloads a **completed** trusted Kaggle comparison
and requires its kernel, exact run ID, expected source-bundle SHA256, and source
commit (defaults to the workflow commit). Configure `KAGGLE_API_TOKEN` as a repository
secret. It retains exact reports and a receipt; downloaded code/weights are not run.
This does **not** train on GitHub or pretend unit tests measure a refactor's new model
accuracy: a completed Kaggle candidate and dispatch are required for that check.
Study test trials also emit standard reports. Use the `selected_trial` recorded in
`study_selection.json` for the sweep candidate, not whichever test trial looks best.

### Progress report

`Minor_Project_Progress_Report.docx` contains the latest results and three embedded
plots in **three total pages**, including the student details and references. The
page count and layout were checked by rendering with LibreOffice 26.8.1; the PDF
proof is in `results/kaggle/controlled-study-v2/report-render/`.

The original report is preserved in
`results/kaggle/controlled-study-v2/report-backup.docx`; the separate `docs/` copy
is unchanged. The one-off generation tools and temporary renderer were removed
after validation; opening the final DOCX requires no additional project dependencies.

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

- Three seeds support Huber's typical-error gain, but its drag RMSE/R² vary severely
  on the same test set. Address out-of-range geometry and low-Re tails before claiming
  a generally reliable surrogate; do not hide them by removing test airfoils.
- `Cdp`, `Cp_ps`/`Cp_ss` (100-point pressure distributions), and the `alpha_min`/`max`/
  `step` fields are present in the archive and unused.
- The leading-edge radius is fitted from the vertical thickness distribution, so for a
  strongly cambered section it runs above the radius measured normal to the camber line.
  It remains a monotonic, physical feature; it is not a certified nose radius.
- More seeds for the sweep winner, geometry-fold evaluation, additional extreme
  geometries, and XFOIL panel-resolution label-noise checks remain future work. A
  longer-patience study and explicitly specified tail-regression thresholds have not
  been run or adopted. The frozen test was
  inspected for diagnosis; new folds or new airfoils are needed for stronger confirmation.

## Citation

Cite the NASA Airfoil Learning Dataset and the original XFOIL-related source material per
the NASA project documentation: <https://github.com/nasa/airfoil-learning>.

## License

Intended to use the MIT License; no license file has been added yet. Dataset licensing and
attribution should be checked independently before redistribution.
