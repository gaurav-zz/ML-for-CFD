# Airfoil ML Surrogate

Machine-learning surrogate modeling for aerodynamic prediction of airfoils.

## Research Question

Can machine-learning models learn the relationship between airfoil geometry, flow
conditions, and aerodynamic coefficients accurately enough to act as a fast surrogate for
conventional aerodynamic analysis — **on airfoils they have never seen**?

That last clause is the generalization claim. Valid geometry and a geometry-disjoint
evaluation protocol are prerequisites; accuracy improvements still require measurement.

## Status

**Latest (verified 29 October 2026):** the selective 1%-RMS floor study completed all
**18 fits**, with **1,205 cloud tests passing**. Pooled training-only out-of-fold R²
is **0.9873 Cl, 0.8983 Cd, and 0.9245 Cm**. The candidate has no member clipping and
passes 16 of 17 relative checks, but the flapped AH 81 airfoil's drag RMSE worsens
55.1%, exceeding the frozen tail limit. **No new family is recommended or promoted.**
Defaults and the pinned CPU reference remain unchanged.

The preceding broad RMS floor reached pooled drag R² 0.8818 and failed four relative
checks. The selective floor improves drag R² and typical errors, but is not a uniform
accuracy improvement or a guarantee for individual airfoils.

The preceding geometry-fold study exposed the near-constant-feature amplification
and saturation failures that motivated this paired experiment. Its original
development decision and all previous artifacts remain unchanged.

The earlier 12-fit Cdp auxiliary study passed its development gate but **failed its
frozen test tail audit**. That historical decision and its limits are unchanged.

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
test drag squared error. Median relative error, absolute error, R², and geometry-tail
diagnostics must therefore be read together; no single aggregate certifies reliability.

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

### Cdp auxiliary study

This completed experiment takes the additional `Cdp` target from
[NASA's training code](https://github.com/nasa/airfoil-learning/blob/f211e62fab6be125657d49c58fdce1bdeb0d12a8/pytorch/train_dnn.py),
not its row-random split or full-data scaling. It compares our unchanged three-output
Huber model against the same model with an internal auxiliary Cdp head.

The audited sidecar covers **all 768,542 existing rows**, with no duplicate keys,
unmatched operating points, missing labels, or imputation. Two Cdp values are
negative (minimum -0.00108), so the auxiliary target uses **identity space**, not
a logarithm. `Cd - Cdp` is positive throughout the audited data. This is a label
consistency check, not a guarantee of XFOIL accuracy on extreme geometries.

```bash
python -m src.data.auxiliary
python -m src.utils.kaggle_sync push <owner>/<new-cdp-dataset> --create --include-auxiliary
python -m src.utils.kaggle_sync kernel <owner>/<cdp-study> --dataset <owner>/<new-cdp-dataset> --accelerator gpu --auxiliary-study
```

Use a **separate dataset**, leaving the original two tables, their hashes, and the
pinned reference untouched. The optional upload adds only `airfoil_auxiliary.csv`
and `auxiliary_report.json`. Dataset and source staging reject nonempty destinations;
personal reports, credentials and raw archives are excluded by explicit allowlists,
not merely by `.gitignore`.

The fixed experiment consists of **12 fits**, with clear phase/seed/arm progress:

- Six development fits: baseline versus auxiliary weight 0.1, seeds 42/43/44.
  Only canonical training geometries are partitioned into fit, stopping, and
  evaluation sets (80/10/10; partition seed 20261020).
- A development-only eligibility decision is saved before production validation
  or test scoring. No best-seed selection is allowed.
- Six confirmation fits: both predeclared arms and all three seeds on the full
  canonical training partition, with the original validation set for stopping.
  Test results audit the frozen decision; they never rewrite it.

Cdp is a **label only** and never joins the 113 input features. The public predictor
still returns Cl, Cd and Cm; auxiliary predictions are separate. The baseline trunk,
primary-head initialization, primary target scalers, and primary loss denominator
are preserved. With H denoting Huber loss in training-standardized target units:

```text
primary loss = (H(Cl) + 2 H(log(Cd)) + 0.5 H(Cm)) / 3.5
auxiliary training loss = primary loss + 0.1 H(Cdp)
```

Only observed auxiliary labels contribute. Scheduling and stopping monitor the
**same primary validation loss** for both arms; improving Cdp alone cannot extend
training or select a checkpoint. Auxiliary validation loss is logged diagnostically,
not used for stopping or eligibility. The 0.1 weight is a declared experiment choice,
not a numerical ratio claimed by NASA.

The new paired development gate supplements the original median-only CI guard:

- Mean median relative error may worsen by at most 0.5 percentage points for each
  primary target; mean drag median error must improve.
- Mean physical Cd RMSE, p95 and worst-airfoil RMSE, and low-Re/thick-airfoil stress
  RMSE may worsen by at most 5%.
- Each paired seed's pooled and worst-airfoil Cd RMSE may worsen by at most 10%.
- Stress boundaries come only from fitting data: the lowest observed Re and the
  90th percentile of unique-airfoil thickness. Missing stress evidence or incomplete
  seeds make a gate inconclusive, never a pass.

These tolerances are **predeclared engineering policies**, not statistical confidence
limits or physical guarantees. A failure retains the baseline as the development
choice; neither a passing gate nor the experiment automatically changes production
defaults. The existing test has already been inspected, so confirmation is not a
new blind benchmark.

**Measured outcome:** run `20261020T104928953059Z` completed all 12 fits on a Tesla
T4 in approximately 26 minutes. Cloud preflight passed **704 tests** (six skipped,
one deselected). The successful notebook source is `07142df`; the earlier precision
assertion and notebook-path failures occurred before study fitting and did not
change the model settings, data, or gate thresholds.

On the 57-airfoil training-only development evaluation, mean median errors were:

| target | baseline | with Cdp |
|---|---|---|
| `Cl` | 8.797% | 7.597% |
| `Cd` | 11.510% | 8.976% |
| `Cm` | 24.265% | 19.971% |

Mean development Cd RMSE decreased from 0.01491 to 0.01091, and all development
checks passed. `auxiliary_selection.json` therefore froze the Cdp arm **before**
canonical validation or test scoring. This decision has not been rewritten.

The full-data confirmation results are the mean and sample standard deviation of
test median relative error across seeds 42/43/44, in percent:

| target | baseline | with Cdp |
|---|---|---|
| `Cl` | 3.834 +/- 0.017% | 3.790 +/- 0.011% |
| `Cd` | 5.612 +/- 0.065% | 5.550 +/- 0.094% |
| `Cm` | 10.124 +/- 0.451% | 10.022 +/- 0.357% |

The three confirmation baselines reproduce the prior GPU Huber run's 45 reported
test scores and stored weights/scalers exactly. Recomputed development and test
gates also match the saved decisions, supporting the intended paired comparison.

Average Cd RMSE decreased from **0.02290 to 0.01407**, helped substantially by
correcting the previous seed-44 failure. The worst of the three seed-level Cd RMSEs
also improved, from 0.04159 to 0.02133. However, paired improvement was not uniform:

| seed | baseline Cd RMSE | Cdp Cd RMSE | baseline Cd R² | Cdp Cd R² |
|---|---|---|---|---|
| 42 | 0.01157 | 0.00908 | 0.8760 | 0.9237 |
| 43 | 0.01553 | 0.02133 | 0.7766 | 0.5784 |
| 44 | 0.04159 | 0.01180 | -0.6021 | 0.8710 |

The paired test-tail audit failed three unchanged limits:

- Mean p95 airfoil Cd RMSE increased from **0.01227 to 0.01313**, about **7.1%**,
  exceeding the 5% allowance.
- Seed 43's pooled Cd RMSE increased about **37.4%**, exceeding the 10% allowance.
- Seed 43's worst-airfoil Cd RMSE increased from **0.14714 to 0.21420**, about
  **45.6%**, exceeding the 10% allowance.

Mean Cl RMSE also increased from 0.10781 to 0.12292, and mean Cm RMSE from 0.01472
to 0.01647. Those regressions must not be hidden behind slightly better medians.
The original median-only guard passes the predeclared seed-42 Cdp candidate against
the pinned CPU reference (Cl -0.0223 pp, Cd +0.0763 pp, Cm +0.0276 pp), but that does
not override the failed paired tail audit.

**Decision:** retain the existing baseline; do not promote the auxiliary model or
select a favorable seed. Cdp supervision is promising on average, but this result
does not meet our predeclared reliability criteria. This was one development
geometry partition with three initialization seeds, not three geometry folds.
The next priority is additional training-only geometry folds and validation of
seed-disagreement/ensemble behavior, not relaxing the limits or tuning on the
known test failures. No further training sweep was automatically launched.

Evidence is preserved under
`results/kaggle/cdp-study-v3/experiments/v2/20261020T104928953059Z/`, including the
immutable selection, all per-seed reports, `summary.csv`, and `tail_audit.json`.
The successful source archive SHA256 is
`4453ecf7c00cef238466d7732b8215990c9902d8a759e89051a35a1ae5999ccf`.

### Geometry-fold and ensemble study

This completed bounded experiment uses **only the original 572 training airfoils**.
Canonical validation and test are checked for identity/coverage but are never
passed to model fitting, stopping, scoring, or selection in this study.

```bash
python -m src.utils.kaggle_sync kernel <owner>/<fold-study> --dataset <owner>/<cdp-dataset> --accelerator gpu --fold-study
```

The existing private Cdp dataset is reused, without uploading the raw archive or
personal reports. The original 113 features, Huber training settings, and auxiliary
weight 0.1 remain fixed; this experiment does not add pressure supervision or change
geometry resolution.

- Three deterministic geometry-ID folds use partition seed 20261026. Evaluation
  groups contain 191, 191, and 190 airfoils and together cover every training ID.
- Within each fold, 38 remaining airfoils form a shared stopping set; the inner fit
  contains 343, 343, or 344 airfoils. Fit/stopping/evaluation are disjoint within a
  fold. All membership and original row positions are saved before fitting.
- **18 fresh fits**: three folds x three seeds (42/43/44) x baseline/Cdp arms.
  Reusing old full-training models here would leak the fold evaluation geometries,
  so those models are not used. Progress logs show the fold and global fit count.
- Each fold's three members form an ensemble without extra fitting: arithmetic
  mean for Cl/Cm, geometric mean for positive Cd. One aggregation rule is declared
  before evaluation, not chosen by trying different rules on the test set.
- The fixed reference is **baseline seed 42**, never the best seed. Candidate
  families are the baseline ensemble and the Cdp ensemble. Eligibility is assessed
  across all folds; eligible families are ranked by equally weighted mean fold Cd
  median error, with baseline-ensemble tie preference and seed-42 fallback.

The numerical tolerances remain 0.5 percentage points for mean primary median
error, 5% for mean Cd tail/stress metrics, and 10% for each fold's pooled/worst-airfoil
Cd RMSE. Before this experiment, **mean Cl and Cm RMSE are additionally constrained
to at most 5% deterioration**, to prevent the negative-transfer pattern seen in the
Cdp study. This is a new fold-study requirement, not a retroactive change to the
previous study's recorded gate. Fold IDs are not misrepresented as random seeds.

Disagreement is the three-member sample standard deviation in Cl/log(Cd)/Cm.
Reports compare its rank correlation with observed log-drag error and show all rows
in fixed low/mid/high spread bins (prediction-only p50/p90 boundaries). This is not
calibrated uncertainty, a confidence interval, an abstention guarantee, or a reason
to drop difficult observations. Folds share training data; their variation is
descriptive and is reported separately from seed variation within each fold.

Outputs include `folds.json`, `fold_rows.csv`, member/ensemble metrics, disagreement
diagnostics, `gates.json`, and an immutable `fold_selection.json`. Any choice is a
development recommendation only: fresh full-training fitting and a locked assessment
would be needed later. Defaults and the pinned reference are not changed, and no
improvement on the old test set is claimed. This is geometry-ID-held-out evaluation,
not a guarantee of withholding every related airfoil family.

**Measured outcome:** run `20261026T182641468688Z`, from source `9cbe549`, completed
18/18 fits and all six ensemble evaluations on a Tesla T4 in about 36.4 minutes,
including setup and evaluation. Cloud preflight passed **907 tests**, with six
skips and one deselected test. All 18 fits restored their best checkpoints.
The original validation and test partitions were not scored.

The following are **unweighted means of the three fold medians**, not pooled
errors or scores on the earlier canonical test set:

| target | fixed baseline seed 42 | baseline ensemble | Cdp ensemble |
|---|---:|---:|---:|
| `Cl` | 4.875% | 4.372% | 4.389% |
| `Cd` | 7.022% | 6.192% | 6.243% |
| `Cm` | 10.925% | 9.942% | 10.046% |

`fold_selection.json` correctly selects **`cdp_ensemble`**: all 17 relative checks
pass. The baseline ensemble has slightly lower median errors, but fails the mean
Cl RMSE gate: **0.18948 to 0.27627 (+45.8%)**, exceeding the 5% allowance. Selection
therefore does not show that Cdp is uniformly more accurate than the other ensemble.
Its drag-median improvement against the fixed reference is **0.778 percentage
points**, not a comparison against the earlier 5.57% canonical-test result.

**A passing relative gate is not an absolute reliability guarantee.** The selected
ensemble's Cd RMSE by fold ID is **0.00984, 0.00908, 0.17813**. Its mean is 0.06568
with sample standard deviation 0.09739; fold 2's Cd R² is **-37.42**. The reference
on that fold is already catastrophically wrong (Cd RMSE **84.37546**), so a large
relative improvement can pass without becoming acceptable in physical units.

The failure is concentrated in **AH 93-W-480B**, all **1,268** of whose rows are in
fold 2's development evaluation, not the canonical holdouts:

- Its thickness is **47.84%**, versus an inner-fit maximum of **30.55%**. Its blunt
  trailing edge and several other descriptors also exceed that fold's fit ranges.
- The trailing-edge feature `camber_049` is only **-0.000688 chord**, but its
  training scale is **5.696e-7**, producing a standardized magnitude of **1,208**.
  Several aft-thickness features are also about 43-48 training standard deviations
  away. Near-constant scaling magnifies this extrapolation problem.
- All three baseline members and two Cdp members reach the numerical Cd upper
  limit of **1,000** on every row of that airfoil. The third Cdp member predicts
  **1e-6 to 3.55e-5**, including 35 lower-limit clips. These are saturation failures,
  not physically credible coefficient estimates.
- Their Cdp geometric ensemble predicts **1.00-3.29**, against observed Cd
  **0.04547-0.26742**. That airfoil still contributes **99.83%** of the ensemble's
  fold-2 drag squared error. Averaging clipped extremes does not establish a
  learned, accurate prediction.
- The baseline ensemble reports **zero log-Cd disagreement** on all 1,268 rows
  because all members were clipped to the same value. Before clipping, their
  log-Cd sample spread is **9.48-11.05**. Low reported spread is therefore not
  evidence of confidence, particularly when numerical limits have been activated.

A diagnostic-only sensitivity replay replaced `camber_049` with each model's
inner-fit mean on this one training-fold airfoil. It removed the inverse-limit
activations, confirming strong sensitivity to that feature, but residual errors
still varied substantially. This was **not** a new trained candidate, a selected
seed, a change to the dataset, or evidence of a general accuracy fix.

**Decision:** preserve the frozen development selection and retain production
defaults. Before another training study, investigate physically appropriate scaling
of near-constant geometry features and saturation-aware prediction/disagreement
reporting, with explicitly declared absolute reliability criteria. Do not alter
the completed study's limits, remove its difficult airfoil, or tune on the canonical
test set. No additional training or confirmation run was launched during this audit.

Artifacts are under
`results/kaggle/fold-study-v1/experiments/v2/20261026T182641468688Z/`. The source
archive SHA256 is
`6d86788aec1e3309a2be893eb57f5c201d158d39bb642c27bfc3cc98dfa5fc7b`.
Fit-free verification reproduced the source snapshot, table hashes, every fold's
row identities and training-derived stress thresholds, all 18 member and six
ensemble score reports, both gate decisions, and the summary statistics. Saved
NumPy inference required no local PyTorch or fitting. The local scaler-version
warning was retained; score replay used `rtol=5e-5, atol=1e-6`. One disagreement
quantile boundary differed by one row across numerical runtimes, recorded rather
than claimed bit-identical. Verification and targeted diagnosis are preserved in
`results/kaggle/fold-study-v1/{verification,saturation_audit,endpoint_sensitivity_audit}.json`.

### Literature-guided input-scaling study

The 28 October comparison uses primary sources, not an inferred R² leaderboard:

| source | verified result/protocol | implication for this project |
|---|---|---|
| [NeuralFoil, Sharpe and Hansman, 2025](https://arxiv.org/html/2503.16323v1#S4.SS3) | Table 4 reports MAE, not coefficient R²; its largest model has Cl MAE 0.012, ln(Cd) MAE 0.020, and Cm MAE 0.002 | These cannot be converted into our geometry-held-out R² or median relative error |
| [NASA/TM-20220006290, Juangphanich et al., 2023](https://ntrs.nasa.gov/citations/20220006290) | DNN/SplineConv, coefficient and pressure outputs, standard/min-max normalization; no comparable aggregate geometry-held-out R² benchmark | Normalization and physical prediction checks matter; adding a GNN is not an established remedy |
| [AirfRANS, corrected paper](https://arxiv.org/html/2212.07564v3) | RANS field prediction, 5-20% thick NACA airfoils, Re 2-6 million; coefficient relative errors and rank correlations, not the same task or labels | Its benchmark does not establish accuracy on our 47.8%-thick extrapolation case |
| [Cao et al., 2025](https://arxiv.org/abs/2501.01165) | The abstract reports 1.1% average surrogate drag error for a laminar parametric solver at Re 100-5,000 | This is a different physical regime and metric, not evidence that a PINN will improve our R² |
| [Ovadia et al., 2019](https://arxiv.org/html/1906.02530v2) | Classification uncertainty deteriorates under dataset shift even when ensembles help | A useful evaluation principle, not an aerodynamic error guarantee |

NeuralFoil's separate 0.37% drag-error benchmark uses 189 easier NACA cases at
alpha=5 degrees, thickness 8-16%, and three Reynolds numbers. Its broad training
data has 7,913,292 generated XFoil attempts, around 56% converged. The checked
[generator/split implementation](https://github.com/peterdsharpe/NeuralFoil/blob/987794104fd07fc279971451ce642e9b92b93b08/training/training_data/load_data.py)
globally shuffles cases rather than withholding whole generated geometries. Neither
benchmark is interchangeable with our three geometry folds, and that mismatch does
not excuse our negative Cd R².

The closest actionable difference is conditioning. NeuralFoil uses 18 CST geometry
parameters and [fixed engineered input scales](https://github.com/peterdsharpe/NeuralFoil/blob/987794104fd07fc279971451ce642e9b92b93b08/neuralfoil/main.py#L163-L177),
not a separate empirical standard-deviation denominator for every sampled endpoint.
Those constants were informed by its training distribution; they are not purely
physics-derived constants. We already use several of its other ideas: SiLU, RAdam,
Huber loss, and reflection symmetry. Its boundary-layer supervision is not our Cdp
auxiliary head, and its convergence confidence is not a calibrated coefficient-error
interval. A later CST experiment should check reconstruction error before changing
our representation; it is not part of the scaling-only experiment below.

**Predeclared experiment:** compare two Cdp models that differ only in profile input
scaling, using the same three training-only folds and seeds 42/43/44: **18 fresh fits**.
Both arms retain auxiliary weight 0.1, architecture, loss, target weights/transforms,
optimizer, feature columns, reflection, and primary-validation stopping. The reference
is `cdp_column_scaling_ensemble`, not a single seed; the candidate is
`cdp_profile_scaling_ensemble`. This avoids confounding scaling with ensembling.

The opt-in model setting is `input_scaling="profile_rms_floor"`; the default remains
`"standard"`. After the existing inner-fit and reflection statistics are computed,
only the 100 thickness/camber station denominators receive this floor:

```text
profile_floor = sqrt(mean(inner_fit_variance_of_100_profile_coordinates))
effective_scale[j] = max(existing_standard_scale[j], profile_floor)
```

These coordinates share chord units. Means, empirical variances, named scalar
descriptors, operating-condition scales, targets, and dataset bytes stay unchanged.
Exactly constant coordinates retain StandardScaler's scale of one. No floor value
is fitted to the evaluation airfoil or tuned in a sweep. The formula is our bounded
engineering experiment motivated by the observed failure, not a method or accuracy
gain claimed by NeuralFoil. Existing saved-model inference retains its stored scales.

The new study preserves the previous numerical relative-gate limits but compares
the two correctly labeled ensembles. Additional, separately recorded minimum sanity
checks require **no inverse-log clipping in any candidate member** and **positive
ensemble R² for every primary target in every fold**. These checks are necessary,
not sufficient, for reliability; they are not paper-derived engineering tolerances.
An unsafe reference is not recommended merely because the candidate fails. No
decision automatically changes production defaults or the pinned reference.

Report all-case per-fold and directly computed pooled out-of-fold R², MAE/RMSE,
relative errors, and per-geometry tails. Pooled R² is not an average of fold R².
Retain row-aligned fitting-space predictions before clipping, clipping masks, and
raw versus post-clipping log-Cd spread so saturation cannot masquerade as agreement.
No rows are dropped, and canonical validation/test remain unscored. Training and
fitting tests run only on Kaggle; local verification remains fit-free.

**Execution:** private kernel
[`purplray/airfoil-v2-input-scaling-study`](https://www.kaggle.com/code/purplray/airfoil-v2-input-scaling-study),
version **1**, was submitted on 28 October with `--input-scaling-study` and
`--machine-shape NvidiaTeslaT4` and has **completed**. It attaches the
existing `purplray/airfoil-ml-surrogate-v2-cdp` dataset. Both primary table hashes
match the preceding fold study. The bundled source SHA256 is
`c6a02c9d127aeebbbf47701b0abeb8300feb470ab0ced1c1e392d8895b757bbf`;
it identifies the uncommitted source snapshot at submission, not just Git HEAD `9cbe549`.

The pre-submit local selection passed **1,059 fit-free tests**. A legacy test-policy gap
was corrected: all **45** fitting cases are now excluded by the local marker
selection and also skipped outside Kaggle even if the opt-in flags are set.
Cloud preflight passed **1,093 tests**, with ten skips and one deselected test.
All 18 experiment fits ran on a Tesla T4 and restored their best checkpoints.
Total cloud execution through the final decision took about **47.3 minutes**,
including the 9.5-minute preflight and evaluation. No experimental model training
ran locally, and no production defaults or personal-report files were changed.

**Measured outcome:** run `20261028T124731457567Z`, independently verified on
29 October. The following pooled R² values are computed directly from all **538,380
canonical-training OOF rows**, each evaluated once per ensemble. They are neither
mean fold R² nor scores on the original canonical validation/test partitions.

| target | column-scaling Cdp ensemble | profile-floor Cdp ensemble |
|---|---:|---:|
| `Cl` | 0.942780 | **0.987426** |
| `Cd` | -10.862452 | **0.881793** |
| `Cm` | 0.703586 | **0.925699** |

Per-fold R² makes the tradeoff visible. Each cell is **reference -> candidate**:

| fold ID | `Cl` | `Cd` | `Cm` |
|---|---|---|---|
| 0 | 0.987929 -> 0.987086 | 0.893289 -> 0.840996 | 0.943361 -> 0.938947 |
| 1 | 0.987344 -> 0.987363 | 0.911707 -> 0.902591 | 0.935499 -> 0.938804 |
| 2 | 0.852872 -> 0.987788 | -37.424160 -> 0.902872 | 0.228881 -> 0.897930 |

The candidate passes all **12 minimum numerical-safety checks**: every fold/target
ensemble R² is positive and none of its **1,615,140 member-row predictions** crosses
the inverse-log limits. The reference has **2,571 clipped member values** on the
1,268 AH 93-W-480B rows in fold 2, and fails the positive drag-R² check there.

Nevertheless, the candidate fails **four of 17 frozen relative checks**:

- Mean-fold median drag error increases from **6.24315% to 6.28591%**. This is only
  +0.04276 percentage points, but the separately declared strict-improvement rule
  still fails; passing the 0.5-percentage-point nondegradation allowance is not enough.
- Fold 0 Cd pooled RMSE increases **22.07%**, exceeding the 10% per-fold allowance.
- Fold 0 maximum-geometry Cd RMSE increases **54.23%**, also exceeding 10%.
- Fold 1 maximum-geometry Cd RMSE increases **38.37%**, also exceeding 10%.

The pooled gains are therefore real within this development experiment, but do not
show uniform improvement across geometries or folds. The paired reference reproduces
the preceding study's nine Cdp members' scores; the data and split were not changed
to obtain the improvement.

**Remaining failures:** AH 93-W-480B's drag RMSE falls from **2.10926 to 0.05571**,
but its own geometry-level drag R² is still **-2.33** and median drag relative error
is **58.46%**. Predicted Cd is **0.01843-0.05549**, against observed
**0.04547-0.26742**. Removing saturation has not made this extreme airfoil accurate.
Its contribution to fold-2 drag squared error falls from 99.83% to 27.55%.

The largest new fold-0 drag squared-error regressions are LIEBECK L1003 (Cd RMSE
**0.05031 -> 0.08556**) and AH 81-K-144 W-F KLAPPE (**0.05548 -> 0.07989**).
Fold 1's leading regression is the 18%-camber, 5%-thickness plate
`cp-180-050-gn` (**0.03142 -> 0.04348**). These are diagnostics, not exclusions.

**Decision:** preserve `selected_family=null`, no fallback, and
`production_promotion=false`. The reference is not assumed safe simply because
the candidate was rejected. The next priority is understanding these residual
geometry-specific errors and coverage limitations, not relaxing the completed
study's limits or adding more epochs without evidence. Any later conditioning or
CST experiment needs a separately declared design; no additional study was launched.

**Verification:** all 18 saved models were replayed with NumPy, and the input and
target scaler moments were independently recomputed from inner-fit data. The effective
profile floors, six saved prediction bundles, row coverage, physical scores,
raw/clipped spread, numerical-safety decisions, 17 relative checks, and 294-row
summary agree with the cloud artifacts. NPZ score comparisons use
`rtol=1e-10, atol=1e-12`; weight replay uses `rtol=5e-5, atol=2e-5`. The sklearn
1.6.1-to-1.9.0 scaler-loading warning is recorded in the receipt rather than hidden.
No local PyTorch import or model fitting was used for this verification.

Cloud outputs are under
`results/kaggle/input-scaling-v1/experiments/v2/20261028T124731457567Z/`.
The one-off audit script and independent receipt are
`results/kaggle/input-scaling-v1/verify_results.py` and
`results/kaggle/input-scaling-v1/verification.json`; the receipt includes artifact hashes
and the geometry-specific diagnostics. The original validation/test sets remain
unscored by this study, and none of these R² values establishes a cross-paper ranking.

### Residual audit and selective-floor plan

The 29 October follow-up audit uses the existing prediction bundles, not new
trained models. It covers all **572** development geometries and retains every row.
The broad floor improves drag RMSE on 210 geometries but worsens it on **362**;
median geometry drag RMSE rises from **0.00628 to 0.00657**, despite the reduction
in mean error caused by fixing the largest failure. Geometry-level negative drag
R² counts rise from two to four. The tradeoff is therefore broader than one bad seed.

There are at least two distinct issues:

- L1003 and AH 81-K-144 W-F KLAPPE do not exceed the individual, reflected inner-fit
  geometry feature ranges. That does not prove joint in-distribution coverage.
  Their candidate drag squared errors are concentrated at Re <= 100,000
  (**90.66%** and **96.45%**, respectively). Angle bands are descriptive, not verified
  stall labels. L1003's drag error worsens for all three candidate seeds.
- Genuine geometry extrapolation remains. The cambered plate's normalized maximum
  camber is **16.36%**, versus its fold's fit maximum **14.70%**. AH 93-W-480B is
  **47.84%** thick versus **30.55%** in its fit partition, with a much larger trailing
  edge and leading-edge radius. Rescaling cannot create missing geometry coverage.

The broad RMS floor changes **57, 57, and 56** sampled coordinates in the three
folds, including many otherwise well-resolved stations. Training-only scale spectra
support a narrower experiment: a fixed **1% of profile RMS** floor changes only
**4, 3, and 4** coordinates. These are `camber_000`, `camber_048`, `camber_049`,
and, in folds 0 and 2, `thickness_000`. No interior coordinate is changed in these
specific fixed fit partitions. The rule will still be applied by variance, not by
hardcoding a list of station indices.

**Frozen hypothesis and design:** `input_scaling="profile_rms_floor_1pct"` applies

```text
R = sqrt(mean(inner_fit_variance_of_100_profile_coordinates))
F = 0.01 * R
effective_scale[j] = max(existing_standard_scale[j], F)
```

Only varying profile coordinates are eligible; exactly constant coordinates keep
their denominator of one. Preserve means, variances, named descriptors, flow scales,
targets, row weighting, reflection, architecture, optimizer, loss, and stopping.
The actual floor values from the existing fit statistics are approximately
**0.00029539, 0.00028209, and 0.00027359 chord**. The 1% ratio is a single engineering
choice fixed before new fitting, not a searched parameter or a paper-derived optimum.
It bounds RMS-relative preprocessing gain by 100 on varying coordinates, **not** the
condition number of the covariance matrix or neural network.

Processed-table to float32 rounding was checked on fit geometries only: maximum
observed coordinate rounding is about **1.47e-8 chord**, and maximum standardized
rounding under the proposed scales is below **6.4e-7**. This measures floating-point
rounding only. No independent physical source/interpolation-error bound has been
established, so the floor is not represented as a calibrated geometric-noise model.
Small endpoint variations may be genuine; no feature is zeroed or discarded.

The separately identified paired experiment used **nine fresh standard-scaling Cdp
fits and nine selective-floor Cdp fits**, with the same three geometry folds, seeds
42/43/44, auxiliary weight 0.1 and fixed fitting settings. The completed broad-floor
model is historical context, not a third selectable arm. Keep the original 17
relative checks and 12 minimum safety checks unchanged. Continue to report every
fold, pooled OOF R², geometry tails, and pre-clipping numerical evidence; no favorable
seed, difficult-row exclusion, unsafe fallback, or automatic promotion is permitted.
Repeated folds are adaptive development evidence, not a fresh blind benchmark.

This experiment tests whether avoiding broad reweighting retains typical/tail
accuracy while containing the endpoint amplification. It does not assume success
on the genuinely extrapolating geometries. CST remains a separate later option
requiring reconstruction checks for blunt, flapped, and extreme geometries first.
No canonical validation/test scoring or production default change is part of this
plan. All new model fitting and fitting tests remain on Kaggle.

The audit tables, including per-member geometry errors, flow-condition slices,
geometry-envelope diagnostics, and scale changes, are saved under
`results/kaggle/input-scaling-v1/residual_audit/`. The fixed proposal's selectivity
and rounding checks are in `selective_floor_assessment.json` in that directory.

**Submission:** private kernel
[`purplray/airfoil-v2-selective-scaling-study`](https://www.kaggle.com/code/purplray/airfoil-v2-selective-scaling-study),
version **1**, was submitted on 29 October with `--selective-scaling-study` and
`--machine-shape NvidiaTeslaT4` and is now **complete and verified**. The mode is
`selective_scaling`; its frozen
variant is `selective_profile_floor_1pct`. The source bundle SHA256 is
`db35561467135a90639bc70cd9b2cd9188b2995df7e66441245f16ec7a54c0b7`, based on Git
`9cbe549` plus the uncommitted implementation. Both primary dataset hashes remain
unchanged. Existing full-RMS and historical study wrappers retain their behavior.

Before submission, **1,167 fit-free tests passed**, and all **53 fitting tests** were
confirmed skipped outside Kaggle even with both opt-in flags enabled. The selective
implementation also reproduced the predeclared 4/3/4 masks from saved inner-fit
moments without fitting or making new candidate predictions. Cloud preflight passed
**1,205 tests**, with 14 skips and one deselected test. All 18 CUDA fits completed and
restored their best checkpoints. Cloud execution took approximately **41.4 minutes**,
including the 10.1-minute preflight and evaluation. The logs confirm two T4 devices
were available; they do not contain a GPU-utilization time series.

**Verified results:** run `20261029T134426839881Z`. The table below uses directly
pooled R² on the same **538,380 training-only OOF rows**, not mean fold R² or the
original canonical test set. The new standard-scaling control reproduces all nine
previous control members' scores.

| target | standard Cdp control | prior broad floor | selective 1% floor |
|---|---:|---:|---:|
| `Cl` | 0.942780 | 0.987426 | 0.987309 |
| `Cd` | -10.862452 | 0.881793 | **0.898320** |
| `Cm` | 0.703586 | 0.925699 | 0.924527 |

Selective candidate R² by fold:

| fold ID | `Cl` | `Cd` | `Cm` |
|---|---:|---:|---:|
| 0 | 0.987743 | 0.871060 | 0.940355 |
| 1 | 0.987521 | 0.918873 | 0.938365 |
| 2 | 0.986631 | 0.904711 | 0.893623 |

Mean-fold median drag error improves from **6.24315% to 6.11128%**; the separate
pooled median is **6.10968%**. The selective model improves geometry-level drag RMSE
on **305 of 572 geometries** and worsens it on 267, compared with 210 improvements
for the broad floor. Its geometry-level negative drag R² count is three, versus
two for the control and four for the broad floor. These are descriptive development
results, not statistical significance or a cross-paper ranking.

**Frozen decision:** all 12 minimum numerical-safety checks pass, including zero
inverse-limit crossings in the candidate's 1,615,140 member-row predictions. Sixteen
relative checks pass. The sole failure is fold 0 maximum-geometry drag RMSE:
**0.055476 -> 0.086058 (+55.13%)**, versus the 10% allowance. The responsible airfoil
is **AH 81-K-144 W-F KLAPPE**, with geometry-level Cd R² **-2.889**. Fold 0 pooled
drag RMSE is also worse (+9.92%), although it narrowly passes its 10% guard.
Consequently `selected_family=null`, no fallback, and `production_promotion=false`
remain unchanged. A nearly passing gate is not a reason to relax its limits.

The focused geometry comparisons show why pooled recovery is insufficient:

| airfoil | control Cd RMSE | broad-floor Cd RMSE | selective Cd RMSE | selective geometry Cd R² |
|---|---:|---:|---:|---:|
| AH 81-K-144 W-F KLAPPE | 0.05548 | 0.07989 | 0.08606 | -2.889 |
| LIEBECK L1003 | 0.05031 | 0.08556 | 0.05417 | -0.141 |
| 18%-camber, 5%-thickness plate | 0.03142 | 0.04348 | 0.02094 | 0.708 |
| AH 93-W-480B | 2.10926 | 0.05571 | 0.05852 | -2.678 |

The next priority is examining the flapped-airfoil representation, low-Re source
polars, and member-level variation before declaring another intervention. No seeds
or rows were removed, no gates were changed, and no further training run was started.

**Audit evidence:** all 18 saved models were replayed with NumPy. Input/target
scaler moments were recomputed from the inner-fit rows; the effective selective
floors and changed-coordinate masks match the predeclared assessment. The six NPZ
bundles, their mode/variant identities, pooled and per-fold scores, clipping/spread,
17 relative checks, 12 safety checks, and 294-row summary reproduce the saved output.
Score tolerance is `rtol=1e-10, atol=1e-12`; weight-replay tolerance is
`rtol=5e-5, atol=2e-5`. The sklearn scaler-version warning is retained in the audit
receipt. No model fitting or PyTorch import was used for this verification.

Artifacts are under
`results/kaggle/selective-scaling-v1/experiments/v2/20261029T134426839881Z/`.
The independent script, receipt, and complete geometry metrics are
`results/kaggle/selective-scaling-v1/{verify_results.py,verification.json,verified_geometry_metrics.csv}`.
The scheduled monitor was found inactive at the later manual status check; these
results were retrieved and verified afterward, not continuously monitored. No
canonical validation/test scoring, default/pin change, or personal-report edit occurred.

### Unused-data audit

A non-fitting audit on **training geometries only**, performed on 26 October 2026,
found all **538,380 training operating points** have finite numeric `Cp_ss` and
`Cp_ps` arrays, each with 100 entries. They match the stored surface-array lengths.
All 1,144 training surface coordinate arrays use a common **uniform 100-point x
grid**, not the cosine grid used by the current model representation. Equal array
length is not, by itself, evidence of physical consistency with the scalar labels.

The pressure values require further scrutiny before supervision:

| diagnostic flag | training rows | training airfoils |
|---|---:|---:|
| any Cp > 2 | 2,408 | 7 |
| any abs(Cp) > 10 | 12,522 | 324 |
| any abs(Cp) > 100 | 842 | 1 |

Global training Cp extrema are **-1585.04 and +1451.85**. Values with magnitude above
10 occur only at stored indices 0 or 1, concentrated at the leading endpoint.
The most extreme examples belong to training airfoil S9037. These are screening
flags, not automatic exclusions: a source/interpolation problem is a hypothesis
to verify, and large suction alone is not proof of an invalid label. No values were
clipped, no rows removed, and Cp was not introduced into training. These unused
pressure values cannot be the direct cause of the existing coefficient-model errors.

The same audit reconstructed geometry at 1,001 cosine-spaced chord locations from
50 versus 100 sampled stations. The reference was the normalized source surfaces,
linearly interpolated with the same nonnegative-thickness rule. Errors are **chord
fractions**, not coefficient errors:

| geometric statistic across training airfoils | 50 stations | 100 stations |
|---|---:|---:|
| median thickness RMSE | 1.559e-4 | 7.321e-5 |
| p95 of maximum thickness deviation | 2.715e-3 | 2.211e-3 |
| median camber RMSE | 2.781e-5 | 9.176e-6 |

The stored 50-station features reproduce the conversion to within 1e-16. Increasing
the station count reduces interpolation error, but the largest deviations remain
near the nose and extra stations cannot restore detail already absent from the raw
uniform grid. This audit does not establish an aerodynamic accuracy gain from higher
resolution. The temporary audit script was removed; no model fitting or local Torch
installation was performed.

### Testing

```bash
python -m pytest tests/test_kaggle_sync.py tests/test_kaggle_notebook.py tests/test_analysis.py tests/test_torch_mlp.py -m 'not gpu'
python -m pytest tests/test_accuracy_guard.py tests/test_study.py
python -m pytest tests/test_auxiliary.py tests/test_tail_risk.py tests/test_auxiliary_study.py
python -m pytest tests/test_ensemble.py tests/test_fold_gate.py tests/test_fold_study.py
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
The personal report is kept locally and ignored by Git to prevent future commits.

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
- `Cdp` has been evaluated as auxiliary supervision but is not enabled by default.
  `Cp_ps`/`Cp_ss` (100-point pressure distributions) and the `alpha_min`/`max`/`step`
  fields remain unused by the coefficient model.
- The leading-edge radius is fitted from the vertical thickness distribution, so for a
  strongly cambered section it runs above the radius measured normal to the camber line.
  It remains a monotonic, physical feature; it is not a certified nose radius.
- The broad and selective scaling studies remove observed clipping and improve
  pooled R², but fail four and one relative checks respectively and leave some
  individual airfoils inaccurate.
  Understanding the remaining conditioning and geometry-coverage errors takes
  priority over another sweep. Additional geometry coverage, XFOIL panel-resolution label
  checks, and later locked confirmation remain future work; the old test was already
  inspected and is not a fresh blind benchmark.

## Citation

Cite the NASA Airfoil Learning Dataset and the original XFOIL-related source material per
the NASA project documentation: <https://github.com/nasa/airfoil-learning>.

## License

Intended to use the MIT License; no license file has been added yet. Dataset licensing and
attribution should be checked independently before redistribution.
