# Benchmarking the v2 validation results

Written 2026-10-18. Scope: compare the v2 validation numbers against published
airfoil-surrogate work, audit whether the comparison is even valid, and list the practices
worth adopting. Four measurements in this document were run for it; everything else is
cited.

---

## 0. Two caveats before any number is quoted

**These are validation numbers.** The queue selects each target's winner on validation RMSE,
so these scores are the ones selection optimised. Validation R² is a selection statistic, not
a result. The frozen test partition (123 airfoils the queue never saw) was scored separately
in §1b, and it changes the headline substantially.

**Single seed.** Every model was fitted once. The MLP's v1→v2 change is +0.0011 on `Cl` and
+0.0010 on `Cd`; HistGradientBoosting's `Cl` change is +0.0008. Those are inside seed noise
and must not be described as improvements. Only changes above roughly 0.01 are worth a claim.

---

## 1. What the numbers actually show

| model | Cl R² | Cd R² | Cm R² | Cd median rel. err | inference (113,678 rows) |
|---|---|---|---|---|---|
| linear | 0.9496 | 0.4918 | 0.7625 | 24.3% | 0.135 s |
| gradient_boosting | 0.9804 | 0.8639 | 0.9133 | 12.2% | 3.86 s |
| random_forest | 0.9702 | 0.8160 | 0.8935 | 14.4% | 0.548 s |
| **mlp** | **0.9880** | **0.9224** | **0.9593** | **8.1%** | **0.224 s** |

The MLP wins all three targets and is second-fastest at inference (17× faster than the
boosting model). It is Pareto-optimal on accuracy and cost, so it remains the right choice —
with the standing caveat that its `max_iter=100` still truncates training.

### v1 → v2, same frozen geometries

| model | Cd R² v1 | Cd R² v2 | Δ |
|---|---|---|---|
| linear | 0.1192 | 0.4918 | **+0.3725** |
| gradient_boosting | 0.8782 | 0.8639 | −0.0143 |
| random_forest | 0.8752 | 0.8160 | **−0.0592** |
| mlp | 0.9214 | 0.9224 | +0.0010 |

This is the most informative table in the run, and it does not say what the README predicts.

The README's thesis is that v2's corrected representation should matter *most* where the model
is flexible, because v1's 100 broken numbers forced flexible models to work from a distorted
coordinate fingerprint. The measurement is the opposite: the fix is worth **+0.37 R² to the
linear model and +0.001 to the MLP**, while random forest gets *worse* by 0.059.

The consistent reading: the geometry fix corrected **conditioning**, not **information**. The
linear model could finally express `Cl ≈ 2π sin α`-like structure from a well-conditioned
physical basis; the tree ensembles never needed the basis because 100 distorted numbers still
carried the geometry. That also explains why `Cd` improved for linear and not for the others.

**Consequence:** the case for v2's representation rests on interpretability, physical
validity, and transfer to new geometry — *not* on a validated accuracy gain. The README's
claim that v1's score was "optimistic for the wrong reason" now needs the measurement in §3
before it can be repeated.

### R² is hiding the real change

`log(Cd)` was introduced to fix a 223× dynamic range. It does not show up in R² at all
(+0.001), because R² is computed in physical units and is therefore still dominated by
high-drag rows. It shows up in relative error: **8.1% median** for the MLP, against a v1
figure that was never recorded, so no v1/v2 comparison is possible.

Mean relative error is unusable and should be dropped from the headline table:

| target | mean rel. err | median rel. err | ratio |
|---|---|---|---|
| Cl (mlp) | 0.503 | 0.0608 | 8.3× |
| Cd (mlp) | 0.117 | 0.0815 | 1.4× |
| Cm (mlp) | 0.651 | 0.1120 | 5.8× |

The 8.3× gap on `Cl` is the near-zero-lift rows the metrics module already warns about.

---

## 1b. The frozen test split, measured

The MLP was refitted locally with the published configuration
(`hidden=(64,32)`, `max_iter=100`, `batch=1024`, `scale_targets=True`, `log(Cd)`) on the frozen
train partition. It reproduces the reported validation scores **exactly** — `Cl` 0.9880,
`Cd` 0.9224, `Cm` 0.9593 to four decimals — so the harness is faithful and the test numbers
below are the same model.

| split | `Cl` R² | `Cd` R² | `Cm` R² | `Cd` MAE | `Cd` RMSE | `Cd` median rel. |
|---|---|---|---|---|---|---|
| validation (selection) | 0.9880 | 0.9224 | 0.9593 | 0.00446 | 0.00835 | 8.15% |
| **test (headline)** | **0.9709** | **0.6393** | **0.8213** | 0.00539 | 0.01973 | **8.11%** |

**`Cd` R² collapses from 0.9224 to 0.6393 while the median relative error does not move at all**
(8.15% → 8.11%). The typical prediction is exactly as good on unseen airfoils; the *tail* is
much worse, and R² is a square-error statistic dominated by the tail and by the validation
set's own variance. RMSE more than doubles (0.0084 → 0.0197) with essentially unchanged MAE.

This settles the reporting question with evidence: **R² is not portable across splits and must
not be the headline.** "R² = 0.92" was a selection statistic. The honest claim is
*median `Cd` error 8.1% on 123 unseen airfoils.*

### Where the error actually lives

Test split, pre-stall (`|α| ≤ 12`), `Ncrit = 9`, by Reynolds number:

| Re | n | `Cd` median rel. | `Cd` MAE | `Cd` R² | `Cl` median rel. |
|---|---|---|---|---|---|
| 5e4 | 8,851 | 9.2% | 0.00854 | 0.758 | 14.4% |
| 1e5 | 9,035 | 8.8% | 0.00621 | 0.734 | 8.9% |
| 2e5 | 9,115 | 8.4% | 0.00413 | 0.697 | 6.5% |
| 5e5 | 9,194 | 8.5% | 0.00278 | 0.715 | 6.0% |
| 1e6 | 9,360 | 9.0% | 0.00255 | 0.773 | 5.9% |

Two distinct behaviours:

- **`Cd` error is flat across Re** (8.4–9.2%), while absolute MAE falls 3.3× as Re rises. In
  relative terms the drag prediction does not care about Reynolds number at all — it is
  limited by **unseen geometry**, not by flow condition. That is the substance of the 8.1%,
  and it is why restricting to "easy" flow conditions does not improve it.
- **`Cl` error is Re-driven**: 14.4% at Re = 5e4 collapsing to 5.9% at Re = 1e6. Low-Reynolds
  laminar separation and transition sensitivity are the failure mode, exactly where XFOIL's
  e^N model is most delicate.

### Post-stall is not working

| subset | n | `Cl` | `Cd` | `Cm` | `Cd` R² |
|---|---|---|---|---|---|
| `|α| ≤ 12`, all Re/Ncrit | 95,895 | 6.8% | 7.8% | 11.8% | 0.821 |
| `|α| ≤ 12`, `Ncrit = 9` | 45,555 | 7.7% | 8.8% | 14.1% | 0.815 |
| `|α| > 12` (post-stall) | 20,589 | 3.6% | 10.0% | 14.5% | **−0.173** |

**Post-stall `Cd` R² is −0.173 — worse than predicting the mean.** The pooled 0.639 is held up
entirely by the pre-stall rows. On unseen airfoils in separated flow the surrogate has no
usable drag capability, and a single pooled number concealed that.

Note the `Cl` figure *improves* post-stall (3.6%). That is a relative-error artefact: stalled
`Cl` is large, so a similar absolute error divides by a bigger denominator. Do not read it as
better physics.

**Claimable today:** median 8% drag error, pre-stall, on 123 unseen airfoils; `Cl` good to ~6%
above Re = 2e5. **Not claimable:** anything post-stall.

---

## 2. Against published work

| study | predicts | data volume | **split protocol** | reported error |
|---|---|---|---|---|
| **NeuralFoil** — Sharpe & Hansman, arXiv:2503.16323 (2025) | XFOIL `Cd`, `Cl`, `Cm`, boundary layer | 7,913,292 generated cases, 56% converged ≈ **4.4 M** solutions | **random 95/5 case split** (395,665 test cases); one separately-designed airfoil checked as unseen | **`Cd` mean rel. 0.37%** on easy cases; **2.0%** on a post-stall/transitional test set |
| **Cao et al.** — arXiv:2501.01165 (2025) | `Cl`, `Cd`, laminar Re 100–5,000 | UIUC + 30% perturbations; PINN, no labels | shape × flow parameter space | mean rel. **`Cl` 4.6%, `Cd` 1.1%** |
| **Bouhlel et al. (2020), Du et al. (2021), Peng et al. (2022)** | aerodynamic coefficients | separate subsonic/transonic models | as tabulated in NeuralFoil Table 6 | "similar accuracy" to NeuralFoil |
| **Airfoil2Vec** — arXiv:2609.38213 (2026) | pressure/velocity fields, ~10 k RANS | NACA 4-digit + non-NACA | **explicit unseen-geometry** interpolation and extrapolation | field errors, not coefficients |
| **AirfRANS / GeoMPNN / AeroDiT** | RANS fields | ~1 k airfoils | unseen airfoils | relative L2 (0.025–0.1) |

**These are not comparable to 8.1%.** Three separate reasons, in descending order of size:

1. **Split protocol.** NeuralFoil's 2.0% is a random split of *cases*: the same airfoil's other
   operating points are in training. Ours is geometry-disjoint. NeuralFoil validates its
   unseen-geometry claim on exactly **one** held-out airfoil, and says so explicitly ("this
   represents a previously-unseen airfoil ... so no unfair advantage is gained by
   memorization"). A single airfoil is a case study, not a generalisation estimate.
2. **Case difficulty.** NeuralFoil's own authors warn against the comparison: their `Cd` error
   is 0.37% on easy cases and 2.0% when post-stall and transitional rows are included, a 5.4×
   spread from difficulty alone. Our 8.1% pools everything, including 17.2% post-stall rows.
3. **Ground-truth quality.** Their reference is XFOIL at 289 points, "the highest resolution
   XFOIL 6.98 allows". Our targets come from the NASA archive at whatever discretisation was
   used to generate it. If that is near 100 points, part of our measured error is
   discretisation noise in the label, not model error.

**What a fair comparison requires:** report error on a matched subset — `|α| ≤ 12`,
`Ncrit = 9`, one Reynolds number — against the frozen test split. That is a two-line filter
once predictions exist, and it is the only number that can sit next to 0.37%.

---

## 3. Controlled test: does a random split actually inflate the score?

The README asserts v1's score was inflated because 100 numbers fingerprint each geometry.
That is a testable claim, and it had never been tested. Design:

- **Validation set held fixed** — the frozen validation partition, 113,678 rows / 123 airfoils
  in *both* arms. Same `SST`, so R² is directly comparable.
- **Arm A (honest)** — train on the frozen train partition: 538,380 rows, 572 airfoils.
- **Arm B (leaky)** — train on train + test partitions, includes every validation airfoil's
  other operating points, then **subsampled to exactly 538,380 rows** so row count is not a
  confound.
- Same 113 features, same `log(Cd)` target, same model settings.

| model | honest (disjoint) | leaky (same airfoils in train) | Δ R² |
|---|---|---|---|
| linear | 0.4918 | 0.4966 | **+0.005** |
| hist. gradient boosting (100 it.) | 0.8331 | 0.8318 | **−0.001** |

The linear arm reproduces the published 0.4918 exactly, which validates the harness.

**Neither model gains from leakage.** For the linear model that is expected — 113 aggregate
features cannot memorise 123 individual polars. For boosted trees it is the interesting
result: heavily regularised trees (31 leaves, lr 0.05, 100 iterations) cannot carve out 123
per-airfoil response surfaces either.

Three implications:

1. **Published random-split numbers are not automatically inflated.** The inflation is real in
   principle but requires capacity, a model that can key on geometry identity, and enough
   rows per airfoil. Do not assume a paper's number is inflated; check what its model could
   plausibly memorise.
2. **The README's explanation of v1 is unsupported.** "100 numbers still fingerprint each
   geometry, which lets a flexible model memorize per-geometry behaviour" cannot explain a
   score measured on *held-out* airfoils — there is nothing to memorise on an airfoil the
   model has never seen. The measurement in §1 gives the better explanation: independent
   centering **distorted** thickness and camber rather than deleting them, and distorted
   geometry is still predictive. That is why fixing it moved the MLP by +0.001.
3. **The leaky arm was only tested on `Cd`, and not on the MLP** (~195 s per fit). The MLP is
   the model with capacity to memorise, so it is the one arm still missing.

---

## 4. Practices worth adopting

### Evaluation protocol

- **Report both protocols.** A number means nothing without its split. Publish
  geometry-disjoint as primary and row-random as a reference point showing what the geometry
  constraint costs.
- **k-fold by geometry, not one 15% holdout.** 123 test airfoils is a small sample; the
  standard error on `Cd` R² across folds is plausibly ±0.02. Five folds × MLP ≈ 16 min on
  Kaggle CPU.
- **Match case difficulty explicitly.** Report an easy subset (`|α| ≤ 12`, `Ncrit = 9`, one
  `Re`) alongside the pooled number — this is precisely what NeuralFoil does with its 0.37%
  vs 2.0% pair, and it is the only way our work can be placed against theirs.
- **Separate held-out-airfoil from held-out-case claims.** NeuralFoil's unseen-shape evidence
  is one airfoil; ours is 123. State the count next to the claim.

### Reporting

- **Lead with median relative error, not R².** For `Cd` that is 8.1%; R² is dominated by
  stalled rows (34.5% of `Cd` validation variance comes from the 17.2% of rows at `|α| > 12`).
- **Drop mean relative error** from headline tables; the 8.3× mean/median gap on `Cl` makes it
  meaningless.
- **Report per-airfoil distributions, not pooled aggregates.** `analysis.py` already does
  this; it needs a run.
- **Report `Cl_max`, `Cd_min`, and `max L/D` error per airfoil.** These are what a designer
  compares sections on, and pooled RMSE is insensitive to them.
- **Multi-seed mean ± std.** Three seeds for the MLP ≈ 10 min.

### Modelling

- **Physics-aware inputs and constraints.** NeuralFoil's gains come from embedded symmetries,
  known limit-case extrapolation, and feature engineering — not from a bigger MLP. We have
  `sin α`, `cos α`, `log Re`, which is the cheap half of that idea.
- **Huber loss instead of MSE** on the transformed targets. NeuralFoil uses Huber (δ≈0.05 in
  normalised units) plus weight decay; they report dropout and batch norm were tested and
  *rejected*. Worth a single controlled comparison.
- **Converge the MLP.** It still hits `max_iter=100`. This is the cheapest accuracy available
  and v1 already flagged it.
- **Target weighting.** NeuralFoil weights drag highest and finds that learning the boundary
  layer *helps* drag. For us the analogue is fitting `Cl`, `Cd`, `Cm` jointly — already done —
  and checking whether joint fitting beats per-target models.
- **Quantify label noise.** Run XFOIL on ~20 airfoils at 100 and 289 points and measure how
  much `Cd` moves. That bounds the achievable error and tells you whether 8% is model error or
  data error. Minutes of compute.

### Engineering (the reproducibility part)

- **Pin and record**: seed, library versions, git commit — already in `run.json`. Keep it.
- **One command to reproduce**, with the dataset fingerprint checked — already there.
- **Report the ablation, not just the endpoint.** v1 vs v2 currently changes the
  representation, the target transform, and adds `Cm` simultaneously, so no single effect is
  attributable. The linear/GB results in §1 show the effects are *not* additive across model
  families; a 3-arm ablation (v1 features / v2 features, each with and without `log(Cd)`) would
  isolate them.

---

## 5. Recommended next actions, costed

| # | action | why | cost | status |
|---|---|---|---|---|
| 1 | Evaluate on the frozen test split | validation scores are selection statistics | seconds | **done, §1b** |
| 2 | Easy-subset and per-Re metrics | the only numbers comparable to NeuralFoil | seconds | **done, §1b** |
| 3 | Report post-stall separately | pooled R² hid a negative `Cd` R² there | seconds | **done, §1b** |
| 4 | Diagnose post-stall `Cd` (R² = −0.17) | the surrogate has no separated-flow capability | 1 session | **next** |
| 5 | MLP leakage arm | closes the open question in §3 | ~7 min | pending |
| 6 | 3 seeds × MLP | determines whether any v1→v2 delta is real | ~10 min | pending |
| 7 | Converge the MLP (`max_iter=500`) | known headroom, cheapest accuracy | ~15 min | pending |
| 8 | Label-noise floor: XFOIL at 100 vs 289 pts | is 8% model error or data error? | minutes | pending |

Items 1–3 needed only the fitted models and are complete. Item 4 is now the highest-value
work, because a negative post-stall `Cd` R² is a capability gap rather than a tuning problem.

Artifacts from the local run: `results/v2_local/mlp_test_report.json`,
`results/v2_local/mlp.joblib` (refit MLP, validation scores reproduce the Kaggle run exactly).

---

## References

- Sharpe, P. & Hansman, R. J. *NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using
  Physics-Informed Machine Learning.* arXiv:2503.16323 (2025). Dataset size, 95/5 split, and
  error figures quoted from §3.5–§4.6 of the preprint.
- Cao, W. et al. *Solving all laminar flows around airfoils all-at-once using a parametric
  neural network solver.* arXiv:2501.01165 (2025).
- *Airfoil2Vec: Spectral Geometry-Conditioned Neural Surrogate Models for Airfoil Aerodynamics.*
  arXiv:2609.38213 (2026).
- Bouhlel, M. A. et al. (2020); Du et al. (2021); Peng et al. (2022) — as tabulated in
  NeuralFoil Table 6. Not read directly; cited for provenance only.
- Bonnet, F. et al. *AirfRANS* (NeurIPS 2022 Datasets & Benchmarks); Helwig, J. et al.
  *GeoMPNN* arXiv:2412.09399 (2024); Wang, C. et al. *AeroDiT* arXiv:2412.17394 (2024).
