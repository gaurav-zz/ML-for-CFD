# Validation And Selective Reliability

## Purpose And Status

The practical objective is useful aerodynamic screening with fewer expensive reference
evaluations, not universal replacement of physics solvers. A model that identifies
where it is useful can be valuable despite failures on the full geometry population.
That claim requires validation of the routing decision as well as the coefficient errors.

Current code provides predictions, numerical diagnostics, and an **offline descriptive
referral analysis**. It does not yet provide calibrated confidence percentages,
guaranteed error bounds, or an operationally qualified automatic acceptance gate.
The existing full-domain model decisions are not retroactively changed by this goal.

## Keep Evaluation Cohorts Separate

| Evidence | Scope | What it establishes |
|---|---|---|
| Early baselines and v1/v2 changes | Different representations; originally selected on validation | Useful development history, not an isolated attribution of every gain |
| Controlled loss/regularization and Cdp confirmation | Original 123-airfoil test, 116,484 rows; already inspected | Historical paired evidence, not a new blind test |
| Geometry-fold study | Three folds of the original 572 training IDs, with three seeds | Geometry-held-out development behavior and initialization variability |
| Broad and selective input-scaling studies | Same 538,380 OOF rows and fold identities; Cdp fixed | Controlled conditioning comparisons |
| Referral analysis | The same inspected OOF predictions | Exploratory error/coverage and FP/FN accounting, not calibrated deployment performance |

Each geometry-fold model fits 343 or 344 geometries and stops on 38 different
geometries. The evaluation folds contain 191, 191, and 190 IDs. All rows remain in
the full-population reports. Fold training sets overlap; fold/seed variability is
descriptive, not a collection of independent confidence trials.

The repeated standard-Cdp control in the broad and selective runs has exactly
equivalent saved numeric arrays and configurations. It is counted once, not as
additional independent replication.

## Current Full-Population Evidence

| Configuration | Cl R² | Cd R² | Cm R² | Cd RMSE |
|---|---:|---:|---:|---:|
| Standard Cdp ensemble | 0.942780 | -10.862452 | 0.703586 | 0.102742 |
| Broad RMS floor | 0.987426 | 0.881793 | 0.925699 | 0.010256 |
| Selective 1%-RMS floor | 0.987309 | 0.898320 | 0.924527 | 0.009512 |

R² is computed directly over aligned OOF observations, not averaged across folds.
Negative R² is not negative Cd: it means worse squared error than predicting the
reference population mean. The selective candidate passes 16/17 frozen relative
checks and all 12 minimum numerical-safety checks, but is not promoted. Its fold-0
worst-geometry Cd RMSE worsens 55.1%, against a 10% allowance.

| Geometry | Control Cd RMSE | Broad floor | Selective floor | Interpretation |
|---|---:|---:|---:|---|
| AH 93-W-480B | 2.10926 | 0.05571 | 0.05852 | Clipping removed; extreme thickness remains poorly predicted |
| AH 81-K-144 W-F KLAPPE | 0.05548 | 0.07989 | 0.08606 | Flapped case remains a tail regression |
| LIEBECK L1003 | 0.05031 | 0.08556 | 0.05417 | Selective scaling recovers much, not all, of the regression |
| 18%-camber, 5%-thickness plate | 0.03142 | 0.04348 | 0.02094 | Selective scaling improves this case |

These names are regression-test cases, not a deployable blacklist. Future inputs do
not arrive with their true prediction error attached.

## Common Success And Failure Patterns

The new analyzer compares all 572 geometries across the three compatible configurations.
It also retains full-row results, including the 1,268 hard-referral rows in the
standard control. Selective scaling improves geometry-level Cd RMSE on 305 geometries
and worsens it on 267; relative to the broad floor it improves 397 and worsens 175.

A diagnostic persistence comparison defines low absolute-error geometries as being
in the lowest Cd-RMSE quartile of **every** configuration, and high absolute-error
geometries as being in the highest decile of **every** configuration:

| Descriptive group | Geometries | Median thickness | Median absolute camber | Median support distance | Outside any fit-feature range |
|---|---:|---:|---:|---:|---:|
| Persistently low absolute error | 116 | 11.99% | 1.88% | 0.142 | 0.86% |
| Persistently high absolute error | 47 | 14.05% | 5.61% | 0.270 | 19.15% |

These are post-hoc associations, not safe shape limits. Absolute-error ranks do not
guarantee small relative errors. Roughly 81% of the persistently high-error group
still lies inside all individual fit-feature ranges, so marginal bounds alone are
not sufficient. Similarity must consider the joint profile, descriptors, and flow
conditions; the fixed support distance itself is not a probability of accuracy.

The recurring signals across compatible evidence are:

- Very thick/extrapolating geometries: FX 79-W-660A dominates drag squared error in
  five of six cached Cdp-confirmation test reports; AH 93-W-480B is the analogous
  training-fold extreme. These are different cohorts and are not pooled together.
- Strong camber and unusual/flapped profiles appear repeatedly among difficult cases.
  L1003 and AH 81 are not outside individual fit-feature ranges, so a range check
  must not automatically approve them.
- Low Reynolds number is recurrent: 90.66% and 96.45% of the broad-floor drag squared
  errors on L1003 and AH 81 occur at Re <= 100,000. This is a diagnostic concentration,
  not a claim that every low-Re case is unsafe.
- Numerical amplification and clipping were established failure mechanisms. Clipping
  can make distinct predictions agree at a common limit. Preserve raw fitting-space
  values and raw disagreement rather than treating clipped agreement as confidence.
- Geometry/reference conventions and source-label quality still need targeted checks.
  A shared geometry transform is implemented, but Alpha/chord/Cm reference alignment
  for unusual source cases has not been fully established. This is not a proven bug.

No model-capacity or irreducible-noise ceiling has been established. Existing input
variation, source precision, physics mismatch, and learning-objective effects must
not be conflated.

## What A Referral Rule Must Use

Only information available at prediction time may determine a referral:

- Geometry validity and reference/units completeness.
- Profile/descriptor support relative to fit geometries, including joint similarity.
- Operating-condition support: angle, Re, Ncrit and applicable physics assumptions.
- Raw numerical saturation/nonfinite outputs and prediction consistency.
- Ensemble disagreement and, later, validated error estimates or intervals.

Do not use the true error, airfoil-name lookup, or a post-hoc list of failures as a
generalization rule. Do not equate a percentile rank with a probability. Low spread
does not detect all shared model bias.

Routing should normally be **geometry plus operating condition**, not a permanent
whole-airfoil label. A geometry can be useful in one region of its polar and unreliable
in another. Report fully retained geometries, partially retained polars, and referred
conditions separately; row coverage is not the fraction of airfoils needing no mesh.

## Risk And Coverage Diagnostics

The current tool ranks saved predictions by raw log-Cd spread, clipped spread,
fixed input-only geometry distance, or the maximum of their empirical percentile
ranks. Cutoffs use predictions/inputs only, with ties retained and raw-member clipping
referred. The percentiles are computed on this inspected OOF population; no selector
is trained and no method is selected for deployment.

| Selective candidate ranking | Actual row coverage | Retained Cd RMSE | Median relative error | P95 absolute error | Fully / partially retained geometries |
|---|---:|---:|---:|---:|---:|
| No referrals | 100% | 0.009512 | 6.1097% | See full report | 572 / 0 |
| Raw spread | 95.00% | 0.007460 | 5.8767% | 0.014371 | 268 / 303 |
| Geometry distance | 95.05% | 0.007791 | 5.9340% | 0.015003 | 545 / 0 |
| Combined maximum percentile | 95.00% | 0.007234 | 5.8944% | 0.014351 | 373 / 180 |
| Raw spread | 50.00% | 0.005167 | 4.7736% | 0.009736 | 0 / 568 |

Subset distributions change when cases are referred. These rows are not new
full-population improvements, and no subset R² gain is claimed. The combined score's
lower RMSE also does not imply the lowest rate of budget exceedances.

## False Accepts And False Referrals

For the **referral** problem, define positive as: the prediction exceeds the agreed
error budget and needs reference evaluation. This is approximation error, not an
airworthiness or physical-safety certification.

| Reference outcome | Referred | Retained |
|---|---|---|
| Over budget | True positive | **False negative: missed referral / false acceptance** |
| Within budget | **False positive: unnecessary referral** | True negative |

Use separate measures: missed-referral rate among over-budget cases, over-budget
fraction among retained cases, unnecessary-referral rate, actual coverage, and
coefficient/decision errors. These denominators answer different questions.

As an illustration only, with a 10% Cd relative-error budget and 95% raw-spread
coverage: TP=17,646; FN=154,267; FP=9,273; TN=357,194. Thus 89.74% of over-budget
rows are retained, and **30.16% of retained rows exceed that budget**. At 50%
coverage, 22.11% of retained rows still exceed it. The combined ranking at 95%
coverage has a 30.25% retained exceedance rate despite its lower RMSE.

The 5%, 10%, and 20% budgets in the tool are **illustrative sensitivity settings**,
not approved engineering requirements. An operational multi-target rule should
support budgets such as `abs(error_y) <= a_y + r_y * abs(reference_y)`, avoiding
unstable relative-only criteria near zero Cl/Cm. Appropriate tolerances, reference
uncertainty, and any L/D or design constraints must be agreed before calibration.

For **design screening**, positive may instead mean "meets the design criterion."
Its false positives are bad candidates retained as promising; false negatives are
good candidates wrongly discarded. That is a different confusion matrix. Validate
top-candidate recall, ranking/regret, and constraints explicitly. An uncertain
negative prediction should be deferred, not automatically discard a potentially
useful design.

## Qualification Plan

1. Define the intended geometry/flow domain, engineering error budgets, and screening
   decision. Keep the full existing benchmark intact and report domain coverage.
2. Freeze prediction-time scores and candidate referral rules. Geometry similarity
   and low spread are evidence features, not already calibrated confidence.
3. Use distinct geometry groups for model fitting, stopping, selector calibration,
   and audit. Early-stopping data are not independent calibration data. Current folds
   have already been inspected; any reuse is development-only evidence.
4. Validate retained error, false accepts/referrals, full/partial geometry coverage,
   and decision quality. Use geometry-level uncertainty assessment rather than
   pretending correlated polar rows are independent trials.
5. Confirm the qualified workflow on genuinely new geometries/families. Any project
   simulation must obey the [online-data-first novelty policy](solver-validation.md).
6. Measure end-to-end utility on matched tasks: cold/warm inference, batch throughput,
   reference lookup, geometry/mesh preparation, solver time, retries and caching.

For a simple equal-cost example, total work is approximately
`N * prediction_cost + (1 - coverage) * N * reference_cost + overhead`.
If inference/overhead were negligible, 90% coverage would suggest an idealized 10x
speedup. This is **not measured project performance**: referred cases can cost more,
meshes are shared per geometry, and fully/partially retained polars change the cost.
Track wall time and CPU/GPU resource consumption, not just the number of predictions.

## Exploratory Calibration Protocol

The exploratory study fixes the selective-floor Cdp ensemble and calibrates referral
thresholds, not surrogate weights. It is explicitly developmental: the source
evaluation folds have already informed model choices, so a new split of these rows
does not become a fresh blind benchmark or establish calibrated confidence.

- Split each original evaluation fold's sorted geometry IDs once, using RNG seed
  `20261105 + fold_id`: the first half is calibration, the remainder audit. This gives
  95 calibration geometries per fold and 96/96/95 audit geometries. Preserve the
  original surrogate fit/stopping partitions and keep both new groups outside them.
- Calibrate independently for each fixed fold ensemble. Do not pool calibration
  examples across folds: other fold models were trained on some audit geometries.
- Compare raw log-Cd spread, the fixed geometry-support distance, and their maximum
  calibration-percentile score. Fit empirical score distributions on eligible
  calibration inputs only; do not rerank audit inputs. For the combined rule, both
  native score components must lie inside their closed calibration min/max intervals
  before percentile transformation; values outside require referral. This conservative
  support guard is additional to the native methods' fixed cutoff and numerical rules.
- Use the approved exploratory Cd relative-error bands 5%, 10%, and 20%. Separately
  examine empirical retained over-budget fractions capped at 5% and 10% during
  calibration. These two quantities are not interchangeable, and neither is an
  approved operational limit or a confidence level.
- Evaluate all 21 row-weighted calibration quantiles `0, .05, ..., 1`, using the
  `lower` interpolation rule on nonclipped calibration rows. Fit CDFs on those same
  eligible rows, without filtering by true error. Retain ties and always refer
  numerical clipping. Choose the largest calibration coverage meeting the
  empirical risk cap and retaining at least 20 distinct calibration geometries.
  This minimum support is an experimental safeguard, not a statistical guarantee.
  Break equal-coverage ties with the lower threshold. Evaluate every candidate:
  conditional retained risk need not be monotone in the threshold. If no threshold
  qualifies, refer all audit requests; do not invent a fallback.
- Freeze each threshold before reading audit errors. Report all three score methods,
  three error bands, and two risk caps for every fold, including empty/no-policy
  outcomes. Do not choose a winning method or threshold from the audit results.
- Compare retained audit errors against the no-referral baseline on those same audit
  rows, not against a different full-population denominator. Also retain full-row
  metrics, FP/FN counts, row- and geometry-level coverage, and Cl/Cm error diagnostics.
  Report `FP / (FP + TN)` separately from the unnecessary-referral share
  `FP / (TP + FP)`; report per-geometry risk concentration without changing the
  row-weighted selection objective. Undefined denominators remain undefined.

This is point-level approximation-error research. It does not qualify complete
polars, design-ranking decisions, multi-target budgets, or formal conditional-risk
guarantees. No primary model fitting, original validation/test scoring, solver runs,
or production/default changes are part of it. Computational tests and the calibration
study run on cloud CPU first, with source, input, and partition provenance retained.

### Findings And Limits

The frozen cloud-CPU experiment completed with 110 passing tests, no surrogate
retraining, and no solver runs. It retained all 54 fold/method/error-band/risk-cap
outcomes, including policies that referred every audit request.

| Cd error band | Policies with a supported calibration threshold | Separate audit outcome |
|---|---:|---|
| 5% | 0 of 18 | No threshold met the calibration cap/support requirements |
| 10% | 0 of 18 | No threshold met the calibration cap/support requirements |
| 20% | 13 of 18 | 12 of the 13 selected policies exceeded their calibration risk cap |

For the 20% error band and 10% empirical calibration-risk cap, the selected rules
retained approximately 89-96% of audit rows, but the retained over-budget fraction
was 10.26-13.14%. The lone selected policy with an audit risk below its cap is not
chosen as a winner after inspecting the audit. Empty retention has undefined risk,
not a successful zero-risk result. Per-geometry and full/partial polar diagnostics
remain important even when a point-level average looks acceptable.

These findings do not prove that selective screening is ineffective. They show that
the tested global thresholds on disagreement and geometry distance are insufficient
to support the requested exploratory error/risk combinations consistently. A later
study should investigate richer prediction-time error signals, including operating
conditions and geometry/flow interactions, rather than relaxing caps to label this
run successful. It must again reserve calibration and audit groups and evaluate the
specific fixed surrogate that will be used; thresholds cannot be assumed transferable
to different weights or a final refitted ensemble. No production policy is qualified.

Evidence is stored under `results/kaggle/referral-calibration-dev-v1/`, from the private
[`airfoil-exploratory-referral-calibration`](https://www.kaggle.com/code/purplray/airfoil-exploratory-referral-calibration)
kernel. Its source SHA256 is
`795c18af7ea300def7c6e8167357bde67e346638287a5e5102d9bc1631660003`.
The archived frozen plan, partitions, calibration-only CDFs, candidates, selected
policies, full audit rows, and metric tables preserve the complete development result.

To reproduce the same fixed study in a prepared cloud CPU environment:

```bash
python -m src.training.referral_calibration \
  --selective-run <completed-selective-run-directory> \
  --polars <airfoil_polars.csv> --split configs/frozen_split.json \
  --geometry-coverage <geometry_coverage.csv> \
  --output-dir <fresh-output-directory>
```

## Reproduce The Offline Analysis

Run computational validation on cloud first. The analyzer uses saved arrays and
does not fit models, run solvers, rescore canonical holdouts, or modify old artifacts:

```bash
python -m src.training.reliability_analysis \
  --broad-run <completed-broad-run-directory> \
  --selective-run <completed-selective-run-directory> \
  --polars <airfoil_polars.csv> --split configs/frozen_split.json \
  --geometry-coverage <geometry_coverage.csv> \
  --output-dir <fresh-output-directory>
```

The completed review is at `results/reliability-review/current-oof-20261105/`:
`analysis.json`, `full_row_summary.csv`, `per_geometry.csv`,
`geometry_comparisons.csv`, `risk_coverage.csv`, and `illustrative_sensitivity.csv`.
It validates row/fold identities, numeric arrays, modes, and repeated-control
equivalence. Geometry support is optional; its fixed table and checksum are recorded.
The persistence table above is derived from per-configuration Cd-RMSE ranks and the
input-only geometry coverage table, not used as a referral feature.

Historical evidence remains at:

- `results/kaggle/controlled-study-v2/experiments/v2/20261019T115249707139Z/`
- `results/kaggle/cdp-study-v3/experiments/v2/20261020T104928953059Z/`
- `results/kaggle/fold-study-v1/experiments/v2/20261026T182641468688Z/`
- `results/kaggle/input-scaling-v1/experiments/v2/20261028T124731457567Z/`
- `results/kaggle/selective-scaling-v1/experiments/v2/20261029T134426839881Z/`

Run logs, failed-run evidence, weights, frozen gates, and verification receipts are
preserved. Detailed chronology is in the [archive](archive/README.md), not README.
