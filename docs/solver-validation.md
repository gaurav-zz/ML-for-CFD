# Independent Reference And Solver Validation

## Policy

This is a future validation/fallback plan, not an instruction to start simulations.
No airfoil has been approved for a new solver run by this document.

The surrogate supports rapid analysis; a referral means that another source of
evidence is needed, not that CFD is automatically required or automatically correct.
All computational work is cloud-first. Local execution is considered only after
cloud feasibility has been checked and a limitation is documented.

Two conditions are mandatory before any new project simulation:

1. **The geometry is absent from the entire source dataset.** Include all 825 source
   cases, not just 818 processed cases or the current training fold. The seven cases
   lacking usable polars are not eligible merely because their labels are missing.
2. **Suitable required reference data are unavailable after a documented online
   search.** Existing suitable data take precedence over a new simulation.

Known difficult dataset airfoils, including AH81, AH93, and L1003, may be diagnosed
using their existing/source records. They must not be simulated under this project
policy. Missing operating points on an already known geometry do not make it a new
airfoil; record the unsupported request rather than bypassing the restriction.

## Novelty And Data Search

Maintain a proposed validation registry before launching jobs. Each entry needs:

- Candidate identity, aliases, source URL/license, retrieval date, and original coordinates.
- Complete-source membership check, including skipped/raw cases.
- Geometry checks after consistent normalization: ordering, reversed surface traversal,
  translation, rotation, scale, reflected copies, and near-duplicates. A renamed or
  rescaled known airfoil is not independent validation.
- Versioned comparison tolerances and any ambiguous match. Unresolved identity is
  **not** verified novelty; no candidate should be declared unseen from its name alone.
- Requested Alpha/Re/Mach/transition/reference conventions and their data coverage.
- Public search results and why any available data are unsuitable for the intended test.
- Final status: use existing reference, ineligible, unresolved, or eligible for a
  separately approved bounded simulation.

Candidate sources include the [UIUC coordinate database](https://m-selig.ae.illinois.edu/ads/coord_database.html),
[UIUC low-speed tests](https://m-selig.ae.illinois.edu/uiuc_lsat.html), and primary
experimental/CFD publications with accessible geometry and conditions. Many familiar
UIUC/NACA airfoils are already in the source corpus. None is presumed novel here.

"No suitable data found" should describe the search performed, not claim exhaustive
proof that no online data exist. Data for a similar airfoil can inform screening,
but must not be presented as reference truth for a different geometry.

## Match The Physics

Before comparing surrogate and reference values, establish:

- Geometry version, flap configuration, chord axis, dimensional chord and any transformation.
- Alpha convention, velocity/fluid properties, chord-based Re, Mach and compressibility.
- Transition/trip/roughness and freestream turbulence assumptions. Ncrit is not a
  universally interchangeable turbulence-intensity input.
- Force directions/signs, reference area/span/length, and Cm moment point/sign.
- Steady versus time-averaged quantities, averaging interval, and source uncertainty.

The current model has no independent Mach or dimensional-chord feature. Keep these
as reference metadata and do not infer validation outside compatible assumptions.
Two-dimensional steady fully turbulent RANS, transitional XFOIL-associated polars,
and three-dimensional separated/unsteady flow are not interchangeable ground truths.

For problematic existing records, inspect original geometry/polar headers and source
processing before assuming a label error or applying an angle correction. The inspected
NASA generation workflow can retain scraped scalar coefficients while regenerating
Cp; Cp regeneration alone therefore does not certify the scalar labels' conventions.

## Cloud-First Execution Options

| Workload | First choice | Local fallback |
|---|---|---|
| Model experiments | Existing private cloud ML workflow | Only after documented cloud infeasibility and a bounded alternative plan |
| Fit-free tests and offline reliability analysis | Cloud CPU/CI runner | Consider only if cloud execution cannot reasonably perform the task |
| Small appropriate OpenFOAM cases | Cloud CPU with a pinned distribution and case setup | A bounded local case only after the cloud-first check |
| Justified high-order/unsteady PyFR cases | Suitable cloud accelerator/runtime with checkpoints | Not a default local workload |

No paid provider, machine size, or long-running case is selected here. Establish
dependency/backend support, precision, memory, mesh/partition size, wall-time and
budget limits before provisioning. Ephemeral notebook runtimes may be unsuitable
for long CFD jobs; a persistent cloud runner can be more appropriate if approved.

### PyFR

[PyFR](https://pyfr.readthedocs.io/en/latest/systems/solver.html) supports high-order
flux-reconstruction formulations with Euler/Navier-Stokes and artificial-compressibility
systems. Official [installation guidance](https://pyfr.readthedocs.io/en/latest/installation.html)
documents OpenMP and accelerator backends including CUDA, HIP, OpenCL, and Metal.
Match backend dependencies, numerical precision, memory and MPI partitioning to the
chosen environment; using cloud or a higher order does not itself guarantee fidelity.

Use documented [meshing](https://pyfr.readthedocs.io/en/latest/meshing.html),
[force/moment extraction](https://pyfr.readthedocs.io/en/latest/plugins/soln-plugin-fluid-force.html),
and [time averaging](https://pyfr.readthedocs.io/en/latest/plugins/soln-plugin-tavg.html).
The synthetic-eddy [turbulence plugin](https://pyfr.readthedocs.io/en/latest/plugins/solver-plugin-turbulence.html)
is not a substitute for an Ncrit transition prescription. Extreme cases may require
unsteady and three-dimensional treatment; do not label an arbitrary 2D result accurate.

### OpenFOAM

Pin the OpenFOAM distribution, version, image and architecture. Do not mix Foundation
and OpenCFD solver/module instructions or dictionaries. The OpenCFD
[Docker guidance](https://gitlab.com/openfoam/core/openfoam/-/wikis/precompiled/docker.md)
warns about AMD64 emulation on ARM; match architecture for serious use. The Foundation
documents [macOS through an Ubuntu VM](https://openfoam.org/download/macos/).
These are possible fallback routes, not installations performed by this project.

Where needed, review the selected distribution's
[transition model](https://doc.openfoam.com/2606/tools/processing/models/turbulence/ras/linear-evm/rtm/kOmegaSSTLM)
and [force-coefficient conventions](https://doc.openfoam.com/2606/tools/post-processing/function-objects/forces/forceCoeffs).
An available turbulence/transition model does not establish that its inputs match
the source polars. A few low-cost attached-flow pilot cases may precede harder cases,
but only on verified eligible new geometries without suitable existing data.

## Validate The Reference Run

Before using a solver result to judge the surrogate:

1. Check geometry and mesh quality, curved/boundary representation, domain extent,
   and wall resolution appropriate to the chosen formulation.
2. Document boundary conditions and all physical/reference assumptions.
3. Check mesh/domain refinement and applicable timestep/order sensitivity.
4. Check mass balance, residual behavior, and converged force/moment histories.
5. For unsteady cases, remove startup transients and establish adequate averaging,
   sampling uncertainty, and where relevant spanwise sensitivity.
6. Preserve failures and reference uncertainty. Do not force an unconverged or
   mismatched case into a scalar accuracy table.

Use published verification cases to validate the solver setup without violating
the project's new-simulation policy: inspect available results/setup documentation
first. If a proposed actual benchmark run uses an in-dataset airfoil, it is not
authorized by the current policy even when it is a standard solver tutorial.

## Evidence, Caching, And Utility

Cache references by geometry and complete physics/numerics configuration hashes,
not just an airfoil name or rounded Re. Retain source-search decisions, novelty
evidence, geometry, mesh, solver/image versions, input dictionaries, hardware,
logs, restart files, force histories, pressure fields and failed runs.

Pressure visualization can use qualified reference/CFD fields. The coefficient-only
surrogate does not supply a pressure distribution; any future Cp predictor requires
separate training data and validation.

Measure cold/warm prediction, batching, lookup, geometry/mesh setup, solver and
post-processing costs separately. Include failed runs, retries, caching, and resource
hours. Record actual hardware and resource telemetry; requested GPU allocation is
not a utilization measurement. Compare matched accuracy/physics tasks and report
the resulting error/coverage/speed tradeoff rather than an unmeasured speedup claim.
