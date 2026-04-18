---
name: numerical-prototype
description: Stage 6 of the research-spark pipeline. Converts the Stage 4-5 formalism into a running JAX-based numerical prototype and produces a concrete predicted observable, after passing three required validation passes (analytic-limit recovery, synthetic benchmark, convergence study). Triggers when the user has a formalized theory and wants to turn it into code, or on phrases like "prototype the model", "build a JAX simulation of the theory", "produce the predicted observable", "validate the solver against analytic limits", "do the convergence study", "implement the formalism from Stage 4", or after Stage 4-5 completes. The predicted observable this emits is the direct input to Stage 7 experimental design. Catches pathologies the analytics hide (stiffness, IC sensitivity, narrow validity) by running the math instead of just staring at it.
---

# numerical-prototype

Stage 6. Converts the formalism into a running solver, runs three validation passes, emits the predicted observable Stage 7 will design a measurement for.

## Why this is a separate stage from Stage 4-5

A derived equation and a running solver reveal different failure modes. The derivation tells you what the equation claims; the prototype tells you whether the equation integrates stably, converges under refinement, and produces a physically interpretable output. In practice, prototyping routinely catches:

- Stiffness the analytics do not flag
- Initial-condition sensitivity that ensemble averages suppress
- Regime narrowing (the theory formally holds for all parameters; the solver produces sensible answers only in a subset)
- Sign and coefficient errors that survived the limit checks

Splitting prototyping from experimental design also creates a clean interface: the predicted observable file. Stage 7 reads it as-is without having to understand the simulation.

## Prerequisites

`04_theory.md` and `05_formalism.tex` must exist, and `05_formalism.tex` must compile. The formalism needs labeled equations; the reconciliation check reads labels.

## Workflow

### 1. Load inputs and the code architecture rules

Read `04_theory.md` and `05_formalism.tex`. Pull out: equations to solve, state variables, parameters with physical ranges (usually from the dimensionless-groups section), boundary and initial conditions.

Before writing any code, load `../_research-commons/code_architecture/` in full. Every file there is part of the contract:
- `jax_first_rules.md` (no Python loops in physics cores, vmap/jit discipline, PRNGkey handling, dtype policy)
- `env_conventions.md` (Python 3.12+, uv, pyproject.toml layout)
- `testing_conventions.md` (pytest structure, property tests, limit tests, shape tests, convergence tests)
- `repo_layout.md` (directory structure matching homodyne / heterodyne / RheoJax)

Code that ignores these conventions fragments the ecosystem and makes future integration painful. The rules are consistent across the stack deliberately.

### 2. Build the minimal JAX implementation

`templates/prototype_skeleton.py` is the starting scaffold. In order:

- **State.** A registered-pytree dataclass holding state variables. Dtype consistent per module policy.
- **Params.** Separate from state. Physical parameters that do not vary during integration.
- **Forward operator.** `step(state, key, params) -> new_state`. Pure, jit-able, no Python loops. `jax.vmap` for batching, `jax.lax.scan` for time integration.
- **Observable extractor.** `extract_observable(trajectory, params) -> Observable` in the format Stage 7 expects (see `templates/predicted_observable.md`).

Keep v1 minimal. Adaptive timestepping, multi-device parallelism, exotic boundary conditions: all deferred. The goal is to run the theory and produce an observable.

### 3. Run the three validation passes

None can be skipped. Each catches a different failure mode.

**Pass 1: analytic-limit recovery.** Switch off the new physics (set the new-physics parameter to zero, or take a known limit) and verify the solver reproduces the analytic reference within documented tolerance. `scripts/limit_recovery_check.py`.

Common tolerance: 1e-3 for stochastic solvers, 1e-6 for deterministic. Signatures of specific failures: a constant ratio between result and reference usually means a factor-of-2 or pi error; timestep-dependent deviation usually means a sign error that cancels in steady state but grows in transient.

Do not proceed if this fails. The error is in the derivation, in the implementation, or in an un-stated assumption. Hunting it down is the work, not a distraction from the work.

**Pass 2: synthetic benchmark.** A problem with a known analytic answer that exercises the *new* physics, not just the limiting case. Examples: Ornstein-Uhlenbeck for stochastic solvers; a simple reversible reaction for bond-exchange work; a sphere form factor for scattering solvers. Same tolerance discipline.

**Pass 3: convergence study.** Vary timestep, grid, or particle count across at least three resolutions. `scripts/convergence_study.py` does Richardson extrapolation and reports empirical convergence order plus uncertainty. The observable at the finest resolution should fall inside the extrapolated uncertainty band, and the empirical order should match the expected order for the integrator.

### 4. Emit the predicted observable

Follow `templates/predicted_observable.md`. Required content: observable values with units; uncertainty decomposed into numerical (from convergence), parametric (from parameter ranges), and statistical (from stochastic ensembles); temporal and spatial structure; noise model for the eventual measurement; expected SNR at typical conditions.

Stage 7 reads these fields directly. A predicted observable without uncertainty bounds is not actionable; Stage 7 cannot build a capability map against it.

### 5. Reconcile code against formalism

`../_research-commons/scripts/formalism_code_reconcile.py` compares symbols in `05_formalism.tex` against symbols in the code. Flags symbols in the LaTeX that do not appear in code (forgot to implement), and vice versa (implemented but not documented). Resolve mismatches before finalizing.

### 6. Write `06_prototype.md`, lint, hand off

Structure: summary of what was implemented; validation results for all three passes (with plots); predicted observable summary and path; known limitations; open questions for Stage 7. Style-lint. Update `_state.yaml`.

## Failure modes worth naming

- **Skipping validation because "the code looks right."** Visual inspection misses sign errors and dropped terms consistently. The passes catch what the eyes do not.
- **Running the convergence study at one resolution and declaring convergence.** Richardson needs three. A single run tells you nothing about whether the result is a numerical artifact.
- **Emitting a scalar predicted observable with no uncertainty.** Stage 7 has nothing to plan against. Worst case, uncertainty is "large and hard to quantify"; that is still more informative than silence.
- **Python loops in the jit-compiled physics core.** If vmap and scan cannot express the computation, the state representation probably needs redesigning. Fix the data layout, not the integrator.

## Templates and scripts

- `templates/prototype_skeleton.py`: JAX scaffold
- `templates/validation_report.md`: structure for the three passes
- `templates/predicted_observable.md`: standardized handoff format for Stage 7
- `scripts/convergence_study.py`: Richardson-extrapolation driver
- `scripts/limit_recovery_check.py`: limit-recovery runner
- `scripts/observable_extractor.py`: pulls observables from trajectory in canonical format
- Shared: `../_research-commons/code_architecture/`

## Fan-out (Claude Code multi-agent)

Natural parallelization points: forward simulation, limit-recovery pass, synthetic-benchmark pass, convergence study, and reconciliation can each run as sub-agents. A synthesis sub-agent merges the validation results into the report.

## Handoff to Stage 7

Stage 7 reads the predicted observable file and uses it as the target signal. If the observable has structure at 10 ms, Stage 7's capability map must show an instrument with temporal resolution meaningfully below 10 ms, or the plan flags high risk and requires a concrete mitigation. That comparison is automatic because the observable file is standardized.
