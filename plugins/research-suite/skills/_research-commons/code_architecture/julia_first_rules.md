# Julia-first code architecture rules

Load this file instead of the JAX quartet (`jax_first_rules.md`, `env_conventions.md`, `testing_conventions.md`, `repo_layout.md`) when Stage 6's language-choice step picked Julia. Do not load both.

## Which environment

- `@sciml` (default): DifferentialEquations.jl, Lux-based UDEs, SINDy equation discovery, DynamicalSystems.jl (bifurcation, chaos, pattern formation), Catalyst, NetworkDynamics.
- `@bayes`: Turing + Pigeons DEV override, for NUTS / Consensus MC / NRPT uncertainty quantification, Bayesian UDEs, GP work.
- `@pinn`: NeuralPDE, for physics-informed neural nets on PDE-posed theories.
- `@gnn`: only if the theory itself is graph-structured (rare for a Stage 6 prototype).

Activate with `julia --project=@sciml` (or `@bayes` / `@pinn` / `@gnn`) before writing any code. Document the chosen environment in `06_prototype.md`.

## Core rules

1. **Type stability.** Every hot-path function must be type-stable: no `Union` return types, no abstract fields in structs used inside a loop, no `Any`. Check with `@code_warntype` before declaring the prototype done.
2. **Allocation-free hot loops.** The physics kernel (`step!`) should not allocate: preallocate buffers outside the loop and mutate in place, the Julia analog of JAX's pure `step(state, ...) -> new_state`. This applies to the kernel itself, not the outer trajectory-capture loop around it (`integrate!` allocating one `State` snapshot per recorded step is expected and fine; only the per-step physics update is held to the zero-allocation bar).
3. **Multiple dispatch over conditionals.** Prefer separate methods on concrete types over `if`/`elseif` branching on a type flag.
4. **Explicit RNG discipline.** Every stochastic function takes an `AbstractRNG` as an explicit argument, never a hidden global. Mirrors JAX's PRNGKey discipline.
5. **DifferentialEquations.jl for anything stiff.** If the governing equation is a stiff ODE, DAE, or PDE (likely the reason Julia was chosen), use `OrdinaryDiffEq.jl`'s implicit solvers (`Rodas5`, `TRBDF2`, and similar) rather than hand-rolling an explicit integrator.
6. **Never `Pkg.add` a dev-overridden package** (`Pigeons`, `GNNLux`/`GNNGraphs`/`GNNlib`) without confirming the user wants to leave dev-override mode. Never use `SequentialMonteCarlo` in a multi-package environment; it SIGABRTs on Julia 1.12 via a `RNGPool.__init__` threading race.

## Package layout

Standard Julia package structure, not the JAX quartet's `pyproject.toml` layout:

```
<repo-root>/
├── Project.toml
├── Manifest.toml               # committed, per-environment
├── src/
│   └── <PackageName>.jl        # module; exports step!, integrate, extract_observable
├── test/
│   └── runtests.jl             # @testset blocks, same four tests as testing_conventions.md
├── examples/
│   └── run_minimal.jl
└── docs/
    ├── derivation.md           # symlink to 04_theory.md, same convention as the JAX layout
    └── predicted_observable.md # symlink to the Stage 6 artifact
```

## Testing

Same four mandatory tests as `testing_conventions.md` (invariant, limit recovery, shape/dtype analog, convergence), written as `@testset` blocks using `≈` for numeric comparison, never `==`:

```julia
using Test, Random

@testset "time advances by n_steps * dt" begin
    rng = MersenneTwister(0)
    params = Params(dt=1e-3)
    state = State(100, 3, rng)
    trajectory = integrate!(state, params, 1000, rng)
    @test trajectory[end].t ≈ 1000 * params.dt rtol=1e-10
end

@testset "recovers free diffusion (limit)" begin
    rng = MersenneTwister(0)
    params = Params(interaction_strength=0.0, diffusion=1.0, dt=1e-3)
    state = State(500, 3, rng)
    trajectory = integrate!(state, params, 2000, rng)
    obs = extract_observable(trajectory, params)
    dims = size(state.positions, 1)
    expected_msd = dims * 2 * params.diffusion * obs.t[end]
    @test obs.values[end] ≈ expected_msd rtol=0.3  # loose: illustrative tolerance, tune per problem
end
```

Run with `] test` from the activated environment.

## Bridging to the Python validation harness

`../../numerical-prototype/scripts/convergence_study.py` and `../../numerical-prototype/scripts/limit_recovery_check.py` are language-agnostic: they call a Python callable and compare array-likes. Bridge Julia into them with a thin `juliacall` shim instead of reimplementing the scripts:

```python
# bridge.py - imported by convergence_study.py / limit_recovery_check.py
from juliacall import Main as jl
jl.seval('include("src/MyPackage.jl"); using .MyPackage')

def run(dt: float):
    return jl.MyPackage.run_prototype(dt)  # juliacall converts the Julia array to NumPy
```

No changes to the validation scripts are needed. `observable_extractor.py`'s array-to-list conversion already handles juliacall-wrapped arrays through its iterable fallback.

## Reconciling code against the formalism

`../scripts/formalism_code_reconcile.py` parses Python via `ast` and does not read Julia. Reconcile a Julia prototype manually: read the labeled equations in `05_formalism.tex`, check each symbol appears in the Julia source (function arguments, struct fields, or named constants), and record any mismatches in `06_prototype.md`'s reconciliation section.

## Why these rules

Type stability and allocation discipline are Julia's analog of JAX's jit/vmap discipline: both exist because the alternative (dynamic dispatch, Python loops) silently kills performance in ways that are easy to miss by inspection. The `@sciml`/`@bayes`/`@pinn`/`@gnn` split matches the 5-environment layout used across the rest of the Julia work this pipeline feeds into; picking the matching environment here keeps Stage 6 prototypes consistent with everything else in the ecosystem.
