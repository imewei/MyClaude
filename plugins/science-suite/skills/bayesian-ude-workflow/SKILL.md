---
name: bayesian-ude-workflow
description: Build end-to-end Bayesian Universal Differential Equation (UDE) workflows in Julia. Use when fitting neural ODEs with posterior uncertainty, embedding Lux neural networks inside Turing @model blocks with DiffEq solves, warm-starting MCMC from a deterministic optimum, or composing UDE training with SINDy for symbolic recovery. Use proactively when the user mentions Bayesian neural ODE, UDE with uncertainty, Turing + DifferentialEquations, or wants confidence intervals on a learned dynamical system.
---

# Bayesian Universal Differential Equation Workflow

End-to-end recipe for fitting a Universal Differential Equation (UDE) — an ODE whose right-hand side mixes known physics with a neural network correction — and obtaining a posterior over both the physical parameters and the neural network weights.

## Mode Flag

- `--mode quick`: routing table + agent delegation only
- `--mode standard` (default): stage outline + sampler selection table
- `--mode deep`: full Stage 2 Turing model code block

---

## Expert Agent

For Bayesian UDE workflows in Julia, delegate to:

- **`julia-pro`**: Julia SciML ecosystem, DifferentialEquations.jl, Turing.jl, ComponentArrays.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
- **`statistical-physicist`** (secondary): Bayesian inference theory, MCMC diagnostics, identifiability.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`

---

## Why a multi-stage workflow?

A Bayesian UDE posterior is brutal for vanilla NUTS:

- **High dimension** — neural network weights stack on top of physical parameters
- **Curved geometry** — the likelihood surface bends sharply along sensitive directions of the ODE flow
- **Multiple basins** — neuron permutation symmetries and sign degeneracies create disjoint modes
- **Cost** — every log-density evaluation requires a full ODE solve plus an adjoint pass

Initializing a sampler at the prior mean almost never converges. The fix is a **staged pipeline**: deterministic optimum → MCMC warm-start → diagnostics → optional tempering escape hatch.

---

## Stage 1 — Deterministic warm-start

Find a maximum a posteriori (MAP) or maximum likelihood point with `Optimization.jl` driving `SciMLSensitivity` adjoints. This gives the sampler a sensible initial position and surfaces obvious problems (stiff solver failures, unbounded gradients) before you commit to MCMC.

```julia
using OrdinaryDiffEq, Lux, ComponentArrays, Optimization, OptimizationOptimJL, SciMLSensitivity, Zygote, Random

# Define the UDE: known physics + Lux neural correction
nn = Chain(Dense(2, 16, tanh), Dense(16, 2))
ps_init, st = Lux.setup(Random.default_rng(), nn)
ps_ca = ComponentArray(ps_init)             # contiguous, AD-friendly packing

function ude!(du, u, p, t)
    correction, _ = nn(u, p.nn, st)
    du[1] = p.α * u[1] - correction[1]
    du[2] = -p.β * u[2] + correction[2]
end

p0 = ComponentArray(α = 1.0, β = 1.0, nn = ps_ca)
prob = ODEProblem(ude!, u0, tspan, p0)

function loss(p)
    sol = solve(prob, Tsit5(); p = p, saveat = t_obs,
                sensealg = GaussAdjoint(autojacvec = ZygoteVJP()))
    sum(abs2, Array(sol) .- y_obs)
end

opt_prob = OptimizationProblem(OptimizationFunction((p, _) -> loss(p),
                                                    Optimization.AutoZygote()), p0)
map_estimate = solve(opt_prob, BFGS(), maxiters = 500)
```

`ComponentArray` is critical: it gives the optimizer (and later, Turing) a flat parameter vector while preserving the structured `(α, β, nn)` view that the ODE function consumes. See `optimization-patterns` for the full Optimization.jl interface and `sciml-modern-stack` for sensealg selection.

---

> **--mode deep required** for the full Turing model code block below.

## Stage 2 — Turing model with embedded ODE

Wrap the same `prob` in a `Turing.@model`. Use `remake` to inject sampled parameters without rebuilding the problem each step.

```julia
using Turing

@model function bayesian_ude(y_obs, t_obs, prob, p_template)
    # Priors over physical parameters
    α ~ truncated(Normal(1.0, 0.5), 0, Inf)
    β ~ truncated(Normal(1.0, 0.5), 0, Inf)

    # Prior over neural network weights (flat vector)
    nn_dim = length(p_template.nn)
    nn_flat ~ MvNormal(zeros(nn_dim), 0.5 * I)

    # Repack into ComponentArray with the same structure
    p = ComponentArray(α = α, β = β,
                       nn = reshape(nn_flat, axes(p_template.nn)))

    # Solve and condition on observations
    sol = solve(remake(prob; p = p), Tsit5();
                saveat = t_obs,
                sensealg = ForwardDiffSensitivity())
    σ ~ truncated(Normal(0, 0.1), 0, Inf)
    y_obs .~ Normal.(Array(sol), σ)
end

model = bayesian_ude(y_obs, t_obs, prob, p0)
```

**Sensealg selection in one breath**:

- **`ForwardDiff` AD**: bypasses `sensealg` entirely (uses Dual numbers for `u0` and `p`). The safe default for ≤100 parameters and stiff systems. Just pass `Optimization.AutoForwardDiff()` to the optimizer or let Turing pick ForwardDiff — no `sensealg` keyword needed.
- **Reverse-mode AD** (`Zygote`, `Enzyme`, `ReverseDiff`) for many parameters: pair with **`GaussAdjoint()`** — the SciMLSensitivity team's current general recommendation. It is `O(n³ + p)` for stiff/implicit problems vs `O((n+p)³)` for `BacksolveAdjoint`/`InterpolatingAdjoint`, supports checkpointing, and avoids the backwards-solve instability that breaks `BacksolveAdjoint` on stiff systems and DAEs.
- **`InterpolatingAdjoint`** / **`QuadratureAdjoint`**: now niche, useful only in benchmarking scenarios where you can prove an advantage.
- **`BacksolveAdjoint`**: lowest memory but **avoid for stiff systems and DAEs** — backwards-solution accuracy degrades fast.

---

## Stage 3 — Sampler selection

| Situation | Sampler | Notes |
|-----------|---------|-------|
| Posterior is unimodal, NUTS converges from MAP init | `NUTS(0.65)` | Fastest. Initialize with `init_params = vec(map_estimate)`. |
| Suspect multimodality (UDE identifiability, neuron permutations) | Pigeons NRPT | See `consensus-mcmc-pigeons` for the Turing → `TuringLogPotential` wrap |
| Very high dimension, NUTS too slow | SVI as a coarse exploration tool | See `variational-inference-patterns`; not a substitute for full MCMC |

```julia
chain = sample(model, NUTS(0.65), MCMCThreads(), 1000, 4;
               init_params = repeat([vec(map_estimate)], 4))
```

---

## Stage 4 — Diagnostics and posterior predictive

Run the standard battery via `mcmc-diagnostics`: R-hat below 1.01, ESS > 400 per parameter, trace plots without trends, BFMI in healthy range, PSIS-LOO for model comparison.

For UDE-specific checks:

- **Posterior predictive ODE solves** — sample parameters from the chain, solve the ODE, overlay against held-out data. Coverage of credible bands is the primary acceptance test.
- **Neural correction visualization** — plot the learned NN output as a function of state. If credible bands are wider than the physics term itself, the data does not constrain the correction.

---

## AD backend choice

| Backend | Pair with sensealg | Use for | Pitfall |
|---------|--------------------|---------|---------|
| `ForwardDiff` | (none — bypassed) | ≤100 params, stiff systems, Stage 2 Turing sampling | Cost scales linearly with parameter count |
| `Zygote` | `GaussAdjoint()` | High-dim NN weights, Stage 1 MAP | Mutating array operations break it |
| `Enzyme` | `GaussAdjoint()` | High-dim params on stiff or fast solvers | Younger ecosystem; check solver support |
| `ReverseDiff` | `GaussAdjoint()` (compiled tape) | Medium-dim with re-used graphs | No GPU; tape recompiles on shape change |

Mixing AD backends across stages is fine — what matters is that **the sensealg matches the AD direction**. ForwardDiff ignores `sensealg` (it uses Dual numbers); reverse-mode backends respect it. See `julia-ad-backends` for the full compatibility matrix and `sciml-modern-stack` for solver-specific tuning.

---

## Common pitfalls

- **Stiff ODE + NUTS** — stiff solvers (`Rodas5`, `KenCarp4`) propagate small input perturbations into large gradients, which curves the posterior geometry sharply. Tighten solver tolerances (`reltol = 1e-8`) or switch to a non-stiff problem formulation.
- **Initial-condition uncertainty** — if `u0` is also uncertain, sample it as a parameter. Don't fix `u0 = y_obs[1]` and pretend the noise vanishes at `t = 0`.
- **Neuron permutation symmetries** — wide NNs have label-switching modes (swap two neurons → identical likelihood). NUTS will not break this symmetry; either constrain the NN architecture (small width, weight sign constraints) or use Pigeons.
- **Forward-mode adjoint through `Tsit5`** — fine for tens of parameters, prohibitive for thousands. Switch to `InterpolatingAdjoint` + Zygote when the NN dominates the parameter count.

---

## Composition with SINDy

A Bayesian UDE often pays off twice. First, the trained neural correction tells you *where* known physics is wrong. Second, you can extract a *symbolic* form of that correction:

1. Sample many parameter draws from the posterior
2. Evaluate the neural correction on a grid of states for each draw
3. Feed the median correction to `DataDrivenDiffEq` SINDy to recover sparse symbolic terms
4. Refit the symbolic UDE (no NN) and compare posteriors — if narrower, the symbolic form is preferred

See `equation-discovery` for the classical SINDy machinery, or `bayesian-sindy-workflow` when you need credible intervals and inclusion probabilities on the extracted symbolic coefficients (the natural match for a Bayesian UDE pipeline).

---

## Python / JAX counterpart

For a JAX-first Bayesian UDE workflow — Diffrax + Equinox + NumPyro + Optax — see the dedicated **[Bayesian UDE in JAX](../bayesian-ude-jax/SKILL.md)** skill. **Stay in Julia / Turing + DiffEq + Lux** for stiff systems where `GaussAdjoint` matters, when the symbolic MTK pipeline pays off, or when Pigeons NRPT is needed for multimodal posteriors. **Drop to the JAX path** when the surrounding training loop is already JAX, when GPU `vmap`-parallel chain sampling dominates wall-clock, when Optax schedulers / HuggingFace integration matters more than SciML tooling, or when the team's codebase is Python-first.

## Composition with neighboring skills

- **Optimization patterns** — Stage 1 driver. See `optimization-patterns`.
- **SciML modern stack** — sensealg selection, solver options. See `sciml-modern-stack`.
- **Turing model design** — `@model` patterns, priors, hierarchical structure. See `turing-model-design`.
- **Consensus MCMC with Pigeons** — multimodal escape hatch. See `consensus-mcmc-pigeons`.
- **MCMC diagnostics** — Stage 4. See `mcmc-diagnostics`.
- **Equation discovery** — classical SINDy machinery for symbolic extraction from trained UDE residuals. See `equation-discovery`.
- **Bayesian SINDy** — credible intervals and inclusion probabilities on the extracted symbolic coefficients. See `bayesian-sindy-workflow`.
- **Differential equations** — solver selection, stiffness handling. See `differential-equations`.

---

## Checklist

- [ ] Defined the UDE with `ComponentArray`-packed parameters from the start
- [ ] Stage 1: warm-started with `Optimization.jl` to a MAP estimate before MCMC
- [ ] Stage 2: used `remake` inside `@model` (do not rebuild `ODEProblem` each step)
- [ ] Paired `sensealg` with a compatible AD backend; `ForwardDiffSensitivity` is the safe default
- [ ] Stage 3: initialized NUTS at the MAP estimate via `init_params`
- [ ] Switched to Pigeons (`consensus-mcmc-pigeons`) if NUTS R-hat stayed elevated after warm-start
- [ ] Ran posterior predictive ODE solves and checked credible-band coverage on held-out data
- [ ] Visualized the neural correction; verified its credible bands are narrower than the physics term
- [ ] If the correction looks structured, attempted SINDy extraction via `equation-discovery`
- [ ] Documented priors, sensealg, and AD backend choices alongside the chain
