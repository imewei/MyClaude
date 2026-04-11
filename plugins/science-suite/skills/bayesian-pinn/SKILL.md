---
name: bayesian-pinn
description: Put posterior uncertainty on PINN / neural-ODE solutions using NeuralPDE.jl's BNNODE (ODEs) and BayesianPINN (PDEs) discretizers. Use when you need credible intervals on a learned PDE solution, an inverse problem with uncertainty on the physical parameters, or HMC sampling over neural-network weights driving a physics residual. Use proactively when the user mentions BPINN, BNNODE, Bayesian PINN, Bayesian neural ODE with NeuralPDE, or wants uncertainty on a neural surrogate of a known ODE/PDE.
---

# Bayesian PINN — BNNODE & BayesianPINN

NeuralPDE.jl ships its own Bayesian engine built directly on `AdvancedHMC` and `MCMCChains`, **independent of Turing** (no `@model` block). Two entry points depending on whether the target equation is an ODE or a PDE.

## Expert Agent

For Bayesian PINN workflows, delegate to:

- **`julia-pro`**: Julia SciML ecosystem, NeuralPDE.jl, AdvancedHMC patterns.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
- **`statistical-physicist`** (secondary): Posterior geometry, prior elicitation for NN weights, MCMC diagnostics.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`

---

## When to reach for BPINN vs neighboring Bayesian options

| Question | Use |
|----------|-----|
| Purely neural surrogate of a known ODE/PDE, need credible intervals on the solution? | **BNNODE / BayesianPINN** (this skill) |
| Known-physics + neural-network-correction hybrid (UDE)? | `bayesian-ude-workflow` (Turing `@model` with embedded `solve`) |
| Inverse problem: unknown PDE coefficients, fit from data with uncertainty? | BNNODE with `param = [prior, ...]` — this skill |
| Multimodal BPINN posterior that NUTS cannot escape? | Wrap the NeuralPDE log-density via `LogDensityProblems` and feed to `consensus-mcmc-pigeons` |

The split from the vanilla `neural-pde` skill is **deterministic PINN vs. Bayesian PINN**. If you just want a point estimate of the PDE solution, stay in `neural-pde`.

---

## ODEs — `BNNODE` solver

The high-level interface: pass `BNNODE` as the algorithm to `solve(prob, alg)`.

```julia
using NeuralPDE, Lux, AdvancedHMC, OrdinaryDiffEq

# Out-of-place form is REQUIRED — `du = f(u, p, t)`, not `f!(du, u, p, t)`
function lotka(u, p, t)
    [p[1] * u[1] - p[2] * u[1] * u[2],
     p[3] * u[1] * u[2] - p[4] * u[2]]
end
prob = ODEProblem(lotka, u0, tspan, p_true)

chain = Chain(Dense(1, 8, tanh), Dense(8, 8, tanh), Dense(8, 2))

alg = BNNODE(
    chain;
    dataset       = [t_obs, u_obs],          # for inverse problems / data fitting
    draw_samples  = 1500,
    priorsNNw     = (0.0, 2.0),              # (mean, std) of Gaussian weight prior — default
    phystd        = [0.05, 0.05],            # std of physics residual likelihood
    l2std         = [0.05, 0.05],            # std of data-fit likelihood
    param         = [Normal(1.0, 2.0)],      # priors on physical parameters (inverse)
    Adaptorkwargs = (Integrator = Leapfrog,),
    progress      = true,
)

sol = solve(prob, alg)
```

The lower-level function `ahmc_bayesian_pinn_ode(prob, chain; ...)` exposes the same kwargs for callers that want to drive HMC directly without going through the SciML solver interface.

> **Critical constraint**: BPINN ODE solvers only accept the **out-of-place** form `du = f(u, p, t)`. The mutating in-place form `f!(du, u, p, t)` will silently misbehave or error. Convert in-place ODEs before passing them to `BNNODE`.

---

## PDEs — `BayesianPINN` discretizer

For full PDE systems (built with `@parameters` / `@variables` / `Differential` — see `neural-pde` for the symbolic-spec layer), use the `BayesianPINN` discretizer instead of `PhysicsInformedNN`:

```julia
using NeuralPDE, Lux, AdvancedHMC

discretization = BayesianPINN(
    chain,
    GridTraining(0.1);
    Kernel  = HMC(0.05, 4),                   # step size, leapfrog steps
    dataset = [u_obs, t_obs],
)

sol = solve(
    pde_system, discretization;
    draw_samples = 1500,
    priorsNNw    = (0.0, 2.0),
)
```

The result of either path (`BNNODE` or `BayesianPINN`) is an object whose `sol.original` field carries the underlying `MCMCChains.Chains`, fully compatible with `mcmc-diagnostics` (R-hat, ESS, ArviZ).

---

## Prior choice for NN weights

The weight prior `priorsNNw = (μ, σ)` directly controls posterior identifiability. Defaults (`0.0, 2.0`) are fine for small networks; for wider networks, **tighten** the prior — wide priors on many weights produce multimodal posteriors that NUTS cannot escape.

| Network width | Recommended σ | Rationale |
|---------------|---------------|-----------|
| ≤ 16 units/layer | 2.0 | Default; data informs the posterior |
| 32–64 units/layer | 1.0 | Reduce neuron-permutation symmetry basin count |
| > 64 units/layer | 0.5 + label-symmetry constraints | Otherwise NUTS almost certainly fails to mix — reach for Pigeons |

---

## Multimodal escape hatch

BPINN posteriors are notoriously multimodal — neural-network weight symmetries (neuron permutation, sign flips) create disjoint basins that NUTS bridges poorly. Symptoms: high per-chain ESS but R-hat > 1.01 across chains.

Wrap the NeuralPDE log-density via `LogDensityProblems` and feed it to `Pigeons.jl`:

```julia
using NeuralPDE, Pigeons, LogDensityProblems

# NeuralPDE gives you a log-density callable via the discretizer internals;
# you can wrap it in a struct implementing the LogDensityProblems interface:
struct BPINNTarget{D}
    discretization::D
    pde_system::Any
end
LogDensityProblems.logdensity(t::BPINNTarget, x) = # (discretizer-specific)
LogDensityProblems.dimension(t::BPINNTarget)     = # (weight count)
LogDensityProblems.capabilities(::Type{<:BPINNTarget}) = LogDensityProblems.LogDensityOrder{0}()

pt = pigeons(
    target      = BPINNTarget(discretization, pde_system),
    n_chains    = 10,
    n_rounds    = 10,
    variational = GaussianReference(),    # critical for NN weight posteriors
    seed        = 1,
)
```

See `consensus-mcmc-pigeons` for the full NRPT pattern and Pigeons tuning.

---

## Diagnostics

Post-process `sol.original` (a `MCMCChains.Chains`) via `mcmc-diagnostics`:

- R-hat < 1.01 on all NN weight + physical parameter chains
- ESS > 400 per parameter
- Trace plots without trends
- Posterior-predictive PDE solves: sample from the chain, run `solve` with each draw, overlay against held-out data

For inverse problems, the physical parameters (`p`) should have narrow posteriors with bands narrower than the priors — if not, the data is not identifying them. Widen the data set or tighten the prior.

---

## Composition with neighboring skills

- **Neural PDE** — deterministic PINN for the same target, discretization strategies, MOL alternative. See `neural-pde`.
- **Modeling toolkit** — symbolic PDE primitives (`@parameters`, `@variables`, `Differential`). See `modeling-toolkit`.
- **Bayesian UDE workflow** — when the target is known-physics + NN correction, not pure neural surrogate. See `bayesian-ude-workflow`.
- **Consensus MCMC with Pigeons** — multimodal escape hatch for BPINN weight posteriors. See `consensus-mcmc-pigeons`.
- **MCMC diagnostics** — post-processing `sol.original` chains. See `mcmc-diagnostics`.
- **Bayesian inference hub** — parent routing. See `bayesian-inference`.

---

## Checklist

- [ ] Converted any in-place `f!(du, u, p, t)` ODEs to out-of-place before passing to `BNNODE`
- [ ] Picked `priorsNNw` deliberately based on network width (not left at default for wide nets)
- [ ] For inverse problems: defined priors on physical parameters via `param = [...]`
- [ ] Ran R-hat / ESS on `sol.original` via `mcmc-diagnostics`
- [ ] Compared the Bayesian PINN against a deterministic PINN (via `neural-pde`) for sanity
- [ ] If R-hat stayed elevated, escalated to Pigeons via `consensus-mcmc-pigeons`
- [ ] Verified posterior predictive coverage against held-out data
