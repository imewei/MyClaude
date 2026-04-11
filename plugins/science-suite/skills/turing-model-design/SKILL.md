---
name: turing-model-design
maturity: "5-Expert"
specialization: Bayesian Modeling
description: Design probabilistic models with Turing.jl including prior selection, hierarchical models, and non-centered parameterization. Use when building Bayesian models for inference. Also use when specifying priors, writing @model functions, implementing mixture models, setting up hierarchical/multilevel structures, or reparameterizing for better NUTS sampling. Use proactively when the user mentions Bayesian modeling in Julia, probabilistic programming, or posterior inference with Turing.jl, even if they only describe the statistical model.
---

# Turing.jl Model Design

Probabilistic model specification for Bayesian inference.

## Expert Agents

For complex Bayesian models, hierarchical inference, and probabilistic programming workflows, delegate to:

- **`julia-pro`** (primary): Julia + Turing.jl specialist.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: MCMC sampling (NUTS/HMC), variational inference, hierarchical modeling, convergence diagnostics.
- **`statistical-physicist`** (secondary): Bayesian inference theory, identifiability, MCMC geometry.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`

---

## Basic Model

```julia
using Turing, Distributions

@model function my_model(data)
    # Priors
    μ ~ Normal(0, 10)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # Likelihood
    for i in eachindex(data)
        data[i] ~ Normal(μ, σ)
    end
end
```

---

## Hierarchical Model

```julia
@model function hierarchical_model(y, groups)
    # Hyperpriors
    μ_global ~ Normal(0, 10)
    σ_global ~ truncated(Normal(0, 5), 0, Inf)

    # Group-level parameters
    n_groups = length(unique(groups))
    μ_group ~ filldist(Normal(μ_global, σ_global), n_groups)

    # Likelihood
    for i in eachindex(y)
        y[i] ~ Normal(μ_group[groups[i]], 1.0)
    end
end
```

---

## Embedding a DiffEq solve

Turing models can call any deterministic function — including an ODE solver. Use `remake` to inject sampled parameters into a pre-built `ODEProblem` so the solver structure is constructed once:

```julia
using Turing, OrdinaryDiffEq, SciMLSensitivity

@model function ode_inference(y_obs, t_obs, prob)
    α ~ truncated(Normal(1.0, 0.5), 0, Inf)
    β ~ truncated(Normal(1.0, 0.5), 0, Inf)
    σ ~ truncated(Normal(0.0, 0.1), 0, Inf)

    sol = solve(remake(prob; p = [α, β]), Tsit5();
                saveat = t_obs,
                sensealg = ForwardDiffSensitivity())
    y_obs .~ Normal.(Array(sol), σ)
end
```

**Sensealg vs AD backend**: when Turing's AD backend is `ForwardDiff` (the default for many models), the `sensealg` keyword is **bypassed entirely** — DiffEq solves with Dual numbers and ignores any sensealg you pass. If you switch Turing to a reverse-mode backend (Zygote, ReverseDiff, Enzyme), pair it with `GaussAdjoint()` — the current SciMLSensitivity default. For full neural-physics workflows including AD backend choice, see `bayesian-ude-workflow`.

## Embedding a Lux neural network

Wide Lux networks have many parameters; pack them in a `ComponentArray` so Turing sees a single named parameter object:

```julia
using Lux, ComponentArrays, Random

nn = Chain(Dense(2, 16, tanh), Dense(16, 2))
ps_init, st = Lux.setup(Random.default_rng(), nn)

@model function nn_regression(x, y, st)
    nn_dim = length(ComponentArray(ps_init))
    θ ~ MvNormal(zeros(nn_dim), I)                     # flat prior on weights
    p = ComponentArray(θ, getaxes(ComponentArray(ps_init)))
    pred, _ = nn(x, p, st)
    y .~ Normal.(vec(pred), 0.1)
end
```

`getaxes` preserves the structured `(layer_1.weight, layer_1.bias, ...)` view so the same flat vector can be repacked deterministically. This is the parameter-packing pattern that powers Bayesian UDEs.

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Weakly informative priors | Regularize without dominating |
| Non-centered parameterization | Better sampling geometry for hierarchies |
| Parameter identifiability | Check posterior correlation structure |
| Generative (prior) predictive testing | Simulate from priors before fitting |
| Multimodal escape hatch | Wrap as `TuringLogPotential`, sample with Pigeons (`consensus-mcmc-pigeons`) |
| Initialization | Warm-start NUTS at a MAP estimate via `init_params` |

## Composition with neighboring skills

- **MCMC diagnostics** — R-hat, ESS, BFMI, ArviZ post-processing. See `mcmc-diagnostics`.
- **Variational inference patterns** — VI alternative for high-dimensional posteriors. See `variational-inference-patterns`.
- **Bayesian UDE workflow** — full hybrid physics + neural correction recipe. See `bayesian-ude-workflow`.
- **Consensus MCMC with Pigeons** — multimodal posteriors where NUTS fails. See `consensus-mcmc-pigeons`.

## Checklist

- [ ] Priors match domain knowledge and were checked via prior-predictive simulation
- [ ] Hierarchical structure appropriate; non-centered where the funnel geometry demands
- [ ] Parameters identifiable (checked posterior pair plots for ridges)
- [ ] If embedding a DiffEq solve: used `remake` and `ForwardDiffSensitivity`
- [ ] If embedding a Lux NN: packed parameters via `ComponentArray` with `getaxes`
- [ ] Ran convergence diagnostics before interpreting results
- [ ] Escalated to tempering if the sampler failed to mix
