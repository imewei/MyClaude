---
name: turing-model-design
maturity: "5-Expert"
specialization: Bayesian Modeling
description: Design probabilistic models with Turing.jl including prior selection, hierarchical models, and non-centered parameterization. Use when building Bayesian models for inference. Also use when specifying priors, writing @model functions, implementing mixture models, setting up hierarchical/multilevel structures, or reparameterizing for better NUTS sampling. Use proactively when the user mentions Bayesian modeling in Julia, probabilistic programming, or posterior inference with Turing.jl, even if they only describe the statistical model.
---

# Turing.jl Model Design

Probabilistic model specification for Bayesian inference.

## Expert Agent

For complex Bayesian models, hierarchical inference, and probabilistic programming workflows, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia optimization, including Bayesian inference with Turing.jl.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: MCMC sampling (NUTS/HMC), variational inference, hierarchical modeling, and convergence diagnostics.

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

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Weakly informative priors | Regularize without dominating |
| Non-centered parameterization | Better sampling geometry |
| Parameter identifiability | Check correlation structure |
| Generative testing | Simulate from priors |

---

## Checklist

- [ ] Priors match domain knowledge
- [ ] Hierarchical structure appropriate
- [ ] Parameters identifiable
- [ ] Model generates realistic data

---

**Version**: 1.0.6
