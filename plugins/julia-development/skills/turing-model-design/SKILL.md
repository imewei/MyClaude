---
name: turing-model-design
description: Master Turing.jl probabilistic model specification, prior selection, likelihood definition, hierarchical models, and parameter identifiability for Bayesian inference. Use when designing Bayesian models (.jl files with @model macro), specifying priors with Distributions.jl (Normal, truncated, filldist), defining likelihoods with tilde notation (~), creating hierarchical models with hyperpriors, working with non-centered parameterization for sampling efficiency, ensuring parameter identifiability, designing generative models, building Bayesian neural networks or Gaussian processes, or structuring complex probabilistic workflows. Essential for all Turing.jl Bayesian modeling and foundational for MCMC and variational inference tasks.
---

# Turing Model Design

Master designing probabilistic models with Turing.jl.

## When to use this skill

- Designing Bayesian models with @model macro in Turing.jl
- Specifying prior distributions (Normal, truncated, filldist, MvNormal)
- Defining likelihood functions with tilde notation (~)
- Creating hierarchical models with hyperpriors and group-level effects
- Implementing non-centered parameterization for efficient sampling
- Ensuring parameter identifiability in complex models
- Designing generative models for data
- Building Bayesian neural networks (BNN) or Gaussian processes (GP)
- Working with model priors for regularization
- Structuring probabilistic models for inference
- Integrating with SciML for Bayesian differential equations

## Basic Model Structure
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

## Resources
- **Turing.jl Tutorials**: https://turinglang.org/tutorials/
