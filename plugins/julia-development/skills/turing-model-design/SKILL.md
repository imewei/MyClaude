---
name: turing-model-design
description: Turing.jl model specification, prior selection, likelihood definition, hierarchical models, and identifiability. Use for designing probabilistic models and specifying Bayesian workflows.
---

# Turing Model Design

Master designing probabilistic models with Turing.jl.

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
