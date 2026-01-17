---
name: turing-model-design
version: "1.0.7"
maturity: "5-Expert"
specialization: Bayesian Modeling
description: Design probabilistic models with Turing.jl including prior selection, hierarchical models, and non-centered parameterization. Use when building Bayesian models for inference.
---

# Turing.jl Model Design

Probabilistic model specification for Bayesian inference.

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

**Version**: 1.0.5
