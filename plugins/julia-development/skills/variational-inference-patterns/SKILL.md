---
name: variational-inference-patterns
description: ADVI with Turing.jl, Bijectors.jl transformations, ELBO monitoring, and VI vs MCMC comparison. Use when MCMC is too slow or for approximate inference.
---

# Variational Inference Patterns

Master variational inference with Turing.jl and ADVI.

## ADVI Pattern
```julia
using Turing, Bijectors

@model function my_model(data)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0, 1), 0, Inf)
    data ~ Normal(μ, σ)
end

model = my_model(data)

# Variational inference
q = vi(model, ADVI(10, 1000))

# Sample from approximation
samples = rand(q, 1000)
```

## When to Use VI
- MCMC too slow
- Need approximate inference quickly
- Large datasets
- Online learning

## Resources
- **Turing VI**: https://turinglang.org/dev/tutorials/9-variationalinference/
