---
name: variational-inference-patterns
description: Master variational inference with Turing.jl ADVI, Bijectors.jl transformations, ELBO monitoring, and VI vs MCMC trade-offs for scalable approximate Bayesian inference. Use when MCMC is too slow for large datasets (.jl files with vi() function), implementing ADVI (Automatic Differentiation Variational Inference), using Bijectors.jl for constrained parameter transformations, monitoring ELBO (Evidence Lower Bound) convergence, comparing VI vs MCMC trade-offs (speed vs accuracy), working with large-scale Bayesian models, implementing stochastic variational inference, sampling from variational approximations, or performing online/streaming Bayesian learning. Essential when MCMC is computationally prohibitive and approximate inference is acceptable.
---

# Variational Inference Patterns

Master variational inference with Turing.jl and ADVI.

## When to use this skill

- MCMC is too slow for large datasets or complex models
- Implementing ADVI (Automatic Differentiation Variational Inference)
- Using Bijectors.jl for constrained parameter transformations
- Monitoring ELBO (Evidence Lower Bound) during optimization
- Comparing VI vs MCMC trade-offs (computational cost vs accuracy)
- Working with large-scale Bayesian models requiring scalability
- Implementing stochastic variational inference (mini-batch training)
- Sampling from variational posterior approximations
- Performing online or streaming Bayesian learning
- Getting approximate posteriors quickly for exploration
- Initializing MCMC with VI posterior as warm start

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
