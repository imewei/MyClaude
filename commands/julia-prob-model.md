---
description: Build probabilistic models in Julia with Turing.jl and parallelize MCMC with Distributed.jl
category: julia-probabilistic
argument-hint: "[--mcmc] [--variational] [--parallel] [--distributed]"
allowed-tools: "*"
---

# /julia-prob-model

Build probabilistic models in Julia with Turing.jl. Parallelize MCMC with Distributed.jl, analogous to Blackjax in JAX.

## Description

Comprehensive probabilistic programming in Julia using Turing.jl for Bayesian modeling. Includes parallel MCMC sampling with Distributed.jl, variational inference, and integration with Julia's scientific computing ecosystem, providing JAX/Blackjax-like functionality.

## Usage

```
/julia-prob-model [--mcmc] [--variational] [--parallel] [--distributed]
```

## What it does

1. Build probabilistic models with Turing.jl (analogous to Numpyro)
2. Implement MCMC algorithms with parallel sampling
3. Use Distributed.jl for multi-process Bayesian inference
4. Apply variational inference for fast approximate inference
5. Integrate with Julia's ML and scientific computing ecosystem

## Example output

```julia
using Turing
using Distributions
using StatsPlots
using LinearAlgebra
using Random
using Distributed  # For parallel computing
using MCMCChains   # For chain analysis
using StatsFuns    # For statistical functions

# Add worker processes for parallel computing
# addprocs(4)  # Uncomment to add 4 worker processes

# Basic probabilistic model (equivalent to Numpyro model)
@model function linear_regression(X, y, ::Type{T} = Float64) where {T}
    # Priors
    n_features = size(X, 2)
    β ~ MvNormal(zeros(n_features), I)  # Weights prior
    σ ~ InverseGamma(2, 3)              # Noise prior

    # Likelihood
    μ = X * β
    y ~ MvNormal(μ, σ^2 * I)

    return β, σ
end

# Generate synthetic data
Random.seed!(42)
n_samples, n_features = 100, 3
X = randn(n_samples, n_features)
true_β = [1.5, -2.0, 0.5]
true_σ = 0.5
y = X * true_β + true_σ * randn(n_samples)

# Sample from the posterior using MCMC
model = linear_regression(X, y)

# NUTS sampler (similar to JAX/Blackjax NUTS)
sampler = NUTS(0.65)  # Target acceptance rate
chain = sample(model, sampler, 2000; progress=true)

println("MCMC sampling completed")
println("Posterior summary:")
println(describe(chain))

# Hierarchical model example
@model function hierarchical_model(group_idx, y, n_groups, ::Type{T} = Float64) where {T}
    # Hyperpriors
    μ_μ ~ Normal(0, 10)     # Population mean
    σ_μ ~ InverseGamma(2, 3) # Population variance

    # Group-level parameters
    μ ~ filldist(Normal(μ_μ, σ_μ), n_groups)

    # Observation-level variance
    σ ~ InverseGamma(2, 3)

    # Likelihood
    for i in eachindex(y)
        y[i] ~ Normal(μ[group_idx[i]], σ)
    end

    return μ, σ, μ_μ, σ_μ
end

# Generate hierarchical data
n_groups = 5
group_sizes = [20, 25, 30, 15, 22]
group_idx = vcat([fill(i, size) for (i, size) in enumerate(group_sizes)]...)
group_means = randn(n_groups)
y_hierarchical = [randn() + group_means[group_idx[i]] for i in eachindex(group_idx)]

# Sample hierarchical model
hierarchical_model_instance = hierarchical_model(group_idx, y_hierarchical, n_groups)
hierarchical_chain = sample(hierarchical_model_instance, NUTS(), 1500)

println("\\nHierarchical model results:")
println(describe(hierarchical_chain))

# Parallel MCMC sampling (equivalent to JAX distributed sampling)
function parallel_mcmc_sampling(model, n_chains::Int = 4, n_samples::Int = 1000)
    """Run parallel MCMC chains (similar to JAX distributed sampling)"""
    println("Running $n_chains parallel MCMC chains...")

    # Sample multiple chains in parallel
    chains = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)

    println("Parallel sampling completed")

    # Diagnostics
    println("\\nMCMC Diagnostics:")
    println("R-hat values (should be close to 1.0):")

    # Get parameter names
    param_names = names(chains)
    for param in param_names
        if occursin("μ", string(param)) || occursin("β", string(param)) || occursin("σ", string(param))
            rhat_val = rhat(chains[:, param, :])
            println("  $param: $(round(rhat_val, digits=3))")
        end
    end

    return chains
end

# Run parallel sampling
parallel_chains = parallel_mcmc_sampling(model, 4, 1000)

# Variational Inference (fast approximate inference)
using AdvancedVI

@model function vi_linear_regression(X, y)
    """Linear regression model for variational inference"""
    n_features = size(X, 2)

    # Priors
    β ~ MvNormal(zeros(n_features), I)
    σ ~ LogNormal(0, 1)

    # Likelihood
    μ = X * β
    y ~ MvNormal(μ, σ^2 * I)
end

# Variational inference
vi_model = vi_linear_regression(X, y)

# Mean-field approximation
q = meanfield(vi_model)

# Optimize variational parameters
vi_result = vi(vi_model, q, 1000; optimizer=Optimisers.Adam(0.01))

println("\\nVariational Inference completed")
println("ELBO (Evidence Lower BOund): ", vi_result.elbo)

# Sample from variational posterior
vi_samples = rand(vi_result.q, 1000)
println("Variational posterior samples shape: ", size(vi_samples))

# Bayesian Neural Network
@model function bayesian_neural_network(X, y, n_hidden::Int = 10)
    """Bayesian neural network with weight uncertainty"""
    n_input = size(X, 2)
    n_output = 1

    # Weight priors for first layer
    W1 ~ filldist(Normal(0, 1), n_hidden, n_input)
    b1 ~ filldist(Normal(0, 1), n_hidden)

    # Weight priors for output layer
    W2 ~ filldist(Normal(0, 1), n_output, n_hidden)
    b2 ~ filldist(Normal(0, 1), n_output)

    # Noise parameter
    σ ~ LogNormal(0, 1)

    # Forward pass
    h = tanh.(W1 * X' .+ b1)  # Hidden layer
    μ = W2 * h .+ b2          # Output layer

    # Likelihood
    for i in 1:length(y)
        y[i] ~ Normal(μ[1, i], σ)
    end
end

# Train Bayesian neural network
bnn_model = bayesian_neural_network(X, y, 5)
bnn_chain = sample(bnn_model, NUTS(), 1000)

println("\\nBayesian Neural Network sampling completed")

# Gaussian Process with Turing
using AbstractGPs
using KernelFunctions

@model function gaussian_process_regression(X, y, kernel_func)
    """Gaussian Process regression model"""
    # Hyperparameters
    σ_f ~ LogNormal(0, 1)  # Signal variance
    ℓ ~ LogNormal(0, 1)    # Length scale
    σ_n ~ LogNormal(-2, 1) # Noise variance

    # Construct kernel
    kernel = σ_f^2 * transform(kernel_func, 1/ℓ)

    # GP prior
    gp = GP(kernel)
    fx = gp(X, σ_n^2)

    # Likelihood
    y ~ fx
end

# Example with RBF kernel
X_gp = collect(range(-3, 3, length=50))
rbf_kernel = SqExponentialKernel()

# Generate GP data
true_gp = GP(0.5^2 * transform(rbf_kernel, 1/0.8))
y_gp = rand(true_gp(X_gp, 0.1^2))

# Sample GP hyperparameters
gp_model = gaussian_process_regression(X_gp, y_gp, rbf_kernel)
gp_chain = sample(gp_model, NUTS(), 500)

println("\\nGaussian Process regression completed")

# Time series modeling
@model function ar_model(y, p::Int)
    """Autoregressive model of order p"""
    n = length(y)

    # AR coefficients
    φ ~ filldist(Normal(0, 0.5), p)

    # Noise variance
    σ ~ InverseGamma(2, 3)

    # Initial values
    y_init ~ MvNormal(zeros(p), I)

    # AR process
    for t in (p+1):n
        μ = sum(φ[i] * y[t-i] for i in 1:p)
        y[t] ~ Normal(μ, σ)
    end
end

# Generate AR data
true_φ = [0.6, -0.3]
y_ar = cumsum(randn(100)) + 0.1 * randn(100)  # Random walk with noise

ar_model_instance = ar_model(y_ar, 2)
ar_chain = sample(ar_model_instance, NUTS(), 1000)

println("\\nAutoregressive model completed")

# Model comparison using WAIC/LOO
using MLJBase
using ParetoSmooth

function model_comparison(models_dict, data)
    """Compare models using information criteria"""
    println("\\n=== Model Comparison ===")

    results = Dict()

    for (name, (model_func, chain)) in models_dict
        # Calculate WAIC (Watanabe-Akaike Information Criterion)
        try
            # Extract log likelihood
            logl = pointwise_loglikelihoods(model_func, chain)
            waic_result = waic(logl)

            results[name] = Dict(
                "waic" => waic_result.waic,
                "se" => waic_result.se_waic,
                "elpd" => waic_result.elpd_waic
            )

            println("$name:")
            println("  WAIC: $(round(waic_result.waic, digits=2)) ± $(round(waic_result.se_waic, digits=2))")
        catch e
            println("$name: WAIC calculation failed")
            results[name] = Dict("error" => string(e))
        end
    end

    return results
end

# Distributed computing setup
function setup_distributed_sampling()
    """Setup distributed computing for large-scale Bayesian inference"""
    println("\\n=== Distributed Computing Setup ===")

    # Check number of workers
    n_workers = nworkers()
    println("Number of worker processes: $n_workers")

    if n_workers == 1
        println("To enable distributed computing, run:")
        println("using Distributed")
        println("addprocs(4)  # Add 4 worker processes")
        println("@everywhere using Turing")
    end

    # Example distributed sampling function
    function distributed_mcmc(model, n_chains_per_worker::Int = 2)
        """Run MCMC across multiple workers"""
        total_chains = n_workers * n_chains_per_worker

        println("Running $total_chains chains across $n_workers workers...")

        # This would run the actual distributed sampling
        # chains = sample(model, NUTS(), MCMCDistributed(), 1000, total_chains)

        println("Distributed sampling would be executed here")
        return nothing
    end

    return distributed_mcmc
end

distributed_mcmc_func = setup_distributed_sampling()

# Performance optimization for MCMC
function optimize_mcmc_performance()
    """Tips for optimizing MCMC performance in Julia"""
    println("""
=== MCMC Performance Optimization ===

1. Model Parameterization:
   ✓ Use non-centered parameterizations for hierarchical models
   ✓ Apply transformations to improve geometry (e.g., log for positive parameters)
   ✓ Use vectorized operations when possible

2. Sampler Configuration:
   ✓ Tune target acceptance rate (0.6-0.8 for NUTS)
   ✓ Adjust maximum tree depth for complex posteriors
   ✓ Use adaptive warmup for automatic tuning

3. Computational Efficiency:
   ✓ Use type-stable model definitions
   ✓ Avoid allocations in likelihood computations
   ✓ Pre-compute constant terms outside the model

4. Parallel/Distributed Computing:
   ✓ Use MCMCThreads() for multi-threading
   ✓ Use MCMCDistributed() for multi-process sampling
   ✓ Consider GPU acceleration for large models

5. Diagnostics and Monitoring:
   ✓ Check R-hat values for convergence
   ✓ Monitor effective sample size (ESS)
   ✓ Use trace plots for visual diagnostics
   ✓ Compute WAIC/LOO for model comparison
    """)
end

optimize_mcmc_performance()

# JAX/Blackjax vs Turing.jl comparison
function jax_vs_turing_comparison()
    """Compare JAX/Blackjax with Turing.jl"""
    println("""
=== JAX/Blackjax vs Turing.jl Comparison ===

Feature                 | JAX/Blackjax              | Turing.jl
------------------------|----------------------------|---------------------------
Model Definition        | Python functions           | @model macro
Automatic Differentiation | JAX AD                   | Zygote.jl/ForwardDiff.jl
MCMC Algorithms         | NUTS, HMC, etc.          | NUTS, HMC, Gibbs, etc.
Parallel Sampling       | jax.pmap, jax.vmap       | MCMCThreads, MCMCDistributed
GPU Support             | Native JAX                | CuArrays.jl integration
Variational Inference   | JAX + Optax               | AdvancedVI.jl
Model Compilation       | XLA JIT                   | Julia's type inference
Ecosystem Integration   | NumPyro, TensorFlow Prob. | StatsPlots, MLJ, Flux.jl

Turing.jl Advantages:
✓ More expressive probabilistic programming language
✓ Better integration with Julia ecosystem
✓ Extensive MCMC algorithm library
✓ Excellent diagnostics and visualization tools

JAX/Blackjax Advantages:
✓ Faster compilation and execution
✓ Better GPU/TPU support
✓ More mature AD system
✓ Easier deployment in production
    """)
end

jax_vs_turing_comparison()

# Complete workflow demonstration
function demonstrate_bayesian_workflow()
    """Demonstrate complete Bayesian modeling workflow"""
    println("\\n=== Complete Bayesian Workflow Demonstration ===")

    # 1. Model definition
    println("1. Defining probabilistic model...")

    @model function demo_model(x, y)
        # Prior
        α ~ Normal(0, 1)
        β ~ Normal(0, 1)
        σ ~ InverseGamma(2, 3)

        # Likelihood
        μ = α .+ β .* x
        y ~ MvNormal(μ, σ^2 * I)
    end

    # 2. Generate data
    println("2. Generating synthetic data...")
    x_demo = randn(50)
    y_demo = 2.0 .+ 1.5 .* x_demo .+ 0.3 .* randn(50)

    # 3. MCMC sampling
    println("3. Running MCMC sampling...")
    demo_model_instance = demo_model(x_demo, y_demo)
    demo_chain = sample(demo_model_instance, NUTS(), 1000)

    # 4. Diagnostics
    println("4. Checking convergence diagnostics...")
    println("α R-hat: ", rhat(demo_chain[:α]))
    println("β R-hat: ", rhat(demo_chain[:β]))
    println("σ R-hat: ", rhat(demo_chain[:σ]))

    # 5. Posterior analysis
    println("5. Posterior analysis:")
    α_samples = Array(demo_chain[:α])
    β_samples = Array(demo_chain[:β])

    println("α posterior mean: $(round(mean(α_samples), digits=3))")
    println("β posterior mean: $(round(mean(β_samples), digits=3))")

    # 6. Posterior predictive checking
    println("6. Posterior predictive sampling...")
    n_pred_samples = 100
    x_new = randn(20)

    # Sample from posterior predictive
    pred_samples = []
    for i in 1:n_pred_samples
        idx = rand(1:length(α_samples))
        α_sample = α_samples[idx]
        β_sample = β_samples[idx]
        σ_sample = Array(demo_chain[:σ])[idx]

        μ_pred = α_sample .+ β_sample .* x_new
        y_pred = rand(MvNormal(μ_pred, σ_sample^2 * I))
        push!(pred_samples, y_pred)
    end

    println("Posterior predictive samples generated: $(length(pred_samples))")

    return demo_chain, pred_samples
end

# Run complete demonstration
demo_results = demonstrate_bayesian_workflow()

# Final summary
println("""
\\n=== Julia Probabilistic Programming Summary ===

You now have a complete toolkit for Bayesian modeling in Julia:

✓ Basic and hierarchical probabilistic models
✓ MCMC sampling with NUTS and parallel chains
✓ Variational inference for fast approximations
✓ Bayesian neural networks and Gaussian processes
✓ Time series modeling with autoregressive models
✓ Model comparison using information criteria
✓ Distributed computing setup for large-scale inference
✓ Performance optimization guidelines

Next steps:
- Explore more complex models in your domain
- Set up distributed computing for large datasets
- Integrate with Julia's ML ecosystem (Flux.jl, MLJ.jl)
- Consider GPU acceleration for computational bottlenecks
""")
```

## Related Commands

- `/julia-ad-grad` - Use AD for gradient-based MCMC samplers
- `/julia-jit-like` - Optimize probabilistic models for performance
- `/jax-numpyro-prob` - Compare with JAX probabilistic programming
- `/python-debug-prof` - Profile Bayesian computation performance