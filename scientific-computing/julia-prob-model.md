---
description: Build probabilistic models in Julia with Turing.jl and parallelize MCMC with Distributed.jl using intelligent 23-agent orchestration
category: julia-probabilistic
argument-hint: "[--mcmc] [--variational] [--parallel] [--distributed] [--agents=auto|julia|scientific|ai|bayesian|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--research]"
allowed-tools: "*"
model: inherit
---

# Julia Probabilistic Programming with Turing.jl

Build probabilistic models in Julia with Turing.jl and parallelize MCMC with Distributed.jl. Provides comprehensive Bayesian modeling capabilities analogous to JAX/Numpyro ecosystem.

## Usage

```bash
/julia-prob-model [--mcmc] [--variational] [--parallel] [--distributed] [--agents=auto|julia|scientific|ai|bayesian|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--research]
```

Builds probabilistic models in Julia with intelligent 23-agent optimization for Turing.jl and Bayesian computation.

## Arguments

- `--mcmc`: Include MCMC algorithms (NUTS, HMC, Gibbs sampling)
- `--variational`: Add variational inference (ADVI, mean-field)
- `--parallel`: Enable parallel sampling across multiple chains
- `--distributed`: Use Distributed.jl for multi-process computation
- `--agents=<agents>`: Agent selection (auto, julia, scientific, ai, bayesian, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with Bayesian intelligence
- `--intelligent`: Enable intelligent agent selection based on probabilistic modeling analysis
- `--breakthrough`: Enable breakthrough Bayesian modeling optimization
- `--optimize`: Apply performance optimization to probabilistic computations
- `--research`: Enable research-grade probabilistic modeling with agent coordination

## What it does

1. Build probabilistic models with Turing.jl (analogous to Numpyro)
2. Implement MCMC algorithms with parallel sampling
3. Use Distributed.jl for multi-process Bayesian inference
4. Apply variational inference for fast approximate inference
5. Integrate with Julia's ML and scientific computing ecosystem
6. **23-Agent Bayesian Intelligence**: Multi-agent collaboration for optimal probabilistic modeling
7. **Advanced MCMC Optimization**: Agent-driven MCMC sampling and convergence optimization
8. **Intelligent Research Integration**: Agent-coordinated research-grade Bayesian computation

## 23-Agent Intelligent Bayesian Modeling System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes probabilistic modeling requirements, Bayesian computation patterns, and research goals to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Bayesian Modeling Pattern Detection → Agent Selection
- Research Projects → research-intelligence-master + bayesian-stats-expert + julia-pro
- Production ML → ai-systems-architect + julia-pro + neural-networks-master
- Scientific Simulation → scientific-computing-master + julia-pro + bayesian-stats-expert
- Statistical Analysis → stats-analysis-expert + bayesian-stats-expert + julia-pro
- High-Performance MCMC → optimization-master + julia-pro + bayesian-stats-expert
```

### Core Julia Bayesian Agents

#### **`julia-pro`** - Julia Ecosystem Bayesian Expert
- **Julia Bayesian Optimization**: Deep expertise in Turing.jl, MCMCChains.jl, and Julia probabilistic ecosystem
- **MCMC Performance**: Julia MCMC sampling optimization and convergence analysis
- **Distributed Computing**: Julia distributed Bayesian computation with Distributed.jl
- **Package Integration**: Julia Bayesian package ecosystem coordination and optimization
- **Type System Integration**: Julia type system optimization for probabilistic programming

#### **`bayesian-stats-expert`** - Advanced Bayesian Statistics
- **Bayesian Methodology**: Advanced Bayesian statistical methods and model design
- **MCMC Diagnostics**: Convergence diagnostics, chain analysis, and sampling optimization
- **Model Selection**: Bayesian model comparison, WAIC, LOO, and information criteria
- **Hierarchical Modeling**: Multi-level Bayesian models and random effects
- **Posterior Analysis**: Posterior inference, uncertainty quantification, and predictive modeling

#### **`research-intelligence-master`** - Research-Grade Bayesian Computing
- **Scientific Research**: Academic and research-grade probabilistic modeling
- **Methodological Innovation**: Advanced Bayesian methods and novel approaches
- **Publication Quality**: Research-grade Bayesian analysis and documentation
- **Cross-Domain Application**: Bayesian methods across scientific domains
- **Theoretical Foundation**: Mathematical foundation and theoretical development

#### **`optimization-master`** - Bayesian Computation Optimization
- **MCMC Performance**: High-performance MCMC sampling and computational efficiency
- **Variational Inference**: Advanced variational inference optimization
- **Distributed Computing**: Large-scale distributed Bayesian computation
- **Memory Management**: Memory-efficient probabilistic computation strategies
- **Numerical Stability**: Numerical stability analysis for Bayesian computation

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Bayesian Modeling
Automatically analyzes Bayesian modeling requirements and selects optimal agent combinations:
- **Model Complexity Analysis**: Detects model complexity and computational requirements
- **Research Assessment**: Evaluates research goals and methodological needs
- **Agent Matching**: Maps Bayesian modeling needs to relevant agent expertise
- **Performance Balance**: Balances computational efficiency with statistical rigor

#### **`julia`** - Julia-Specialized Bayesian Team
- `julia-pro` (Julia Bayesian ecosystem lead)
- `bayesian-stats-expert` (statistical methodology)
- `optimization-master` (performance optimization)
- `neural-networks-master` (Bayesian neural networks)

#### **`scientific`** - Scientific Computing Bayesian Team
- `scientific-computing-master` (lead)
- `julia-pro` (Julia implementation)
- `bayesian-stats-expert` (statistical methods)
- `research-intelligence-master` (research methodology)

#### **`ai`** - AI/ML Bayesian Team
- `neural-networks-master` (lead)
- `julia-pro` (Julia optimization)
- `bayesian-stats-expert` (Bayesian ML)
- `ai-systems-architect` (production systems)

#### **`bayesian`** - Bayesian-Specialized Team
- `bayesian-stats-expert` (lead)
- `julia-pro` (Julia implementation)
- `research-intelligence-master` (research methods)
- `optimization-master` (computational efficiency)

#### **`all`** - Complete 23-Agent Bayesian Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough Bayesian modeling.

### Advanced 23-Agent Bayesian Examples

```bash
# Intelligent auto-selection for Bayesian modeling
/julia-prob-model --agents=auto --intelligent --optimize

# Scientific research with specialized agents
/julia-prob-model --agents=scientific --breakthrough --orchestrate --research

# High-performance MCMC computation
/julia-prob-model --agents=bayesian --optimize --parallel --distributed

# AI/ML Bayesian development
/julia-prob-model --agents=ai --breakthrough --orchestrate --variational

# Research-grade Bayesian analysis
/julia-prob-model --agents=all --breakthrough --research --intelligent

# Complete 23-agent Bayesian ecosystem
/julia-prob-model --agents=all --orchestrate --breakthrough --intelligent --optimize
```

## Basic Probabilistic Model Setup

```julia
using Turing
using Distributions
using StatsPlots
using LinearAlgebra
using Random
using MCMCChains
using StatsFuns

# Basic Bayesian linear regression
@model function linear_regression(x, y)
    # Priors
    α ~ Normal(0, 10)     # Intercept
    β ~ Normal(0, 10)     # Slope
    σ ~ InverseGamma(2, 3) # Noise standard deviation

    # Likelihood
    for i in eachindex(y)
        y[i] ~ Normal(α + β * x[i], σ)
    end
end

# Generate synthetic data
Random.seed!(123)
x = randn(100)
y_true = 2.0 .+ 3.0 * x + 0.5 * randn(100)

# Sample from posterior
model = linear_regression(x, y_true)
chain = sample(model, NUTS(), 2000)

# Extract parameters
α_samples = chain[:α]
β_samples = chain[:β]
σ_samples = chain[:σ]
```

## MCMC Algorithms

```julia
using AdvancedHMC

# NUTS (No-U-Turn Sampler) - default and recommended
chain_nuts = sample(model, NUTS(0.65), 2000)

# Hamiltonian Monte Carlo with custom parameters
chain_hmc = sample(model, HMC(0.01, 10), 2000)

# Metropolis-Hastings for discrete parameters
@model function discrete_model()
    p ~ Uniform(0, 1)
    k ~ Binomial(10, p)
end

discrete_mod = discrete_model()
chain_mh = sample(discrete_mod, MH(), 5000)

# Gibbs sampling for mixed discrete/continuous
@model function mixed_model(y)
    # Continuous parameter
    μ ~ Normal(0, 1)

    # Discrete parameter (mixture component)
    z ~ Categorical([0.3, 0.7])

    # Likelihood depends on discrete parameter
    if z == 1
        y ~ Normal(μ, 1)
    else
        y ~ Normal(μ + 2, 1)
    end
end

mixed_mod = mixed_model([1.2])
chain_gibbs = sample(mixed_mod, Gibbs(HMC(0.01, 5, :μ), PG(20, :z)), 2000)
```

## Parallel MCMC Sampling

```julia
using Distributed

# Add worker processes
addprocs(4)
@everywhere using Turing

# Parallel sampling across multiple chains
@model function parallel_model(x, y)
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)
    σ ~ InverseGamma(2, 3)

    y .~ Normal.(α .+ β .* x, σ)
end

# Sample multiple chains in parallel
parallel_mod = parallel_model(x, y_true)
chains = sample(parallel_mod, NUTS(), MCMCThreads(), 2000, 4)

# Alternative: use multiple processes
chains_proc = sample(parallel_mod, NUTS(), MCMCDistributed(), 2000, 4)

# Combine and analyze chains
combined_chain = chainscat(chains...)
plot(combined_chain)
```

## Distributed Computing Setup

```julia
using Distributed
using ClusterManagers

# Setup for SLURM cluster
addprocs(SlurmManager(10))

# Setup for multi-machine cluster
machines = [("worker1", 4), ("worker2", 4)]
addprocs(machines)

@everywhere begin
    using Turing
    using Distributions
    using LinearAlgebra
end

# Distributed data and model
@everywhere function distributed_regression(local_x, local_y)
    @model function local_model(x, y)
        α ~ Normal(0, 5)
        β ~ Normal(0, 5)
        σ ~ InverseGamma(2, 3)

        for i in eachindex(y)
            y[i] ~ Normal(α + β * x[i], σ)
        end
    end

    model = local_model(local_x, local_y)
    return sample(model, NUTS(), 1000)
end

# Distribute data across workers
n_per_worker = 100
distributed_chains = @distributed (vcat) for i in 1:nworkers()
    local_x = randn(n_per_worker)
    local_y = 2.0 .+ 3.0 * local_x + 0.5 * randn(n_per_worker)
    [distributed_regression(local_x, local_y)]
end
```

## Variational Inference

```julia
using AdvancedVI

# Automatic Differentiation Variational Inference (ADVI)
@model function vi_model(x, y)
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)
    σ ~ LogNormal(0, 1)

    y .~ Normal.(α .+ β .* x, σ)
end

vi_mod = vi_model(x, y_true)

# Mean-field variational inference
vi_result = vi(vi_mod, ADVI(10, 5000))

# Extract approximate posterior
posterior_samples = rand(vi_result, 2000)

# Normalizing flows for more flexible posteriors
using Bijectors

# Planar flow
flow = PlanarFlow(3)  # 3 parameters: α, β, log(σ)
vi_flow = vi(vi_mod, ADVI(10, 5000, flow))
```

## Hierarchical Models

```julia
# Hierarchical linear regression
@model function hierarchical_regression(x, y, groups)
    n_groups = maximum(groups)

    # Population-level parameters
    μ_α ~ Normal(0, 5)
    σ_α ~ InverseGamma(2, 3)
    μ_β ~ Normal(0, 5)
    σ_β ~ InverseGamma(2, 3)

    # Group-level parameters
    α = Vector{Float64}(undef, n_groups)
    β = Vector{Float64}(undef, n_groups)

    for j in 1:n_groups
        α[j] ~ Normal(μ_α, σ_α)
        β[j] ~ Normal(μ_β, σ_β)
    end

    # Observation noise
    σ ~ InverseGamma(2, 3)

    # Likelihood
    for i in eachindex(y)
        group = groups[i]
        y[i] ~ Normal(α[group] + β[group] * x[i], σ)
    end
end

# Synthetic hierarchical data
n_groups = 5
n_per_group = 20
groups = repeat(1:n_groups, inner=n_per_group)
x_hier = randn(length(groups))
y_hier = randn(length(groups))

hier_model = hierarchical_regression(x_hier, y_hier, groups)
hier_chain = sample(hier_model, NUTS(), 2000)
```

## Bayesian Neural Networks

```julia
using Flux

# Bayesian neural network with Turing
@model function bnn(x, y)
    # Network architecture: input -> 10 hidden -> output
    input_dim, hidden_dim, output_dim = size(x, 1), 10, 1

    # Priors for weights and biases
    w1 ~ filldist(Normal(0, 0.1), hidden_dim, input_dim)
    b1 ~ filldist(Normal(0, 0.1), hidden_dim)
    w2 ~ filldist(Normal(0, 0.1), output_dim, hidden_dim)
    b2 ~ filldist(Normal(0, 0.1), output_dim)

    # Observation noise
    σ ~ InverseGamma(2, 3)

    # Forward pass
    h = tanh.(w1 * x .+ b1)
    μ = w2 * h .+ b2

    # Likelihood
    for i in 1:size(y, 2)
        y[:, i] ~ MvNormal(μ[:, i], σ^2 * I)
    end
end

# Generate data for BNN
x_train = reshape(randn(100), 1, :)
y_train = reshape(sin.(x_train[:]) .+ 0.1 * randn(100), 1, :)

bnn_model = bnn(x_train, y_train)
bnn_chain = sample(bnn_model, NUTS(), 1000)

# Prediction with uncertainty
function predict_bnn(chain, x_test)
    predictions = []
    for i in 1:length(chain)
        w1 = chain[:w1][i]
        b1 = chain[:b1][i]
        w2 = chain[:w2][i]
        b2 = chain[:b2][i]

        h = tanh.(w1 * x_test .+ b1)
        pred = w2 * h .+ b2
        push!(predictions, pred)
    end
    return hcat(predictions...)
end
```

## Model Comparison and Selection

```julia
using MLJ
using ArviZ

# Model comparison using WAIC/LOO
@model function model1(x, y)
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)
    σ ~ InverseGamma(2, 3)
    y .~ Normal.(α .+ β .* x, σ)
end

@model function model2(x, y)
    α ~ Normal(0, 1)
    β1 ~ Normal(0, 1)
    β2 ~ Normal(0, 1)
    σ ~ InverseGamma(2, 3)
    y .~ Normal.(α .+ β1 .* x .+ β2 .* x.^2, σ)
end

# Fit both models
chain1 = sample(model1(x, y_true), NUTS(), 2000)
chain2 = sample(model2(x, y_true), NUTS(), 2000)

# Model comparison
using ParetoSmooth

waic1 = waic(chain1)
waic2 = waic(chain2)

loo1 = loo(chain1)
loo2 = loo(chain2)

println("Model 1 WAIC: $(waic1.waic)")
println("Model 2 WAIC: $(waic2.waic)")
println("Lower WAIC is better")
```

## Advanced Diagnostics

```julia
using MCMCDiagnosticTools

# Convergence diagnostics
rhat_values = rhat(chain)
ess_bulk = ess_bulk(chain)
ess_tail = ess_tail(chain)

println("R-hat values (should be < 1.01):")
display(rhat_values)

println("Effective sample sizes:")
println("Bulk ESS: $ess_bulk")
println("Tail ESS: $ess_tail")

# Trace plots and diagnostics
using StatsPlots

# Trace plots
plot(chain)

# Autocorrelation plots
autocorplot(chain)

# Posterior predictive checks
function posterior_predictive_check(model, chain, x_obs, y_obs)
    n_samples = min(100, length(chain))
    y_pred = []

    for i in 1:n_samples
        α = chain[:α][i]
        β = chain[:β][i]
        σ = chain[:σ][i]

        y_sim = α .+ β .* x_obs .+ σ .* randn(length(x_obs))
        push!(y_pred, y_sim)
    end

    # Plot observed vs predicted
    plot(y_obs, alpha=0.7, label="Observed", linewidth=2)
    for (i, y_sim) in enumerate(y_pred[1:min(20, end)])
        plot!(y_sim, alpha=0.1, color=:blue, label=i==1 ? "Predicted" : "")
    end
    xlabel!("Data point")
    ylabel!("Value")
    title!("Posterior Predictive Check")
end

posterior_predictive_check(model, chain, x, y_true)
```

## Performance Optimization

```julia
using BenchmarkTools
using Profile

# Benchmark different samplers
function benchmark_samplers(model)
    println("NUTS sampler:")
    @btime sample($model, NUTS(), 1000)

    println("HMC sampler:")
    @btime sample($model, HMC(0.01, 10), 1000)

    println("Variational inference:")
    @btime vi($model, ADVI(10, 1000))
end

# Profile MCMC sampling
Profile.clear()
@profile sample(model, NUTS(), 1000)
Profile.print()

# Memory-efficient sampling for large datasets
function minibatch_sampling(x, y, batch_size=100)
    n = length(y)
    n_batches = ceil(Int, n / batch_size)

    @model function minibatch_model(x_batch, y_batch)
        α ~ Normal(0, 1)
        β ~ Normal(0, 1)
        σ ~ InverseGamma(2, 3)

        # Scale likelihood by full dataset size
        scale_factor = n / length(y_batch)
        Turing.@addlogprob! scale_factor * sum(logpdf.(Normal.(α .+ β .* x_batch, σ), y_batch))
    end

    # Sample on random minibatch
    batch_idx = sample(1:n, batch_size, replace=false)
    batch_model = minibatch_model(x[batch_idx], y[batch_idx])
    return sample(batch_model, NUTS(), 1000)
end
```

## Integration with Julia Ecosystem

```julia
# Integration with MLJ.jl
using MLJ

mutable struct TuringRegressor <: MLJ.Probabilistic
    n_samples::Int
    sampler
end

TuringRegressor(; n_samples=1000, sampler=NUTS()) = TuringRegressor(n_samples, sampler)

function MLJ.fit(::TuringRegressor, verbosity, X, y)
    x_vec = X.x  # Assuming single feature

    @model function mlj_model(x, y)
        α ~ Normal(0, 1)
        β ~ Normal(0, 1)
        σ ~ InverseGamma(2, 3)
        y .~ Normal.(α .+ β .* x, σ)
    end

    model = mlj_model(x_vec, y)
    chain = sample(model, NUTS(), n_samples)

    return (chain=chain, model=model), nothing, nothing
end

function MLJ.predict(::TuringRegressor, fitresult, Xnew)
    chain, _ = fitresult
    x_new = Xnew.x

    # Posterior predictive distribution
    predictions = []
    for i in 1:length(chain)
        α = chain[:α][i]
        β = chain[:β][i]
        σ = chain[:σ][i]

        pred_mean = α .+ β .* x_new
        pred_dist = Normal.(pred_mean, σ)
        push!(predictions, pred_dist)
    end

    return predictions
end

# Use with MLJ ecosystem
turing_model = TuringRegressor(n_samples=2000)
mach = machine(turing_model, (x=x,), y_true)
fit!(mach)
predictions = predict(mach, (x=x,))
```

## Agent-Enhanced Bayesian Modeling Integration Patterns

### Complete Julia Bayesian Development Workflow
```bash
# Intelligent Bayesian modeling development pipeline
/julia-prob-model --agents=auto --intelligent --optimize --research
/julia-ad-grad --agents=auto --intelligent --higher-order
/julia-jit-like --agents=bayesian --intelligent --performance
```

### Scientific Research Bayesian Pipeline
```bash
# High-performance scientific Bayesian workflow
/julia-prob-model --agents=scientific --breakthrough --orchestrate --mcmc
/jax-numpyro-prob --agents=julia --intelligent --inference=mcmc
/generate-tests --agents=scientific --type=scientific --coverage=90
```

### Production ML Bayesian Infrastructure
```bash
# Large-scale production Bayesian ML optimization
/julia-prob-model --agents=ai --optimize --distributed --variational
/run-all-tests --agents=bayesian --scientific --performance
/check-code-quality --agents=julia --language=julia --analysis=scientific
```

## Related Commands

**Julia Ecosystem Development**: Enhanced Julia Bayesian development with agent intelligence
- `/julia-ad-grad --agents=auto` - Use AD for gradient-based MCMC samplers with agent optimization
- `/julia-jit-like --agents=bayesian` - Optimize probabilistic models for performance with Bayesian agents
- `/optimize --agents=julia` - Julia optimization with specialized agents

**Cross-Language Probabilistic Computing**: Multi-language Bayesian integration
- `/jax-numpyro-prob --agents=auto` - Compare with JAX probabilistic programming
- `/python-debug-prof --agents=bayesian` - Profile Bayesian computation performance with agents
- `/jax-performance --agents=julia` - JAX performance comparison with Julia agents

**Quality Assurance**: Bayesian validation and optimization
- `/generate-tests --agents=bayesian --type=scientific` - Generate Bayesian tests with agent intelligence
- `/run-all-tests --agents=julia --scientific` - Comprehensive Julia Bayesian testing with specialized agents
- `/check-code-quality --agents=auto --language=julia` - Julia Bayesian code quality with agent optimization

ARGUMENTS: [--mcmc] [--variational] [--parallel] [--distributed] [--agents=auto|julia|scientific|ai|bayesian|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--research]