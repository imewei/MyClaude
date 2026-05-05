# Hierarchical Bayesian model using Turing.jl with NUTS sampler

using Turing
using Distributions
using MCMCChains
using ArviZ
using LinearAlgebra
using Random

Random.seed!(42)

"""
Hierarchical normal model for grouped data:
    mu_global ~ Normal(0, 10)
    sigma_global ~ HalfCauchy(5)
    mu_group[j] ~ Normal(mu_global, sigma_global)
    sigma_obs ~ HalfCauchy(2)
    y[i] ~ Normal(mu_group[group[i]], sigma_obs)
"""
@model function hierarchical_normal(y, group, n_groups)
    # Hyperpriors
    mu_global ~ Normal(0.0, 10.0)
    sigma_global ~ truncated(Cauchy(0.0, 5.0); lower=0.0)

    # Group-level parameters
    mu_group ~ filldist(Normal(mu_global, sigma_global), n_groups)

    # Observation noise
    sigma_obs ~ truncated(Cauchy(0.0, 2.0); lower=0.0)

    # Likelihood
    for i in eachindex(y)
        y[i] ~ Normal(mu_group[group[i]], sigma_obs)
    end
end

function generate_synthetic_data(; n_groups=5, n_per_group=30, seed=42)
    rng = MersenneTwister(seed)
    mu_true = 5.0
    sigma_group_true = 2.0
    sigma_obs_true = 1.0

    group_means = mu_true .+ sigma_group_true .* randn(rng, n_groups)
    y = Float64[]
    group = Int[]
    for j in 1:n_groups
        obs = group_means[j] .+ sigma_obs_true .* randn(rng, n_per_group)
        append!(y, obs)
        append!(group, fill(j, n_per_group))
    end
    return y, group, n_groups
end

function fit_model(; n_samples=2000, n_chains=4, target_accept=0.85)
    y, group, n_groups = generate_synthetic_data()

    model = hierarchical_normal(y, group, n_groups)

    chain = sample(model, NUTS(target_accept), MCMCThreads(), n_samples, n_chains)

    # Diagnostics
    println("R-hat values:")
    display(summarystats(chain))

    ess_vals = ess(chain)
    println("\nEffective sample sizes:")
    display(ess_vals)

    # Convert to ArviZ InferenceData for extended diagnostics
    idata = from_mcmcchains(chain)

    return chain, idata
end

function prior_predictive_check(; n_samples=500)
    y, group, n_groups = generate_synthetic_data()
    model = hierarchical_normal(y, group, n_groups)
    prior_chain = sample(model, Prior(), n_samples)
    return prior_chain
end
