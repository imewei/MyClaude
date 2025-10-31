---
name: mcmc-diagnostics
description: Master MCMC convergence diagnostics with trace plots, R-hat (Gelman-Rubin statistic), effective sample size (ESS), divergence checking, and chain mixing analysis for validating Bayesian inference. Use when analyzing MCMC results from Turing.jl (MCMCChains objects), checking R-hat values (should be < 1.01 for convergence), computing effective sample size (ESS > 400 recommended), inspecting trace plots for good mixing, identifying divergent transitions in NUTS/HMC sampling, analyzing autocorrelation in chains, validating multi-chain convergence, diagnosing sampling problems (funnel geometry, poor initialization), or ensuring MCMC reliability. Essential for all Bayesian inference workflows and critical for validating posterior samples before making inferences.
---

# MCMC Diagnostics

Master MCMC convergence diagnostics with MCMCChains.jl.

## When to use this skill

- Analyzing MCMC results from Turing.jl (MCMCChains objects)
- Checking R-hat (Gelman-Rubin) convergence (target: < 1.01)
- Computing effective sample size (ESS) (target: > 400 per chain)
- Inspecting trace plots for good mixing and stationarity
- Identifying divergent transitions in NUTS/HMC samplers
- Analyzing autocorrelation in MCMC chains
- Validating multi-chain convergence
- Diagnosing sampling problems (funnel geometry, initialization issues)
- Ensuring posterior sample reliability before inference
- Visualizing posterior distributions
- Comparing chain diagnostics across parameters

## Essential Diagnostics
```julia
using MCMCChains

# Summary statistics
summarize(chain)

# R-hat (should be < 1.01)
rhat(chain)

# Effective sample size (should be > 400)
ess(chain)

# Trace plots
plot(chain[:parameter])

# Autocorrelation
autocorplot(chain[:parameter])
```

## Convergence Criteria
- R-hat < 1.01: Good convergence
- ESS > 400: Sufficient samples
- Trace plots: Good mixing
- No divergent transitions

## Resources
- **MCMCChains.jl**: https://github.com/TuringLang/MCMCChains.jl
