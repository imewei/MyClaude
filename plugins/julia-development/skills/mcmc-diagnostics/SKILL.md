---
name: mcmc-diagnostics
description: MCMC convergence checking with trace plots, R-hat (Gelman-Rubin), effective sample size, divergence checking, and mixing analysis. Essential for validating MCMC results.
---

# MCMC Diagnostics

Master MCMC convergence diagnostics with MCMCChains.jl.

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
