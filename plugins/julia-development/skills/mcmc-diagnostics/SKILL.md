---
name: mcmc-diagnostics
version: "1.0.7"
maturity: "5-Expert"
specialization: Bayesian Diagnostics
description: Master MCMC convergence diagnostics with R-hat, ESS, trace plots, and divergence checking. Use when validating Bayesian inference results from Turing.jl.
---

# MCMC Diagnostics

Convergence diagnostics with MCMCChains.jl.

---

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

---

## Convergence Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| R-hat | < 1.01 | Chains converged |
| ESS | > 400 | Sufficient samples |
| Trace plots | Good mixing | Stationarity |
| Divergences | 0 | No sampling issues |

---

## Common Problems

| Issue | Solution |
|-------|----------|
| R-hat > 1.1 | Run longer, reparameterize |
| Low ESS | More samples, thin chains |
| Divergences | Non-centered parameterization |
| Poor mixing | Adjust step size |

---

## Checklist

- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 per chain
- [ ] Trace plots show good mixing
- [ ] No divergent transitions

---

**Version**: 1.0.5
