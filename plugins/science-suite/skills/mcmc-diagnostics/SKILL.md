---
name: mcmc-diagnostics
maturity: "5-Expert"
specialization: Bayesian Diagnostics
description: Master MCMC convergence diagnostics with R-hat, ESS, trace plots, and divergence checking. Use when validating Bayesian inference results from Turing.jl. Also use when chains look poorly mixed, R-hat is too high, effective sample size is low, or divergent transitions appear. Use proactively when interpreting posterior samples, debugging sampler performance, or deciding whether to reparameterize a model, even if the user only mentions "my MCMC isn't converging."
---

# MCMC Diagnostics

## Expert Agent

For MCMC convergence diagnostics and Bayesian validation with Turing.jl, delegate to:

- **`julia-pro`**: Julia Bayesian inference, Turing.jl, and MCMC workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

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
