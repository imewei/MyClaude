---
name: mcmc-diagnostics
description: Master MCMC convergence diagnostics with R-hat, ESS, trace plots, and divergence checking. Use when validating Bayesian inference results from Turing.jl. Also use when chains look poorly mixed, R-hat is too high, effective sample size is low, or divergent transitions appear. Use proactively when interpreting posterior samples, debugging sampler performance, or deciding whether to reparameterize a model, even if the user only mentions "my MCMC isn't converging."
---

# MCMC Diagnostics

## Expert Agents

MCMC diagnostics applies to any Bayesian workflow — Turing.jl, NumPyro, or
Pigeons-tempered chains. Delegate to:

- **`julia-pro`**: Julia/Turing.jl + MCMCChains workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
- **`statistical-physicist`**: Bayesian inference theory, sampler geometry,
  PSIS-LOO model comparison, ArviZ post-processing across PPLs.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
- **`jax-pro`**: NumPyro-side diagnostics and JAX-accelerated post-processing.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`

Used by `turing-model-design`, `consensus-mcmc-pigeons`, `bayesian-ude-workflow`,
`numpyro-core-mastery`, and `neural-pde` (BPINN section) — convergence checks
are the common acceptance gate for every Bayesian workflow in this suite.

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

## Python / NumPyro Workflow

```python
import arviz as az
import numpyro
from numpyro.infer import MCMC, NUTS

mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(rng_key, data)

# Convert to ArviZ InferenceData
idata = az.from_numpyro(mcmc)

az.summary(idata)                 # R-hat, ESS bulk/tail, MCSE, HDI
az.rhat(idata)                    # Target < 1.01
az.ess(idata, method="bulk")      # Target > 400 (also "tail")
az.plot_trace(idata)
az.plot_rank(idata)               # Rank plots — robust alternative to trace
az.plot_energy(idata)             # HMC energy / BFMI
```

**Model comparison** (PSIS-LOO / WAIC):
```python
az.compare({"model_a": idata_a, "model_b": idata_b}, ic="loo")
az.loo_pit(idata)                 # posterior-predictive uniformity check
```

> **ArviZ v1.0 (2025) is a major-version bump**. The `az.rhat`, `az.ess`, `az.summary`, and `from_numpyro` function names are preserved but internal behavior changed — verify compatibility of pipelines written against `arviz<1.0` before upgrading.

---

## Checklist

- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 per chain
- [ ] Trace plots show good mixing
- [ ] No divergent transitions

---

**Version**: 1.0.5
