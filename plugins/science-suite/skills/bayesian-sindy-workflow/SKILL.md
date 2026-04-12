---
name: bayesian-sindy-workflow
description: Build Bayesian SINDy workflows with horseshoe priors, ensemble SINDy, and UQ-SINDy. Use when you need credible intervals on SINDy coefficients, inclusion probabilities for library terms, noise-robust sparse regression, or Bayesian model comparison between candidate dynamical systems. Use proactively when the user mentions horseshoe prior, spike-and-slab, Bayesian SINDy, UQ-SINDy, inclusion probability, posterior SINDy, or sparse Bayesian regression on a dynamical-system library.
---

# Bayesian SINDy Workflow

Classical SINDy (see `equation-discovery`) returns a single point estimate of the sparse coefficient vector via sequentially-thresholded least squares. When measurement noise is non-trivial, library terms are correlated, or the application requires credible intervals and inclusion probabilities, use a **Bayesian SINDy** formulation instead.

## Expert Agent

- **`statistical-physicist`** (primary) — Bayesian inference on sparse regression, horseshoe-prior construction, MCMC convergence for high-dimensional posteriors
- **`julia-pro`** (secondary) — for the Turing UQ-SINDy sidebar and Julia-side workflows

Location: `plugins/science-suite/agents/statistical-physicist.md` and `plugins/science-suite/agents/julia-pro.md`.

## When to prefer Bayesian SINDy vs classical SINDy

| Situation | Use |
|-----------|-----|
| Fast point estimate on clean data, single right answer | Classical SINDy (`equation-discovery`) |
| Need credible intervals on each active coefficient | **Bayesian SINDy** (this skill) |
| Library terms are correlated, STLSQ threshold is ambiguous | **Bayesian SINDy** — posterior uncertainty reveals the correlation |
| Measurement noise is high, want robust sparsity | **Bayesian SINDy** with horseshoe prior |
| Model comparison between candidate libraries (which subset is best?) | **Bayesian SINDy** + PSIS-LOO via `mcmc-diagnostics` |
| Working inside a Bayesian UDE workflow that also needs symbolic extraction | **Bayesian SINDy** composed with `bayesian-ude-workflow` |

## Three Bayesian SINDy routes

1. **Spike-and-slab / horseshoe prior on `Xi`** — place a sparsifying prior directly on the coefficient vector and sample with NUTS. Each library term gets a marginal posterior probability of inclusion and a coefficient credible interval. Most principled approach and the focus of the worked example below.

2. **Ensemble SINDy (PySINDy `EnsembleOptimizer`)** — bootstrap the data, run classical SINDy repeatedly, collect an empirical distribution over inclusion frequencies and coefficient values. Cheap, parallel, and good enough for exploratory uncertainty. Not a proper posterior but useful as a baseline.

3. **UQ-SINDy (DataDrivenDiffEq.jl `ImplicitOptimizer` + Turing)** — Julia-side Bayesian posterior on the sparse coefficients through `Turing.@model` wrappers around the DataDrivenDiffEq library. See the Julia sidebar below and `turing-model-design` for the full workflow.

## Worked example — Lorenz-63 with horseshoe prior

The canonical SINDy benchmark. Lorenz-63 has 3 state variables and, in a second-order polynomial library of 10 terms, only 7 active terms across the 3 state equations (2 in `dx/dt`, 3 in `dy/dt`, 2 in `dz/dt`). A correct Bayesian SINDy fit should identify exactly those 7 terms with posterior inclusion probability near 1 and leave the 23 inactive terms near 0.

### Stage 1 — Generate ground-truth data

```python
import numpy as np
from scipy.integrate import solve_ivp

def lorenz(t, u, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = u
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

t_eval = np.linspace(0, 10, 5000)
sol = solve_ivp(lorenz, (0, 10), [1.0, 1.0, 1.0], t_eval=t_eval, rtol=1e-10)
X = sol.y.T                                          # (5000, 3)

rng = np.random.default_rng(0)
X_noisy = X + 0.01 * X.std(axis=0) * rng.standard_normal(X.shape)
dXdt = (X_noisy[2:] - X_noisy[:-2]) / (t_eval[2:] - t_eval[:-2])[:, None]
X_mid = X_noisy[1:-1]                                # (4998, 3)
```

Real workflows use smoothing splines or weak-form derivatives instead of central differences; the central-difference version here is kept minimal for the worked example.

### Stage 2 — Build the candidate library

Second-order polynomial library: `1, x, y, z, x², y², z², xy, xz, yz` (10 features).

```python
def build_library(X):
    x, y, z = X.T
    return np.column_stack([
        np.ones_like(x), x, y, z,
        x*x, y*y, z*z, x*y, x*z, y*z,
    ])

Theta = build_library(X_mid)                         # (4998, 10)
# Ground truth coefficients:
#   dx/dt = -10 x + 10 y             -> terms: x, y
#   dy/dt =  28 x - y - xz           -> terms: x, y, xz
#   dz/dt = -(8/3) z + xy            -> terms: z, xy
# 7 active terms across 30 possible (10 features x 3 states)
```

### Stage 3 — Fit horseshoe prior with NumPyro + NUTS

```python
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def bayesian_sindy(Theta, dXdt):
    n_feat, n_states = Theta.shape[1], dXdt.shape[1]
    tau = numpyro.sample("tau", dist.HalfCauchy(0.1))
    lam = numpyro.sample("lam", dist.HalfCauchy(jnp.ones((n_feat, n_states))))
    Xi  = numpyro.sample("Xi",  dist.Normal(0.0, tau * lam))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(jnp.ones(n_states)))
    numpyro.sample("obs", dist.Normal(Theta @ Xi, sigma), obs=dXdt)

kernel = NUTS(bayesian_sindy, target_accept_prob=0.9)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4, progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), Theta=jnp.asarray(Theta), dXdt=jnp.asarray(dXdt))
samples = mcmc.get_samples()                         # {"Xi": (8000, 10, 3), ...}
```

The horseshoe prior has two scale parameters: a global shrinkage `tau` pulling every coefficient toward zero, and a per-coefficient local scale `lam` that allows genuinely-active terms to escape the global shrinkage. Terms with weak posterior support stay near zero, terms with strong evidence get pushed away from zero.

Horseshoe posteriors are notoriously funnel-shaped (the `bad_posterior_geometry` notebook in NumPyro documents this). If R-hat stays elevated, reparameterize the model via `numpyro.handlers.reparam` with `LocScaleReparam` — see `numpyro-core-mastery`.

### Stage 4 — Diagnostics + PSIS-LOO

```python
import arviz as az
idata = az.from_numpyro(mcmc)

print(az.rhat(idata))                                # all < 1.01
print(az.ess(idata, method="bulk"))                  # > 400 per parameter

loo_full = az.loo(idata, pointwise=True)
print(loo_full)
```

See `mcmc-diagnostics` for the full R-hat / ESS / PSIS-LOO workflow and `numpyro-core-mastery` for advanced NUTS tuning. If R-hat is elevated after 1000 warmup + 2000 samples, increase `target_accept_prob` to 0.95 or bump warmup to 2000.

### Stage 5 — Extract inclusion probabilities and credible intervals

```python
Xi_samples = samples["Xi"]                            # (8000, 10, 3)
inclusion_prob = (jnp.abs(Xi_samples) > 0.1).mean(axis=0)   # (10, 3)
q05, q95 = jnp.quantile(Xi_samples, jnp.array([0.05, 0.95]), axis=0)
# Report terms with inclusion_prob > 0.9 and their (q05, q95) credible intervals.
```

Expected: the 7 ground-truth-active terms all flagged with `P(in) > 0.9`, credible intervals straddling sigma=10, rho=28, beta=8/3. Spurious terms should have `P(in) < 0.1`. The `0.1` threshold on `|Xi|` is a post-hoc inclusion rule — for a cleaner criterion use the projection-predictive variable-selection procedure on top of the posterior draws.

## Prior-sensitivity analysis

Sweep the `tau` scale in `dist.HalfCauchy(tau_scale)` across `[0.01, 0.1, 1.0, 10.0]` and re-run NUTS each time — smaller values over-shrink and miss true active terms, larger values admit false positives. For a principled choice use the "p0" parameterization from Piironen & Vehtari (2017) where `tau ~ HalfCauchy(p0 / (D - p0) * sigma / sqrt(n))` and `p0` is the prior-expected number of active terms. For Finnish / regularized horseshoe variants see `numpyro-core-mastery`.

## Julia sidebar — Turing UQ-SINDy pattern

The Julia equivalent uses `DataDrivenDiffEq.jl`'s `ImplicitOptimizer` wrapped in a `Turing.@model`. A minimal sketch:

```julia
using Turing, DataDrivenDiffEq, LinearAlgebra

@model function bayesian_sindy_turing(Theta, dXdt)
    n_feat = size(Theta, 2)
    tau ~ truncated(Cauchy(0, 0.1); lower=0)
    lam ~ filldist(truncated(Cauchy(0, 1); lower=0), n_feat)
    Xi  ~ MvNormal(zeros(n_feat), tau .* lam)
    sigma ~ truncated(Cauchy(0, 1); lower=0)
    dXdt ~ MvNormal(Theta * Xi, sigma^2 * I)
end

chain = sample(bayesian_sindy_turing(Theta, dXdt), NUTS(0.9), 2000)
```

`truncated(...; lower=0)` is the Turing 0.37+ keyword form (the DynamicPPL v0.35 breaking changes moved away from the positional `truncated(dist, 0, Inf)` form). See `turing-model-design` for the full Turing workflow, `consensus-mcmc-pigeons` if the posterior turns out multimodal (which can happen for SINDy with highly correlated library terms), and `bayesian-ude-workflow` for the combined Bayesian UDE + Bayesian SINDy story where the neural-network correction is interpreted back into symbolic terms.

## Composition with neighboring skills

- **Classical SINDy machinery** (library construction, STLSQ, symbolic regression, DataDrivenDiffEq.jl, PySINDy, PyDMD, PySR) -> `equation-discovery`
- **Parent Bayesian inference routing** -> `bayesian-inference` (hub)
- **NumPyro NUTS mechanics and horseshoe-prior reparameterization** -> `numpyro-core-mastery`
- **Julia-side Turing `@model` patterns** -> `turing-model-design`
- **R-hat / ESS / PSIS-LOO / ArviZ workflows** -> `mcmc-diagnostics`
- **Combined Bayesian UDE + Bayesian SINDy** for symbolic extraction from learned neural corrections -> `bayesian-ude-workflow`
- **Multimodal posterior escape hatch** (if SINDy's correlated library produces disjoint modes) -> `consensus-mcmc-pigeons`

## Checklist

- [ ] Built a polynomial library `Theta` matching the suspected dynamics order
- [ ] Chose a horseshoe `tau` scale appropriate for the signal-to-noise ratio (default 0.1 is a good start)
- [ ] Ran NUTS with at least 4 chains, 1000 warmup, 2000 samples
- [ ] Verified R-hat < 1.01 and bulk ESS > 400 per parameter (see `mcmc-diagnostics`)
- [ ] Reparameterized horseshoe funnel via `LocScaleReparam` if R-hat stayed elevated
- [ ] Computed inclusion probabilities via `(jnp.abs(Xi_samples) > threshold).mean(axis=0)`
- [ ] Identified active terms with `P(in) > 0.9` and recorded their credible intervals
- [ ] Ran a prior-sensitivity sweep on `tau_scale` to confirm robustness
- [ ] Compared the Bayesian posterior against a classical SINDy point estimate to ensure consistency
- [ ] For Lorenz-63 sanity check: verified the 7-active-out-of-30 sparsity structure before applying to real data
- [ ] For multimodal posteriors: switched to `consensus-mcmc-pigeons` and re-ran via Turing + Pigeons.jl
