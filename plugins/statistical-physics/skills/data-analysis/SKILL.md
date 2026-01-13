---
name: data-analysis
version: "1.0.7"
maturity: "5-Expert"
specialization: Experimental Data Analysis
description: Analyze experimental correlation data from DLS, SAXS/SANS, rheology, and microscopy using Green-Kubo relations, Bayesian inference (MCMC), and model validation. Use when interpreting scattering data, validating non-equilibrium theories, or predicting transport coefficients.
---

# Data Analysis & Model Validation

Statistical analysis of experimental data with Bayesian inference and model validation.

---

## Correlation Functions

| Type | Formula | Application |
|------|---------|-------------|
| Velocity (VAC) | C_v(t) = ⟨v(t)·v(0)⟩/⟨v²⟩ | Diffusion: D = ∫C_v(t)dt |
| Stress | C_σ(t) = ⟨σ(t)σ(0)⟩ | Viscosity: η = (V/kT)∫C_σ(t)dt |
| Orientational | C_l(t) = ⟨P_l[cos(Δθ)]⟩ | Rotational diffusion: τ_r = 1/D_r |
| Pair (RDF) | g(r) = ⟨ρ(r)⟩/ρ_bulk | Structure at equilibrium |
| Structure | S(q) = 1 + ρ∫[g(r)-1]e^{iq·r}dr | Scattering experiments |

---

## Experimental Techniques

### Dynamic Light Scattering (DLS)

```python
def fit_dls_correlation(tau, g1, q):
    """Extract diffusion from DLS correlation."""
    from scipy.optimize import curve_fit

    def model(tau, D):
        return np.exp(-q**2 * D * tau)

    popt, pcov = curve_fit(model, tau, g1)
    D, D_err = popt[0], np.sqrt(pcov[0, 0])
    # Stokes-Einstein: R = kT / (6πηD)
    return D, D_err
```

### Rheology

| Quantity | Formula | Physical Meaning |
|----------|---------|------------------|
| G*(ω) | G'(ω) + iG''(ω) | Complex modulus |
| G'(ω) | Storage modulus | Elastic response |
| G''(ω) | Loss modulus | Viscous dissipation |
| η*(ω) | √(G'² + G''²)/ω | Complex viscosity |
| τ_relax | 1/ω_c where G' = G'' | Relaxation time |

### Small-Angle Scattering (SAXS/SANS)

| Regime | Formula | Interpretation |
|--------|---------|----------------|
| Guinier (qR < 1) | I(q) ~ exp(-q²R_g²/3) | Radius of gyration |
| Porod (qR > 1) | I(q) ~ q^(-d) | Fractal dimension |
| Peak at q* | λ = 2π/q* | Characteristic length |

---

## Bayesian Inference

```python
import emcee

def log_probability(params, data, model, sigma):
    # Prior: physical constraints
    if any(p < 0 for p in params):
        return -np.inf

    # Likelihood: chi-squared
    theory = model(params)
    chi2 = np.sum((data - theory)**2 / sigma**2)
    return -0.5 * chi2

# Run MCMC
ndim, nwalkers = len(params_init), 32
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability,
    args=(data, model, sigma)
)
sampler.run_mcmc(initial_state, 5000)

# Extract posteriors
samples = sampler.get_chain(discard=1000, flat=True)
# Report: median, 16th/84th percentiles (68% CI)
```

---

## Model Selection

| Method | Formula | Use Case |
|--------|---------|----------|
| BIC | -2ln(L) + k ln(n) | Balance fit vs complexity |
| Bayes Factor | Z₁/Z₂ | Compare model evidence |
| Cross-validation | k-fold | Prediction error |
| Posterior predictive | P(D_new\|D_obs) | Model validation |

---

## Green-Kubo Relations

| Property | Correlation | Formula |
|----------|-------------|---------|
| Diffusion | Velocity | D = ∫⟨v(t)·v(0)⟩dt |
| Viscosity | Stress | η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩dt |
| Thermal conductivity | Heat flux | κ = (V/kT²)∫⟨J_q(t)·J_q(0)⟩dt |

---

## Active Matter Observables

| Observable | Formula | Meaning |
|------------|---------|---------|
| Polar order | φ = \|⟨v_i⟩\|/v₀ | Alignment |
| Enhanced diffusion | D_eff = D_t + v₀²τ_p/d | Activity contribution |
| Effective temperature | T_eff > T | Non-equilibrium |

---

## Computational Tools

```python
def compute_time_correlation(trajectory, obs_func, max_lag=1000):
    """Time-correlation from trajectory."""
    obs = obs_func(trajectory)
    correlations = []
    for lag in range(max_lag):
        C = np.mean(obs[lag:] * obs[:-lag if lag > 0 else None])
        correlations.append(C)
    return np.array(correlations)

from scipy import stats

def compare_distributions(data1, data2):
    """Statistical tests for comparison."""
    ks_stat, ks_pval = stats.ks_2samp(data1, data2)
    u_stat, u_pval = stats.mannwhitneyu(data1, data2)
    return {'KS': (ks_stat, ks_pval), 'U': (u_stat, u_pval)}
```

---

## Best Practices

| Area | Practice |
|------|----------|
| Data quality | Check SNR, remove outliers, sufficient sampling |
| Model fitting | Physical priors, check identifiability |
| Uncertainty | Report posteriors, propagate errors |
| Reproducibility | Document pipeline, provide scripts |

---

## Checklist

- [ ] Correlation functions computed correctly
- [ ] Proper error propagation applied
- [ ] Bayesian inference with physical priors
- [ ] Model validated on independent data
- [ ] Uncertainties reported (credible intervals)
- [ ] Statistical vs systematic errors distinguished
- [ ] Analysis scripts documented and reproducible

---

**Version**: 1.0.5
