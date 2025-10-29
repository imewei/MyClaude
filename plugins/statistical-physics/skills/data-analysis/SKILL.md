---
name: data-analysis
description: Analyze experimental data from scattering, rheology, and microscopy using correlation functions, Bayesian inference, and model validation. Use when interpreting DLS, SAXS/SANS, rheological measurements, or validating non-equilibrium theories against experiments with uncertainty quantification and inverse problem solving.
---

# Data Analysis & Model Validation

Interpret experimental data, validate theoretical predictions, and perform inverse problems for non-equilibrium systems using rigorous statistical methods.

## Correlation Function Analysis

### Time-Correlation Functions

**Velocity Autocorrelation (VAC):**
C_v(t) = ⟨v(t)·v(0)⟩ / ⟨v²⟩
- Diffusion: D = ∫₀^∞ C_v(t) dt (Green-Kubo)
- Decay time: τ ~ 1/friction

**Stress Autocorrelation:**
C_σ(t) = ⟨σ(t)σ(0)⟩
- Viscosity: η = (V/kT) ∫₀^∞ C_σ(t) dt
- Relaxation modulus: G(t) = C_σ(t)

**Orientational Correlation:**
C_l(t) = ⟨P_l[cos(θ(t)-θ(0))]⟩
- Rotational diffusion: τ_r = 1/D_r
- Active matter: Persistence time

### Spatial Correlation Functions

**Pair Distribution Function (RDF):**
g(r) = ⟨ρ(r)⟩ / ρ_bulk
- Structure at equilibrium
- Peak positions: nearest-neighbor distances

**Structure Factor:**
S(q) = 1 + ρ ∫ [g(r)-1] e^(iq·r) dr
- Connection to scattering experiments
- Peak at q*: characteristic length scale

**Spatial Velocity Correlation:**
C_vv(r) = ⟨v(x)·v(x+r)⟩
- Active matter: Long-range correlations
- Hydrodynamic interactions

## Experimental Data Interpretation

### Dynamic Light Scattering (DLS)

**Intensity correlation:**
g₂(τ) = ⟨I(t)I(t+τ)⟩ / ⟨I⟩²

**Field correlation (Siegert relation):**
g₂(τ) = 1 + β|g₁(τ)|²

**Brownian particles:**
g₁(τ) = exp(-q²Dτ)
- Extract: D from exponential decay
- Stokes-Einstein: D = kT/(6πηR)

**Active matter deviations:**
- Ballistic regime: g₁(τ) ≈ 1 - (qv₀)²τ²/2 (short time)
- Diffusive regime: g₁(τ) ~ exp(-D_eff q²τ) (long time)
- Enhanced diffusion: D_eff > D_thermal

```python
def fit_dls_correlation(tau, g1, q):
    """
    Fit DLS data to extract diffusion coefficient
    """
    def model(tau, D):
        return np.exp(-q**2 * D * tau)
    
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(model, tau, g1)
    D, D_err = popt[0], np.sqrt(pcov[0,0])
    return D, D_err
```

### Rheology

**Linear viscoelasticity:**
G*(ω) = G'(ω) + iG''(ω)
- Storage modulus: G'(ω) (elastic response)
- Loss modulus: G''(ω) (viscous dissipation)

**Relaxation spectrum:**
G(t) = ∫ H(τ) exp(-t/τ) dτ
- Extract relaxation times from frequency sweep

**Non-linear rheology:**
- Shear thinning: η(γ̇) ~ γ̇^(n-1), n < 1
- Yield stress: τ > τ_y for flow
- Thixotropy: Time-dependent viscosity

```python
def analyze_rheology(omega, Gp, Gpp):
    """
    Analyze oscillatory rheology data
    """
    # Complex viscosity
    eta_star = np.sqrt(Gp**2 + Gpp**2) / omega
    
    # Crossover frequency (G' = G'')
    idx = np.argmin(np.abs(Gp - Gpp))
    omega_c = omega[idx]
    tau_relax = 1 / omega_c
    
    return eta_star, tau_relax
```

### Scattering (SAXS/SANS/X-ray)

**Static structure factor:**
I(q) ∝ S(q)
- Peak positions: Real-space periodicities
- Peak width: Correlation length ξ ~ 1/Δq

**Dynamic structure factor:**
S(q,ω) = ∫ F(q,t) e^(iωt) dt
- F(q,t): Intermediate scattering function
- Hydrodynamic modes (sound, diffusion, thermal)

**Small-angle scattering:**
- Guinier regime (qR < 1): I(q) ~ exp(-q²R_g²/3), radius of gyration
- Porod regime (qR > 1): I(q) ~ q^(-d), fractal dimension

## Bayesian Inference & Inverse Problems

### Parameter Estimation

**Bayes' theorem:**
P(θ|D) ∝ P(D|θ) P(θ)
- Prior: P(θ) (physical constraints)
- Likelihood: P(D|θ) (measurement model)
- Posterior: P(θ|D) (updated belief)

**Markov Chain Monte Carlo (MCMC):**
```python
import emcee

def log_likelihood(params, data, model):
    """Log-likelihood for MCMC"""
    theory = model(params)
    chi2 = np.sum((data - theory)**2 / sigma**2)
    return -0.5 * chi2

def log_prior(params):
    """Physical constraints on parameters"""
    if all(p > 0 for p in params):  # Positivity
        return 0.0
    return -np.inf

def log_probability(params, data, model):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, data, model)

# Run MCMC
ndim, nwalkers = len(params_init), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data, model))
sampler.run_mcmc(initial_state, 5000)
```

**Uncertainty quantification:**
- Posterior distributions for parameters
- Credible intervals (e.g., 68%, 95%)
- Parameter correlations from covariance

### Model Selection

**Bayesian Information Criterion (BIC):**
BIC = -2 ln(L) + k ln(n)
- Balance fit quality vs. model complexity
- Lower BIC: Better model

**Evidence calculation:**
Z = ∫ P(D|θ) P(θ) dθ
- Bayes factor: Z₁/Z₂ for model comparison

## Model Validation

### Predictive Checks

**Posterior predictive distribution:**
P(D_new|D_obs) = ∫ P(D_new|θ) P(θ|D_obs) dθ
- Simulate from fitted model
- Compare to held-out data

**Residual analysis:**
- Systematic deviations indicate model inadequacy
- Check normality, autocorrelation

### Cross-Validation

**k-fold CV:**
- Train on k-1 folds, test on 1 fold
- Average prediction error

**Leave-one-out (LOO):**
- For small datasets
- Computationally expensive but unbiased

### Experimental Validation

**Consistency checks:**
- Multiple observables from same theory
- Temperature/concentration scaling laws
- Symmetry relations (Onsager, FDT)

**Sensitivity analysis:**
- Which parameters affect which observables?
- Optimal experiments to constrain parameters

## Materials Property Prediction

### From Microscopic to Macroscopic

**Multiscale bridging:**
1. MD: Atomic trajectories → time-correlation functions
2. Theory: Green-Kubo → transport coefficients
3. Continuum: Transport laws → macroscopic behavior

**Example: Viscosity prediction**
- MD: σ_xy(t) stress tensor from simulation
- Green-Kubo: η = (V/kT) ∫₀^∞ ⟨σ_xy(t)σ_xy(0)⟩ dt
- Validation: Compare to rheology measurements

### Active Matter Observables

**Order parameters:**
- Polar order: φ = |⟨v_i⟩| / v₀
- Nematic order: S = ⟨cos(2θ)⟩

**Phase transitions:**
- Critical density/noise for onset of collective motion
- Correlation length divergence near transition

**Effective properties:**
- Enhanced diffusion: D_eff = D_t + v₀²τ_p/d
- Effective temperature: T_eff > T from activity

## Computational Tools

### Correlation Analysis
```python
def compute_time_correlation(trajectory, observable_func, max_lag=1000):
    """
    Compute time-correlation function from trajectory
    """
    obs = observable_func(trajectory)
    correlations = []
    
    for lag in range(max_lag):
        C = np.mean(obs[lag:] * obs[:-lag if lag > 0 else None])
        correlations.append(C)
    
    return np.array(correlations)
```

### Statistical Tests
```python
from scipy import stats

def compare_distributions(data1, data2):
    """Statistical tests for distribution comparison"""
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(data1, data2)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(data1, data2)
    
    return {'KS': (ks_stat, ks_pval), 'U': (u_stat, u_pval)}
```

## Best Practices

### Data Quality
- [ ] Check signal-to-noise ratio, remove outliers
- [ ] Sufficient sampling (time/ensemble averaging)
- [ ] Control variables (temperature, concentration)

### Model Fitting
- [ ] Use physical priors to constrain parameters
- [ ] Check identifiability (can all parameters be determined?)
- [ ] Validate on independent datasets

### Uncertainty Reporting
- [ ] Report parameter uncertainties from posterior
- [ ] Propagate uncertainties to predictions
- [ ] Distinguish statistical vs. systematic errors

### Reproducibility
- [ ] Document data processing pipeline
- [ ] Provide analysis scripts and random seeds
- [ ] Report computational environment

References for advanced methods: information-theoretic model selection, non-parametric Bayesian inference, Gaussian process regression for surrogate models.
