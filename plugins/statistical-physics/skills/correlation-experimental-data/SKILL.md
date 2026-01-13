---
name: correlation-experimental-data
version: "1.0.7"
maturity: "5-Expert"
specialization: Experimental Correlation Analysis
description: Interpret experimental correlation data from DLS (g₂(τ), Siegert relation, Stokes-Einstein), SAXS/SANS (S(q), g(r), Guinier/Porod analysis), XPCS (two-time correlation, aging), FCS (diffusion time, binding kinetics), and rheology (Green-Kubo, microrheology). Extract physical parameters, perform Bayesian model fitting, and validate against theoretical predictions.
---

# Experimental Data Interpretation

Correlation measurements from scattering, microscopy, and spectroscopy experiments.

---

## Technique Overview

| Technique | Measured | Timescale | Key Parameters |
|-----------|----------|-----------|----------------|
| DLS | g₂(τ) intensity correlation | μs - s | D, R_h, β_stretch |
| SAXS/SANS | I(q), S(q) | Static | R_g, S(q→0), fractal d |
| XPCS | C(t₁,t₂) | ms - hours | Aging, α exponent |
| FCS | G(τ) fluorescence | μs - s | τ_D, N, binding |
| Rheology | G*(ω), σ(t) | ms - s | η, G', G'' |

---

## DLS Analysis

### Core Equations
```
g₂(τ) = 1 + β|g₁(τ)|²         # Siegert relation
g₁(τ) = exp(-Γτ)              # Brownian: Γ = Dq²
D = kT/(6πηR)                 # Stokes-Einstein
q = (4πn/λ)sin(θ/2)           # Scattering vector
```

### Extraction Code
```python
def analyze_dls(tau, g2, T=298, viscosity=0.001, wavelength=632.8e-9, angle=90, n=1.33):
    from scipy.optimize import curve_fit

    q = 4*np.pi*n/wavelength * np.sin(np.radians(angle)/2)

    def g2_model(tau, beta, Gamma):
        return 1 + beta * np.exp(-2*Gamma*tau)

    popt, pcov = curve_fit(g2_model, tau, g2, p0=[0.8, 1000])
    beta, Gamma = popt

    D = Gamma / q**2
    kT = 1.38e-23 * T
    R = kT / (6*np.pi*viscosity*D)

    return {'radius': R, 'diffusion': D, 'beta': beta}
```

### Stretched Exponential (KWW)
```python
def fit_kww(tau, g2):
    def g2_kww(tau, beta_coh, tau_c, beta_stretch):
        return 1 + beta_coh * np.exp(-2*(tau/tau_c)**beta_stretch)

    popt, _ = curve_fit(g2_kww, tau, g2, p0=[0.8, tau[len(tau)//2], 0.7],
                        bounds=([0, 0, 0.1], [1, tau[-1]*10, 1.0]))
    return {'tau_c': popt[1], 'beta_stretch': popt[2]}
```

| β Value | System Type |
|---------|-------------|
| 1.0 | Single exponential (Brownian) |
| 0.5-0.7 | Supercooled liquids, glasses |
| < 0.5 | Broad distribution |

---

## SAXS/SANS Analysis

### Structure Factor
```python
def extract_structure_factor(q, I_conc, I_dilute):
    """S(q) = I_concentrated / I_dilute"""
    return I_conc / I_dilute

def pair_distribution(q, S_q, rho):
    """g(r) via Fourier transform"""
    from scipy.integrate import simps
    r = np.linspace(0.1, 50, 500)
    g_r = np.zeros_like(r)
    for i, r_val in enumerate(r):
        integrand = q**2 * (S_q - 1) * np.sin(q * r_val)
        g_r[i] = rho + (1/(2*np.pi**2*r_val)) * simps(integrand, q)
    return r, g_r
```

### Guinier Analysis
```python
def guinier_analysis(q, I, q_max=0.1):
    """R_g from ln(I) vs q² linear fit (qR_g < 1)"""
    mask = q < q_max
    slope, intercept, r, _, _ = linregress(q[mask]**2, np.log(I[mask]))
    R_g = np.sqrt(-3 * slope)
    return {'R_g': R_g, 'I_0': np.exp(intercept), 'r_squared': r**2}
```

### Porod Analysis

| Exponent d | Surface Type |
|------------|--------------|
| 4 | Sharp interface (Porod law) |
| < 4 | Fractal surface |
| > 4 | Diffuse interface |

---

## XPCS Analysis

### Two-Time Correlation
```python
def two_time_correlation(intensity_frames, q_idx):
    """C(t₁,t₂) for aging detection"""
    I = intensity_frames[:, q_idx, :]
    n = len(I)
    C = np.zeros((n, n))
    for t1 in range(n):
        for t2 in range(t1, n):
            C[t1, t2] = np.mean(I[t1] * I[t2]) / np.mean(I[t1]) / np.mean(I[t2])
            C[t2, t1] = C[t1, t2]
    return C
```

### Compressed Exponential

| α Value | Motion Type |
|---------|-------------|
| 1 | Brownian diffusion |
| 1.5 | Intermediate |
| 2 | Ballistic (active matter) |

---

## FCS Analysis

### 3D Diffusion Model
```
G(τ) = (1/N) × (1/(1+τ/τ_D)) × (1/√(1+τ/(ω²τ_D)))
D = r₀²/(4τ_D)
```

```python
def fcs_analysis(tau, G, r0=0.2e-6, omega=5):
    def G_3d(tau, N, tau_D):
        return (1/N) * (1/(1+tau/tau_D)) * (1/np.sqrt(1+tau/(omega**2*tau_D)))

    popt, _ = curve_fit(G_3d, tau, G, p0=[10, 1e-3])
    N, tau_D = popt
    D = r0**2 / (4*tau_D)
    return {'N': N, 'tau_D': tau_D, 'D': D}
```

---

## Green-Kubo Relations

| Transport Coefficient | Formula |
|----------------------|---------|
| Diffusion | D = ∫⟨v(t)·v(0)⟩dt |
| Viscosity | η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩dt |
| Thermal conductivity | κ = (V/kT²)∫⟨J_q(t)·J_q(0)⟩dt |

---

## Sum Rule Validation

| Sum Rule | Expression | Check |
|----------|------------|-------|
| Compressibility | S(0) = ρkTκ_T | Compare to measured S(q→0) |
| Number conservation | ∫[S(q)-1]dq = 0 | Should be ~0 |
| Kramers-Kronig | χ''/χ' relation | Response functions |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Background subtraction | Required for scattering |
| Calibration | Known standards (latex spheres for DLS) |
| Model selection | Start simple, add complexity if justified |
| Error analysis | Bootstrap for non-linear fits |
| Physical constraints | D > 0, 0 < β ≤ 1 |

---

## Checklist

- [ ] Background/baseline corrected
- [ ] Calibration with known standards
- [ ] Appropriate model selected (F-test, AIC)
- [ ] Physical constraints enforced
- [ ] Errors propagated through derived quantities
- [ ] Results validated against sum rules
- [ ] Confidence intervals reported

---

**Version**: 1.0.5
