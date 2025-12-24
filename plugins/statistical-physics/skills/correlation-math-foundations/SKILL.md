---
name: correlation-math-foundations
version: "1.0.5"
maturity: "5-Expert"
specialization: Correlation Theory
description: Mathematical foundations of correlation functions including two-point C(r), higher-order χ₄(t), cumulants, Fourier/Laplace/wavelet transforms, Wiener-Khinchin theorem, Ornstein-Zernike equations, fluctuation-dissipation theorem, and Green's functions. Use when developing correlation theory or connecting microscopic correlations to macroscopic response.
---

# Mathematical Foundations of Correlation Functions

Theory connecting fluctuations to observables via transform methods and response theory.

---

## Two-Point Correlation

**Definition**: C(r) = ⟨φ(r)φ(0)⟩ - ⟨φ⟩²

| Property | Expression | Meaning |
|----------|------------|---------|
| Variance | C(0) = ⟨(δφ)²⟩ | Fluctuation magnitude |
| Decay | C(r→∞) → 0 | Short-range correlations |
| Symmetry | C(r) = C(-r) | Equilibrium systems |

**Time-Dependent**: C(t) = ⟨A(t)A(0)⟩ - ⟨A⟩²

---

## Higher-Order Correlations

### Three-Point
C³(r₁,r₂) = ⟨φ(0)φ(r₁)φ(r₂)⟩ - factorizable terms

### Four-Point (Dynamic Heterogeneity)
χ₄(t) = ⟨[C(0,t)]²⟩ - ⟨C(0,t)⟩²

- Growing correlation length near glass transition
- Signature of cooperative dynamics

### Cumulants

| Order | Expression | Measures |
|-------|------------|----------|
| κ₂ | ⟨φ²⟩ - ⟨φ⟩² | Variance |
| κ₃ | Third moment (connected) | Skewness |
| κ₄ | Fourth moment - 3σ⁴ | Non-Gaussian tails |

---

## Transform Methods

### Fourier Transform

**Spatial**: C̃(k) = ∫ C(r) e^(ik·r) dr

**Temporal**: C̃(ω) = ∫ C(t) e^(iωt) dt

```python
import numpy as np

def structure_factor(positions, box_size, k_max=10):
    """S(k) via FFT of density field."""
    bins = int(box_size * k_max / (2*np.pi))
    rho, _ = np.histogramdd(positions, bins=bins, range=[(0,box_size)]*3)
    S_k = np.abs(np.fft.fftn(rho))**2
    return radial_average(S_k)

def spectral_density(time_series, dt):
    """Power spectrum via Wiener-Khinchin."""
    C = np.correlate(time_series - time_series.mean(),
                     time_series - time_series.mean(), mode='full')
    C = C[len(C)//2:]
    S = np.fft.fft(C)
    return np.fft.fftfreq(len(C), dt)[:len(C)//2], S[:len(S)//2].real
```

### Laplace Transform

C̃(s) = ∫₀^∞ C(t) e^(-st) dt

- Continuous relaxation spectra
- Non-exponential decay analysis

### Wavelet Transform

W(a,b) = ∫ C(t) ψ*((t-b)/a) dt / √a

- Multi-scale time-frequency analysis
- Intermittent dynamics detection

---

## Wiener-Khinchin Theorem

**Statement**: S(ω) = ∫₋∞^∞ C(t) e^(-iωt) dt

| Application | Observable |
|-------------|------------|
| DLS | S(ω) from g₂(τ) |
| Johnson-Nyquist noise | Voltage correlations |
| Brillouin peaks | Density correlations |

---

## Ornstein-Zernike Equations

**Real space**: h(r) = c(r) + ρ ∫ c(|r-r'|) h(r') dr'

**Fourier space**: h̃(k) = c̃(k) / [1 - ρc̃(k)]

| Closure | Expression | Use Case |
|---------|------------|----------|
| Percus-Yevick | c(r) = [1-e^(βu)]g(r) | Hard spheres |
| HNC | c(r) = g(r)-1-ln g(r)-βu(r) | Charged systems |

---

## Fluctuation-Dissipation Theorem

**FDT**: χ_AB(t) = β d/dt ⟨A(t)B(0)⟩_eq

| Response | Correlation |
|----------|-------------|
| Conductivity σ | ⟨j(t)j(0)⟩ (current-current) |
| Susceptibility χ | ⟨M(t)M(0)⟩ (magnetization) |
| Dielectric ε(ω) | Dipole correlations |

**Generalized FDT** (non-equilibrium):
χ(t,t') = β T_eff ∂C(t,t')/∂t'

- T_eff > T for driven systems

---

## Green's Functions

**Definition**: G(r,t) = ⟨φ(r,t)φ(0,0)⟩

| System | Green's Function |
|--------|-----------------|
| Diffusion | G(r,t) = (4πDt)^(-d/2) exp(-r²/4Dt) |
| Wave | G(r,t) for acoustic modes |
| Quantum | Feynman propagator |

**Observables**:
- Density of states: ρ(ω) ∝ Im G(ω)
- Spectral function: A(k,ω) = -2Im G(k,ω)/π

---

## Finite-Size Effects

| Regime | Behavior |
|--------|----------|
| ξ << L | Exponential decay, minimal effects |
| ξ ~ L | Finite-size scaling: ξ(L) ~ L^ν |
| ξ >> L | Dominated by finite-size, extrapolate L→∞ |

**Critical**: T_c(L) = T_c(∞) + aL^(-1/ν)

---

## Sum Rules & Constraints

| Rule | Expression |
|------|------------|
| Compressibility | S(k→0) = ρkTκ_T |
| Number conservation | ∫[S(k)-1]dk = 0 |
| Moments | ⟨ω^n⟩ = i^n d^n/dt^n C(t)|_{t=0} |

**Physical Constraints**:
- Non-negativity: C(0) ≥ |C(t)|
- Causality: χ(t<0) = 0
- Kramers-Kronig: Re χ(ω) ↔ Im χ(ω)

---

## Implementation

```python
class CorrelationAnalyzer:
    def two_point(self, data, max_lag=None):
        data = data - data.mean()
        C = np.correlate(data, data, mode='full')
        C = C[len(C)//2:] / C[len(C)//2]
        return C[:max_lag] if max_lag else C

    def cumulant(self, data, order):
        if order == 2: return np.var(data)
        elif order == 3: return np.mean((data - data.mean())**3)
        elif order == 4:
            return np.mean((data - data.mean())**4) - 3*np.var(data)**2
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Finite-size checks | Verify ξ << L for bulk properties |
| Long-time decay | Ensure C(t→∞) → 0 |
| Sum rules | Validate compressibility, conservation |
| Sampling | Sufficient for Fourier accuracy |

---

## Checklist

- [ ] Correlation type identified (two-point, higher-order)
- [ ] Transform method selected (Fourier/Laplace/wavelet)
- [ ] Finite-size effects assessed
- [ ] Sum rules validated
- [ ] Physical constraints verified

---

**Version**: 1.0.5
