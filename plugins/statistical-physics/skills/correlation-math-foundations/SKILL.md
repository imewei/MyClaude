---
name: correlation-math-foundations
version: "1.0.7"
description: Mathematical foundations of correlation functions including two-point C(r), higher-order χ₄(t), cumulants, Fourier/Laplace/wavelet transforms, Wiener-Khinchin theorem, Ornstein-Zernike equations, fluctuation-dissipation theorem, and Green's functions. Use when developing correlation theory or connecting microscopic correlations to macroscopic response.
---

# Mathematical Foundations of Correlation Functions

## Two-Point Correlation

**Definition**: C(r) = ⟨φ(r)φ(0)⟩ - ⟨φ⟩²

| Property | Expression |
|----------|------------|
| Variance | C(0) = ⟨(δφ)²⟩ |
| Decay | C(r→∞) → 0 |
| Symmetry | C(r) = C(-r) |
| Time | C(t) = ⟨A(t)A(0)⟩ - ⟨A⟩² |

## Higher-Order

- **Three-point**: C³(r₁,r₂) = ⟨φ(0)φ(r₁)φ(r₂)⟩ - factorizable
- **Four-point**: χ₄(t) = ⟨[C(0,t)]²⟩ - ⟨C(0,t)⟩² (dynamic heterogeneity)
- **Cumulants**: κ₂ (variance), κ₃ (skewness), κ₄ (non-Gaussian tails)

## Transforms

### Fourier
**Spatial**: C̃(k) = ∫ C(r) e^(ik·r) dr
**Temporal**: C̃(ω) = ∫ C(t) e^(iωt) dt

```python
def structure_factor(positions, box_size, k_max=10):
    bins = int(box_size * k_max / (2*np.pi))
    rho, _ = np.histogramdd(positions, bins=bins, range=[(0,box_size)]*3)
    return radial_average(np.abs(np.fft.fftn(rho))**2)

def spectral_density(time_series, dt):
    C = np.correlate(time_series - time_series.mean(),
                     time_series - time_series.mean(), mode='full')
    C = C[len(C)//2:]
    S = np.fft.fft(C)
    return np.fft.fftfreq(len(C), dt)[:len(C)//2], S[:len(S)//2].real
```

### Laplace & Wavelet
- **Laplace**: C̃(s) = ∫₀^∞ C(t) e^(-st) dt (relaxation spectra)
- **Wavelet**: W(a,b) = ∫ C(t) ψ*((t-b)/a) dt / √a (multi-scale)

## Wiener-Khinchin

S(ω) = ∫₋∞^∞ C(t) e^(-iωt) dt

| Application | Observable |
|-------------|------------|
| DLS | S(ω) from g₂(τ) |
| Johnson-Nyquist | Voltage correlations |

## Ornstein-Zernike

**Fourier**: h̃(k) = c̃(k) / [1 - ρc̃(k)]

| Closure | Use |
|---------|-----|
| Percus-Yevick | Hard spheres |
| HNC | Charged systems |

## Fluctuation-Dissipation

**FDT**: χ_AB(t) = β d/dt ⟨A(t)B(0)⟩_eq

| Response | Correlation |
|----------|-------------|
| σ (conductivity) | ⟨j(t)j(0)⟩ |
| χ (susceptibility) | ⟨M(t)M(0)⟩ |
| ε(ω) (dielectric) | Dipole |

**Non-equilibrium**: χ(t,t') = β T_eff ∂C/∂t' (T_eff > T for driven)

## Green's Functions

G(r,t) = ⟨φ(r,t)φ(0,0)⟩

| System | G(r,t) |
|--------|--------|
| Diffusion | (4πDt)^(-d/2) exp(-r²/4Dt) |
| DOS | ρ(ω) ∝ Im G(ω) |

## Finite-Size

| Regime | Behavior |
|--------|----------|
| ξ << L | Minimal effects |
| ξ ~ L | ξ(L) ~ L^ν |
| ξ >> L | Extrapolate L→∞ |

**Critical**: T_c(L) = T_c(∞) + aL^(-1/ν)

## Sum Rules

| Rule | Expression |
|------|------------|
| Compressibility | S(k→0) = ρkTκ_T |
| Conservation | ∫[S(k)-1]dk = 0 |
| Moments | ⟨ω^n⟩ = i^n d^n/dt^n C(t)|_{t=0} |

**Constraints**: C(0) ≥ |C(t)|, χ(t<0) = 0, Kramers-Kronig

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

**Outcome**: Transform methods, FDT, Ornstein-Zernike, finite-size scaling, sum rules
