---
name: correlation-math-foundations
description: Master mathematical foundations of correlation functions including two-point C(r) = ⟨φ(r)φ(0)⟩ - ⟨φ⟩² and higher-order (three-point, four-point χ₄(t) for dynamic heterogeneity) correlations, cumulants (κ₂ = variance, κ₃ = skewness, κ₄ connects to non-Gaussian fluctuations), transform methods (Fourier for structure factor S(k) and spectral density I(ω), Laplace for continuous relaxation spectra, wavelet for multi-scale time-frequency analysis), Wiener-Khinchin theorem S(ω) = ∫C(t)e^(-iωt)dt connecting time correlation to spectral density, Ornstein-Zernike equations h̃(k) = c̃(k)/[1-ρc̃(k)] separating direct vs indirect correlations with Percus-Yevick and hypernetted chain closures, fluctuation-dissipation theorem χ_AB(t) = β d/dt⟨A(t)B(0)⟩ connecting equilibrium correlations to linear response, and Green's functions G(r,t) = ⟨φ(r,t)φ(0,0)⟩ as propagators. Use when developing correlation theory for novel materials, analyzing finite-size scaling near critical points (ξ ~ L), connecting microscopic correlations to macroscopic response functions, or deriving transport laws from time-correlation functions.
---

# Mathematical Foundations & Transform Methods

## When to use this skill

- Deriving two-point correlation functions C(r) = ⟨φ(r)φ(0)⟩ - ⟨φ⟩² for density, spin, or field fluctuations in statistical mechanics with proper symmetry properties (*.py theory development, *.tex mathematical derivations)
- Computing higher-order correlation functions: three-point C³(r₁,r₂) for three-body interactions, four-point χ₄(t) for dynamic heterogeneity in glasses, or six-point for critical phenomena
- Calculating cumulants to remove factorizable contributions: κ₂ = variance, κ₃ = skewness detecting asymmetry, κ₄ measuring non-Gaussian tails in fluctuation distributions
- Implementing Fourier transforms for spatial correlations to structure factor S(k) = 1 + ρ∫C(r)e^(ik·r)dr or temporal correlations to spectral density via FFT (*.py NumPy/SciPy)
- Performing inverse Fourier transforms from scattering data S(k) to real-space pair distribution g(r) using C(r) = (2π)^(-d)∫C̃(k)e^(-ik·r)dk with proper normalization
- Applying Laplace transforms C̃(s) = ∫₀^∞ C(t)e^(-st)dt for continuous relaxation spectra, non-exponential decay analysis, or impedance spectroscopy interpretation
- Implementing wavelet transforms W(a,b) for multi-scale time-frequency analysis with scale parameter a and translation b to detect intermittent dynamics, bursts, or non-stationary correlations
- Deriving Wiener-Khinchin theorem applications: compute power spectral density from autocorrelation for DLS, electrical noise (Johnson-Nyquist), or hydrodynamic modes (Brillouin peaks)
- Solving Ornstein-Zernike integral equations h(r) = c(r) + ρ∫c(|r-r'|)h(r')dr' to separate direct correlation c(r) from total correlation h(r) = g(r)-1 in liquid structure theory
- Applying Percus-Yevick closure c(r) = [1-e^(βu(r))]g(r) or hypernetted chain closure c(r) = g(r)-1-ln g(r)-βu(r) to solve for structure factors from pair potentials
- Deriving fluctuation-dissipation theorem χ_AB(t) = β d/dt⟨A(t)B(0)⟩_{eq} to connect equilibrium correlations (conductivity σ from current-current ⟨j(t)j(0)⟩, susceptibility χ from magnetization ⟨M(t)M(0)⟩) to linear response
- Detecting fluctuation-dissipation violations in non-equilibrium systems: generalized FDT with effective temperature T_eff > T for driven, active, or glassy systems
- Computing Green's functions G(r,t) = ⟨φ(r,t)φ(0,0)⟩ as propagators for diffusion G(r,t) = (4πDt)^(-d/2)exp(-r²/4Dt), wave propagation, or quantum systems
- Extracting density of states ρ(ω) ∝ Im G(ω) or spectral function A(k,ω) = -2Im G(k,ω)/π from Green's functions for photoemission spectroscopy, tunneling, or optical absorption
- Analyzing finite-size effects: correlation length ξ vs system size L with finite-size scaling ξ(L) ~ L^ν at criticality, T_c shift T_c(L) = T_c(∞) + aL^(-1/ν), or rounding of phase transitions
- Correcting finite-size artifacts in simulations using periodic boundary condition corrections for long-range correlations with image summations or Ewald summation techniques
- Validating sum rules: compressibility S(k→0) = ρkTκ_T, number conservation ∫[S(k)-1]dk = 0, moment relations ⟨ω^n⟩ = ∫ω^n S(ω)dω = i^n d^n/dt^n C(t)|_{t=0}
- Verifying physical constraints: non-negativity C(0) ≥ |C(t)|, causality χ(t<0) = 0, Kramers-Kronig relations connecting real and imaginary parts of response functions
- Implementing translational invariance for homogeneous systems C(r,r') = C(|r-r'|) to simplify Fourier transforms and use periodic boundary conditions
- Checking stationarity C(t,t') = C(t-t') for equilibrium or steady-state systems enabling Wiener-Khinchin theorem and spectral analysis
- Deriving symmetry relations: time-reversal C(t) = C(-t) for equilibrium, Onsager relations C_AB(t) = C_BA(t), or spatial symmetries (isotropic, cubic) for crystalline systems
- Computing critical exponents from correlation functions: C(r) ~ r^(-(d-2+η)) at T_c, correlation length divergence ξ ~ |T-T_c|^(-ν), or universal amplitude ratios

Master the mathematical theory of correlation functions, transform methods, and fundamental theorems connecting correlations to physical observables.

## Fundamental Theory

### Two-Point Correlation Functions

**Definition:**
C(r) = ⟨φ(r)φ(0)⟩ - ⟨φ⟩²

- Measures joint fluctuations at two points
- Central quantity in statistical mechanics
- Relates to structure and dynamics

**Properties:**
- C(0) = ⟨(δφ)²⟩ = variance
- C(r→∞) → 0 for short-range correlations
- Symmetric: C(r) = C(-r) for equilibrium systems

**Time-Dependent:**
C(t) = ⟨A(t)A(0)⟩ - ⟨A⟩²
- Autocorrelation function
- Measures memory/persistence
- Decay rate: relaxation time τ

### Higher-Order Correlation Functions

**Three-Point:**
C³(r₁,r₂) = ⟨φ(0)φ(r₁)φ(r₂)⟩ - ⟨φ⟩³ - ⟨φ⟩[C(r₁) + C(r₂) + C(|r₁-r₂|)]

- Captures three-body interactions
- Non-zero for non-Gaussian fluctuations
- Important in critical phenomena

**Four-Point (Dynamic Heterogeneity):**
χ₄(t) = ⟨[C(0,t)]²⟩ - ⟨C(0,t)⟩²

- Measures spatial correlations of dynamics
- Growing length scale near glass transition
- Signature of cooperative motion

### Cumulants

**Cumulant expansion:**
κ₁ = ⟨φ⟩
κ₂ = ⟨φ²⟩ - ⟨φ⟩² = variance
κ₃ = ⟨φ³⟩ - 3⟨φ⟩⟨φ²⟩ + 2⟨φ⟩³ = skewness
κ₄ = ⟨φ⁴⟩ - 3⟨φ²⟩² - 4⟨φ⟩⟨φ³⟩ + 12⟨φ⟩²⟨φ²⟩ - 6⟨φ⟩⁴

**Purpose:**
- Remove trivial factorizable contributions
- κₙ = 0 for n ≥ 3 if Gaussian distribution
- Measure "true" correlations beyond mean-field

**Applications:**
- Spin models: Cumulants reveal phase transitions
- Density fluctuations: Connect to compressibility
- Critical phenomena: Universal ratios of cumulants

## Statistical Properties

### Translational Invariance

**Homogeneous systems:**
C(r,r') = C(r-r') = C(|r-r'|)

- Depends only on separation, not absolute position
- Simplifies Fourier transforms
- Basis for periodic boundary conditions

### Stationarity

**Time-translation invariance:**
C(t,t') = C(t-t') = C(τ)

- Equilibrium or steady-state systems
- Enables Wiener-Khinchin theorem
- Required for spectral analysis

### Symmetry Relations

**Time-reversal (equilibrium):**
C(t) = C(-t)

**Spatial symmetry:**
- Isotropic: C(r) = C(r) depends only on |r|
- Cubic: C(x,y,z) respects lattice symmetry

**Onsager relations:**
C_AB(t) = C_BA(t) for equilibrium

### Finite-Size Effects

**Correlation length ξ vs system size L:**

**Short-range (ξ << L):**
- Correlations decay exponentially: C(r) ~ e^(-r/ξ)
- Minimal finite-size effects
- Safe to use periodic boundaries

**Critical regime (ξ ~ L):**
- Finite-size scaling: ξ(L) ~ L^ν
- Shift critical temperature: T_c(L) = T_c(∞) + aL^(-1/ν)
- Rounding of phase transitions

**Long-range (ξ >> L):**
- Dominated by finite-size effects
- Need careful extrapolation L → ∞
- Universal scaling functions

**Applications:**
- Ising model: Finite-size scaling at criticality
- Polymers: Chain length vs persistence length
- Active matter: Correlation length from activity

## Transform Methods

### Fourier Transform

**Spatial:**
C̃(k) = ∫ C(r) e^(ik·r) dr
C(r) = (2π)^(-d) ∫ C̃(k) e^(-ik·r) dk

**Temporal:**
C̃(ω) = ∫ C(t) e^(iωt) dt
C(t) = (2π)^(-1) ∫ C̃(ω) e^(-iωt) dω

**Applications:**
- Structure factor: S(k) = 1 + ρC̃(k)
- Spectral density: I(ω) ∝ C̃(ω)
- Efficient computation via FFT: O(N log N)

```python
import numpy as np

def spatial_fft_correlation(positions, values, k_grid):
    """
    Compute spatial correlation via FFT
    """
    # Grid data
    grid = bin_data_to_grid(positions, values)
    
    # FFT
    C_k = np.fft.fftn(grid)
    C_k = np.abs(C_k)**2  # Power spectrum
    
    # Interpolate to k_grid
    return interpolate_to_k(C_k, k_grid)

def temporal_fft_correlation(time_series, dt):
    """
    Compute power spectral density via FFT
    """
    N = len(time_series)
    C_omega = np.fft.fft(time_series)
    frequencies = np.fft.fftfreq(N, dt)
    
    # Power spectrum
    power = np.abs(C_omega)**2 / N
    
    return frequencies[:N//2], power[:N//2]
```

### Laplace Transform

**Definition:**
C̃(s) = ∫₀^∞ C(t) e^(-st) dt

**Applications:**
- Continuous relaxation spectra
- Non-exponential decay analysis
- Connection to impedance spectroscopy

**Inverse Laplace:**
- Numerically ill-posed
- Regularization methods (Tikhonov, MaxEnt)
- Extract distribution of relaxation times

### Wavelet Transform

**Definition:**
W(a,b) = ∫ C(t) ψ*((t-b)/a) dt / √a

- a: scale parameter
- b: translation parameter
- ψ: mother wavelet

**Advantages:**
- Multi-scale analysis
- Time-frequency localization
- Adaptive to signal features

**Applications:**
- Intermittent dynamics in turbulence
- Burst analysis in active matter
- Non-stationary correlations

## Advanced Methods

### Wiener-Khinchin Theorem

**Statement:**
S(ω) = ∫₋∞^∞ C(t) e^(-iωt) dt

- Connects time correlation to spectral density
- Valid for stationary processes
- Basis for spectral analysis

**Applications:**
- DLS: S(ω) from intensity autocorrelation g₂(τ)
- Electrical noise: Johnson-Nyquist from voltage correlations
- Hydrodynamic modes: Brillouin peaks from density correlations

**Computational:**
```python
def wiener_khinchin_spectrum(time_series, dt):
    """
    Compute spectral density via autocorrelation
    """
    # Autocorrelation
    C = np.correlate(time_series - time_series.mean(), 
                     time_series - time_series.mean(), 
                     mode='full')
    C = C[len(C)//2:]  # Positive lags
    
    # FFT to spectrum
    S = np.fft.fft(C)
    freqs = np.fft.fftfreq(len(C), dt)
    
    return freqs[:len(freqs)//2], S[:len(S)//2].real
```

### Ornstein-Zernike Equations

**Integral equation:**
h(r) = c(r) + ρ ∫ c(|r-r'|) h(r') dr'

- h(r) = g(r) - 1: total correlation
- c(r): direct correlation
- Separates direct vs indirect correlations

**Fourier space:**
h̃(k) = c̃(k) + ρc̃(k)h̃(k)
→ h̃(k) = c̃(k) / [1 - ρc̃(k)]

**Closure relations:**
- Percus-Yevick: c(r) = [1 - e^(βu(r))]g(r)
- Hypernetted chain: c(r) = g(r) - 1 - ln g(r) - βu(r)

**Applications:**
- Liquid structure theory
- Structure factors for interacting systems
- Pair potentials from scattering data

### Response Theory

**Linear response:**
⟨A(t)⟩ - ⟨A⟩₀ = ∫₀^t χ_AB(t-t') h_B(t') dt'

- χ_AB: response function
- h_B: external perturbation

**Fluctuation-dissipation theorem (FDT):**
χ_AB(t) = β d/dt ⟨A(t)B(0)⟩_{eq}

- Connects equilibrium correlations to response
- Valid for linear regime near equilibrium
- Violations in non-equilibrium systems

**Applications:**
- Conductivity: σ = β∫⟨j(t)j(0)⟩dt from current correlations
- Susceptibility: χ = β⟨(ΔM)²⟩ from magnetization fluctuations
- Dielectric response: ε(ω) from dipole correlations

**Generalized FDT (non-equilibrium):**
χ(t,t') = β T_eff(t,t') ∂/∂t' C(t,t')

- T_eff: effective temperature
- T_eff > T for driven systems
- Measure of non-equilibrium driving

### Green's Functions

**Definition:**
G(r,t) = ⟨φ(r,t)φ(0,0)⟩

- Propagator for fluctuations
- Satisfies equation of motion

**Applications:**
- Diffusion: G(r,t) = (4πDt)^(-d/2) exp(-r²/4Dt)
- Wave propagation: G(r,t) for acoustic modes
- Quantum systems: Feynman propagator

**Connection to observables:**
- Density of states: ρ(ω) ∝ Im G(ω)
- Spectral function: A(k,ω) = -2 Im G(k,ω)/π

## Sum Rules and Constraints

### Sum Rules

**Static structure factor:**
S(k→0) = ρkTκ_T (compressibility)
∫ [S(k) - 1] dk = 0 (number conservation)

**Time-integrated:**
∫₀^∞ C(t) dt = D (diffusion from velocity autocorrelation)

**Moments:**
⟨ω^n⟩ = ∫ ω^n S(ω) dω = i^n d^n/dt^n C(t)|_{t=0}

### Physical Constraints

- Non-negativity: C(0) ≥ |C(t)| for all t
- Causality: χ(t) = 0 for t < 0
- Kramers-Kronig: Re χ(ω) and Im χ(ω) related by Hilbert transform

## Computational Implementation

```python
import numpy as np
from scipy import fft, signal

class CorrelationAnalyzer:
    """Mathematical foundations for correlation analysis"""
    
    def two_point_correlation(self, data, max_lag=None):
        """Compute two-point autocorrelation"""
        data = data - data.mean()
        C = np.correlate(data, data, mode='full')
        C = C[len(C)//2:]  # Positive lags
        C = C / C[0]  # Normalize
        
        if max_lag:
            C = C[:max_lag]
        
        return C
    
    def higher_order_correlation(self, data, order=3):
        """Compute higher-order correlations (cumulants)"""
        if order == 2:
            return np.var(data)
        elif order == 3:
            mean = data.mean()
            return np.mean((data - mean)**3)
        elif order == 4:
            mean, var = data.mean(), np.var(data)
            return np.mean((data - mean)**4) - 3*var**2
    
    def structure_factor(self, positions, box_size, k_max=10):
        """Compute structure factor via spatial FFT"""
        # Create density field
        bins = int(box_size * k_max / (2*np.pi))
        rho, edges = np.histogramdd(positions, bins=bins, 
                                     range=[(0,box_size)]*3)
        
        # FFT
        S_k = np.abs(fft.fftn(rho))**2
        
        # Radial average
        return self.radial_average(S_k)
    
    def spectral_density(self, time_series, dt):
        """Wiener-Khinchin spectral density"""
        C = self.two_point_correlation(time_series)
        S = fft.fft(C)
        freqs = fft.fftfreq(len(C), dt)
        
        return freqs[:len(freqs)//2], S[:len(S)//2].real
```

## Best Practices

- **Finite-size checks**: Verify ξ << L for bulk properties
- **Long-time behavior**: Ensure decay to zero for short-range correlations
- **Sum rules**: Validate against analytical constraints
- **Transform accuracy**: Sufficient sampling for Fourier/Laplace transforms

References for advanced topics: field-theoretic correlation functions, path integral formulations, renormalization group analysis.
