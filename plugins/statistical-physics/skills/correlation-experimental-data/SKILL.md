---
name: correlation-experimental-data
description: Interpret experimental correlation data from DLS (dynamic light scattering with g₂(τ) intensity correlation, Siegert relation for field correlation g₁(τ), particle size from Stokes-Einstein with hydrodynamic radius R, stretched exponential KWW fits for polydisperse systems, multi-angle analysis for diffusion coefficient D), SAXS/SANS (structure factor S(q) extraction, pair distribution function g(r) via Fourier transform, Guinier analysis for radius of gyration R_g, Porod regime for surface properties with fractal dimension d), XPCS (two-time correlation C(t₁,t₂) for aging detection, slow dynamics milliseconds to hours, compressed exponential for ballistic/active motion), FCS (3D diffusion model with focal volume, multi-component analysis for free/bound populations, binding kinetics from τ_D), rheology (stress correlations for viscosity η via Green-Kubo, microrheology from particle tracking MSD, viscoelastic moduli G'(ω) and G''(ω)), and scattering experiments. Extract physical parameters (diffusion coefficients, particle sizes, relaxation times, interaction potentials), validate theoretical predictions, perform model fitting with Bayesian inference and MCMC uncertainty quantification, and connect correlation measurements to materials properties.
---

# Experimental Data Interpretation

## When to use this skill

- Analyzing DLS autocorrelation data g₂(τ) to extract particle sizes using Siegert relation g₂(τ) = 1 + β|g₁(τ)|² and Stokes-Einstein equation D = kT/(6πηR) (*.py analysis scripts, *.csv data files)
- Fitting stretched exponential (KWW) functions to DLS data from polydisperse systems, supercooled liquids, or colloidal glasses with β < 1 indicating distribution of relaxation times
- Performing multi-angle DLS analysis to extract diffusion coefficient D from linear fit of decay rate Γ vs q² across multiple scattering angles (90°, 60°, 120°, etc.)
- Extracting structure factor S(q) from concentrated colloidal suspensions using SAXS/SANS by comparing I(q)_concentrated / I(q)_dilute with dilute reference measurements
- Computing pair distribution function g(r) from structure factor S(q) via Fourier transform ρg(r) = ρ + (1/2π²r)∫q²[S(q)-1]sin(qr)dq for liquid structure analysis
- Performing Guinier analysis on small-angle scattering data to extract radius of gyration R_g from linear fit of ln(I) vs q² in qR_g < 1 regime without assuming particle shape
- Analyzing Porod regime at large q to determine surface properties: I(q) ~ q⁻⁴ for sharp interfaces, q⁻ᵈ for fractal surfaces with dimension d < 4
- Interpreting XPCS two-time correlation C(t₁,t₂) to detect aging, non-stationarity, or dynamic heterogeneity in glassy systems, gels, or driven suspensions
- Fitting compressed exponential (α > 1) to XPCS data from active matter, ballistic motion, or flowing systems where α = 2 indicates ballistic regime
- Analyzing FCS autocorrelation G(τ) to extract diffusion time τ_D, diffusion coefficient D = r₀²/(4τ_D), average number of molecules N, and concentration from focal volume
- Performing multi-component FCS analysis to separate fast/slow populations (free vs bound proteins), extract binding kinetics, or analyze oligomerization states
- Extracting complex viscosity η* = √(G'² + G''²)/ω from oscillatory rheology data with storage modulus G'(ω) and loss modulus G''(ω) measurements
- Computing relaxation time τ from rheology crossover frequency where G'(ω) = G''(ω) to characterize polymer relaxation, gel formation, or yield stress fluids
- Performing microrheology from particle tracking MSD to extract complex modulus G*(ω) = kT/(πa⟨Δr²(ω)⟩) using generalized Stokes-Einstein relation
- Calculating transport coefficients from Green-Kubo relations: viscosity η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩dt or diffusion D = ∫⟨v(t)·v(0)⟩dt from correlation functions
- Validating theoretical predictions against experimental correlations using sum rules: S(q→0) = ρkTκ_T (compressibility), ∫[S(q)-1]dq = 0 (number conservation), Kramers-Kronig for response functions
- Performing Bayesian parameter estimation with MCMC (emcee, PyMC3) to extract parameter posteriors, credible intervals, and uncertainty quantification from experimental correlation data
- Conducting model selection using Bayesian Information Criterion (BIC = -2ln(L) + k ln(n)) or Bayes factors to compare single exponential vs stretched exponential vs multi-component models
- Fitting non-exponential relaxation models to experimental data: stretched exponential for glasses, compressed exponential for active systems, multi-exponential for multiple relaxation processes
- Extracting interaction potentials from structure factor measurements using Ornstein-Zernike integral equations with Percus-Yevick or hypernetted chain closures
- Analyzing temperature, concentration, or angle-dependent scattering data to validate scaling laws, identify phase transitions, or extract critical exponents
- Reporting experimental results with proper uncertainty propagation: radius R_err = R(D_err/D) from diffusion uncertainty, confidence intervals from bootstrap resampling (N=1000 samples)

Interpret correlation measurements from scattering, microscopy, and spectroscopy experiments, connecting raw data to physical parameters and materials properties.

## Dynamic Light Scattering (DLS)

### Intensity Correlation Function

**Measured Quantity:**
g₂(τ) = ⟨I(t)I(t+τ)⟩ / ⟨I⟩²

**Siegert Relation:**
g₂(τ) = 1 + β|g₁(τ)|²
- β: coherence factor (instrument-dependent, typically 0.5-1.0)
- g₁(τ): field correlation function (what we want)

**For Brownian particles:**
g₁(τ) = exp(-Γτ)
- Γ = Dq²: decay rate
- D: diffusion coefficient
- q = (4πn/λ)sin(θ/2): scattering vector

**Extract particle size:**
D = kT/(6πηR) (Stokes-Einstein)
- R: hydrodynamic radius
- η: solvent viscosity
- T: temperature

```python
def analyze_dls(tau, g2, temperature=298, viscosity=0.001, 
                wavelength=632.8e-9, angle=90, n_solvent=1.33):
    """
    Extract particle size from DLS autocorrelation
    
    Parameters:
    -----------
    tau : array, lag times (seconds)
    g2 : array, intensity correlation g₂(τ)
    temperature : float, K
    viscosity : float, Pa·s
    wavelength : float, m
    angle : float, degrees
    n_solvent : float, refractive index
    
    Returns:
    --------
    R : float, hydrodynamic radius (m)
    D : float, diffusion coefficient (m²/s)
    beta : float, coherence factor
    """
    from scipy.optimize import curve_fit
    
    # Scattering vector
    q = 4*np.pi*n_solvent/wavelength * np.sin(np.radians(angle)/2)
    
    # Fit g₂(τ) = 1 + β exp(-2Γτ)
    def g2_model(tau, beta, Gamma):
        return 1 + beta * np.exp(-2*Gamma*tau)
    
    popt, pcov = curve_fit(g2_model, tau, g2, p0=[0.8, 1000])
    beta, Gamma = popt
    beta_err, Gamma_err = np.sqrt(np.diag(pcov))
    
    # Diffusion coefficient
    D = Gamma / q**2
    D_err = Gamma_err / q**2
    
    # Hydrodynamic radius
    kT = 1.38e-23 * temperature
    R = kT / (6*np.pi*viscosity*D)
    R_err = R * (D_err / D)  # Propagate error
    
    return {
        'radius': R,
        'radius_err': R_err,
        'diffusion': D,
        'diffusion_err': D_err,
        'beta': beta,
        'Gamma': Gamma
    }
```

### Non-Exponential Relaxation

**Stretched Exponential (KWW):**
g₁(τ) = exp[-(τ/τ_c)^β]
- β < 1: Distribution of relaxation times
- β → 0: Broad distribution (glassy systems)
- β = 1: Single exponential (Brownian)

**Physical Interpretation:**
- β = 0.5-0.7: Supercooled liquids, colloidal glasses
- Distribution width: Δτ/τ_c ~ β^(-1)

```python
def fit_stretched_exponential(tau, g2):
    """
    Fit stretched exponential to DLS data
    
    g₂(τ) = 1 + β_coherence exp(-2(τ/τ_c)^β_stretch)
    """
    from scipy.optimize import curve_fit
    
    def g2_kww(tau, beta_coh, tau_c, beta_stretch):
        return 1 + beta_coh * np.exp(-2*(tau/tau_c)**beta_stretch)
    
    # Initial guess
    p0 = [0.8, tau[len(tau)//2], 0.7]
    
    # Fit with bounds
    bounds = ([0, 0, 0.1], [1, tau[-1]*10, 1.0])
    popt, pcov = curve_fit(g2_kww, tau, g2, p0=p0, bounds=bounds)
    
    beta_coh, tau_c, beta_stretch = popt
    errors = np.sqrt(np.diag(pcov))
    
    # Compute average relaxation time
    from scipy.special import gamma
    tau_avg = tau_c * gamma(1/beta_stretch) / beta_stretch
    
    return {
        'tau_c': tau_c,
        'beta_stretch': beta_stretch,
        'tau_avg': tau_avg,
        'errors': errors
    }
```

### Multi-Angle DLS

**q-Dependence:**
- Translational diffusion: Γ = Dq²
- Internal modes (polymers, micelles): Additional q-dependent terms

```python
def multi_angle_analysis(angles, Gamma, wavelength=632.8e-9, n=1.33):
    """
    Extract diffusion from multi-angle DLS
    
    Plot Γ vs q² should be linear: Γ = Dq²
    """
    q = 4*np.pi*n/wavelength * np.sin(np.radians(angles)/2)
    q_squared = q**2
    
    # Linear fit
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(q_squared, Gamma)
    
    D = slope
    D_err = std_err
    
    # Check linearity (should have intercept ≈ 0)
    if abs(intercept) > 0.1*slope*q_squared.max():
        print("Warning: Non-zero intercept suggests internal modes or polydispersity")
    
    return D, D_err, r_value**2
```

## SAXS/SANS - Small-Angle Scattering

### Structure Factor Extraction

**Measured Intensity:**
I(q) = P(q) S(q)
- P(q): form factor (single particle shape)
- S(q): structure factor (inter-particle correlations)

**For dilute systems:** S(q) ≈ 1, I(q) ≈ P(q)
**For concentrated systems:** Must extract S(q)

```python
def extract_structure_factor(q, I_concentrated, I_dilute):
    """
    Extract structure factor from SAXS/SANS
    
    S(q) = I_concentrated(q) / I_dilute(q)
    
    Requires dilute reference measurement
    """
    # Normalize concentrations
    # c_dilute / c_concentrated ratio
    
    S_q = I_concentrated / I_dilute
    
    # Check sum rule: S(q→0) = ρkTκ_T
    # For hard spheres: S(0) < 1 (repulsion)
    # For attractive: S(0) can be > 1
    
    return S_q

def pair_distribution_from_saxs(q, S_q, rho):
    """
    Fourier transform S(q) to get g(r)
    
    ρg(r) = ρ + (1/(2π²r)) ∫ q²[S(q)-1] sin(qr) dq
    """
    from scipy.integrate import simps
    
    r = np.linspace(0.1, 50, 500)  # Angstroms
    g_r = np.zeros_like(r)
    
    for i, r_val in enumerate(r):
        integrand = q**2 * (S_q - 1) * np.sin(q * r_val)
        g_r[i] = rho + (1/(2*np.pi**2*r_val)) * simps(integrand, q)
    
    return r, g_r
```

### Guinier Analysis

**Small-q Regime (qR_g < 1):**
I(q) ≈ I(0) exp(-q²R_g²/3)
- R_g: radius of gyration
- Extract size without assuming shape

```python
def guinier_analysis(q, I, q_max=None):
    """
    Guinier analysis for radius of gyration
    
    ln(I) vs q² should be linear for qR_g < 1
    """
    if q_max is None:
        q_max = 0.1  # nm⁻¹, adjust based on expected size
    
    mask = q < q_max
    q_fit = q[mask]
    I_fit = I[mask]
    
    # Linear fit to ln(I) vs q²
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        q_fit**2, np.log(I_fit)
    )
    
    R_g = np.sqrt(-3 * slope)
    R_g_err = np.sqrt(3 * std_err / abs(slope))
    
    I_0 = np.exp(intercept)
    
    return {
        'R_g': R_g,
        'R_g_err': R_g_err,
        'I_0': I_0,
        'q_max': q_max,
        'r_squared': r_value**2
    }
```

### Porod Regime

**Large-q (qR >> 1):**
I(q) ~ q^(-d)
- d = 4: Sharp interface (Porod law)
- d < 4: Fractal surface
- d > 4: Diffuse interface

```python
def porod_analysis(q, I, q_min=0.5):
    """
    Porod analysis for surface properties
    
    Log-log plot: slope gives dimensionality
    """
    mask = q > q_min
    q_fit = q[mask]
    I_fit = I[mask]
    
    # Log-log fit
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log(q_fit), np.log(I_fit)
    )
    
    d = -slope
    
    # Interpret
    if abs(d - 4) < 0.2:
        surface_type = "Sharp interface (Porod)"
    elif d < 4:
        surface_type = f"Fractal surface, dimension {d:.2f}"
    else:
        surface_type = "Diffuse interface"
    
    return {
        'exponent': d,
        'exponent_err': std_err,
        'surface_type': surface_type,
        'r_squared': r_value**2
    }
```

## XPCS - X-Ray Photon Correlation Spectroscopy

### Slow Dynamics Analysis

**XPCS probes slower dynamics than DLS:**
- DLS: microseconds to seconds
- XPCS: milliseconds to hours

**Two-Time Correlation:**
C(t₁, t₂) = ⟨I(q, t₁)I(q, t₂)⟩ / ⟨I(q)⟩²
- Non-stationary: Aging, drift
- Two-time plot reveals heterogeneity

```python
def two_time_correlation(intensity_frames, q_index):
    """
    Compute two-time correlation C(t₁, t₂)
    
    Parameters:
    -----------
    intensity_frames : array, shape (n_times, n_q, n_pixels)
    q_index : int, which q to analyze
    
    Returns:
    --------
    C : array, shape (n_times, n_times)
    """
    I = intensity_frames[:, q_index, :]
    n_times = len(I)
    
    C = np.zeros((n_times, n_times))
    
    for t1 in range(n_times):
        for t2 in range(t1, n_times):
            C[t1, t2] = np.mean(I[t1] * I[t2]) / np.mean(I[t1]) / np.mean(I[t2])
            C[t2, t1] = C[t1, t2]  # Symmetric
    
    return C

def analyze_aging(C_two_time):
    """
    Detect aging from two-time correlation
    
    Stationary: C depends only on t₂-t₁
    Aging: C depends on both t₁ and t₂
    """
    n_times = len(C_two_time)
    
    # Diagonal: Short-time dynamics
    diag_values = [C_two_time[i, i+10] for i in range(n_times-10)]
    
    # Check if decay rate changes with observation time
    aging_index = np.std(diag_values) / np.mean(diag_values)
    
    if aging_index > 0.1:
        print("Aging detected: Dynamics depend on observation time")
    else:
        print("Stationary dynamics")
    
    return aging_index
```

### Compressed Exponential

**Ballistic Motion:**
g₁(τ) = exp(-Γτ)^α with α > 1
- α = 2: Ballistic (v²t²)
- α = 1: Brownian diffusion
- Active matter, driven systems

```python
def fit_compressed_exponential(tau, g2):
    """
    Fit compressed exponential (α > 1)
    
    Common in active matter, XPCS of flowing systems
    """
    from scipy.optimize import curve_fit
    
    def g2_compressed(tau, beta, Gamma, alpha):
        return 1 + beta * np.exp(-2*(Gamma*tau)**alpha)
    
    p0 = [0.8, 1.0, 1.5]
    bounds = ([0, 0, 1.0], [1, np.inf, 3.0])
    
    popt, pcov = curve_fit(g2_compressed, tau, g2, p0=p0, bounds=bounds)
    
    return {
        'beta': popt[0],
        'Gamma': popt[1],
        'alpha': popt[2],
        'errors': np.sqrt(np.diag(pcov))
    }
```

## FCS - Fluorescence Correlation Spectroscopy

### Autocorrelation Analysis

**FCS correlation:**
G(τ) = ⟨δF(t)δF(t+τ)⟩ / ⟨F⟩²

**For 3D diffusion:**
G(τ) = (1/N) × (1/(1+τ/τ_D)) × (1/√(1+τ/(ω²τ_D)))
- N: average number of molecules in focal volume
- τ_D: diffusion time through focal volume
- ω: aspect ratio of focal volume (z₀/r₀)

**Extract diffusion:**
D = r₀²/(4τ_D)
- r₀: focal volume radius (from calibration)

```python
def fcs_analysis(tau, G, r0=0.2e-6, omega=5):
    """
    Analyze FCS autocorrelation
    
    Parameters:
    -----------
    tau : array, lag times (seconds)
    G : array, correlation G(τ)
    r0 : float, focal radius (m)
    omega : float, aspect ratio z₀/r₀
    
    Returns:
    --------
    N : average number of molecules
    D : diffusion coefficient (m²/s)
    tau_D : diffusion time (s)
    """
    from scipy.optimize import curve_fit
    
    def G_3d_diffusion(tau, N, tau_D):
        return (1/N) * (1/(1+tau/tau_D)) * (1/np.sqrt(1+tau/(omega**2*tau_D)))
    
    # Fit
    popt, pcov = curve_fit(G_3d_diffusion, tau, G, p0=[10, 1e-3])
    N, tau_D = popt
    N_err, tau_D_err = np.sqrt(np.diag(pcov))
    
    # Diffusion coefficient
    D = r0**2 / (4*tau_D)
    D_err = D * (tau_D_err / tau_D)
    
    # Concentration
    V_eff = np.pi**(3/2) * r0**3 * omega
    C = N / (V_eff * 6.022e23)  # Molar
    
    return {
        'N': N,
        'N_err': N_err,
        'tau_D': tau_D,
        'tau_D_err': tau_D_err,
        'D': D,
        'D_err': D_err,
        'concentration': C
    }
```

### Multi-Component Analysis

**Two species:**
G(τ) = (1/(N₁+N₂)²) × [N₁²G₁(τ) + N₂²G₂(τ)]

**Binding kinetics:**
- Free + Bound populations
- Different diffusion times
- Extract binding constants

```python
def fcs_two_component(tau, G, r0=0.2e-6):
    """
    Fit two-component FCS (e.g., free + bound)
    """
    from scipy.optimize import curve_fit
    
    def G_two_species(tau, N1, tau_D1, N2, tau_D2):
        G1 = (1/(1+tau/tau_D1)) * (1/np.sqrt(1+tau/(25*tau_D1)))
        G2 = (1/(1+tau/tau_D2)) * (1/np.sqrt(1+tau/(25*tau_D2)))
        
        N_tot = N1 + N2
        return (1/N_tot**2) * (N1**2*G1 + N2**2*G2)
    
    # Initial guess: fast component 10× faster
    p0 = [5, 1e-3, 5, 1e-2]
    
    popt, pcov = curve_fit(G_two_species, tau, G, p0=p0)
    N1, tau_D1, N2, tau_D2 = popt
    
    # Diffusion coefficients
    D1 = r0**2 / (4*tau_D1)
    D2 = r0**2 / (4*tau_D2)
    
    # Fractions
    frac1 = N1 / (N1 + N2)
    frac2 = N2 / (N1 + N2)
    
    return {
        'D_fast': max(D1, D2),
        'D_slow': min(D1, D2),
        'frac_fast': frac1 if D1 > D2 else frac2,
        'frac_slow': frac2 if D1 > D2 else frac1
    }
```

## Rheology - Stress Correlations

### Microrheology from DLS

**Generalized Stokes-Einstein:**
G*(ω) = kT/(πaΔr²(ω))
- a: particle radius
- Δr²(ω): mean-squared displacement at frequency ω

```python
def microrheology_from_msd(omega, msd, a, T=298):
    """
    Extract complex modulus from MSD
    
    G*(ω) from particle tracking or DLS
    """
    kT = 1.38e-23 * T
    
    # Generalized Stokes-Einstein
    G_star = kT / (np.pi * a * msd)
    
    return G_star

def extract_viscosity_from_correlation(C_v, dt, T=298, m=1e-15):
    """
    Viscosity from velocity autocorrelation (Green-Kubo)
    
    η = ∫₀^∞ ⟨v(t)v(0)⟩ dt
    """
    # Integrate correlation
    from scipy.integrate import simps
    
    times = np.arange(len(C_v)) * dt
    eta = simps(C_v, times) * m / (kT/T)
    
    return eta
```

## Connection to Theory

### Green-Kubo Relations

**Transport Coefficients from Correlations:**

**Diffusion:**
D = ∫₀^∞ ⟨v(t)·v(0)⟩ dt

**Viscosity:**
η = (V/kT) ∫₀^∞ ⟨σ_xy(t)σ_xy(0)⟩ dt

**Thermal conductivity:**
κ = (V/kT²) ∫₀^∞ ⟨J_q(t)·J_q(0)⟩ dt

```python
def green_kubo_transport(correlation, dt, volume, T):
    """
    Calculate transport coefficient from time-correlation
    
    Parameters:
    -----------
    correlation : array, C(t) = ⟨A(t)A(0)⟩
    dt : float, time step
    volume : float, system volume
    T : float, temperature
    
    Returns:
    --------
    L : transport coefficient
    """
    from scipy.integrate import simps
    
    kT = 1.38e-23 * T
    times = np.arange(len(correlation)) * dt
    
    # Integrate
    L = (volume/kT) * simps(correlation, times)
    
    return L
```

### Validation Against Sum Rules

**Check Consistency:**
- S(q→0) = ρkTκ_T (compressibility sum rule)
- ∫ [S(q)-1] dq = 0 (number conservation)
- Kramers-Kronig relations for response functions

```python
def validate_sum_rules(q, S_q, rho, T, kappa_T):
    """
    Check structure factor sum rules
    """
    kT = 1.38e-23 * T
    
    # Compressibility sum rule
    S_0_theory = rho * kT * kappa_T
    S_0_measured = np.interp(0, q, S_q)
    
    print(f"S(0) theory: {S_0_theory:.3f}")
    print(f"S(0) measured: {S_0_measured:.3f}")
    
    # Number conservation
    from scipy.integrate import simps
    integral = simps(S_q - 1, q)
    print(f"∫[S(q)-1]dq = {integral:.6f} (should be ≈ 0)")
    
    if abs(S_0_measured - S_0_theory) / S_0_theory > 0.1:
        print("Warning: S(0) deviates from theory")
```

## Best Practices

### Data Quality
- Background subtraction for scattering
- Baseline correction for DLS
- Calibration with known standards
- Multiple measurements for statistics

### Model Selection
- Start with simplest model (single exponential)
- Add complexity only if statistically justified (F-test, AIC)
- Physical constraints (D > 0, β ∈ [0,1])
- Cross-validate on independent data

### Error Analysis
- Bootstrap for non-linear fits
- Propagate errors through derived quantities
- Report confidence intervals, not just point estimates
- Check residuals for systematic deviations

### Reporting
- Include experimental conditions (T, concentration, solvent)
- Specify instrument parameters (wavelength, angle, focal volume)
- Provide raw data and fitting code
- Compare to literature values when available

References for advanced techniques: time-resolved correlation spectroscopy, multidimensional correlation analysis, machine learning for correlation pattern recognition.
