---
name: stochastic-dynamics
description: Model stochastic processes using master equations (continuous-time Markov chains dP_n/dt = Σ[W_mn P_m - W_nm P_n] for discrete states with Gillespie algorithm for exact stochastic simulation), Fokker-Planck equations (∂P/∂t = -∂(μF·P)/∂x + D·∂²P/∂x² governing probability distribution evolution with drift μF and diffusion D), and Langevin dynamics (dx/dt = μF(x) + √(2D)ξ(t) generating stochastic trajectories with white noise ⟨ξ(t)ξ(t')⟩ = δ(t-t') using Euler-Maruyama numerical integration). Calculate transport coefficients via Green-Kubo relations: diffusion D = ∫⟨v(t)·v(0)⟩dt from velocity autocorrelation, viscosity η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩dt from stress correlation, thermal conductivity κ = (V/kT²)∫⟨J_q(t)·J_q(0)⟩dt from heat flux correlation. Use when simulating noise-driven systems (Brownian motion, polymer dynamics, protein folding, population genetics), predicting transport properties (diffusion in electrolytes/polymers, viscosity of complex fluids, ionic conductivity in solid electrolytes) from microscopic correlation functions, modeling rare events (transition path sampling, metadynamics, forward flux sampling), implementing Brownian dynamics in overdamped limit for colloids/polymers, or calculating relaxation dynamics (glass transition, aging, stress relaxation, creep) in complex fluids and active matter.
---

# Stochastic Dynamics & Transport

## When to use this skill

- Implementing master equations dP_n/dt = Σ[W_mn P_m - W_nm P_n] for continuous-time Markov chains modeling chemical kinetics, population dynamics, or ion channel gating with discrete states (*.py, *.jl simulation codes)
- Performing exact stochastic simulation using Gillespie algorithm: compute total rate Σ propensities, draw exponential waiting time τ ~ Exp(1/total_rate), select reaction probabilistically for chemical reaction networks
- Solving Fokker-Planck equations ∂P/∂t = -∂(μF·P)/∂x + D·∂²P/∂x² for probability distribution evolution with drift term μF (deterministic force) and diffusion term D (noise strength) using finite differences or spectral methods
- Finding steady-state distributions P_ss from Fokker-Planck by balancing drift and diffusion: 0 = -∂(μF·P)/∂x + D·∂²P/∂x² for equilibrium or non-equilibrium steady states
- Implementing Langevin dynamics dx/dt = μF(x) + √(2D)ξ(t) with white noise ⟨ξ(t)ξ(t')⟩ = δ(t-t') for stochastic trajectories using Euler-Maruyama numerical integration x_{n+1} = x_n + μF(x_n)dt + √(2Ddt)·randn()
- Simulating Brownian motion of particles in potential landscapes: harmonic wells for optical tweezers, double-wells for bistable systems, periodic potentials for molecular motors or ratchets
- Modeling polymer dynamics using Langevin with bead-spring chains: Rouse model (no hydrodynamics), Zimm model (with hydrodynamic interactions), or reptation for entangled polymers
- Computing diffusion coefficient D = ∫₀^∞ ⟨v(t)·v(0)⟩dt from velocity autocorrelation via Green-Kubo relations for molecular dynamics, Brownian particles, or active matter systems
- Calculating viscosity η = (V/kT)∫₀^∞ ⟨σ_xy(t)σ_xy(0)⟩dt from stress autocorrelation using Green-Kubo for polymer melts, colloidal suspensions, or complex fluids with proper convergence analysis
- Extracting thermal conductivity κ = (V/kT²)∫₀^∞ ⟨J_q(t)·J_q(0)⟩dt from heat flux correlation for molecular crystals, amorphous solids, or thermal interface materials
- Validating Einstein relations D = kT·μ connecting diffusion D to mobility μ, or Drude conductivity σ = ne²τ/m for free electron gas with scattering time τ
- Performing NEMD (non-equilibrium molecular dynamics) with applied gradients: shear flow for viscosity η = -σ_xy/(dv_x/dy), temperature gradient for thermal conductivity κ, electric field for ionic conductivity
- Implementing rare event sampling techniques: transition path sampling for reactive trajectories, forward flux sampling crossing interfaces sequentially, metadynamics adding history-dependent bias to escape wells
- Modeling protein folding using Langevin dynamics on energy landscape: identify folding pathways, compute mean first passage time MFPT for folding rate k_fold = 1/MFPT, analyze transition states
- Simulating Brownian dynamics in overdamped limit dx/dt = μF + √(2Dμ)ξ neglecting inertia for high-friction systems like colloids, polymers in solution, or biological macromolecules
- Predicting diffusion in electrolytes from MD simulations: compute velocity autocorrelation, integrate via Green-Kubo D = ∫C_v(t)dt, compare to experimental measurements from NMR or impedance spectroscopy
- Calculating viscosity of polymer melts: run MD at different temperatures, compute stress autocorrelation η(T) = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩dt, validate against rheology experiments
- Modeling ionic conductivity in solid electrolytes (Li-ion batteries): simulate Li+ hopping, compute current autocorrelation, extract conductivity σ via Green-Kubo for materials design
- Analyzing relaxation dynamics in glasses: stress relaxation G(t) = ⟨σ(t)σ(0)⟩ for aging, stretched exponential fits φ(t) ~ exp[-(t/τ)^β], dynamic heterogeneity from four-point correlations
- Simulating active matter using Langevin with self-propulsion: active Brownian particles dx/dt = v₀n̂ + √(2D_t)ξ_t, dn̂/dt = √(2D_r)ξ_r × n̂ for motility-induced phase separation
- Computing transport in driven systems: sheared colloids, electrophoresis, sedimentation with hydrodynamic interactions using Brownian dynamics with external forces
- Implementing path integral formulations for stochastic processes: action S = ∫L(x,ẋ,t)dt, saddle-point approximation for large deviation theory of rare events

Model stochastic processes and calculate transport properties using master equations, Fokker-Planck/Langevin dynamics, and Green-Kubo relations.

## Stochastic Process Frameworks

### Master Equations
**Continuous-time Markov chains:**
dP_n/dt = ∑_m [W_{mn}P_m - W_{nm}P_n]
- Discrete states, continuous time
- Applications: Chemical kinetics, population dynamics

**Gillespie Algorithm (exact stochastic simulation):**
```python
def gillespie_step(state, propensities):
    total_rate = sum(propensities)
    tau = np.random.exponential(1/total_rate)
    reaction = np.random.choice(len(propensities), p=propensities/total_rate)
    return tau, reaction
```

### Fokker-Planck Equation
**Probability distribution evolution:**
∂P/∂t = -∂(μF·P)/∂x + D·∂²P/∂x²
- Drift term: μF (deterministic force)
- Diffusion term: D (noise strength)

**Steady state**: Balance drift and diffusion
**Applications**: Brownian motion, polymer dynamics, population genetics

### Langevin Dynamics
**Stochastic trajectories:**
dx/dt = μF(x) + √(2D)ξ(t)
- ⟨ξ(t)ξ(t')⟩ = δ(t-t'), white noise

**Numerical integration (Euler-Maruyama):**
```python
def langevin_step(x, F, mu, D, dt):
    return x + mu*F(x)*dt + np.sqrt(2*D*dt)*np.random.randn()
```

**Applications**: Molecular motors, active particles, protein folding

## Transport Theory

### Green-Kubo Relations

**General form:**
L_αβ = β ∫₀^∞ ⟨J_α(t)J_β(0)⟩ dt
- Transport coefficient from equilibrium time-correlation

**Diffusion:**
D = ∫₀^∞ ⟨v(t)·v(0)⟩ dt (velocity autocorrelation)

**Viscosity:**
η = (V/kT) ∫₀^∞ ⟨σ_xy(t)σ_xy(0)⟩ dt (stress autocorrelation)

**Thermal conductivity:**
κ = (V/kT²) ∫₀^∞ ⟨J_q(t)·J_q(0)⟩ dt (heat flux correlation)

### Einstein Relations
D = kT·μ (diffusion-mobility)
σ = ne²τ/m (Drude conductivity)

### NEMD (Non-Equilibrium MD)
- Apply gradients (shear, temperature, electric field)
- Measure response (stress, heat flux, current)
- Extract: η = -σ_xy/(dv_x/dy) from shear flow

## Computational Methods

### Rare Event Sampling
**Forward Flux Sampling**: Cross interfaces sequentially
**Transition Path Sampling**: Sample reactive trajectories
**Metadynamics**: Add history-dependent bias to escape wells

### Brownian Dynamics
Overdamped limit: dx/dt = μF + √(2Dμ)ξ
- Neglect inertia (high friction)
- Applications: Colloids, polymers, proteins

## Materials Prediction

**Transport Properties:**
- Predict diffusion in electrolytes, polymers
- Calculate viscosity of complex fluids
- Model ionic conductivity in solid electrolytes

**Relaxation Dynamics:**
- Glass transition, aging
- Stress relaxation, creep

References for advanced methods: field-theoretic approaches, path integrals, large deviation theory for rare events.
