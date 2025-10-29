---
name: stochastic-dynamics
description: Model stochastic processes using master equations, Fokker-Planck equations, and Langevin dynamics. Use when simulating noise-driven systems, calculating transport coefficients via Green-Kubo relations, or predicting diffusion, viscosity, and conductivity from microscopic dynamics in complex fluids and active matter.
---

# Stochastic Dynamics & Transport

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
