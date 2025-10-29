---
name: non-equilibrium-theory
description: Apply non-equilibrium thermodynamics and statistical mechanics frameworks including fluctuation theorems, entropy production, linear response theory, and Onsager relations. Use when modeling irreversible processes, driven systems, or deriving transport laws from microscopic dynamics for materials far from thermal equilibrium.
---

# Non-Equilibrium Theory & Methods

Apply rigorous theoretical frameworks for systems far from thermal equilibrium, including fluctuation theorems, entropy production, and linear response theory.

## Core Theoretical Frameworks

### Entropy Production & Irreversibility

**Entropy production rate:**
σ = ∑_i J_i X_i ≥ 0
- J_i: thermodynamic fluxes (heat, particle, charge)
- X_i: thermodynamic forces (gradients)
- σ > 0: Irreversibility, arrow of time

**Applications**: Quantify dissipation in driven systems, efficiency bounds

### Fluctuation Theorems

**Crooks Fluctuation Theorem:**
P_F(W)/P_R(-W) = exp(βW - β∆F)
- Work distributions for forward/reverse processes
- Extract free energy differences from non-equilibrium measurements

**Jarzynski Equality:**
⟨exp(-βW)⟩ = exp(-β∆F)
- Free energy from non-equilibrium work measurements

**Applications**: Single-molecule pulling experiments, molecular motors, nanoscale energy conversion

### Linear Response Theory

**Response function:**
χ(ω) = ∫₀^∞ dt e^(iωt) ⟨A(t)B(0)⟩
- Relates applied perturbation to system response
- Connects microscopic correlations to macroscopic observables

**Kubo Formula (conductivity):**
σ = β ∫₀^∞ ⟨j(t)j(0)⟩ dt
- Transport from equilibrium current-current correlation

**Applications**: Electrical/thermal conductivity, dielectric response, viscosity

### Onsager Reciprocal Relations

**Near-equilibrium:** L_ij = L_ji
- Symmetry of transport coefficients
- Connects different irreversible processes

**Example**: Thermoelectric effects, cross-diffusion

## Mathematical Methods

### Variational Principles
- Minimum entropy production (near equilibrium)
- Maximum path probability (far from equilibrium)

### Green's Function Methods
- Propagators for stochastic processes
- Solve Fokker-Planck equations

### Perturbation Theory
- Systematic expansion around equilibrium or known solutions

## Materials Design Applications

**Self-Assembling Structures:**
- Balance driving forces vs. dissipation
- Predict steady-state morphologies

**Energy Harvesting:**
- Optimize efficiency using fluctuation theorems
- Design Brownian motors, ratchets

**Adaptive Materials:**
- Predict response to external fields
- Model non-equilibrium phase transitions

References for advanced topics: non-equilibrium ensembles, large deviation theory, path integral formulations.
