---
name: non-equilibrium-theory
description: Apply non-equilibrium thermodynamics and statistical mechanics frameworks including fluctuation theorems (Crooks P_F(W)/P_R(-W) = exp(βW - β∆F) relating forward/reverse work distributions, Jarzynski equality ⟨exp(-βW)⟩ = exp(-β∆F) extracting free energy from non-equilibrium measurements), entropy production rate σ = Σ J_i X_i ≥ 0 quantifying irreversibility with thermodynamic fluxes J_i and forces X_i, linear response theory connecting response function χ(ω) = ∫₀^∞ ⟨A(t)B(0)⟩e^(iωt)dt to equilibrium correlations with Kubo formula σ = β∫⟨j(t)j(0)⟩dt for conductivity, fluctuation-dissipation theorem χ_AB(t) = β d/dt⟨A(t)B(0)⟩_{eq} relating equilibrium fluctuations to linear response with violations T_eff > T in non-equilibrium systems, and Onsager reciprocal relations L_ij = L_ji for near-equilibrium transport symmetry. Use when modeling irreversible processes (diffusion, heat conduction, chemical reactions), analyzing driven systems (molecular motors, active matter, sheared fluids), deriving transport laws (viscosity, conductivity, thermal conductivity) from microscopic correlation functions via Green-Kubo relations, validating thermodynamic consistency, or designing efficient non-equilibrium processes (energy harvesting, self-assembly, adaptive materials) for materials far from thermal equilibrium.
---

# Non-Equilibrium Theory & Methods

## When to use this skill

- Deriving entropy production rate σ = Σ J_i X_i for irreversible processes with thermodynamic fluxes (heat, particle, charge flow) and forces (temperature, chemical potential, voltage gradients) to quantify dissipation (*.py theory codes, *.tex derivations)
- Applying Crooks fluctuation theorem P_F(W)/P_R(-W) = exp(βW - β∆F) to analyze work distributions from forward/reverse non-equilibrium processes in single-molecule pulling experiments, optical tweezers, or AFM force spectroscopy
- Extracting free energy differences ∆F from non-equilibrium measurements using Jarzynski equality ⟨exp(-βW)⟩ = exp(-β∆F) for protein folding, ligand binding, or conformational changes
- Computing linear response function χ(ω) = ∫₀^∞ ⟨A(t)B(0)⟩e^(iωt)dt relating applied perturbation to system response via equilibrium time-correlation function for electrical/thermal/mechanical response
- Deriving Kubo formula for conductivity σ = β∫₀^∞ ⟨j(t)j(0)⟩dt connecting electrical conductivity to equilibrium current-current correlation for transport in metals, electrolytes, or ionic conductors
- Validating fluctuation-dissipation theorem χ_AB(t) = β d/dt⟨A(t)B(0)⟩_{eq} by comparing response function measured under perturbation to equilibrium correlation from spontaneous fluctuations
- Detecting FDT violations in non-equilibrium systems: extract effective temperature T_eff from χ(t) vs C(t) where T_eff > T indicates driving in active matter, sheared fluids, or glassy aging
- Implementing Onsager reciprocal relations L_ij = L_ji to connect different irreversible processes near equilibrium: thermoelectric effects (Seebeck/Peltier), cross-diffusion in multicomponent systems
- Modeling molecular motors using fluctuation theorems to optimize efficiency: extract free energy transduction from ATP hydrolysis to mechanical work in kinesin, myosin, or F₁-ATPase
- Designing energy harvesting devices using non-equilibrium thermodynamics: Brownian ratchets, flashing ratchets, or information engines with Landauer bound kT ln2 per bit erasure
- Analyzing active matter non-equilibrium driving: compute entropy production in self-propelled particles, measure violations of detailed balance σ = k_B Σ (P_ij/P_ji - 1) > 0
- Deriving transport coefficients from microscopic dynamics via Green-Kubo: viscosity η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩dt, thermal conductivity κ = (V/kT²)∫⟨J_q(t)·J_q(0)⟩dt
- Implementing stochastic thermodynamics framework for small systems: define work W, heat Q, entropy production ∆S = β(Q - ∆F) along single trajectories with trajectory-dependent fluctuation theorems
- Modeling self-assembly processes using non-equilibrium thermodynamics: balance driving forces (chemical gradients, active transport) with dissipation to predict steady-state structures
- Analyzing dissipative structures: reaction-diffusion patterns (Turing), convection rolls (Bénard), chemical oscillations (Belousov-Zhabotinsky) sustained by continuous energy input
- Computing minimum entropy production principle near equilibrium: steady states minimize σ = Σ L_ij X_i X_j subject to constraints for linear transport laws
- Deriving maximum path probability for systems far from equilibrium using path integral formulations or large deviation theory for rare event statistics
- Validating thermodynamic consistency: check Onsager symmetry L_ij = L_ji, entropy production positivity σ ≥ 0, second law compliance ∆S_total ≥ 0 for driven processes
- Designing adaptive materials with non-equilibrium responsiveness: self-healing polymers driven by chemical reactions, thermally-responsive gels, pH-sensitive drug delivery
- Modeling non-equilibrium phase transitions: absorbing-state transitions in reaction-diffusion systems, flocking transitions in active matter, dynamic phase transitions in driven lattice gases

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
