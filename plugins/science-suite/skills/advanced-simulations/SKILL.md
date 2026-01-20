---
name: advanced-simulations
version: "1.0.1"
description: Master advanced simulation techniques including non-equilibrium thermodynamics, stochastic dynamics, and multiscale modeling. Bridge scales from atomistic to mesoscale.
---

# Advanced Simulations & Statistical Physics

Comprehensive framework for modeling complex physical systems across multiple scales and environments.

## Expert Agent

For complex simulation workflows, multi-scale modeling, and HPC execution, delegate to the expert agent:

- **`simulation-expert`**: Unified specialist for Molecular Dynamics, Computational Physics, and HPC.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`
  - *Capabilities*: Large-scale MD (LAMMPS/GROMACS), differentiable physics (JAX-MD), and non-equilibrium thermodynamics.

## 1. Stochastic Dynamics & Transport

### Framework Selection
- **Langevin Dynamics**: For noise-driven trajectories (Brownian motion, polymers).
- **Master Equations**: For discrete state transitions (chemical kinetics).
- **Gillespie Algorithm**: Exact stochastic simulation for Markov processes.

### Transport Coefficients (Green-Kubo)
Calculate macroscopic transport properties from microscopic fluctuations:
- **Diffusion**: $D = \int_0^\infty \langle v(t) \cdot v(0) \rangle dt$
- **Viscosity**: $\eta = \frac{V}{kT} \int_0^\infty \langle \sigma_{xy}(t) \sigma_{xy}(0) \rangle dt$

## 2. Non-Equilibrium Theory

### Key Theorems
- **Jarzynski Equality**: $\langle \exp(-\beta W) \rangle = \exp(-\beta \Delta F)$. Extract free energy from non-equilibrium work.
- **Crooks Fluctuation Theorem**: Relates forward and reverse work distributions.
- **Fluctuation-Dissipation Theorem (FDT)**: Relates response to equilibrium fluctuations.

### Entropy Production
Quantify irreversibility in driven systems: $\sigma = \sum J_i X_i \geq 0$, where $J_i$ are fluxes and $X_i$ are forces.

## 3. Multiscale Modeling & Coarse-Graining

### Scaling Strategies
- **Coarse-Graining (CG)**: Reduce degrees of freedom (e.g., MARTINI, force matching) to reach larger length/time scales.
- **Dissipative Particle Dynamics (DPD)**: Mesoscale simulation for soft matter and fluid dynamics.
- **Scale Coupling**: Concurrent (QM/MM) or sequential (DFT $\to$ MD $\to$ CG) methods.

## 4. Sampling & Convergence Checklist

- [ ] **Ergodicity**: Ensure the system explores sufficient phase space.
- [ ] **Timescale Separation**: Verify that the integration timestep is significantly smaller than the fastest relaxation time.
- [ ] **Rare Events**: Use Forward Flux Sampling or Metadynamics for high-barrier transitions.
- [ ] **Thermodynamic Consistency**: Check Onsager symmetry and positive entropy production.
