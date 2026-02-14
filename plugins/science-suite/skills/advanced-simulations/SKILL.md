---
name: advanced-simulations
version: "2.2.1"
description: Master advanced simulation techniques including non-equilibrium thermodynamics, stochastic dynamics, and multiscale modeling. Bridge scales from atomistic to mesoscale.
---

# Advanced Simulations

Comprehensive framework for high-performance computational physics workflows and multiscale modeling.

## Expert Agent

For complex simulation workflows, multi-scale modeling, and HPC execution, delegate to the expert agent:

- **`simulation-expert`**: Unified specialist for Molecular Dynamics, Computational Physics, and HPC.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`
  - *Capabilities*: Large-scale MD (LAMMPS/GROMACS), differentiable physics (JAX-MD), and HPC cluster optimization.

## Core Skills

### [MD Simulation Setup](./md-simulation-setup/SKILL.md)
Configuring force fields, ensembles, and boundary conditions.

### [ML Force Fields](./ml-force-fields/SKILL.md)
Integrating machine learning potentials into physics simulations.

### [Multiscale Modeling](./multiscale-modeling/SKILL.md)
Bridging scales from atomistic MD to mesoscopic continuum models.

### [Trajectory Analysis](./trajectory-analysis/SKILL.md)
Computing structural and dynamic properties from simulation data.

## 1. Molecular Dynamics Workflows

### Force Field Integration
- **Classical**: CHARMM, AMBER, GROMOS.
- **Machine Learning**: DeepMD-kit, NequIP, MACE.
- **Differentiable**: JAX-MD neighbor lists and energy functions.

### HPC Scaling
- **Parallelization**: Domain decomposition and MPI sharded execution.
- **Acceleration**: GPU kernels for neighbor list builds and force calculations.

## 2. Multiscale Modeling & Coarse-Graining

### Scaling Strategies
- **Coarse-Graining (CG)**: Reduce degrees of freedom (e.g., MARTINI, force matching) to reach larger length/time scales.
- **Dissipative Particle Dynamics (DPD)**: Mesoscale simulation for soft matter and fluid dynamics.
- **Scale Coupling**: Concurrent (QM/MM) or sequential (DFT $\to$ MD $\to$ CG) methods.

## 3. Trajectory Generation & Sampling

### Ensemble Control
- **NVE/NVT/NPT**: Proper thermostat and barostat selection for physical ensembles.
- **Replica Exchange**: Parallel tempering for enhanced sampling of rugged landscapes.

### Transition Paths
- **Rare Events**: Forward Flux Sampling, Metadynamics, and Umbrella Sampling workflows.

## 4. Performance & Convergence Checklist

- [ ] **Cell Lists**: Optimized neighbor search for $O(N)$ scaling.
- [ ] **Symplectic Integration**: Energy conservation drift $< 10^{-4}$.
- [ ] **Strong/Weak Scaling**: Benchmarks for HPC efficiency.
- [ ] **Checkpointing**: Periodic state saving for long-running jobs.
