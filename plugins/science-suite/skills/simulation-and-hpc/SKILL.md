---
name: simulation-and-hpc
description: Meta-orchestrator for simulation, HPC, and computational methods. Routes to MD setup, trajectory analysis, ML force fields, parallel computing, GPU acceleration, numerical methods, and applied math methods (signal processing, time series, optimization, control theory). Use when setting up MD simulations, analyzing trajectories, training ML force fields, implementing parallel computing, writing GPU kernels, solving numerical problems, or applying computational methods.
---

# Simulation and HPC

Orchestrator for simulation and high-performance computing. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`simulation-expert`**: Specialist for MD simulations, HPC workflows, GPU programming, and numerical methods.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`
  - *Capabilities*: GROMACS/LAMMPS, trajectory analysis, ML force fields, MPI/OpenMP, GPU kernels, numerical algorithms.

## Core Skills

### [MD Simulation Setup](../md-simulation-setup/SKILL.md)
Molecular dynamics setup: GROMACS/LAMMPS force fields, system preparation, and equilibration protocols.

### [Trajectory Analysis](../trajectory-analysis/SKILL.md)
Post-processing MD trajectories: MDAnalysis, RMSD, RDF, free energy, and clustering.

### [ML Force Fields](../ml-force-fields/SKILL.md)
Machine-learned interatomic potentials: NequIP, MACE, DeePMD, and active learning workflows.

### [Parallel Computing](../parallel-computing/SKILL.md)
MPI, OpenMP, Dask, and Ray: parallel algorithm design and scalability analysis.

### [GPU Acceleration](../gpu-acceleration/SKILL.md)
CUDA, ROCm, JAX pmap, and GPU-optimized algorithms for scientific computing.

### [Numerical Methods Implementation](../numerical-methods-implementation/SKILL.md)
Numerical algorithms: finite difference/element, spectral methods, iterative solvers, and quadrature.

### [Signal Processing](../signal-processing/SKILL.md)
FFT, filtering, spectral estimation, wavelet transforms, and time-frequency analysis.

### [Time Series Analysis](../time-series-analysis/SKILL.md)
Time series: ARIMA, state space models, changepoint detection, and Fourier decomposition.

### [Advanced Optimization](../advanced-optimization/SKILL.md)
Global and combinatorial optimization: genetic algorithms, simulated annealing, and basin hopping.

### [Control Theory](../control-theory/SKILL.md)
Control systems: PID, LQR, MPC, stability analysis, and reinforcement learning for control.

## Routing Decision Tree

```
What is the simulation / HPC task?
|
+-- Set up or run MD simulations?
|   --> md-simulation-setup
|
+-- Analyze MD trajectories?
|   --> trajectory-analysis
|
+-- ML-based interatomic potentials?
|   --> ml-force-fields
|
+-- MPI / multi-node parallelism?
|   --> parallel-computing
|
+-- GPU kernel / device programming?
|   --> gpu-acceleration
|
+-- FD / FE / spectral numerical algorithms?
|   --> numerical-methods-implementation
|
+-- FFT / filtering / spectral analysis?
|   --> signal-processing
|
+-- Time series modeling / forecasting?
|   --> time-series-analysis
|
+-- Global / combinatorial optimization?
|   --> advanced-optimization
|
+-- Control systems / MPC / stability?
    --> control-theory
```

## Checklist

- [ ] Select sub-skill using routing tree before implementation
- [ ] Verify MD system is properly equilibrated before production runs
- [ ] Profile parallel code for load balance before scaling to more nodes
- [ ] Validate GPU kernel results against CPU reference implementation
- [ ] Check numerical method stability (CFL condition, step size convergence)
- [ ] Apply windowing functions before FFT to reduce spectral leakage
- [ ] Use proper statistical tests when comparing time series models
- [ ] Validate ML force fields against DFT benchmarks before production MD
