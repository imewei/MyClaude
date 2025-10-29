# Molecular Simulation Plugin

Molecular dynamics and multiscale simulation for atomistic modeling with ML force fields, DPD, coarse-graining, and nanoscale DEM.

## Overview

This plugin provides comprehensive molecular dynamics and multiscale simulation capabilities through a specialized agent and four core skills covering classical MD, machine learning force fields, multiscale methods, and trajectory analysis.

## Agent

### simulation-expert

Expert in molecular dynamics and multiscale simulation specializing in:

1. **MD Simulation Setup & Execution** - LAMMPS, GROMACS, HOOMD-blue, force fields
2. **ML Force Fields Development** - NequIP, MACE, DeepMD, active learning
3. **Multiscale Modeling** - DPD, coarse-graining, nanoscale DEM
4. **Trajectory Analysis** - Property calculations, RDF, diffusion, validation

## Skills

### 1. md-simulation-setup
Set up and execute MD simulations using LAMMPS (materials), GROMACS (biomolecules), and HOOMD-blue (soft matter) with appropriate force fields (AMBER, CHARMM, ReaxFF, EAM), thermostats/barostats, and parallel optimization.

### 2. ml-force-fields
Train and deploy ML force fields (NequIP, MACE, DeepMD) achieving near-DFT accuracy (~1 meV/atom) with 1000-10000x speedups through active learning, uncertainty quantification, and LAMMPS/GROMACS integration.

### 3. multiscale-modeling
Bridge atomistic MD to mesoscale using systematic coarse-graining, dissipative particle dynamics (DPD), and nanoscale discrete element method (DEM) for soft matter, polymers, and nanoparticles.

### 4. trajectory-analysis
Extract structural (RDF, S(q)), thermodynamic (density, Cp), mechanical (elastic constants), and transport (diffusion, viscosity) properties from MD trajectories and validate against experiments.

## Technology Stack

- **MD Engines**: LAMMPS, GROMACS, NAMD, HOOMD-blue, ESPResSo
- **ML Force Fields**: NequIP, Allegro, MACE, DeepMD-kit, SchNet
- **Analysis**: MDAnalysis, MDTraj, OVITO, VMD, PyMOL
- **Force Fields**: AMBER, CHARMM, OPLS-AA, ReaxFF, EAM, MARTINI
- **HPC**: GPU (CUDA), MPI, domain decomposition

## Usage

The agent is invoked for MD simulations, ML force field development, multiscale modeling, and trajectory analysis. Skills can be used individually or combined for comprehensive materials prediction workflows.

## Requirements

- Python 3.12+
- LAMMPS, GROMACS, or HOOMD-blue
- Optional: ML frameworks (PyTorch, JAX) for ML force fields
- Analysis tools: MDAnalysis, MDTraj

## License

MIT

## Author

Scientific Computing Team
