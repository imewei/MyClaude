# HPC Computing Plugin

High-Performance Computing and Numerical Methods for Scientific Computing across Python, Julia/SciML, C++, and Rust.

## Overview

This plugin provides comprehensive HPC and numerical computing capabilities through a coordinating agent and four specialized skills covering the complete spectrum of scientific computing workflows.

## Agent

### hpc-numerical-coordinator

Expert HPC and numerical methods coordinator specializing in:

1. **Numerical Methods Implementation** - ODE/PDE solvers, optimization, linear algebra
2. **Parallel Computing Strategy** - MPI/OpenMP, job scheduling, workflow orchestration
3. **GPU Acceleration** - CUDA/ROCm, memory optimization, hybrid CPU-GPU pipelines
4. **Ecosystem Selection** - Python vs Julia evaluation, hybrid integration, toolchain management

## Skills

### 1. numerical-methods-implementation
Implement numerical algorithms for ODE/PDE solvers, optimization techniques, and linear algebra operations in Python (SciPy) and Julia (DifferentialEquations.jl, Optim.jl).

### 2. parallel-computing-strategy
Design parallel computing workflows using MPI/OpenMP for distributed and shared-memory systems with SLURM/PBS job scheduling and Dask/Dagger.jl orchestration.

### 3. gpu-acceleration
Implement GPU acceleration using CUDA/ROCm with CuPy/Numba (Python) or CUDA.jl (Julia), including memory optimization and hybrid CPU-GPU pipelines.

### 4. ecosystem-selection
Select optimal scientific computing ecosystems (Python vs Julia), implement Python-Julia hybrid integrations (PyJulia, PyCall.jl), and manage reproducible toolchains.

## Technology Stack

- **Python**: NumPy, SciPy, Numba, CuPy, Dask, JAX
- **Julia/SciML**: DifferentialEquations.jl, Optim.jl, CUDA.jl, Dagger.jl
- **Systems**: C/C++, Rust, CUDA, OpenMP, MPI
- **HPC Tools**: SLURM, PBS, Nsight, Scalasca, Intel VTune

## Usage

The agent is invoked for HPC workflows, numerical methods, parallel computing, GPU acceleration, and ecosystem selection tasks. Skills can be used individually or combined for comprehensive scientific computing solutions.

## Requirements

- Python 3.12+
- Optional: Julia, MPI, CUDA toolkit
- NumPy, SciPy for Python
- SciML ecosystem for Julia

## License

MIT

## Author

Scientific Computing Team
