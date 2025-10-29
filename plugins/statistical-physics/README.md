# Statistical Physics Plugin

Non-equilibrium statistical physics for driven systems, active matter, and complex dynamics with rigorous theory, computational methods, and experimental validation. Includes comprehensive correlation function analysis bridging theoretical foundations to experimental data interpretation.

## Overview

This plugin provides comprehensive statistical physics capabilities through two specialized agents and eight core skills covering non-equilibrium theory, stochastic dynamics, active matter, correlation function mathematics, physical system applications, computational algorithms, and experimental data interpretation.

## Agents

### non-equilibrium-expert

Expert in non-equilibrium statistical physics specializing in:

1. **Non-Equilibrium Theory & Methods** - Fluctuation theorems, entropy production, linear response theory
2. **Stochastic Dynamics & Transport** - Master equations, Fokker-Planck, Langevin dynamics, Green-Kubo relations
3. **Active Matter & Complex Systems** - Self-propelled particles, pattern formation, collective behavior
4. **Data Analysis & Model Validation** - Correlation functions, Bayesian inference, experimental validation

### correlation-function-expert

Expert in correlation functions and their applications across scientific disciplines specializing in:

1. **Mathematical Foundations & Transform Methods** - Two-point/higher-order correlations, cumulants, Wiener-Khinchin theorem, Ornstein-Zernike equations
2. **Physical Systems & Applications** - Condensed matter, soft matter, biological systems, non-equilibrium dynamics
3. **Computational Methods & Algorithms** - FFT-based O(N log N) algorithms, multi-tau correlators, JAX acceleration
4. **Experimental Data Interpretation** - DLS, SAXS/SANS, XPCS, FCS analysis with uncertainty quantification

## Skills

### 1. non-equilibrium-theory

Apply rigorous theoretical frameworks for systems far from thermal equilibrium including:
- **Entropy production** and irreversibility quantification
- **Fluctuation theorems** (Crooks, Jarzynski equality) for free energy from non-equilibrium measurements
- **Linear response theory** and Kubo formulas connecting microscopic correlations to macroscopic observables
- **Onsager reciprocal relations** for transport coefficient symmetries

Applications: Single-molecule experiments, molecular motors, energy harvesting, adaptive materials design.

### 2. stochastic-dynamics

Model stochastic processes and calculate transport properties:
- **Master equations** for discrete-state continuous-time Markov chains (Gillespie algorithm)
- **Fokker-Planck equations** for probability distribution evolution
- **Langevin dynamics** for stochastic trajectories with noise (Euler-Maruyama integration)
- **Green-Kubo relations** for transport coefficients (diffusion, viscosity, conductivity) from equilibrium correlations

Applications: Brownian motion, molecular motors, protein folding, transport in complex fluids.

### 3. active-matter

Model active matter, pattern formation, and emergent collective behavior:
- **Active Brownian particles** with self-propulsion and rotational diffusion
- **Motility-induced phase separation (MIPS)** in bacterial colonies and active colloids
- **Vicsek model** for flocking and collective motion transitions
- **Reaction-diffusion systems** with Turing instabilities for pattern formation

Applications: Bacterial colonies, cytoskeletal dynamics, bio-inspired materials, artificial cilia, adaptive camouflage.

### 4. data-analysis

Analyze experimental data and validate theoretical predictions:
- **Correlation functions** (time and spatial) for transport properties and relaxation dynamics
- **Experimental data interpretation** from DLS, rheology, SAXS/SANS, microscopy
- **Bayesian inference** for parameter estimation and uncertainty quantification
- **Model validation** with predictive checks, cross-validation, and consistency checks

Applications: Interpret scattering experiments, extract transport coefficients, validate non-equilibrium theories, inverse problem solving.

### 5. correlation-math-foundations

Master mathematical theory of correlation functions:
- **Two-point and higher-order** correlations, cumulants for multi-particle interactions
- **Transform methods**: Fourier (FFT), Laplace, wavelet for spectral analysis
- **Wiener-Khinchin theorem** connecting time correlations to spectral density
- **Ornstein-Zernike equations** for liquid structure, fluctuation-dissipation theorem

Applications: Theoretical foundation for all correlation analysis, finite-size scaling, sum rule validation.

### 6. correlation-physical-systems

Apply correlation functions to physical systems:
- **Condensed matter**: Spin correlations (Ising, Heisenberg), electronic correlations, density correlations
- **Soft matter**: Polymer dynamics (Rouse-Zimm), colloidal interactions, glass transitions
- **Biological systems**: Protein folding, membrane fluctuations, molecular motor correlations
- **Non-equilibrium**: Active matter, dynamic heterogeneity, information transfer

Applications: Map physical systems to appropriate correlation functions, materials characterization.

### 7. correlation-computational-methods

Implement efficient correlation algorithms:
- **FFT-based O(N log N)** methods for fast correlation calculation
- **Multi-tau correlators** for wide dynamic range (10⁻⁶ to 10³ seconds)
- **Multi-scale analysis** from femtoseconds to hours, nanometers to micrometers
- **Statistical validation**: Bootstrap, uncertainty quantification, convergence analysis
- **JAX-accelerated GPU** computation for large datasets

Applications: High-performance correlation analysis, real-time experimental data processing.

### 8. correlation-experimental-data

Interpret experimental correlation data:
- **DLS (Dynamic Light Scattering)**: Extract diffusion coefficients, particle sizes, dynamic heterogeneity
- **SAXS/SANS (Small-Angle Scattering)**: Structure factors S(q), pair distribution g(r), Guinier/Porod analysis
- **XPCS (X-ray Photon Correlation)**: Slow dynamics, aging, two-time correlations
- **FCS (Fluorescence Correlation)**: Diffusion, concentration, binding kinetics

Applications: Connect experimental measurements to physical parameters with uncertainty quantification.

## Technology Stack

- **Theory**: Fluctuation theorems, linear response theory, Onsager relations, correlation function theory
- **Stochastic Methods**: Master equations, Fokker-Planck, Langevin dynamics, Gillespie algorithm
- **Simulation**: NEMD, stochastic ODE/PDE solvers, agent-based models
- **Correlation Analysis**: FFT-based algorithms, multi-tau correlators, JAX GPU acceleration, KD-trees
- **Analysis**: Green-Kubo, Bayesian inference (emcee, PyMC3), bootstrap resampling, statistical tests
- **Experimental**: DLS, SAXS/SANS, XPCS, FCS, rheology, neutron scattering, microscopy

## Usage

Agents are invoked for non-equilibrium systems, correlation analysis, and experimental data interpretation. Skills can be used individually or combined for comprehensive materials prediction workflows.

**Example use cases for non-equilibrium-expert:**
- Predict transport properties (viscosity, diffusion, conductivity) from microscopic dynamics
- Model active matter systems (bacterial colonies, self-propelled particles, cytoskeletal dynamics)
- Analyze experimental data from scattering or rheology with uncertainty quantification
- Design energy harvesting systems using fluctuation theorems
- Study pattern formation and self-organization in reaction-diffusion systems

**Example use cases for correlation-function-expert:**
- Analyze DLS data to extract particle sizes and dynamic heterogeneity
- Compute structure factors S(q) and pair distribution functions g(r) from SAXS/SANS
- Calculate correlation functions from MD trajectories with FFT-based algorithms
- Interpret XPCS two-time correlations for aging and slow dynamics
- Extract diffusion coefficients from FCS autocorrelation analysis
- Validate theoretical predictions against experimental correlation measurements

## Integration with Other Plugins

**Delegates to:**
- **molecular-simulation**: MD trajectory generation, NEMD simulations
- **hpc-computing**: Parallel stochastic simulations, GPU acceleration
- **machine-learning**: Hybrid physics-ML models, neural ODEs for dynamics

**Provides to:**
- Theoretical frameworks for interpreting simulation results
- Transport coefficient predictions from Green-Kubo relations
- Guidance for experimental design and data interpretation
- New theories for materials with adaptive or emergent properties

## Requirements

- Python 3.12+
- Core: NumPy, SciPy, Matplotlib, SymPy, pandas
- Stochastic: Gillespie, stochastic ODE solvers
- Correlation: JAX (GPU acceleration), scipy.spatial (KD-trees), scipy.fft
- Bayesian: emcee, PyMC3, ArviZ
- Visualization: Plotly, Seaborn
- Optional: MD analysis tools (MDAnalysis) for trajectory post-processing

## License

MIT

## Author

Scientific Computing Team
