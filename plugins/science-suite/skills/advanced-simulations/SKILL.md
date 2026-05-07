---
name: advanced-simulations
description: Hub for advanced simulation workflows that combine MD setup, ML force fields, multiscale modeling, trajectory analysis, rare-event sampling, and non-equilibrium transport. Use when setting up molecular dynamics, choosing or training force fields, bridging atomistic-to-mesoscale models, analyzing trajectories, computing transport properties, or planning rare-event/weighted-ensemble simulations. For general HPC/GPU/numerical-method routing, use simulation-and-hpc.
---

# Advanced Simulations

Comprehensive framework for high-performance computational physics workflows and multiscale modeling.

## Expert Agent

For complex simulation workflows, multi-scale modeling, and HPC execution, delegate to the expert agent:

- **`simulation-expert`**: Unified specialist for Molecular Dynamics, Computational Physics, and HPC.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`
  - *Capabilities*: Large-scale MD (LAMMPS/GROMACS), differentiable physics (JAX-MD), and HPC cluster optimization.

## Core Skills

### [MD Simulation Setup](../md-simulation-setup/SKILL.md)
Configuring force fields, ensembles, and boundary conditions.

### [ML Force Fields](../ml-force-fields/SKILL.md)
Integrating machine learning potentials into physics simulations.

### [Multiscale Modeling](../multiscale-modeling/SKILL.md)
Bridging scales from atomistic MD to mesoscopic continuum models.

### [Trajectory Analysis](../trajectory-analysis/SKILL.md)
Computing structural and dynamic properties from simulation data.

## Routing Decision Tree

```
What is the simulation concern?
|
+-- Preparing a classical MD system / topology / ensemble?
|   --> science-suite:md-simulation-setup
|
+-- Choosing, training, or deploying ML interatomic potentials?
|   --> science-suite:ml-force-fields
|
+-- Bridging atomistic, coarse-grained, mesoscale, or continuum models?
|   --> science-suite:multiscale-modeling
|
+-- Extracting RDF/MSD/transport/mechanical observables from trajectories?
|   --> science-suite:trajectory-analysis
|
+-- Rare-event sampling / weighted ensemble / path sampling?
|   --> [cross-hub] science-suite:statistical-physics-hub → science-suite:rare-events-sampling
|       (FFS, TIS/RETIS, WE/WESTPA, OPS, AMS, milestoning; see Section 4 here for Python reference stack)
|
+-- None of the above / concern is ambiguous?
    --> Delegate to simulation-expert for open-ended triage, or clarify the
        primary concern and re-enter the routing decision tree.
```

## Skill Selection Table

| Trigger | Skill |
|---|---|
| LAMMPS, GROMACS, OpenMM setup, topology, ensemble | `science-suite:md-simulation-setup` |
| MLIP, MLFF, MACE, NequIP, ACE, active learning | `science-suite:ml-force-fields` |
| coarse-graining, MARTINI, DPD, QM/MM, mesoscale | `science-suite:multiscale-modeling` |
| trajectory post-processing, RDF, MSD, viscosity | `science-suite:trajectory-analysis` |
| WESTPA, weighted ensemble, TPS/TIS, rare event | `science-suite:statistical-physics-hub → science-suite:rare-events-sampling` |

## 1. Molecular Dynamics Workflows

### Force Field Integration
- **Classical**: CHARMM, AMBER, GROMOS.
- **Machine Learning**: NequIP / Allegro / MACE / SchNetPack / PaiNN / fairchem UMA (Python) and ACEpotentials.jl / PotentialLearning.jl / Molly.jl (Julia). See [ML Force Fields](../ml-force-fields/SKILL.md) for the architecture decision matrix and deployment paths.
- **Differentiable**: JAX-MD neighbor lists and energy functions (Python); Molly.jl differentiable MD (Julia).

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

## 4. Rare-Event Samplers (Python reference stack)

Neither JAX nor Julia has mature native forward-flux / weighted-ensemble samplers. The reference Python tools remain NumPy-era but actively maintained:

| Framework | Methods | MD backends |
|-----------|---------|-------------|
| **WESTPA** (`westpa`) | Weighted Ensemble (WE), adaptive MAB binning, HaMSM/RED post-analysis, unbiased rate constants | AMBER, GROMACS, NAMD, OpenMM |
| **OpenPathSampling** (`openpathsampling`) | TPS, TIS, RETIS, MSTIS, MISTIS, committor analysis, flux calculations | OpenMM (primary) |

```bash
# WESTPA typical workflow
w_init         # create west.h5
w_run          # run WE propagation
w_assign       # assign segments to progress-coordinate bins
w_direct       # direct rate analysis
```

Both support unbiased rate constants and path-ensemble analysis. WESTPA scales to HPC via MPI/ZMQ work managers; OPS stores full path ensembles in NetCDF.

### Agent-based modeling (ABM) with cloning / branching populations

| Framework | Role |
|-----------|------|
| **`mesa`** (`projectmesa/mesa`) | Python ABM reference: `mesa.Agent` / `mesa.Model`, spaces (`MultiGrid`, `SingleGrid`, `ContinuousSpace`, `NetworkGrid`, `HexGrid`, `OrthogonalMoore`), Mesa 3.x `AgentSet` API (`shuffle_do`, `do`, `agg`), `DataCollector` → pandas, and the SolaraViz browser front-end for interactive parameter sweeps |

Use `mesa` for cloning-style rare-event sampling (agent = replica), population-level splitting, and coupled heterogeneous-agent models where WESTPA's flat weighted-ensemble formulation is the wrong shape. `mesa` is NumPy-based — pair with `numpy` / `polars` for analysis, not JAX.

> **JAX-native gap**: A weighted-ensemble sampler leveraging `jax.vmap` for replica parallelism remains an open research opportunity — no mature JAX-native rare-event framework exists.

## Checklist

- [ ] **Cell Lists**: Optimized neighbor search for $O(N)$ scaling.
- [ ] **Symplectic Integration**: Energy conservation drift $< 10^{-4}$.
- [ ] **Strong/Weak Scaling**: Benchmarks for HPC efficiency.
- [ ] **Checkpointing**: Periodic state saving for long-running jobs.
