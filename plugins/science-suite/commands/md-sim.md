---
name: md-sim
description: Set up and run a molecular dynamics simulation — topology prep, force field selection, equilibration protocol, and trajectory analysis.
argument-hint: "[--engine gromacs|openmm|jax-md] [--system path/to/pdb] [--steps N]"
allowed-tools: ["Read", "Write", "Bash", "Edit", "Glob"]
---

# /md-sim — Molecular Dynamics Simulation

Routes to `simulation-expert` via `science-suite:simulation-and-hpc`.

## Usage

```
/md-sim --engine gromacs --system protein.pdb --steps 50000
/md-sim --engine jax-md --system lj_fluid.pdb --steps 100000
/md-sim --engine openmm --system membrane.pdb --steps 200000
```

## What This Does

1. Reads the `--system` PDB file and validates topology
2. Selects force field appropriate for the molecular system
3. Generates equilibration protocol (energy minimization → NVT → NPT)
4. Runs production MD for `--steps` steps
5. Outputs trajectory file and basic analysis (RMSD, energy)

## Engine Routing

| `--engine` | Routes To | Notes |
|---|---|---|
| `gromacs` | simulation-expert | GROMACS `.mdp` file generation |
| `openmm` | simulation-expert | Python OpenMM system builder |
| `jax-md` | simulation-expert → jax-pro | JAX-MD + NVE/NVT ensemble |

## Token Strategy

Engine-specific reference sections (GROMACS MDP templates, OpenMM force field tables) load only when `--engine` is specified. Default invocation loads the routing tree only.
