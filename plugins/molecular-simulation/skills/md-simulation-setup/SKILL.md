---
name: md-simulation-setup
description: Set up and execute molecular dynamics simulations using LAMMPS (materials, nanoscale), GROMACS (biomolecules), and HOOMD-blue (soft matter). Use when preparing atomistic systems, selecting force fields (AMBER, CHARMM, ReaxFF, EAM), configuring thermostats/barostats, running equilibration/production MD, or optimizing parallel HPC execution.
---

# MD Simulation Setup & Execution

## Overview

Set up, execute, and optimize molecular dynamics simulations across LAMMPS, GROMACS, and HOOMD-blue for materials, biomolecules, and soft matter systems with appropriate force fields and simulation protocols.

## Core MD Engines

### LAMMPS - Materials & Nanoscale

**Use for**: Metals, ceramics, polymers, nanomaterials, general-purpose MD

**Input Script Example:**
```lammps
units metal
atom_style atomic
read_data system.data

pair_style eam/alloy
pair_coeff * * Al_zhou.eam.alloy Al

minimize 1.0e-4 1.0e-6 1000 10000

velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1
timestep 0.001
run 10000    # NVT equilibration

unfix 1
fix 2 all npt temp 300.0 300.0 0.1 iso 1.0 1.0 1.0
dump 1 all custom 1000 dump.lammpstrj id type x y z
run 1000000  # NPT production
```

**Force Fields:**
- Metals: EAM/alloy, ADP
- Covalent: Tersoff (Si, C), REBO
- Reactive: ReaxFF (bond breaking)
- Polymers: OPLS-AA, TraPPE

**Parallel:**
```bash
mpirun -np 16 lmp -in in.lammps     # MPI
lmp -sf gpu -pk gpu 4 -in in.lammps # GPU
```

### GROMACS - Biomolecules

**Use for**: Proteins, DNA, lipid membranes, solvated biomolecules

**Workflow:**
```bash
# 1. Topology
gmx pdb2gmx -f protein.pdb -ff amber99sb-ildn -water tip3p

# 2. Solvate & neutralize
gmx editconf -f conf.gro -d 1.0 -bt cubic
gmx solvate -cp box.gro -p topol.top
gmx genion -s ions.tpr -neutral

# 3. Minimization
gmx grompp -f minim.mdp -o em.tpr
gmx mdrun -v -deffnm em

# 4. NVT equilibration
gmx grompp -f nvt.mdp -o nvt.tpr
gmx mdrun -v -deffnm nvt

# 5. NPT equilibration
gmx grompp -f npt.mdp -o npt.tpr
gmx mdrun -v -deffnm npt

# 6. Production
gmx grompp -f md.mdp -o md.tpr
gmx mdrun -v -deffnm md -nb gpu
```

**Force Fields:**
- AMBER: amber99sb-ildn, amber14sb
- CHARMM: charmm27, charmm36
- OPLS: opls-aa

### HOOMD-blue - Soft Matter & GPU

**Use for**: Polymers, colloids, anisotropic particles, GPU-native simulations

**Python Script:**
```python
import hoomd
import hoomd.md as md

hoomd.context.initialize()
system = hoomd.init.read_gsd('initial.gsd')

nl = md.nlist.cell()
lj = md.pair.lj(r_cut=2.5, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

md.integrate.mode_standard(dt=0.005)
md.integrate.nvt(group=hoomd.group.all(), kT=1.0, tau=1.0)

hoomd.dump.gsd('traj.gsd', period=1000, group=hoomd.group.all())
hoomd.run(1e6)
```

**DPD in HOOMD:**
```python
dpd = md.pair.dpd(r_cut=1.0, nlist=nl, kT=1.0, seed=42)
dpd.pair_coeff.set('A', 'A', A=25.0, gamma=4.5)
md.integrate.nve(group=hoomd.group.all())
```

## Simulation Protocols

### Standard Equilibration
```
1. Energy Minimization (Fmax < 10 kJ/mol/nm)
2. NVT (100-500 ps): Heat to target T
3. NPT (1-5 ns): Equilibrate density
4. Production (10-100 ns): Data collection
```

### Best Practices

**Timesteps:**
- All-atom + constraints: 2 fs
- All-atom: 0.5-1 fs
- Coarse-grained: 10-50 fs

**Thermostats:**
- NVT: Nosé-Hoover (τ = 0.1-1.0 ps)
- Langevin (γ = 1-10 ps⁻¹)

**Barostats:**
- NPT: Parrinello-Rahman (τ = 1-2 ps)

**Cutoffs:**
- VDW: 1.0-1.4 nm
- Electrostatics: PME, cutoff 1.0-1.2 nm

## Performance Optimization

### LAMMPS
```bash
# Domain decomposition
mpirun -np 16 lmp -partition 16x1x1 -in in.lammps

# GPU
lmp -sf gpu -pk gpu 4 neigh yes -in in.lammps
```

### GROMACS
```bash
# GPU acceleration
gmx mdrun -deffnm md -nb gpu -pme gpu -bonded gpu

# Thread-MPI
gmx mdrun -deffnm md -ntmpi 4 -ntomp 8
```

### HOOMD-blue
```python
# Multi-GPU
hoomd.context.initialize("--mode=gpu --gpu=0,1,2,3")
```

## Troubleshooting

- **System explodes**: Minimize first, reduce timestep
- **T/P oscillations**: Longer τ, check ensemble
- **Slow**: Optimize neighbor list, reduce output, use GPU
- **Density drift**: Longer equilibration, check barostat

References available for force field parameters, example scripts, and detailed troubleshooting.
