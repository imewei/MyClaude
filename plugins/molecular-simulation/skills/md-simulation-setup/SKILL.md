---
name: md-simulation-setup
version: "1.0.7"
description: Set up classical MD simulations using LAMMPS, GROMACS, and HOOMD-blue for materials and biomolecular systems. Use when writing input scripts, selecting force fields, configuring ensembles, or optimizing parallel execution.
---

# MD Simulation Setup

## Engine Selection

| Engine | Best For | Force Fields |
|--------|----------|--------------|
| LAMMPS | Metals, polymers, nanomaterials | EAM, Tersoff, ReaxFF, OPLS-AA |
| GROMACS | Biomolecules, solvation | AMBER, CHARMM, OPLS |
| HOOMD-blue | Soft matter, GPU-native | LJ, DPD, custom |

## LAMMPS

```lammps
units metal; atom_style atomic; read_data system.data
pair_style eam/alloy; pair_coeff * * Al_zhou.eam.alloy Al
minimize 1.0e-4 1.0e-6 1000 10000
velocity all create 300.0 12345
fix 1 all nvt temp 300.0 300.0 0.1; timestep 0.001
run 10000  # Equilibration
unfix 1; fix 2 all npt temp 300.0 300.0 0.1 iso 1.0 1.0 1.0
dump 1 all custom 1000 dump.lammpstrj id type x y z
run 1000000  # Production
```

## GROMACS

```bash
gmx pdb2gmx -f protein.pdb -ff amber99sb-ildn -water tip3p
gmx editconf -f conf.gro -d 1.0 -bt cubic; gmx solvate -cp box.gro -p topol.top
gmx genion -s ions.tpr -neutral
gmx grompp -f em.mdp -o em.tpr && gmx mdrun -deffnm em
gmx grompp -f nvt.mdp -o nvt.tpr && gmx mdrun -deffnm nvt
gmx grompp -f npt.mdp -o npt.tpr && gmx mdrun -deffnm npt
gmx grompp -f md.mdp -o md.tpr && gmx mdrun -deffnm md -nb gpu
```

## Equilibration

| Stage | Purpose | Duration |
|-------|---------|----------|
| Minimization | Remove clashes | Fmax < 10 kJ/mol/nm |
| NVT | Heat to T | 100-500 ps |
| NPT | Equilibrate density | 1-5 ns |
| Production | Data collection | 10-100+ ns |

## Parameters

| Parameter | Value |
|-----------|-------|
| Timestep (all-atom) | 2 fs with SHAKE/LINCS |
| Timestep (no constraints) | 0.5-1 fs |
| Thermostat | Nosé-Hoover (τ = 0.1-1.0 ps) |
| Barostat | Parrinello-Rahman (τ = 1-2 ps) |
| VDW cutoff | 1.0-1.4 nm |
| Electrostatics | PME, cutoff 1.0-1.2 nm |

## Parallel

| Engine | MPI | GPU |
|--------|-----|-----|
| LAMMPS | `mpirun -np 16 lmp -in script` | `-sf gpu -pk gpu 4` |
| GROMACS | `-ntmpi 4 -ntomp 8` | `-nb gpu -pme gpu` |
| HOOMD | N/A | `--mode=gpu --gpu=0,1` |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| System explodes | Minimize first, reduce timestep |
| T/P oscillations | Longer coupling time |
| Slow performance | Optimize neighbor list |
| Density drift | Longer equilibration |

**Outcome**: Properly configured MD simulation with appropriate force field, equilibration, and parallelization
