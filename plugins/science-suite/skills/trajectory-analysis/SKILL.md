---
name: trajectory-analysis
maturity: "5-Expert"
specialization: MD Analysis
description: Analyze MD trajectories to extract structural, thermodynamic, mechanical, and transport properties. Use when calculating RDF, MSD, viscosity, or validating simulations against experimental data. Also use when computing diffusion coefficients, density profiles, stress-strain curves, pair correlation functions, or Green-Kubo transport coefficients from simulation output. Use proactively when the user has MD trajectory files and needs post-processing analysis, property extraction, or comparison with experiments.
---

# MD Trajectory Analysis

## Expert Agent

For MD trajectory analysis, property extraction, and simulation validation, delegate to:

- **`simulation-expert`**: Molecular dynamics, trajectory analysis, and HPC simulations.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`

Extract properties from LAMMPS/GROMACS trajectories and validate against experiments.

---

## Analysis Tools

| Tool | Use Case |
|------|----------|
| MDAnalysis | General-purpose Python |
| MDTraj | Fast trajectory processing |
| OVITO | Visualization + analysis |
| GROMACS tools | gmx rdf, msd, energy |

---

## Property Calculations

### Structural (RDF)

```python
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

u = mda.Universe('topology.pdb', 'trajectory.xtc')
rdf_calc = rdf.InterRDF(u.select_atoms('name O'), u.select_atoms('name O'))
rdf_calc.run()
```

### Transport (Diffusion)

```python
from MDAnalysis.analysis import msd

msd_calc = msd.EinsteinMSD(u, select='all')
msd_calc.run()
D = msd_calc.results.timeseries[-1] / (6 * time)  # Einstein relation
```

---

## Property Types

| Category | Properties |
|----------|------------|
| Structural | RDF g(r), structure factor S(q) |
| Thermodynamic | Density, heat capacity, phase |
| Mechanical | Elastic constants, stress-strain |
| Transport | Diffusion (MSD), viscosity (Green-Kubo) |

---

## GROMACS Analysis

```bash
gmx rdf -f traj.xtc -s topol.tpr -o rdf.xvg
gmx msd -f traj.xtc -s topol.tpr -o msd.xvg
gmx energy -f ener.edr -o energy.xvg
gmx hbond -f traj.xtc -s topol.tpr
```

---

## Experimental Validation

| Simulation | Experiment |
|------------|------------|
| S(q) | SAXS/SANS scattering |
| Viscosity | Rheology |
| Elastic moduli | Mechanical testing |
| Diffusion | NMR/PFG |

## Parallelization

| Task | Strategy | Tooling |
|------|----------|---------|
| **Frame Processing** | Map function over frames | `multiprocessing`, `dask` |
| **Spatial Analysis** | Decomposition by region | MPI (via `mpi4py`) |
| **GPU Acceleration** | RDF/Distance matrices | `numba.cuda`, `cupy` |
| **Vectorization** | NumPy broadcasting | `numpy`, `jax.vmap` |

---

## Checklist

- [ ] Trajectory files loaded correctly
- [ ] Analysis method appropriate for property
- [ ] Sufficient sampling for convergence
- [ ] Statistical errors estimated
- [ ] Compared to experimental data

---

**Version**: 1.0.5
