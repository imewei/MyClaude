---
name: trajectory-analysis
description: Analyze MD trajectories to calculate structural (RDF, S(q)), thermodynamic (density, Cp), mechanical (elastic constants, stress-strain), and transport (diffusion, viscosity) properties. Use when processing MD outputs, validating against experiments (SAXS/SANS), or extracting materials predictions from simulations.
---

# MD Trajectory Analysis

Extract structural, thermodynamic, mechanical, and transport properties from MD trajectories and validate against experiments.

## Analysis Tools

**MDAnalysis**: General-purpose Python library
**MDTraj**: Fast trajectory processing
**OVITO**: Visualization and analysis
**VMD**: Visualization and scripting

## Property Calculations

### Structural

**RDF (Radial Distribution Function):**
```python
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

u = mda.Universe('topology.pdb', 'trajectory.xtc')
rdf_calc = rdf.InterRDF(u.select_atoms('name O'), u.select_atoms('name O'))
rdf_calc.run()
```

**Structure Factor S(q)**: FFT of g(r), compare with SAXS/SANS

### Thermodynamic

- Density: `ρ = N·m / V`
- Heat capacity: From energy fluctuations
- Phase transitions: Order parameters

### Mechanical

**Elastic Constants**: Stress-strain from NPT
**Stress-Strain Curves**: NEMD simulations
**Fracture**: Crack propagation analysis

### Transport

**Diffusion (MSD):**
```python
from MDAnalysis.analysis import msd

msd_calc = msd.EinsteinMSD(u, select='all')
msd_calc.run()
D = msd_calc.results.timeseries[-1] / (6 * time)  # Einstein relation
```

**Viscosity**: Green-Kubo or NEMD

## Validation

**SAXS/SANS**: Compare S(q) with scattering data
**Rheology**: Match viscosity measurements
**Mechanical Testing**: Compare elastic moduli

## GROMACS Analysis

```bash
gmx rdf -f traj.xtc -s topol.tpr -o rdf.xvg
gmx msd -f traj.xtc -s topol.tpr -o msd.xvg
gmx energy -f ener.edr -o energy.xvg
gmx hbond -f traj.xtc -s topol.tpr
```

References for advanced analysis techniques and experimental validation protocols.
