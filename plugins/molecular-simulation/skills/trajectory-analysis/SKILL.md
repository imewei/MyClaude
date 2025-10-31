---
name: trajectory-analysis
description: Analyze molecular dynamics trajectories to extract and calculate structural (radial distribution function, structure factor), thermodynamic (density, heat capacity, phase transitions), mechanical (elastic constants, stress-strain curves, fracture), and transport (diffusion coefficients, viscosity, thermal conductivity) properties from MD simulation outputs. Use this skill when working with trajectory files from LAMMPS (.lammpstrj, dump.*), GROMACS (.xtc, .trr, .gro), or other MD packages. Use when writing Python analysis scripts using MDAnalysis, MDTraj, or similar trajectory processing libraries. Use when calculating radial distribution functions (RDF/g(r)) to characterize local structure and compare with experimental neutron/X-ray scattering. Use when computing structure factors S(q) for validation against SAXS/SANS experimental data. Use when extracting transport properties through mean-squared displacement (MSD) for diffusion coefficients or Green-Kubo integrals for viscosity. Use when analyzing stress-strain relationships to determine elastic moduli, yield points, or fracture behavior. Use when using GROMACS analysis tools (gmx rdf, gmx msd, gmx energy, gmx hbond) for trajectory post-processing. Use when validating MD simulation results against experimental measurements (rheology, scattering, thermal analysis, mechanical testing). Use when visualizing trajectories or properties using OVITO, VMD, or matplotlib/seaborn for publication-quality figures. Use when calculating thermodynamic properties from ensemble fluctuations or time-averaged quantities. Use when performing convergence analysis to ensure sufficient sampling and statistical reliability of computed properties.
---

# MD Trajectory Analysis

## When to use this skill

- When analyzing LAMMPS trajectory files (.lammpstrj, dump.*, dump.custom)
- When processing GROMACS trajectories (.xtc, .trr, .gro) or energy files (.edr)
- When writing Python analysis scripts using MDAnalysis, MDTraj, or numpy/scipy for trajectory processing
- When calculating radial distribution functions (RDF/g(r)) to characterize local atomic structure
- When computing structure factors S(q) for comparison with SAXS/SANS experimental scattering data
- When extracting diffusion coefficients from mean-squared displacement (MSD) using Einstein relation
- When calculating viscosity from stress autocorrelation functions using Green-Kubo relations
- When performing NEMD (non-equilibrium MD) analysis for transport properties
- When analyzing stress-strain curves to determine elastic constants, Young's modulus, or yield points
- When using GROMACS built-in analysis tools (gmx rdf, gmx msd, gmx energy, gmx hbond, gmx gyrate)
- When validating simulation results against experimental data (rheology, DSC, mechanical testing, scattering)
- When visualizing trajectories using OVITO, VMD, or PyMOL for molecular graphics
- When creating publication-quality plots using matplotlib, seaborn, or plotly
- When calculating thermodynamic properties (density, heat capacity, pressure) from ensemble averages
- When performing convergence analysis to assess statistical sampling quality
- When analyzing hydrogen bonding patterns, crystallinity, or phase transitions in trajectories
- When extracting time-correlation functions or autocorrelation functions for dynamic properties

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
