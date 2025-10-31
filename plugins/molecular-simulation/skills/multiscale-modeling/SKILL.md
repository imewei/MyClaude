---
name: multiscale-modeling
description: Design and execute multiscale simulations bridging atomistic molecular dynamics to mesoscale and continuum models using coarse-graining methods, dissipative particle dynamics (DPD), and nanoscale discrete element methods (DEM). Use this skill when developing coarse-grained models from all-atom MD trajectories using force-matching, relative entropy minimization, or MARTINI mapping schemes. Use when implementing DPD simulations for soft matter systems (polymer blends, vesicles, surfactants, colloids) accessing microsecond to millisecond timescales. Use when working with HOOMD-blue DPD implementations or LAMMPS DPD packages for mesoscale modeling. Use when designing systematic coarse-graining protocols that preserve thermodynamic and structural properties from atomistic simulations. Use when performing backmapping to reconstruct atomistic configurations from coarse-grained structures. Use when coupling nanoscale DEM with molecular forces (van der Waals, capillary, solvation, electrostatic) for nanoparticle systems below 25 nm. Use when implementing QM/MM coupling for multiscale quantum-classical simulations. Use when working with large-scale systems (>1 million atoms) or long timescales (>100 nanoseconds) requiring coarse-graining. Use when parameterizing DPD interaction parameters (conservative, dissipative, random forces) from Flory-Huggins χ-parameters. Use when simulating lipid bilayers, polymer melts, or colloidal suspensions at mesoscale resolution.
---

# Multiscale Modeling

## When to use this skill

- When developing coarse-grained models from all-atom MD trajectories using force-matching or relative entropy methods
- When implementing MARTINI coarse-grained force fields for lipids, proteins, or polymers
- When setting up DPD (dissipative particle dynamics) simulations for soft matter systems
- When writing HOOMD-blue Python scripts for DPD or coarse-grained simulations
- When configuring LAMMPS DPD pair styles and integration schemes
- When designing systematic coarse-graining protocols that preserve structural and thermodynamic properties
- When creating mapping schemes (4-to-1, 10-to-1 atom-to-bead ratios) for coarse-grained models
- When performing backmapping to reconstruct atomistic structures from coarse-grained configurations
- When working with nanoscale DEM (discrete element method) for nanoparticle aggregation or sintering
- When coupling DEM with capillary forces, solvation forces, or van der Waals interactions for particles <25 nm
- When implementing QM/MM (quantum mechanics/molecular mechanics) multiscale coupling
- When simulating systems requiring long timescales (microseconds to milliseconds) beyond atomistic MD capabilities
- When working with large systems (>1 million atoms) where atomistic resolution is computationally prohibitive
- When parameterizing DPD interaction parameters from Flory-Huggins χ-parameters or solubility parameters
- When simulating polymer blends, vesicle formation, surfactant self-assembly, or colloidal suspensions at mesoscale

Bridge atomistic MD to mesoscale using coarse-graining, DPD, and nanoscale DEM for soft matter and nanoparticles.

## Coarse-Graining

**Methods**: Force matching, relative entropy, MARTINI
**Mapping**: 4-to-1 (4 atoms per CG bead)
**Applications**: Polymers, lipids, proteins

## DPD (Dissipative Particle Dynamics)

**Forces**: F = F_C + F_D + F_R (conservative + dissipative + random)
**Parameters**: Map χ-parameter → a_ij repulsion
**Timescale**: μs to ms, length 1-100 nm
**Applications**: Polymer blends, vesicles, surfactants

**HOOMD-blue DPD:**
```python
dpd = md.pair.dpd(r_cut=1.0, kT=1.0, seed=42)
dpd.pair_coeff.set('A', 'A', A=25.0, gamma=4.5)
```

## Nanoscale DEM

**Forces**: Van der Waals, capillary, solvation, electrostatic
**LIGGGHTS-NANO**: Contact models for < 25 nm particles
**Applications**: Nanoparticle aggregation, nanocomposites

## Multiscale Coupling

**Sequential**: DFT → MD → CG → Continuum
**Concurrent**: QM/MM
**Backmapping**: CG → Atomistic reconstruction

References for coarse-graining protocols, DPD parameterization, and advanced coupling strategies.
