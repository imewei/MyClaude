---
name: multiscale-modeling
description: Design multiscale simulations bridging atomistic MD to mesoscale using coarse-graining, DPD (dissipative particle dynamics), and nanoscale DEM. Use for soft matter (polymers, colloids, lipids), systematic coarse-graining (force-matching, relative entropy), or coupling DEM with capillary/solvation forces for nanoparticles.
---

# Multiscale Modeling

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
