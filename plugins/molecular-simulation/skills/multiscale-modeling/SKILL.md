---
name: multiscale-modeling
version: "1.0.7"
maturity: "5-Expert"
specialization: Multiscale Simulation
description: Bridge atomistic MD to mesoscale using coarse-graining, DPD, and nanoscale DEM. Use when developing CG models, implementing DPD simulations, or coupling scales.
---

# Multiscale Modeling

Coarse-graining and mesoscale methods bridging atomistic to continuum.

---

## Methods Overview

| Method | Scale | Application |
|--------|-------|-------------|
| Coarse-Graining | nm → 10nm | Polymers, proteins |
| DPD | 1-100 nm | Soft matter, surfactants |
| Nanoscale DEM | < 25 nm | Nanoparticle aggregation |
| QM/MM | Å → nm | Reactions at interfaces |

---

## Coarse-Graining

**Methods**: Force matching, relative entropy, MARTINI

**Workflow**:
1. All-atom MD reference
2. Define mapping (e.g., 4 atoms → 1 bead)
3. Parameterize CG potential
4. Validate structure/thermodynamics
5. Backmapping for atomistic detail

---

## DPD (Dissipative Particle Dynamics)

```python
# HOOMD-blue DPD
dpd = md.pair.dpd(r_cut=1.0, kT=1.0, seed=42)
dpd.pair_coeff.set('A', 'A', A=25.0, gamma=4.5)
```

**Forces**: F = F_C + F_D + F_R
**Parameterization**: χ-parameter → a_ij repulsion

---

## Scale Coupling

| Coupling | Strategy |
|----------|----------|
| Sequential | DFT → MD → CG → Continuum |
| Concurrent | QM/MM boundary |
| Adaptive | Resolution switching |

---

## Checklist

- [ ] Mapping scheme defined
- [ ] CG potential validated
- [ ] Thermodynamic properties preserved
- [ ] Backmapping tested if needed

---

**Version**: 1.0.5
