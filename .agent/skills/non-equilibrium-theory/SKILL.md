---
name: non-equilibrium-theory
version: "1.0.7"
maturity: "5-Expert"
specialization: Non-Equilibrium Physics
description: Apply non-equilibrium thermodynamics including fluctuation theorems, entropy production, and linear response theory. Use when modeling irreversible processes, analyzing driven systems, or deriving transport coefficients.
---

# Non-Equilibrium Theory

Theoretical frameworks for systems far from thermal equilibrium.

---

## Core Theorems

| Theorem | Formula | Application |
|---------|---------|-------------|
| Entropy Production | σ = Σ J_i X_i ≥ 0 | Quantify irreversibility |
| Crooks | P_F(W)/P_R(-W) = exp(β(W-ΔF)) | Work distributions |
| Jarzynski | ⟨exp(-βW)⟩ = exp(-βΔF) | Free energy from non-eq |
| FDT | χ(t) = β d/dt⟨A(t)B(0)⟩ | Response from fluctuations |
| Onsager | L_ij = L_ji | Transport symmetry |

---

## Linear Response Theory

**Response function**:
```
χ(ω) = ∫₀^∞ dt e^(iωt) ⟨A(t)B(0)⟩
```

**Kubo formula (conductivity)**:
```
σ = β ∫₀^∞ ⟨j(t)j(0)⟩ dt
```

---

## Entropy Production

```
σ = Σ J_i X_i ≥ 0
```

| Symbol | Meaning |
|--------|---------|
| J_i | Thermodynamic fluxes (heat, particle, charge) |
| X_i | Thermodynamic forces (gradients) |
| σ > 0 | Irreversibility, arrow of time |

---

## Fluctuation Theorems

### Jarzynski Equality

Extract free energy from non-equilibrium measurements:
- Single-molecule pulling
- Molecular motors
- Nanoscale energy conversion

### FDT Violations

| Observation | Interpretation |
|-------------|----------------|
| T_eff > T | Non-equilibrium driving |
| χ ≠ βdC/dt | Active matter, aging |

---

## Applications

| Application | Principle |
|-------------|-----------|
| Molecular motors | Extract work from ATP hydrolysis |
| Energy harvesting | Brownian ratchets, Landauer bound |
| Active matter | Compute entropy production |
| Self-assembly | Balance driving vs dissipation |
| Dissipative structures | Turing patterns, convection rolls |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Check thermodynamic consistency | σ ≥ 0, Onsager symmetry |
| Validate FDT | Compare response to correlation |
| Measure effective temperature | T_eff from χ vs C |
| Use Green-Kubo | Transport from equilibrium correlations |

---

## Checklist

- [ ] Entropy production positive
- [ ] Onsager symmetry checked
- [ ] Response functions computed
- [ ] FDT validity assessed
- [ ] Transport coefficients extracted
- [ ] Thermodynamic consistency verified

---

**Version**: 1.0.5
