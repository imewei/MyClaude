---
name: nanocomposites-and-adaptive-materials
description: Nanocomposite and adaptive-material modeling — effective-medium theory (Halpin-Tsai, Mori-Tanaka), percolation-aware property prediction, and self-healing/responsive composite behavior. Use when predicting effective modulus/conductivity from filler volume fraction and aspect ratio, or modeling self-healing composites.
---

# Nanocomposites & Adaptive Materials

Predicting effective properties of filler-reinforced and adaptive composite materials.

## Expert Agent

- **`continuum-mechanics-engineer`** — Effective-medium modeling and composite constitutive prediction.

## Effective-Medium Theories

| Method | Predicts | Best For |
|--------|----------|----------|
| Halpin-Tsai | Effective modulus from filler aspect ratio + volume fraction | Aligned or randomly-oriented short-fiber/platelet composites, low-to-moderate loading |
| Mori-Tanaka | Effective modulus via Eshelby inclusion theory, accounts for filler-filler interaction | Moderate-to-high filler loading where Halpin-Tsai's dilute assumption breaks down |

Both are mean-field approximations — they assume filler is well-dispersed. Neither captures percolation-driven property jumps (see below); use them for modulus prediction below the percolation threshold, not across it.

## Percolation Threshold

Near a critical filler volume fraction (the percolation threshold `φc`), the filler network transitions from disconnected to fully connected, causing a sharp (often orders-of-magnitude) jump in properties governed by connectivity — most dramatically electrical/thermal conductivity, but also a stiffness upturn beyond what Halpin-Tsai/Mori-Tanaka predict.

**Do not derive percolation theory (critical exponents, universality class, lattice vs. continuum percolation) here** — cross-reference `science-suite:statistical-physics-hub`'s percolation/collective-phenomena content (see `glass-and-collective-dynamics`) for that derivation. This skill's job is applying the percolation threshold as an input to composite property prediction, not deriving it from first principles.

## Self-Healing & Responsive Composites

Self-healing nanocomposites typically combine:
1. A transient-network matrix (see `science-suite:transient-networks-and-can`) providing the reversible bond-breaking/reforming mechanism.
2. Filler reinforcement (this skill's effective-medium methods) providing mechanical stiffness.

Model these as a composition: fit the matrix's transient-network behavior independently, then apply effective-medium theory using the matrix's (rate-dependent) modulus as the base-material modulus input — not as a single combined model from scratch.

## Workflow

1. Determine filler volume fraction, aspect ratio, and whether the system is below or near the percolation threshold.
2. Below threshold: apply Halpin-Tsai (dilute) or Mori-Tanaka (concentrated) for modulus prediction.
3. Near/above threshold: flag the effective-medium prediction as unreliable and delegate the connectivity-driven property jump to `statistical-physics-hub`.
4. For adaptive/self-healing systems, decompose into transient-network matrix + filler reinforcement per the composition approach above.

## Delegation

For percolation-threshold statistical mechanics, delegate to `statistical-physicist` via `science-suite:statistical-physics-hub`. For the matrix's own transient-network rheology, see `science-suite:transient-networks-and-can`.
