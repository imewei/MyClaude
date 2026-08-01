---
name: constitutive-equations
description: Constitutive modeling — linear elasticity, hyperelasticity (Neo-Hookean, Mooney-Rivlin, Ogden), and viscoelasticity (Maxwell, Kelvin-Voigt, generalized Maxwell/Prony series). Use when selecting or fitting a stress-strain relation, choosing between hyperelastic strain-energy functions, or fitting a Prony series to relaxation/DMA data.
---

# Constitutive Equations

Selecting and fitting stress-strain relations for continuum materials.

## Expert Agent

- **`continuum-mechanics-engineer`** — Constitutive model selection and parameter fitting.

## Model Catalog

| Model | Form | Regime | Use Case |
|-------|------|--------|----------|
| Linear elastic | σ = Cε | Small strain (< ~5%) | Metals, stiff isotropic solids |
| Neo-Hookean | W = C₁(I₁ - 3) | Large strain | Rubber elasticity, simplest hyperelastic model |
| Mooney-Rivlin | W = C₁(I₁-3) + C₂(I₂-3) | Large strain | Rubber, better fit at moderate strain than Neo-Hookean |
| Ogden | W = Σ (μᵢ/αᵢ)(λ₁^αᵢ + λ₂^αᵢ + λ₃^αᵢ - 3) | Large strain | Most flexible hyperelastic fit, more parameters to calibrate |
| Maxwell | dσ/dt + σ/τ = E dε/dt | Linear viscoelastic | Single-relaxation-time fluid-like response |
| Kelvin-Voigt | σ = Eε + η dε/dt | Linear viscoelastic | Single-relaxation-time solid-like (creep) response |
| Generalized Maxwell (Prony series) | G(t) = G_∞ + Σᵢ Gᵢ exp(-t/τᵢ) | Linear viscoelastic | Multi-relaxation-time solids — the standard fit target for DMA data |

## Fitting Workflow (Prony Series to DMA Data)

1. Convert time-domain relaxation modulus or frequency-domain storage/loss modulus (from `science-suite:dma-rheology`) to the target quantity.
2. Choose the number of Prony terms (start with 3-5; more terms fit better but risk overfitting — check residuals, not just R²).
3. Fit `Gᵢ`, `τᵢ` via nonlinear least squares (delegate large-scale fits to `jax-pro`'s NLSQ domain).
4. Validate: reconstructed G'(ω)/G''(ω) should match the measured data across the full frequency range, not just where fit weight was concentrated.

## Selecting a Hyperelastic Model

- **Neo-Hookean**: default starting point — 1 parameter, correct at small strain, reasonable up to moderate strain.
- **Mooney-Rivlin**: use when Neo-Hookean under/overshoots at moderate strain — 2 parameters give one more degree of freedom.
- **Ogden**: use only when simpler models fail to capture the strain-stiffening or softening behavior across the full strain range — more parameters need more data to constrain without overfitting.

## Delegation

For fitting against experimental DMA/rheology data, see `science-suite:dma-rheology` for how to read the raw data first. For plugging the fitted law into a finite element simulation, see `science-suite:fem-fea`.
