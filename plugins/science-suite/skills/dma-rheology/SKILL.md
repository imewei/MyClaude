---
name: dma-rheology
description: Dynamic Mechanical Analysis (DMA) and rheology — storage/loss modulus interpretation, tan delta, oscillatory shear rheology, and shear/extensional flow curves. Use when interpreting DMA or rheometer output, identifying the linear viscoelastic regime, or distinguishing shear from extensional rheological behavior.
---

# DMA & Rheology

Interpreting dynamic mechanical and rheological measurements.

## Expert Agent

- **`continuum-mechanics-engineer`** — DMA/rheology data interpretation and constitutive parameterization.

## Dynamic Mechanical Analysis (DMA)

- **Storage modulus (E' or G')**: in-phase (elastic) component of the response to oscillatory strain — energy stored and recovered per cycle.
- **Loss modulus (E'' or G'')**: out-of-phase (viscous) component — energy dissipated as heat per cycle.
- **tan δ = E''/E'**: damping ratio; peaks at transitions (e.g. glass transition temperature Tg in a temperature sweep).
- **Complex modulus**: E* = E' + iE'', |E*| = √(E'² + E''²).

## Rheology

- **Oscillatory (small-amplitude) shear**: apply sinusoidal strain, measure stress response — determines G'(ω), G''(ω) within the linear viscoelastic (LVE) regime.
- **Amplitude sweep**: increase strain amplitude at fixed frequency to find the LVE limit (where G'/G'' become strain-dependent — data outside this regime isn't usable for linear constitutive fitting).
- **Flow curves (steady shear)**: shear stress or viscosity vs. shear rate — captures shear-thinning/thickening, yield stress.
- **Extensional rheology**: distinct from shear — relevant for fiber spinning, film blowing, and any process with a strong extensional-flow component. Requires different instrumentation (CaBER for capillary breakup, FiSER for filament stretching) since shear rheometers cannot impose pure extensional flow.

## Workflow: From Raw Data to Constitutive Model

1. Confirm the measurement is within the LVE regime (amplitude sweep first).
2. Run a frequency sweep to get G'(ω), G''(ω) across the accessible frequency range.
3. If a wider frequency range is needed, use `science-suite:harmonic-response-superposition` for time-temperature superposition to build a master curve.
4. Fit a generalized Maxwell (Prony series) model via `science-suite:constitutive-equations`.

## Common Pitfalls

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Fitting outside the LVE regime | Non-physical fit parameters | Always run an amplitude sweep first to confirm LVE limits |
| Confusing shear and extensional viscosity | Wrong process prediction (e.g. fiber spinning uses extensional, not shear) | Match the rheological measurement to the actual flow geometry of the application |
| Ignoring temperature dependence | Master curve construction fails | Use time-temperature superposition (WLF) rather than assuming a single curve holds across all temperatures |

## Delegation

For building master curves via time-temperature superposition, see `science-suite:harmonic-response-superposition`. For fitting a constitutive model to the extracted moduli, see `science-suite:constitutive-equations`.
