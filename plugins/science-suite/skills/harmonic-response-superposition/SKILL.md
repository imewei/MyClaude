---
name: harmonic-response-superposition
description: Harmonic response analysis and time-temperature superposition (TTS) — complex modulus under sinusoidal loading, WLF equation, and master curve construction. Use when building a master curve from multi-temperature frequency sweeps, applying the WLF equation, or analyzing steady-state harmonic loading response.
---

# Harmonic Response & Time-Temperature Superposition

Building master curves and analyzing harmonic loading response.

## Expert Agent

- **`continuum-mechanics-engineer`** — TTS master-curve construction and harmonic response analysis.

## Harmonic Response

Steady-state response to sinusoidal loading is characterized by the complex modulus `E*(ω) = E'(ω) + iE''(ω)` (see `science-suite:dma-rheology` for the physical meaning of E'/E''). The phase lag between stress and strain is `δ = arctan(E''/E')`.

## Time-Temperature Superposition (TTS)

Many polymers are thermorheologically simple: their relaxation spectrum shifts uniformly with temperature without changing shape. This lets frequency sweeps measured at several temperatures be shifted horizontally along the frequency axis and stitched into one "master curve" spanning far more decades of effective frequency than any single measurement could reach directly.

### WLF Equation

```
log(a_T) = -C1 (T - T_ref) / (C2 + T - T_ref)
```

- `a_T`: horizontal shift factor for temperature T relative to reference temperature `T_ref`.
- `C1`, `C2`: material-specific constants (WLF's universal-average values, C1≈17.44, C2≈51.6 K, are a starting guess only — always fit to the material's own data).
- Shift each isothermal curve by `a_T` along the log-frequency axis; overlapping segments should collapse onto a single smooth curve if the material is thermorheologically simple.

### Validity Check

If shifted curves do NOT collapse onto a smooth single curve, the material is not thermorheologically simple (e.g. it may be undergoing a phase transition or chemical change across the temperature range tested) — do not force a WLF fit in that case; flag the discrepancy instead of reporting a low-quality master curve as if it were valid.

## Workflow

1. Collect frequency sweeps at multiple temperatures (from `science-suite:dma-rheology`).
2. Pick a reference temperature `T_ref` (often Tg or Tg+50K by convention).
3. Fit `a_T` per temperature (either via WLF or empirically via curve-shifting software) to maximize overlap between adjacent curves.
4. Verify the shift factors themselves vary smoothly and monotonically with temperature — a non-monotonic `a_T(T)` is a red flag for the thermorheological-simplicity assumption.

## Delegation

For fitting the resulting master curve to a Prony series constitutive model, see `science-suite:constitutive-equations`.
