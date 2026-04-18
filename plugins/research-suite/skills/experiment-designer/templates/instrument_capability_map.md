# Instrument capability map

For each observable dimension the measurement depends on, compare the predicted signal (from Stage 6) against the instrument's capability. The 3× margin rule is a heuristic default; override with an explicit, logged reason if Wei's experience with the specific instrument indicates a different threshold.

## Format

One row per observable dimension. Every row has a margin, a risk level, and a mitigation path if the margin is low.

| Dimension | Predicted signal | Instrument capability | Margin | Risk | Mitigation |
|-----------|------------------|-----------------------|--------|------|------------|
| Temporal resolution | 2.0 s characteristic timescale | 0.1 s minimum frame | 20× | LOW | none needed |
| Total duration | 120 s window | up to 3600 s continuous | 30× | LOW | none needed |
| Signal amplitude | Δ gap ≈ 1.5 (dimensionless) | typical variance < 0.05 | 30× | LOW | none needed |
| SNR at typical conditions | expected 15 | detectability threshold 3 | 5× | MEDIUM | increase photon flux, or ensemble 5 runs |
| Spatial resolution | 2 μm feature | beamline focus 500 nm | 4× | MEDIUM | acceptable; flag for monitoring |
| Shear-rate control | 0.1 /s setpoint | rheometer stable at 0.05 /s | 2× | HIGH | use fixture-matched Rheo-XPCS cell; alternative: drop setpoint to 1.0 /s and rescope |

## Risk levels

- **LOW** (margin ≥ 3×): no action required
- **MEDIUM** (1.5× ≤ margin < 3×): mitigation required; measurement is feasible but sensitive
- **HIGH** (margin < 1.5×): measurement is borderline; rescoping may be needed
- **UNFEASIBLE** (margin < 1×): measurement cannot resolve the predicted signal; plan must change

## Required capability dimensions

At minimum, include:
- Temporal resolution (sampling rate)
- Total duration (measurement window)
- Signal amplitude vs baseline noise
- Expected SNR
- Control variables (shear rate, temperature, concentration, whatever the measurement varies)

Add dimensions specific to the technique:
- For XPCS: coherence length, q-range, count rate
- For SAXS: q-range, beam divergence, sample thickness
- For rheology: gap geometry, torque resolution, slip detection

## Mitigation patterns

When a dimension is MEDIUM or HIGH, the mitigation must be specific, not aspirational:

- **"Increase photon flux":** specify by what factor and at what cost (beam damage, reduced coherence, longer exposure)
- **"Ensemble multiple runs":** specify how many and what that does to total duration
- **"Alternative observable":** specify which; Stage 6 may need a new extraction
- **"Rescope the claim":** specify what the rescoped claim would be; this sends the plan back to Stage 3

Aspirational mitigations ("we will be careful", "we will calibrate") are not mitigations.

## 3× threshold: when to override

The 3× default is a rule of thumb. Override with an explicit reason when:

- Wei's experience with this specific instrument and observable indicates a tighter ratio is routinely achievable (e.g., "at 8-ID-I I have repeatedly resolved Δ gap of 0.02 against variance 0.05, so 1.5× is achievable for this specific observable")
- The measurement is a pilot, with a follow-on campaign that will refine margins (lower bar acceptable for pilots)
- The observable has structure beyond simple amplitude that the capability map does not capture (e.g., a low-amplitude signal whose temporal pattern is diagnostic)

Every override is recorded in the artifact with a one-line justification.

## What to do if most dimensions are MEDIUM or HIGH

Three or more MEDIUM-or-higher rows usually means one of:
1. The predicted signal is near the instrument's limit across the board: the measurement is a genuine stretch and the plan should acknowledge this upfront
2. The Stage 6 prediction is too optimistic about signal amplitude: revisit the prototype's predicted uncertainty
3. The instrument is not a good match for this claim: consider a different measurement approach

Do not let a capability map full of MEDIUM rows pass without comment. Flag in the executive summary that the measurement is near capability limits.

## Example for Rheo-XPCS at APS 8-ID-I (illustrative)

| Dimension | Predicted | Capability | Margin | Risk |
|-----------|-----------|------------|--------|------|
| Frame rate | 5 Hz needed | up to 1 kHz | 200× | LOW |
| Coherence length | probe 1 μm | 1-10 μm routinely | OK | LOW |
| Q-range | 0.001-0.1 Å⁻¹ | supported | OK | LOW |
| Count rate | moderate signal | high flux available | OK | LOW |
| Rheometer shear rate | 0.1-10 /s | stable in range | OK | LOW |
| Rheo-XPCS cell matching | required | in-house cell exists | 1× | MEDIUM (availability risk) |

Here the technical dimensions are fine, but the logistical dimension (access to the Rheo-XPCS cell) is the binding constraint. That is typical for beamline work and deserves to be in the capability map explicitly.
