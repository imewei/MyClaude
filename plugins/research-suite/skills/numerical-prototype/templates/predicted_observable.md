# Predicted observable

Standardized format for the handoff from Stage 6 to Stage 7. The predicted observable file tells the experimental designer what signal the measurement needs to resolve.

## Required fields

### 1. Observable identity

- **Name:** Short identifier (e.g., "g1_correlation_function", "spectral_gap_time_series", "mean_square_displacement")
- **Type:** Time series / spectrum / scalar / tensor / distribution
- **Physical meaning:** 1-2 sentences linking this observable to the claim from Stage 3.

### 2. Predicted values

The observable's values with explicit structure:

- If time series: `t` array and `values` array, with units on each
- If spectrum: frequency array and amplitude array, with units
- If scalar: single value with units
- If tensor: shape and axis semantics documented

### 3. Uncertainty bounds

Every reported value has an uncertainty. The uncertainty has three components, reported separately:

- **Numerical:** from the convergence study (discretization error)
- **Parametric:** from sensitivity to physical parameters within their validity range (often the largest)
- **Statistical:** if the observable is derived from stochastic simulation, the standard error from the ensemble

Total uncertainty at the expected measurement point is the quadrature sum, but reporting the components lets Stage 7 reason about which source dominates.

### 4. Temporal structure (for time-dependent observables)

- **Characteristic timescale:** the timescale of the feature Stage 7 needs to resolve
- **Minimum required sampling rate:** 10× the inverse characteristic timescale, at minimum, to avoid aliasing
- **Total duration:** how long the measurement window must be

### 5. Spatial structure (if applicable)

- **Characteristic length:** the length scale of the feature
- **Minimum spatial resolution:** 10× finer, at minimum
- **Spatial extent:** how large the field of view must be

### 6. Noise model

What background noise would be present in a real measurement? Known sources:

- Shot noise (photon-limited measurements): Poisson statistics with mean proportional to flux
- Thermal noise (detector-limited): Gaussian with temperature-dependent amplitude
- Read noise (electronic): Gaussian with detector-specific amplitude
- Speckle statistics (scattering): exponential intensity distribution for fully developed speckle

The noise model matters because it sets the signal-to-noise ratio the experimental design must achieve.

### 7. Expected signal-to-noise at typical measurement conditions

Estimate: given the observable amplitude and the noise model, what SNR would a typical measurement achieve? This is the number Stage 7 will try to improve.

## File format

Emit as YAML or JSON, with arrays stored as either nested lists or referenced from NumPy `.npz` files for large data. Prefer YAML for human-readability, reference numerical arrays out if they exceed a few thousand values.

Example:

```yaml
name: "spectral_gap_time_series"
type: "time_series"
physical_meaning: "First-to-second eigenvalue gap of the stress-response operator, predicted to collapse >30s before flocculation onset."

predicted_values:
  t_file: "predicted_observable_t.npy"
  values_file: "predicted_observable_values.npy"
  units:
    t: "seconds"
    values: "dimensionless"

uncertainty:
  numerical:
    type: "relative"
    value: 0.03
    source: "convergence_study (dt: 1e-4 to 1e-2, Richardson)"
  parametric:
    type: "relative"
    value: 0.12
    source: "phi range 0.45-0.55"
  statistical:
    type: "absolute"
    value: 0.02
    source: "ensemble of 200 realizations"

temporal_structure:
  characteristic_timescale_s: 2.0
  min_sampling_rate_hz: 5.0
  total_duration_s: 120.0

noise_model:
  type: "speckle"
  fully_developed: true
  additional_electronic:
    type: "gaussian"
    std_relative: 0.01

expected_snr_at_typical_conditions: 15
```

## What Stage 7 does with this

Stage 7 compares each field against the available instrument's capability. The 3× margin rule applies on every dimension (time, space, SNR): if the margin is less than 3, the measurement is flagged high-risk and requires an explicit mitigation in the plan.
