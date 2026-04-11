---
name: correlation-analysis
description: Meta-orchestrator for correlation function analysis. Routes to mathematical foundations, physical systems, computational methods, and experimental data interpretation skills. Use when computing correlation functions, analyzing DLS/SAXS/XPCS data, implementing FFT-based correlators, or connecting microscopic correlations to macroscopic response.
---

# Correlation Analysis

Orchestrator for correlation function analysis across statistical physics, condensed matter, and experimental data. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`statistical-physicist`**: Specialist for correlation functions, spectral analysis, and statistical mechanics.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - *Capabilities*: Green's functions, response theory, critical phenomena, scattering, and experimental data analysis.

## Core Skills

### [Correlation Math Foundations](../correlation-math-foundations/SKILL.md)
Mathematical theory: Green's functions, Wiener-Khinchin theorem, fluctuation-dissipation, and spectral representations.

### [Correlation Physical Systems](../correlation-physical-systems/SKILL.md)
Physical applications: pair correlations, structure factors, dynamic susceptibilities, and critical exponents.

### [Correlation Computational Methods](../correlation-computational-methods/SKILL.md)
Numerical computation: FFT-based correlators, Monte Carlo estimators, and GPU-accelerated correlation kernels.

### [Correlation Experimental Data](../correlation-experimental-data/SKILL.md)
Experimental analysis: noise correction, resolution deconvolution, and fitting correlation models to scattering data.

> **Python `freud` ecosystem** (RDF / S(q) / Steinhardt / F(q,t) / neighbor lists): see `correlation-physical-systems` for worked examples + `correlation-computational-methods` for algorithmic notes. No native Julia equivalent — Julia users go through `PythonCall.jl` per the handoff pattern in `chaos-attractors`.

## Routing Decision Tree

```
What is the correlation analysis task?
|
+-- Derive or understand correlation formalism?
|   --> correlation-math-foundations
|
+-- Compute correlations for a physical model?
|   --> correlation-physical-systems
|
+-- Implement numerical correlation computation?
|   --> correlation-computational-methods
|
+-- Analyze experimental scattering / spectroscopy data?
    --> correlation-experimental-data
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| Green's functions, FDT, spectral rep. | `correlation-math-foundations` |
| Structure factor, critical exponents | `correlation-physical-systems` |
| FFT correlators, MC estimators | `correlation-computational-methods` |
| Noise correction, model fitting | `correlation-experimental-data` |

## Checklist

- [ ] Identify whether the task is theory, simulation, or experiment before routing
- [ ] Verify the correlation function is properly normalized (C(0) = variance)
- [ ] Check ergodicity assumptions when applying time-average = ensemble-average
- [ ] Use FFT-based methods (O(N log N)) rather than direct summation (O(N²))
- [ ] Confirm statistical uncertainty estimates are included in all fitted parameters
- [ ] Validate computational correlators against analytic results for simple test cases
- [ ] Account for finite-size effects in simulation correlation lengths near criticality
