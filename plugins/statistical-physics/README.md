# Statistical Physics

Non-equilibrium statistical physics for driven systems, active matter, and complex dynamics with theory, simulation, and experimental validation. Includes correlation function analysis bridging theoretical foundations to experimental data interpretation.

**Version:** 1.0.1 | **Category:** uncategorized | **License:** Unknown

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/statistical-physics.html)

## What's New in v1.0.1

### 🚀 Agent Performance Optimization

Both agents now feature comprehensive systematic development processes with self-verification checkpoints:

**correlation-function-expert**
- ✅ 8-step systematic workflow for correlation analysis (data requirements → computational methods → statistical validation → experimental comparison → physical parameter extraction → consistency validation → documentation)
- ✅ 8 quality assurance principles with constitutional AI checkpoints (computational rigor, physical validity, statistical robustness, experimental accuracy, theoretical consistency, cross-validation, reproducibility, scientific communication)
- ✅ 16 clarifying questions for handling ambiguity across 4 domains (data characteristics, correlation requirements, computational constraints, deliverables)
- ✅ Comprehensive tool usage guidelines with delegation patterns to jax-pro, hpc-numerical-coordinator, data-scientist
- ✅ 3 detailed examples (DLS with stretched exponential: R = 125 ± 8 nm, β = 0.72 ± 0.04; SAXS with Ornstein-Zernike; common pitfalls to avoid)
- ✅ 3 common patterns (DLS autocorrelation 9 steps, MD g(r) 8 steps with KD-tree, Green-Kubo transport 10 steps)

**non-equilibrium-expert**
- ✅ 8-step systematic workflow for non-equilibrium systems (system characterization → theoretical framework → calculations → simulations → thermodynamic validation → experimental comparison → uncertainty quantification → documentation)
- ✅ 8 quality assurance principles (thermodynamic rigor σ ≥ 0, mathematical precision, computational robustness, statistical validity, physical interpretation, experimental validation, reproducibility, scientific communication)
- ✅ 16 clarifying questions across 4 domains (system characteristics, theoretical framework, computational constraints, deliverables)
- ✅ Comprehensive delegation patterns to simulation-expert, correlation-function-expert, hpc-numerical-coordinator, ml-pipeline-coordinator
- ✅ 3 detailed examples (Fluctuation theorem analysis: ∆F = 23.4 ± 1.2 kT from Jarzynski, R² = 0.97 Crooks validation; Langevin with Green-Kubo: D_GK = 2.15 ± 0.08 × 10⁻¹² m²/s)
- ✅ 3 common patterns (Langevin-Green-Kubo 10 steps, NEMD viscosity 9 steps, Active matter MIPS 8 steps)

### 🎯 Skills Discoverability Enhancement

All 8 skills enhanced with comprehensive "When to use this skill" sections (167 total use cases, 15-22 per skill):

- **active-matter** (21 use cases): ABP dynamics, MIPS simulation, Vicsek flocking, Turing patterns, bacterial colonies, phase diagrams Pe-ρ, T_eff extraction, active turbulence
- **correlation-computational-methods** (21 use cases): FFT O(N log N) optimization, multi-tau correlators (10 ns-10 s), JAX GPU 200× speedup, KD-tree O(N log N) g(r), bootstrap N=1000
- **correlation-experimental-data** (21 use cases): DLS g₂(τ) with Siegert, MCMC Bayesian inference, SAXS/SANS S(q), XPCS F(q,t), FCS, rheology G'(ω)/G''(ω), stretched exponential fitting
- **correlation-math-foundations** (21 use cases): Two-point/higher-order correlations, cumulants, S(q) transforms, Wiener-Khinchin, Ornstein-Zernike, FDT validation, sum rules
- **correlation-physical-systems** (20 use cases): Spin/electronic correlations, polymer entanglements, colloidal g(r), glass χ₄(t), biological membranes, active matter C_vv(r)
- **data-analysis** (21 use cases): C_v(t) for D, C_σ(t) for η, DLS fitting, Bayesian MCMC, BIC model selection, FDT violation T_eff > T
- **non-equilibrium-theory** (20 use cases): σ = ∑J_i X_i, Crooks/Jarzynski, Kubo formula, Onsager relations, molecular motors, stochastic thermodynamics
- **stochastic-dynamics** (21 use cases): Master equations, Gillespie algorithm, Fokker-Planck, Langevin Euler-Maruyama, Green-Kubo, NEMD, rare event sampling

**Expected Impact**: +50-75% improvement in automatic skill discovery through detailed use case specifications with file patterns (*.py, *.jl, *.csv), quantitative details (200× speedup, N=1000 bootstrap), and specific scenarios.

### 📋 Summary

- 2 agents enhanced with systematic processes (~400 lines each)
- 8 skills enhanced with comprehensive use cases (167 total)
- Backward compatible with existing workflows
- Follows established pattern from research-methodology plugin v1.0.1

## Agents (2)

### non-equilibrium-expert

**Status:** active

Non-equilibrium statistical physicist expert with 8-step systematic development process and thermodynamic rigor checkpoints. Specializes in driven systems, active matter, fluctuation theorems (Crooks, Jarzynski), transport theory (Green-Kubo), and stochastic dynamics (Langevin, Fokker-Planck, NEMD).

### correlation-function-expert

**Status:** active

Correlation function expert with 8-step systematic workflow and statistical validation principles. Specializes in higher-order correlations, FFT-based O(N log N) algorithms, JAX-accelerated GPU computation, and experimental data interpretation (DLS, SAXS/SANS, XPCS, FCS) with comprehensive uncertainty quantification.

## Skills (8)

### non-equilibrium-theory

Apply non-equilibrium thermodynamics frameworks including fluctuation theorems, entropy production, linear response theory, and Onsager relations.

### stochastic-dynamics

Model stochastic processes using master equations, Fokker-Planck equations, Langevin dynamics, and calculate transport coefficients via Green-Kubo relations.

### active-matter

Model active matter and complex systems including self-propelled particles, flocking, pattern formation, and collective behavior.

### data-analysis

Analyze experimental data from scattering, rheology, and microscopy using correlation functions, Bayesian inference, and model validation.

### correlation-math-foundations

Master mathematical foundations of correlation functions including two-point and higher-order correlations, cumulants, transform methods (Fourier, Laplace, wavelet), Wiener-Khinchin theorem, and Ornstein-Zernike equations.

### correlation-physical-systems

Apply correlation function analysis to condensed matter, soft matter, biological systems, and non-equilibrium systems for materials characterization and property prediction.

### correlation-computational-methods

Implement efficient algorithms for correlation analysis including FFT-based O(N log N) methods, multi-tau correlators, multi-scale analysis, statistical validation, and JAX-accelerated GPU computation.

### correlation-experimental-data

Interpret experimental correlation data from DLS, SAXS/SANS, XPCS, FCS, and rheology experiments. Extract physical parameters and validate theoretical predictions with uncertainty quantification.

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `statistical-physics` plugin
3. Activate an agent (e.g., `@non-equilibrium-expert`)

## Integration

See the full documentation for integration patterns and compatible plugins.

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/statistical-physics.html)

To build documentation locally:

```bash
cd docs/
make html
```
