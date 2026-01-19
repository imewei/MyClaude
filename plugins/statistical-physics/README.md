# Statistical Physics

Non-equilibrium statistical physics for driven systems, active matter, and complex dynamics with theory, simulation, and experimental validation. Includes correlation function analysis bridging theoretical foundations to experimental data interpretation.

**Version:** 2.1.0 | **Category:** scientific-computing | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/statistical-physics.html)


## What's New in v2.1.0

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


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
make -j4 html
```
