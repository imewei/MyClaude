# Gray-box boundary specification

For SciML work (UDEs, PINNs, gray-box ML) this template specifies which terms in the governing equations are prescribed from physics, which are learned from data, and how each learned component is validated.

## Why this template matters

Gray-box models that are not specified component-wise fail in ways that are hard to diagnose once everything is coupled. A learned residual term can compensate for an error in the physics, a discretization artifact, or a data-preprocessing bug, and the loss function cannot distinguish these sources. The discipline below prevents that compensation by validating each learned component in isolation before coupling.

## The specification

### 1. Governing equations (from Stage 4-5)

Write the governing equations in their prescribed form, with placeholders for the terms that will be learned.

Example:
$$\frac{\partial \phi}{\partial t} = D \nabla^2 \phi + \mathcal{F}_{\text{learned}}(\phi, \dot\gamma)$$

where $D \nabla^2 \phi$ is prescribed diffusion and $\mathcal{F}_{\text{learned}}$ is the closure term to be learned.

### 2. Physics-prescribed terms

List every term that is NOT learned. For each:
- What physical principle justifies it
- What its input and output variables are
- What its known domain of validity is

Example:
- $D \nabla^2 \phi$: Fickian diffusion, valid for volume fractions below 0.4
- $\mathbf{v} \cdot \nabla \phi$: standard advection, no learned parameters

### 3. Learned terms

For each learned component $\mathcal{F}_{\text{learned}, i}$:

**What is being learned:** A map from (inputs) to (outputs), e.g., from (local volume fraction, shear rate) to (stress correction).

**Why a learned term rather than a prescribed one:** The closure is not known from first principles, or the analytic closure is only valid outside the regime of interest.

**Architecture choice:** Neural network (name architecture, depth, width), Gaussian process, symbolic regression target, etc.

**Training data:** Where does the data come from? Pre-existing experimental dataset, synthetic data from a higher-fidelity simulation, or data collected during the same project. Specify variable ranges.

**Inductive biases built in:** Monotonicity, positivity, symmetry, known asymptotic limits. Every bias that can be hard-coded should be hard-coded; learning a known symmetry is a waste of data.

### 4. Per-component validation plan

This is the critical part. Before coupling learned components into the full system, each learned component is validated against an analytic or high-fidelity synthetic benchmark.

For each learned component:

**Standalone benchmark:** A test case where the correct output is known. Examples:
- For a learned stress closure, run at a regime where the analytic closure IS known and compare
- For a learned transport coefficient, run on synthetic data from direct simulation and compare

**Pass criterion:** Quantitative threshold. "Relative error below 5% across the training range" or "correct asymptotic behavior in both zero-shear and high-shear limits".

**Failure mode flags:** What would indicate the component is wrong? Overfitting signatures, extrapolation failures, violation of known bounds.

### 5. Coupling validation plan

Once each component passes its standalone benchmark, the full gray-box system is validated against:
- Conservation laws (mass, momentum, energy)
- Symmetries (the learned components should preserve the system's symmetries)
- Behavior in limits where the learned components can be switched off (the system should reduce to the fully prescribed theory)

### 6. Training/inference separation

Specify:
- Training dataset: exact file paths or generation recipe
- Validation dataset: held-out from training
- Inference regime: where the deployed model will be used
- Safeguards for out-of-distribution inputs: bounds checks, uncertainty estimates, fallback to prescribed physics

## Worked example (sketch, not full)

**Equation:** $\partial_t \phi = D_0 \nabla^2 \phi + \nabla \cdot \mathcal{J}_{\text{learned}}(\phi, \nabla \phi, \dot\gamma)$

**Prescribed:** $D_0 \nabla^2 \phi$ (thermal diffusion).

**Learned:** $\mathcal{J}_{\text{learned}}$, a correction flux with inputs (local volume fraction, volume-fraction gradient, local shear rate) and output (3-component flux vector).

**Architecture:** MLP with 3 hidden layers, 64 units each, tanh activation. Output passed through a divergence-free projection to enforce local mass conservation.

**Training data:** 10^5 snapshots from Brownian dynamics at volume fractions 0.2-0.45, shear rates 10^{-2} to 10^2 Péclet.

**Inductive biases:** Zero flux at zero gradient and zero shear. Flux reverses sign under shear-rate reversal.

**Standalone benchmark:** At volume fraction 0.3 and zero shear, the learned flux should match Fickian diffusion with the known volume-fraction-dependent $D(\phi)$. Pass: relative error below 5%.

**Coupling validation:** Mass conservation to 10^{-8} across a full run. Limit where $\mathcal{J}_{\text{learned}} \to 0$ (zero gradient) recovers linear diffusion exactly.

## Notes

- If a learned component has no standalone benchmark, it cannot be validated. Either find one (by constructing a synthetic case) or reconsider whether the term should be learned.
- A coupled validation that passes while a standalone validation fails is a warning sign: the system may be compensating for an un-validated component with other learned degrees of freedom.
- Training-time and inference-time data distributions should match. When they will not, specify the domain-shift handling.
