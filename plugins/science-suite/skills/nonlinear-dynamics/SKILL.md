---
name: nonlinear-dynamics
description: Meta-orchestrator for nonlinear dynamics analysis. Routes to specialized skills for bifurcation analysis, chaos/attractors, network dynamics, pattern formation, and equation discovery. Use when analyzing dynamical systems, coupled oscillators, spatiotemporal chaos, or data-driven model discovery.
---

# Nonlinear Dynamics

Orchestrator for nonlinear dynamics analysis across JAX and Julia ecosystems. Routes problems to the appropriate specialized skill based on the analysis type and computational requirements.

## Expert Agent

For complex nonlinear dynamics problems requiring deep domain expertise, delegate to the expert agent:

- **`nonlinear-dynamics-expert`**: Specialist for dynamical systems, bifurcation theory, chaos, and data-driven model discovery.
  - *Location*: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
  - *Capabilities*: Bifurcation analysis, Lyapunov exponents, coupled oscillator networks, spatiotemporal pattern formation, and equation discovery from data.

## Core Skills

### [Bifurcation Analysis](./bifurcation-analysis/SKILL.md)
Parameter continuation, codimension-1/2 bifurcations, and stability diagrams. Primary ecosystem: Julia (BifurcationKit.jl) with JAX vmap for parameter sweeps.

### [Chaos & Attractors](./chaos-attractors/SKILL.md)
Lyapunov exponents, attractor reconstruction, fractal dimensions, and ergodic measures. Primary ecosystem: Julia (DynamicalSystems.jl) with JAX for parallel Lyapunov computation.

### [Network-Coupled Dynamics](./network-coupled-dynamics/SKILL.md)
Coupled oscillators, synchronization, master stability function, and chimera states. Primary ecosystem: Julia for small networks (<1K nodes), JAX for large-scale GPU-accelerated networks (>1K nodes).

### [Pattern Formation](./pattern-formation/SKILL.md)
Turing patterns, reaction-diffusion systems, spatiotemporal chaos, and amplitude equations. Primary ecosystem: Julia for symbolic PDE analysis, JAX for GPU-accelerated spatial simulations.

### [Equation Discovery](./equation-discovery/SKILL.md)
SINDy, symbolic regression, neural ODEs, and data-driven model identification. Primary ecosystem: Julia (DataDrivenDiffEq.jl) and Python (PySINDy).

## Related Skills

### [JAX-Julia Interop](./jax-julia-interop/SKILL.md)
Cross-language workflows for hybrid analysis pipelines. Use when a problem benefits from both Julia's symbolic tools and JAX's GPU acceleration.

## Routing Decision Tree

```
What is the analysis goal?
|
+-- Parameter dependence / stability boundaries?
|   --> bifurcation-analysis (Julia BifurcationKit, JAX vmap sweeps)
|
+-- Long-term chaotic behavior / attractor geometry?
|   --> chaos-attractors (Julia DynamicalSystems.jl, JAX parallel Lyapunov)
|
+-- Coupled network / synchronization?
|   --> network-coupled-dynamics (Julia <1K nodes, JAX >1K nodes)
|
+-- Spatial or temporal pattern emergence?
|   --> pattern-formation (Julia symbolic PDE, JAX GPU simulations)
|
+-- Discover governing equations from data?
|   --> equation-discovery (Julia DataDrivenDiffEq, Python PySINDy)
|
+-- Need both ecosystems in one workflow?
    --> jax-julia-interop
```

## Ecosystem Selection

| Task                     | Julia                        | JAX                          |
|--------------------------|------------------------------|------------------------------|
| Symbolic analysis        | ModelingToolkit.jl           | --                           |
| Analytical continuation  | BifurcationKit.jl            | --                           |
| GPU parameter sweeps     | --                           | vmap + JIT                   |
| Large networks (>1K)     | --                           | sparse GPU graphs            |
| Small networks (<1K)     | DynamicalSystems.jl          | --                           |
| Equation discovery       | DataDrivenDiffEq.jl          | --                           |
| Neural ODEs              | DiffEqFlux.jl                | Diffrax                      |
