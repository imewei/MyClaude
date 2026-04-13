---
name: nonlinear-dynamics
description: Meta-orchestrator for nonlinear dynamics analysis. Routes to specialized skills for bifurcation analysis, chaos/attractors, network dynamics, pattern formation, and equation discovery. Use when analyzing dynamical systems, computing Lyapunov exponents, discovering governing equations from data (SINDy), reconstructing attractors, studying coupled oscillators, spatiotemporal chaos, or data-driven model discovery.
---

# Nonlinear Dynamics

Orchestrator for nonlinear dynamics analysis across JAX and Julia ecosystems. Routes problems to the appropriate specialized skill based on the analysis type and computational requirements.

## Expert Agent

For complex nonlinear dynamics problems requiring deep domain expertise, delegate to the expert agent:

- **`nonlinear-dynamics-expert`**: Specialist for dynamical systems, bifurcation theory, chaos, and data-driven model discovery.
  - *Location*: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
  - *Capabilities*: Bifurcation analysis, Lyapunov exponents, coupled oscillator networks, spatiotemporal pattern formation, and equation discovery from data.

## Core Skills

### [Bifurcation Analysis](../bifurcation-analysis/SKILL.md)
Parameter continuation, codimension-1/2 bifurcations, and stability diagrams. Primary ecosystem: Julia (BifurcationKit.jl) with JAX vmap for parameter sweeps.

### [Chaos & Attractors](../chaos-attractors/SKILL.md)
Lyapunov exponents, attractor reconstruction, fractal dimensions, and ergodic measures. Primary ecosystem: Julia (DynamicalSystems.jl) with JAX for parallel Lyapunov computation.

### [Network-Coupled Dynamics](../network-coupled-dynamics/SKILL.md)
Coupled oscillators, synchronization, master stability function, and chimera states. Primary ecosystem: Julia for small networks (<1K nodes), JAX for large-scale GPU-accelerated networks (>1K nodes).

### [Pattern Formation](../pattern-formation/SKILL.md)
Turing patterns, reaction-diffusion systems, spatiotemporal chaos, and amplitude equations. Primary ecosystem: Julia for symbolic PDE analysis, JAX for GPU-accelerated spatial simulations.

### [Equation Discovery](../equation-discovery/SKILL.md)
SINDy, symbolic regression, neural ODEs, and data-driven model identification. Primary ecosystem: Julia (DataDrivenDiffEq.jl) and Python (PySINDy).

## Related Skills

### [JAX-Julia Interop](../jax-julia-interop/SKILL.md)
Cross-language workflows for hybrid analysis pipelines. Use when a problem benefits from both Julia's symbolic tools and JAX's GPU acceleration.

### [Bayesian UDE Workflow](../bayesian-ude-workflow/SKILL.md)
Hybrid physics + neural-network ODEs with posterior uncertainty. A UDE is a dynamical-systems technique as much as a SciML technique — reach for this when the goal is to *identify* the unknown part of a dynamical system from data while retaining known physics.

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
+-- Identify unknown dynamics as a hybrid physics + NN with uncertainty?
|   --> bayesian-ude-workflow (Turing + DiffEq + Lux, NUTS/Pigeons)
|
+-- Working inside the Python/NumPy ecosystem (nolds, pyunicorn, antropy,
|   pyEDM, IDTxl, arch, statsmodels, ewstools)?
|   --> chaos-attractors (Python alternatives documented in Ecosystem Selection below)
|
+-- Need both Julia and JAX in one workflow?
    --> jax-julia-interop
```

## Ecosystem Selection

| Task                        | Julia                       | JAX                         | Python (NumPy-based)                                   |
|-----------------------------|-----------------------------|-----------------------------|--------------------------------------------------------|
| Symbolic analysis           | ModelingToolkit.jl          | --                          | SymPy (no DAE/index reduction)                         |
| Analytical continuation     | BifurcationKit.jl           | --                          | **AUTO-07p** (Fortran + Python CLI, installation friction); **recommended practical path**: `juliacall` → `BifurcationKit.jl` — see `bifurcation-analysis` |
| GPU parameter sweeps        | --                          | vmap + JIT                  | --                                                     |
| Large networks (>1K)        | --                          | sparse GPU graphs           | networkx (small-graph reference, not GPU)              |
| Small networks (<1K)        | DynamicalSystems.jl         | --                          | nolds + pyunicorn + antropy (fragmented)               |
| Equation discovery          | DataDrivenDiffEq.jl         | --                          | PySINDy + PyDMD + PySR (PySR bridges to Julia)         |
| Neural ODEs                 | DiffEqFlux.jl               | Diffrax                     | Diffrax (JAX-based)                                    |
| Lyapunov / Hurst / DFA      | ChaosTools.jl               | vmap + scan                 | nolds (Julia-first codebase? → PythonCall.jl handoff in `chaos-attractors`) |
| Recurrence analysis (RQA)   | RecurrenceAnalysis.jl       | --                          | pyunicorn.RecurrencePlot (pyRQA is stale)              |
| Complexity / entropy        | ComplexityMeasures.jl       | --                          | antropy, EntropyHub (Julia? → PythonCall.jl in `chaos-attractors`) |
| Transfer entropy / MI       | Associations.jl             | --                          | IDTxl (Julia? → PythonCall.jl in `chaos-attractors`)   |
| Tipping indicators (EWS)    | TransitionsInTimeseries.jl  | --                          | ewstools (repo active, PyPI lags)                      |
| CCM / empirical dynamics    | --                          | --                          | pyEDM                                                  |
| GARCH / unit roots          | --                          | --                          | arch + statsmodels.tsa                                 |

## Checklist

- [ ] Identify the analysis goal using the routing decision tree before selecting a sub-skill
- [ ] Verify ecosystem selection matches problem scale (Julia for symbolic/small, JAX for GPU/large)
- [ ] Confirm the dynamical system is well-posed (existence and uniqueness of solutions)
- [ ] Check that initial conditions and parameter ranges cover the regime of interest
- [ ] Validate numerical integration accuracy with convergence tests (halve dt, compare results)
- [ ] Ensure bifurcation analysis uses continuation (not brute-force parameter sweeps) for critical points
- [ ] Cross-validate Julia symbolic results with JAX numerical sweeps when using hybrid workflows
- [ ] Document all discovered dynamical features (fixed points, limit cycles, chaos) with parameter values
