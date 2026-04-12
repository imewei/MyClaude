---
name: bayesian-inference
description: Meta-orchestrator for Bayesian inference and probabilistic programming. Routes to NumPyro, Turing.jl, Pigeons consensus MCMC, Bayesian UDE workflows, Bayesian PINNs, point-process / Hawkes inference, variational inference, and MCMC diagnostics skills. Use when building probabilistic models with NumPyro or Turing.jl, sampling multimodal posteriors with parallel tempering, fitting Bayesian neural ODEs or PINNs, modeling self-exciting event data, running MCMC inference, implementing variational inference, or diagnosing sampler convergence.
---

# Bayesian Inference

Orchestrator for Bayesian inference and probabilistic programming. Routes problems to the appropriate specialized skill.

## Expert Agents

- **`statistical-physicist`**: Bayesian inference theory, MCMC methods, posterior analysis.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
- **`julia-pro`** (secondary): Julia + Turing + Pigeons + Bayesian SciML.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

## Core Skills

### [NumPyro Core Mastery](../numpyro-core-mastery/SKILL.md)
NumPyro: model definition, NUTS/HMC, SVI, and integration with JAX for GPU-accelerated inference.

### [Turing Model Design](../turing-model-design/SKILL.md)
Turing.jl: probabilistic model design, sampler selection, embedded DiffEq + Lux integration patterns.

### [Consensus MCMC with Pigeons](../consensus-mcmc-pigeons/SKILL.md)
Non-reversible parallel tempering for multimodal posteriors where NUTS fails to mix.

### [Bayesian UDE Workflow](../bayesian-ude-workflow/SKILL.md)
End-to-end Bayesian Universal Differential Equations in Julia: Turing + DiffEq + Lux + warm-start + NUTS/Pigeons.

### [Bayesian SINDy Workflow](../bayesian-sindy-workflow/SKILL.md)
Sparse Bayesian regression for discovering governing equations with horseshoe priors, ensemble SINDy, and UQ-SINDy. Credible intervals on coefficients, inclusion probabilities for library terms, and Bayesian model comparison via PSIS-LOO. Full Lorenz-63 worked example with NumPyro + NUTS.

### [Bayesian UDE in JAX](../bayesian-ude-jax/SKILL.md)
Python / JAX counterpart: Diffrax + Equinox + NumPyro + Optax. Same staged pipeline, GPU-vectorized chain sampling.

### [Bayesian PINN](../bayesian-pinn/SKILL.md)
NeuralPDE's BNNODE (ODEs) and BayesianPINN (PDEs) discretizers — credible intervals on a purely-neural ODE/PDE surrogate via an internal AdvancedHMC sampler (no Turing).

### [Point Processes](../point-processes/SKILL.md)
Parametric Hawkes with `tick`, Bayesian Hawkes with HSGP background via NumPyro, Julia `PointProcesses.jl`. Use for irregular event-time data (aftershocks, spikes, trades, cascades) rather than regularly-sampled series.

### [Variational Inference Patterns](../variational-inference-patterns/SKILL.md)
VI methods: ELBO optimization, mean-field approximations, normalizing flows, and amortized inference.

### [MCMC Diagnostics](../mcmc-diagnostics/SKILL.md)
Posterior validation: R-hat, ESS, BFMI, trace plots, and convergence diagnostics with ArviZ.

> **Bayesian PINN note**: Bayesian PINNs (BPINN / BNNODE) live in `../bayesian-pinn/SKILL.md` — separated from the deterministic `../neural-pde/SKILL.md` to keep both skills under context budget. NeuralPDE uses its own internal AdvancedHMC integration that does not go through Turing.

## JAX Bayesian Ecosystem Map

The analysis-doc-aligned picture of the active Python / JAX Bayesian stack — each entry covers a different niche from NumPyro:

| Package | Niche | When to pick it |
|---------|-------|-----------------|
| **NumPyro** | Full PPL with distributions, guides, inference algorithms | Default starting point for JAX Bayesian modeling |
| **BlackJAX** | Composable MCMC / SMC kernels (NUTS, HMC, MALA, MCLMC, tempered SMC, Pathfinder, SVGD) with a low-level `init`/`step` API | You want to write the sampler loop yourself, or plug kernels into a NumPyro log-prob, or you need adaptive tempered SMC |
| **GPJax** | Full Gaussian process framework: kernels, exact + sparse variational GPs, Laplace / MCMC hyperparameters | You're doing GP regression / classification inside a JAX pipeline |
| **tinygp** | Minimalist JAX GP with quasi-separable kernels (celerite2-style O(N) for 1D GPs) | Light GP embedded in a NumPyro `@model`; timeseries GPs with long sequences |
| **emcee** | Goodman-Weare affine-invariant ensemble sampler (NumPy) — the astronomy-community default | Legacy pipelines with black-box log-probs, non-JAX codebases, or when NUTS over-tunes on cheap likelihoods |
| **pocoMC** | Preconditioned tempered SMC with normalizing-flow reparameterization (PyTorch) | Multimodal / strongly-correlated / expensive-likelihood posteriors; also returns marginal likelihood |
| **corner** | Matplotlib-based 1D/2D marginal grid plots | Publication-quality posterior summary from `(nsamples, ndim)` arrays or ArviZ `InferenceData` |

> **No Non-Reversible Parallel Tempering in JAX**: If NUTS still fails to mix after BlackJAX tempered SMC, fall back to `consensus-mcmc-pigeons` — `Pigeons.jl` remains the only production NRPT option in either ecosystem. Cross-language bridging works if the rest of the project is Python-first.

## Routing Decision Tree

```
What is the Bayesian inference task?
|
+-- Python / JAX-based Bayesian modeling?
|   --> numpyro-core-mastery
|
+-- Julia-based Bayesian modeling?
|   --> turing-model-design
|
+-- Multimodal posterior / NUTS R-hat stays > 1.01?
|   --> consensus-mcmc-pigeons
|
+-- Neural ODE / UDE with posterior uncertainty (Julia Turing+DiffEq+Lux)?
|   --> bayesian-ude-workflow
|
+-- Neural ODE / UDE with posterior uncertainty (Python/JAX Diffrax+Equinox+NumPyro)?
|   --> bayesian-ude-jax
|
+-- Sparse Bayesian regression / credible intervals on SINDy coefficients?
|   --> bayesian-sindy-workflow (horseshoe prior, ensemble SINDy, UQ-SINDy)
|
+-- Bayesian PINN (BPINN / BNNODE)?
|   --> bayesian-pinn (internal AdvancedHMC, not Turing)
|
+-- Event-time / self-exciting / Hawkes data?
|   --> point-processes
|
+-- Approximate inference / scalable VI?
|   --> variational-inference-patterns
|
+-- Diagnose MCMC convergence / posterior quality?
    --> mcmc-diagnostics
```

## Skill Selection Table

| Task | Skill |
|------|-------|
| NumPyro models, NUTS, SVI | `numpyro-core-mastery` |
| Turing.jl, Julia samplers | `turing-model-design` |
| Multimodal posterior, parallel tempering | `consensus-mcmc-pigeons` |
| Bayesian neural ODE / UDE (Julia) | `bayesian-ude-workflow` |
| Bayesian neural ODE / UDE (Python / JAX) | `bayesian-ude-jax` |
| Bayesian SINDy with credible intervals | `bayesian-sindy-workflow` |
| Bayesian PINN (BPINN / BNNODE) | `bayesian-pinn` |
| Hawkes / self-exciting event data | `point-processes` |
| ELBO, flows, amortized VI | `variational-inference-patterns` |
| R-hat, ESS, ArviZ plots | `mcmc-diagnostics` |

## Checklist

- [ ] Run MCMC diagnostics (`mcmc-diagnostics`) on every posterior before interpreting results
- [ ] Verify R-hat < 1.01 and ESS > 400 per parameter before reporting
- [ ] Use NLSQ warm-start to initialize MCMC chains near the posterior mode
- [ ] Check prior predictive distribution before running inference
- [ ] Validate posterior predictive against held-out data
- [ ] Prefer NumPyro for GPU-accelerated large-scale inference; Turing.jl for Julia workflows
- [ ] Use VI (`variational-inference-patterns`) for exploratory modeling; NUTS for final results
- [ ] If R-hat persists above 1.01, switch to Pigeons via `consensus-mcmc-pigeons` before tweaking priors
- [ ] For neural ODEs / UDEs with uncertainty, follow the staged pipeline in `bayesian-ude-workflow`
- [ ] Document model assumptions, priors, and likelihood choices explicitly
