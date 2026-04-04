---
name: bayesian-inference
description: Meta-orchestrator for Bayesian inference and probabilistic programming. Routes to NumPyro, Turing.jl, variational inference, and MCMC diagnostics skills. Use when building probabilistic models with NumPyro or Turing.jl, running MCMC inference, implementing variational inference, or diagnosing sampler convergence.
---

# Bayesian Inference

Orchestrator for Bayesian inference and probabilistic programming. Routes problems to the appropriate specialized skill.

## Expert Agent

- **`statistical-physicist`**: Specialist for Bayesian inference, probabilistic modeling, and MCMC methods.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - *Capabilities*: NumPyro, Turing.jl, variational inference, MCMC diagnostics, and posterior analysis with ArviZ.

## Core Skills

### [NumPyro Core Mastery](../numpyro-core-mastery/SKILL.md)
NumPyro: model definition, NUTS/HMC, SVI, and integration with JAX for GPU-accelerated inference.

### [Turing Model Design](../turing-model-design/SKILL.md)
Turing.jl: probabilistic model design, sampler selection, and Julia-native Bayesian workflows.

### [Variational Inference Patterns](../variational-inference-patterns/SKILL.md)
VI methods: ELBO optimization, mean-field approximations, normalizing flows, and amortized inference.

### [MCMC Diagnostics](../mcmc-diagnostics/SKILL.md)
Posterior validation: R-hat, ESS, BFMI, trace plots, and convergence diagnostics with ArviZ.

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
- [ ] Document model assumptions, priors, and likelihood choices explicitly
