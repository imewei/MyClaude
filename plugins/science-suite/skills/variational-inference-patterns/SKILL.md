---
name: variational-inference-patterns
description: Master ADVI variational inference with Turing.jl and Bijectors.jl for scalable approximate Bayesian inference. Use when MCMC is too slow for large datasets. Also use when exploring VI vs MCMC trade-offs, warm-starting MCMC with VI, implementing online/streaming Bayesian learning, or needing fast approximate posteriors. Use proactively when the user mentions slow sampling, scalability concerns with Bayesian models, or wants quick posterior exploration before full MCMC.
---

# Variational Inference

## Expert Agents

Variational inference is a cross-PPL technique — applicable to Turing.jl
(`Turing.Variational`), NumPyro (`numpyro.infer.SVI` with AutoGuides), and
custom JAX/PyTorch ELBO loops. Delegate to:

- **`julia-pro`**: Turing.jl + Bijectors.jl ADVI workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
- **`statistical-physicist`**: VI theory, ELBO geometry, normalizing flow
  design, and the VI-vs-MCMC trade-off across PPLs.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
- **`jax-pro`**: NumPyro SVI / AutoGuide / amortized inference and
  JAX-based normalizing flow implementations.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`

VI is most useful as a warm-start for MCMC (see `bayesian-ude-workflow`,
`consensus-mcmc-pigeons`) or as a fast posterior approximation when full
sampling is too expensive. Always validate VI posteriors against a short
NUTS run via `mcmc-diagnostics` before reporting.

---

## ADVI Pattern

```julia
using Turing, Bijectors

@model function my_model(data)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0, 1), 0, Inf)
    data ~ Normal(μ, σ)
end

model = my_model(data)

# Variational inference
q = vi(model, ADVI(10, 1000))

# Sample from approximation
samples = rand(q, 1000)
```

---

## VI vs MCMC Trade-offs

| Aspect | VI | MCMC |
|--------|----|----|
| Speed | Fast | Slow |
| Accuracy | Approximate | Exact (asymptotic) |
| Scalability | Large data | Limited |
| Uncertainty | May underestimate | Reliable |

---

## When to Use VI

- MCMC too slow
- Large datasets
- Quick exploration
- Online/streaming learning
- Warm-starting MCMC

---

## Checklist

- [ ] ELBO converged
- [ ] Approximation quality assessed
- [ ] Trade-offs understood
- [ ] MCMC comparison if needed

---

**Version**: 1.0.5
