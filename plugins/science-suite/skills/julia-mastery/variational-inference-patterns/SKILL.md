---
name: variational-inference-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: Variational Inference
description: Master ADVI variational inference with Turing.jl and Bijectors.jl for scalable approximate Bayesian inference. Use when MCMC is too slow for large datasets.
---

# Variational Inference

ADVI for scalable approximate Bayesian inference.

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
