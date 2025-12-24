---
name: nlsq-core-mastery
version: "1.0.5"
maturity: "5-Expert"
specialization: GPU/TPU-Accelerated Nonlinear Least Squares
description: Master NLSQ library for high-performance curve fitting (150-270x faster than SciPy). Use when fitting >10K points, parameter estimation, robust optimization, streaming datasets (100M+ points), or migrating from SciPy.
---

# NLSQ Core Mastery

GPU/TPU-accelerated nonlinear least squares optimization with JAX, delivering 150-270x speedups for large-scale problems.

---

## API Decision Matrix

| Dataset Size | API | Memory | Use Case |
|-------------|-----|--------|----------|
| < 1M | `CurveFit` | Low | Standard fitting |
| 1M - 4M | `curve_fit_large()` | Managed | Auto chunking |
| 4M - 100M | `LargeDatasetFitter` | Configurable | Manual control |
| > 100M | `StreamingOptimizer` | Constant | Epoch-based |

---

## Core APIs

### Standard Optimization (CurveFit)

```python
from nlsq import CurveFit
import jax.numpy as jnp

def model(x, params):
    A, lambda_, c = params
    return A * jnp.exp(-lambda_ * x) + c

result = CurveFit(
    model=model, x=x, y=y,
    p0=jnp.array([5.0, 0.5, 1.0]),
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
    method='trf', loss='huber'
).fit()
```

### Large Dataset (curve_fit_large)

```python
from nlsq import curve_fit_large

result = curve_fit_large(
    model=model, x=x_5M, y=y_5M,
    p0=p0, bounds=bounds,
    loss='huber', show_progress=True
)
```

### Streaming (100M+ points)

```python
from nlsq import StreamingOptimizer
from nlsq.config import StreamingConfig

config = StreamingConfig(batch_size=100_000, n_epochs=15, optimizer='adam')
opt = StreamingOptimizer(model, p0, config=config)

for epoch in range(15):
    for x_batch, y_batch in data_generator:
        opt.update(x_batch, y_batch)
result = opt.result()
```

### Hybrid Streaming (Multi-scale Parameters)

```python
from nlsq import curve_fit, HybridStreamingConfig

# Handles parameters differing by 10^6×
popt, pcov = curve_fit(model, x, y, p0, method='hybrid_streaming')

# Presets: .aggressive(), .conservative(), .memory_optimized()
config = HybridStreamingConfig.conservative()
```

---

## Loss Function Selection

| Data Quality | Loss | Outlier % | Speed |
|-------------|------|-----------|-------|
| Clean, Gaussian | `linear` | 0% | Very Fast |
| Minor outliers | `soft_l1` | <5% | Fast |
| Moderate outliers | `huber` | 5-10% | Fast |
| Heavy outliers | `cauchy` | 10-20% | Moderate |
| Extreme outliers | `arctan` | >20% | Slow |

---

## Algorithm Selection

| Criterion | TRF | LM |
|-----------|-----|-----|
| Bounds | Required | None |
| Scale | >10K points | <10K points |
| Priority | Robustness | Speed |

---

## Memory Estimation

```python
def estimate_memory_gb(n_points, n_params, dtype='float32'):
    dtype_size = 4 if dtype == 'float32' else 8
    jacobian_gb = n_points * n_params * dtype_size * 4 / 1e9
    return jacobian_gb

# 10M points, 3 params: ~1.34 GB
# 100M points, 5 params: ~8.4 GB
```

---

## Quick Reference

### Mixed Precision

```python
from nlsq.config import configure_mixed_precision
configure_mixed_precision(enable=True)  # Up to 50% memory savings
```

### Callbacks

```python
from nlsq.callbacks import ProgressBar, EarlyStopping
optimizer = CurveFit(model, x, y, p0, callbacks=[
    ProgressBar(),
    EarlyStopping(min_delta=1e-8, patience=10)
])
```

### Diagnostics

```python
from scripts.diagnose_optimization import diagnose_result
diagnose_result(result)
# Check: cost reduction >50%, grad_norm <1e-6, jac_cond <1e10
```

### Parameter Normalization

```python
from nlsq.parameter_normalizer import ParameterNormalizer
normalizer = ParameterNormalizer(p0, bounds, strategy='bounds')
```

---

## Common Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| Divergence | Poor initial guess | Domain knowledge, multi-start |
| Ill-conditioned | Large parameter spread | Use `hybrid_streaming` |
| OOM | Large dataset | Use appropriate API tier |
| Slow convergence | Multi-scale params | Parameter normalization |
| JAX tracer errors | Python conditionals | Use `jnp.where`, `lax.cond` |

---

## Workflow Patterns

### Quick Fit
```python
result = CurveFit(model, x, y, p0).fit()
```

### Production Fit
```python
result = CurveFit(model, x, y, p0, bounds=bounds,
    method='trf', loss='soft_l1',
    ftol=1e-10, xtol=1e-10, max_nfev=500
).fit()
```

### Multi-start (difficult landscapes)
```python
for i in range(n_starts):
    p0_random = uniform(bounds[0], bounds[1])
    result = CurveFit(model, x, y, p0_random).fit()
    if result.cost < best_cost:
        best_result = result
```

---

## Performance Benchmarks

| Size | SciPy | NLSQ GPU | Speedup |
|------|-------|----------|---------|
| 10K | 0.1s | 0.05s | 2x |
| 100K | 10s | 0.1s | 100x |
| 1M | 100s | 0.5s | 200x |
| 10M | OOM | 5s | ∞ |

---

## Resources

- `scripts/benchmark_comparison.py` - NLSQ vs SciPy
- `scripts/diagnose_optimization.py` - Convergence diagnostics
- `assets/nlsq_quickstart.ipynb` - 10-min tutorial
- `references/loss_functions.md` - Loss function theory
- `references/convergence_diagnostics.md` - Troubleshooting

---

**Skill Version**: 1.0.5
**NLSQ Library**: v0.2.1+
