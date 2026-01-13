---
name: nlsq-core-mastery
version: "1.0.7"
description: Master NLSQ library for high-performance curve fitting (150-270x faster than SciPy). Use when fitting >10K points, parameter estimation, robust optimization, streaming datasets (100M+ points), or migrating from SciPy.
---

# NLSQ Core Mastery

## API Selection

| Dataset | API | Memory |
|---------|-----|--------|
| <1M | `CurveFit` | Low |
| 1M-4M | `curve_fit_large()` | Managed |
| 4M-100M | `LargeDatasetFitter` | Configurable |
| >100M | `StreamingOptimizer` | Constant |

## Standard Fitting

```python
from nlsq import CurveFit
def model(x, params):
    A, lambda_, c = params
    return A * jnp.exp(-lambda_ * x) + c

result = CurveFit(model=model, x=x, y=y, p0=jnp.array([5.0, 0.5, 1.0]),
                  bounds=([0,0,0], [np.inf,np.inf,np.inf]),
                  method='trf', loss='huber').fit()
```

## Large Datasets

```python
from nlsq import curve_fit_large
result = curve_fit_large(model, x_5M, y_5M, p0, bounds, loss='huber')
```

## Streaming (100M+)

```python
from nlsq import StreamingOptimizer, StreamingConfig
config = StreamingConfig(batch_size=100_000, n_epochs=15, optimizer='adam')
opt = StreamingOptimizer(model, p0, config=config)
for epoch in range(15):
    for x_batch, y_batch in data_gen:
        opt.update(x_batch, y_batch)
```

## Loss Selection

| Data | Loss | Outliers |
|------|------|----------|
| Clean | `linear` | 0% |
| Minor | `soft_l1` | <5% |
| Moderate | `huber` | 5-10% |
| Heavy | `cauchy` | 10-20% |
| Extreme | `arctan` | >20% |

## Algorithm: TRF vs LM

- **TRF**: Bounded, >10K points, robust
- **LM**: Unbounded, <10K points, fast

## Mixed Precision

```python
from nlsq.config import configure_mixed_precision
configure_mixed_precision(enable=True)  # 50% memory savings
```

## Diagnostics

```python
from scripts.diagnose_optimization import diagnose_result
diagnose_result(result)  # Check convergence metrics
```

## Pitfalls

| Issue | Solution |
|-------|----------|
| Divergence | Multi-start optimization |
| Ill-conditioned | `hybrid_streaming` method |
| OOM | Use appropriate API tier |
| JAX tracer errors | `jnp.where`, `lax.cond` |

**Outcome**: Fast GPU/TPU curve fitting with automatic memory management
