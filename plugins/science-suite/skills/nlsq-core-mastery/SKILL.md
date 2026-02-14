---
name: nlsq-core-mastery
version: "2.2.0"
description: Master NLSQ library for high-performance curve fitting (150-270x faster than SciPy). Use when fitting >10K points, parameter estimation, robust optimization, streaming datasets (100M+ points), or migrating from SciPy.
sources:
  - https://github.com/imewei/NLSQ
  - https://pypi.org/project/nlsq/
  - https://nlsq.readthedocs.io/
---

# NLSQ Core Mastery

## Expert Agent

For complex optimization problems, GPU acceleration setup, and large-scale curve fitting, delegate to the expert agent:

- **`jax-pro`**: Unified specialist for Nonlinear Least Squares (NLSQ) and Core JAX optimization.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Large-scale fitting (100M+ points), hybrid streaming optimization, and custom loss functions.

## Workflow Selection (v0.6.6)

| Workflow | Description | Memory Strategy | Use Case |
|----------|-------------|-----------------|----------|
| `auto` | **Default**. Local optimization | Auto-select | Standard fitting, known initial guess |
| `auto_global` | Global optimization (multi-start) | Auto-select | Multi-modal, unknown initial guess |
| `hpc` | Checkpointed global search | Streaming | Long-running HPC jobs |

## Standard Fitting

```python
from nlsq import fit
import jax.numpy as jnp

def model(x, params):
    A, lambda_, c = params
    return A * jnp.exp(-lambda_ * x) + c

# workflow="auto" handles memory management automatically
result = fit(model, x, y, p0=[5.0, 0.5, 1.0], workflow="auto")
```

## Global Optimization

```python
# workflow="auto_global" enables multi-start optimization
# Requires bounds for sampling space
result = fit(
    model, x, y,
    p0=[5.0, 0.5, 1.0],
    bounds=([0,0,0], [10, 5, 5]),
    workflow="auto_global"
)
```

## Large Datasets (Streaming)

The `workflow` parameter automatically selects streaming for large datasets (>100M points).

```python
# workflow="hpc" ensures checkpoints and streaming
result = fit(
    model, x_100M, y_100M, p0, bounds,
    workflow="hpc",
    checkpoint_dir="/scratch/checkpoints"
)
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

## Parallel Optimization

| Strategy | Implementation | Use Case |
|----------|----------------|----------|
| **Batching** | `jax.vmap(fit_fn)` | 100k independent curves |
| **Data Parallel** | `jax.pmap` / DDP | 100M+ data points (1 curve) |
| **Hybrid** | `StreamingOptimizer` | Infinite data stream (online) |
| **Pipeline** | `jax.lax.scan` | Sequential dependencies |

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
