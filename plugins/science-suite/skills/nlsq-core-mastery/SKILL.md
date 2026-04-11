---
name: nlsq-core-mastery
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

## Workflow Selection

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

## Related JAX Optimization Tools

NLSQ is specialized for **large-scale curve fitting** (150–270× faster than SciPy). For other optimization tasks in JAX, use the following companion libraries — all three are Patrick Kidger / DeepMind successors to the now-archived JAXopt:

| Task | Tool | Key API |
|------|------|---------|
| General root-finding / NLSQ / minimization | **Optimistix** | `optx.root_find`, `optx.least_squares`, `optx.minimise`; `Newton`, `LevenbergMarquardt`, `Dogleg`, `GaussNewton`, `BFGS`, `NonlinearCG`, `NelderMead` |
| Gradient-based optimization (training) | **Optax** | `optax.adam`/`adamw`/`sgd`/`lbfgs`, `optax.chain`, schedules, clipping; also hosts losses/projections from JAXopt |
| Linear solvers | **Lineax** | `lineax.linear_solve`, operator-based: `LU`, `Cholesky`, `QR`, `SVD`, `CG`, `GMRES`, `BiCGStab` |
| ODE-constrained fitting | **Diffrax + Optimistix/NLSQ** | Fit parameters of a dynamical system via `diffeqsolve` inside the loss (Neural ODE / UDE style) |

```python
# NLSQ  = GPU curve fit (large data, a single model)
from nlsq import fit
result = fit(model, x, y, p0=p0, workflow="auto")

# Optimistix  = general NLSQ / root-finding (arbitrary residual, small scale)
import optimistix as optx
solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8)
sol = optx.least_squares(residual_fn, solver, y0=p0, args=(x, y))
```

> **Rule of thumb**: use NLSQ for ≥10⁴-point curve fits (hot GPU path). Use Optimistix when residuals are few but the Jacobian or constraints are complex. They compose — NLSQ for the outer data fit, Optimistix for inner sub-problems.

## Checklist

- [ ] Verify workflow selection matches dataset size: `auto` for standard, `auto_global` for multi-modal, `hpc` for >100M points
- [ ] Confirm initial guess `p0` is physically reasonable to avoid local minima
- [ ] Check that bounds are set for `auto_global` and `hpc` workflows
- [ ] Select loss function based on outlier fraction (linear for clean, cauchy/arctan for heavy outliers)
- [ ] Choose TRF for bounded problems >10K points, LM for unbounded <10K points
- [ ] Enable mixed precision (`configure_mixed_precision`) for memory-constrained workloads
- [ ] Run `diagnose_result()` to verify convergence metrics after fitting
- [ ] Validate fitted parameters against physical constraints and expected ranges
- [ ] Confirm checkpoint directory exists and is writable for `hpc` workflow
