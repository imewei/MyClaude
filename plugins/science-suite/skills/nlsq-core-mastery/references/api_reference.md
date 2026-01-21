# API Reference: NLSQ v0.6.6

## Core Function: `fit()`

The unified entry point for all curve fitting tasks.

```python
from nlsq import fit

result = fit(
    f: Callable,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    # ... standard arguments ...
    workflow: str | None = None,  # PRIMARY CONFIGURATION
    # ... advanced arguments ...
)
```

### Workflows

| Workflow | Description | Backend | Use Case |
|----------|-------------|---------|----------|
| `auto` | **Default**. Local optimization. Auto-selects strategy based on memory. | `CurveFit` or `LargeDatasetFitter` | Standard fitting, known initial guess. |
| `auto_global` | Global optimization with multi-start. | `MultiStartOrchestrator` | Multi-modal landscapes, unknown `p0`. Requires `bounds`. |
| `hpc` | Robust streaming with checkpoints. | `StreamingOptimizer` | Long-running jobs, cluster environments. |

### Parameters

- **f**: Model function `f(x, *params)`. Must use `jax.numpy`.
- **xdata**: Independent variable.
- **ydata**: Dependent variable.
- **p0**: Initial parameter guess. Optional if `workflow="auto_global"`.
- **bounds**: Parameter bounds `([min...], [max...])`. Required for `auto_global`.
- **workflow**: One of `"auto"`, `"auto_global"`, `"hpc"`.
- **memory_limit_gb**: Max memory to use (for `auto` decisions).
- **checkpoint_dir**: Directory for `hpc` workflow checkpoints.

### Returns

Returns a `CurveFitResult` object that behaves like a tuple `(popt, pcov)` but contains additional metadata.

```python
# Tuple unpacking (standard)
popt, pcov = fit(...)

# Object access (advanced)
result = fit(...)
print(result.popt)
print(result.cost)
print(result.message)
```

## Legacy API (Deprecated)

- `curve_fit()`: Use `fit(workflow="auto")`
- `curve_fit_large()`: Use `fit(workflow="auto")` with large data
- `preset="fast"`, `preset="robust"`: Use `workflow` parameter

## Configuration

```python
from nlsq import MemoryConfig, memory_context

# Temporary memory limit
with memory_context(MemoryConfig(memory_limit_gb=4.0)):
    fit(...)
```
