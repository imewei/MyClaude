---
name: nlsq-core-mastery
description: Comprehensive guide for GPU/TPU-accelerated nonlinear least squares optimization using the NLSQ library with JAX. Use this skill when working with curve fitting, parameter estimation, large-scale optimization (millions+ data points), robust fitting with outliers, or when GPU/TPU acceleration is needed for nonlinear least squares problems. Covers CurveFit API, StreamingOptimizer for massive datasets, loss function selection, convergence diagnostics, JAX integration patterns, and real-world applications in physics, biology, and engineering.
---

# NLSQ Core Mastery

## Overview

Master high-performance nonlinear least squares optimization with the NLSQ library—a JAX-based, GPU/TPU-accelerated alternative to SciPy's curve_fit delivering 150-270x speedups for large-scale problems.

**When to use NLSQ:**
- Curve fitting with >10K data points
- Parameter estimation requiring GPU acceleration
- Large-scale optimization (millions+ points)
- Robust fitting with outliers
- Streaming optimization for unbounded datasets
- Real-time fitting applications

**Key advantages over SciPy:**
- 150-270x faster for large datasets (via GPU/TPU)
- Handles 50M+ data points via StreamingOptimizer
- Robust loss functions (Huber, Cauchy, Arctan)
- JAX ecosystem integration (autodiff, JIT, vmap)
- Streaming optimization for memory-constrained scenarios

## Core Capabilities

### 1. Standard Optimization with CurveFit

Fit nonlinear models to data using GPU/TPU-accelerated least squares:

```python
from nlsq import CurveFit
import jax.numpy as jnp

# Define model as pure JAX function
def exponential_decay(t, params):
    A, lambda_, c = params
    return A * jnp.exp(-lambda_ * t) + c

# Prepare data
t = jnp.linspace(0, 10, 100_000)  # 100K points
y = measured_data  # Your data here

# Initial guess
p0 = jnp.array([5.0, 0.5, 1.0])

# Create optimizer
optimizer = CurveFit(
    model=exponential_decay,
    x=t,
    y=y,
    p0=p0,
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),  # Physical constraints
    method='trf',  # Trust Region Reflective
    loss='huber'   # Robust to outliers
)

# Fit
result = optimizer.fit()

# Access results
print(f"Parameters: {result.x}")
print(f"Success: {result.success}")
print(f"Iterations: {result.nfev}")
print(f"Final cost: {result.cost}")
```

**Key parameters:**
- `model`: Pure JAX function `f(x, params) -> predictions`
- `p0`: Initial parameter guess (critical for convergence)
- `bounds`: Parameter constraints (use TRF method)
- `method`: `'trf'` (bounded) or `'lm'` (unbounded)
- `loss`: `'linear'`, `'soft_l1'`, `'huber'`, `'cauchy'`, `'arctan'`
- `ftol`, `xtol`, `gtol`: Convergence tolerances

### 2. Large-Scale Optimization with StreamingOptimizer

Handle datasets that don't fit in GPU memory:

```python
from nlsq import StreamingOptimizer

# Initialize streaming optimizer
optimizer = StreamingOptimizer(
    model=model,
    p0=initial_params,
    chunk_size=100_000,  # Process 100K points at a time
    loss='huber',
    method='trf'
)

# Stream data in chunks
for chunk_idx, (x_chunk, y_chunk) in enumerate(data_loader):
    # Update with new data
    convergence = optimizer.update(x_chunk, y_chunk)

    # Monitor convergence
    print(f"Chunk {chunk_idx}: cost={convergence.cost:.2e}")

    # Early stopping if converged
    if convergence.converged:
        break

# Get final result
result = optimizer.result()
```

**When to use streaming:**
- Dataset > GPU memory
- Estimated memory: `n_points × n_params × 4 bytes`
- Example: 10M points, 20 params = ~800MB Jacobian
- Use streaming when Jacobian > 80% GPU memory

**Memory estimation:**
```python
def estimate_memory(n_points, n_params):
    """Estimate GPU memory (GB) for NLSQ."""
    jacobian_gb = n_points * n_params * 4 / 1e9  # float32
    total_gb = jacobian_gb * 4  # Includes intermediate arrays
    return total_gb

memory = estimate_memory(10_000_000, 20)
print(f"Estimated memory: {memory:.2f} GB")

if memory > 8:  # Assuming 8GB GPU
    print("Use StreamingOptimizer")
    chunk_size = int(n_points * 8 / memory)
```

### 3. Loss Function Selection

Choose loss function based on data quality:

```python
# Decision tree for loss selection:

# Clean data, Gaussian noise → 'linear'
optimizer = CurveFit(model, x, y, p0, loss='linear')  # Fastest

# Few outliers (<5%) → 'soft_l1' or 'huber'
optimizer = CurveFit(model, x, y, p0, loss='soft_l1')  # Good balance

# Many outliers (5-20%) → 'cauchy'
optimizer = CurveFit(model, x, y, p0, loss='cauchy')  # Strong robustness

# Extreme outliers (>20%) → 'arctan'
optimizer = CurveFit(model, x, y, p0, loss='arctan')  # Maximum robustness
```

**Loss function characteristics:**

| Loss | Outlier Sensitivity | Speed | Use Case |
|------|---------------------|-------|----------|
| linear | High (r²) | Very Fast | Clean data |
| soft_l1 | Low-Medium | Fast | Minor outliers |
| huber | Medium | Fast | Moderate outliers |
| cauchy | Low | Moderate | Heavy outliers |
| arctan | Very Low | Slow | Extreme outliers |

**Compare loss functions:**
```python
# Test different loss functions on your data
for loss in ['linear', 'soft_l1', 'huber', 'cauchy']:
    result = CurveFit(model, x, y, p0, loss=loss).fit()
    print(f"{loss:8s}: cost={result.cost:.2e}, params={result.x}")
```

For detailed loss function theory and selection guidance, see `references/loss_functions.md`.

### 4. Algorithm Selection: TRF vs LM

**Trust Region Reflective (TRF):**
- Best for: Bounded problems, large-scale, robust convergence
- Use when: Parameters have physical bounds, N > 10K points

```python
optimizer = CurveFit(
    model, x, y, p0,
    bounds=([0, 0, 0], [100, 2, 10]),  # Bounds required
    method='trf'  # Trust Region Reflective
)
```

**Levenberg-Marquardt (LM):**
- Best for: Unbounded problems, well-conditioned, fast convergence
- Use when: No bounds needed, N < 10K points, good conditioning

```python
optimizer = CurveFit(
    model, x, y, p0,
    method='lm'  # Levenberg-Marquardt
)
```

**Decision matrix:**

| Criterion | Use TRF | Use LM |
|-----------|---------|--------|
| **Bounds** | Yes (required) | No bounds |
| **Scale** | >10K points | <10K points |
| **Robustness** | High priority | Speed priority |

### 5. JAX Integration Patterns

NLSQ requires pure JAX functions for JIT compilation:

**✅ Good: Pure function**
```python
def model(x, params):
    """Pure function: output depends only on inputs."""
    a, b, c = params
    return a * jnp.exp(-b * x) + c
```

**❌ Bad: Side effects**
```python
count = 0
def model_bad(x, params):
    global count
    count += 1  # Side effect breaks JIT!
    return params[0] * x + params[1]
```

**Efficient Jacobian strategies:**

```python
# Strategy 1: Auto-differentiation (default, works for any model)
optimizer = CurveFit(model, x, y, p0)  # Uses jax.jacfwd

# Strategy 2: Analytical Jacobian (fastest for simple models)
def analytical_jac(x, params):
    A, lambda_, c = params
    exp_term = jnp.exp(-lambda_ * x)
    return jnp.stack([
        exp_term,                 # ∂f/∂A
        -A * x * exp_term,       # ∂f/∂λ
        jnp.ones_like(x)         # ∂f/∂c
    ], axis=-1)

optimizer = CurveFit(model, x, y, p0, jac=analytical_jac)

# Strategy 3: jacrev for many parameters (>20)
from jax import jacrev
jac_fn = jacrev(lambda p: model(x, p))
optimizer = CurveFit(model, x, y, p0, jac=jac_fn)
```

**Vectorization with vmap:**
```python
# Fit multiple datasets in parallel
def fit_batch(x_batch, y_batch, p0_batch):
    def fit_single(x, y, p0):
        return CurveFit(model, x, y, p0).fit().x

    # Vectorize over datasets (GPU parallelism!)
    return jax.vmap(fit_single)(x_batch, y_batch, p0_batch)

# Fit 100 datasets simultaneously
all_params = fit_batch(x_batch, y_batch, p0_batch)
```

### 6. Convergence Diagnostics

Always diagnose optimization results:

```python
# Use provided diagnostic script
from scripts.diagnose_optimization import diagnose_result

result = optimizer.fit()
diagnostics = diagnose_result(result)  # Prints detailed analysis

# Check key metrics
if not result.success:
    print(f"Convergence failed: {result.message}")
    # See references/convergence_diagnostics.md for troubleshooting

# Check cost reduction
cost_reduction = (result.initial_cost - result.cost) / result.initial_cost
if cost_reduction < 0.1:
    print("⚠️  Poor cost reduction - check initial guess")

# Check gradient
grad_norm = jnp.linalg.norm(result.grad)
if grad_norm > 1e-4:
    print("⚠️  Large gradient - may not be at minimum")

# Check Jacobian conditioning
jac_cond = jnp.linalg.cond(result.jac)
if jac_cond > 1e10:
    print("⚠️  Ill-conditioned - consider parameter scaling")
```

**Quick diagnostic checklist:**
- ✅ Cost reduction > 50%
- ✅ Gradient norm < 1e-6
- ✅ Jacobian condition < 1e10
- ✅ No parameters at bounds
- ✅ Residuals symmetric around zero

For comprehensive diagnostics, see `references/convergence_diagnostics.md`.

### 7. Performance Optimization

**Benchmark NLSQ vs SciPy:**
```bash
# Run provided benchmark script
python scripts/benchmark_comparison.py
```

**Expected speedups:**

| Dataset Size | CPU (SciPy) | GPU (NLSQ) | Speedup |
|--------------|-------------|------------|---------|
| < 1K | Instant | JIT overhead | 0.5x |
| 1K-10K | ~0.1s | ~0.05s | 2-5x |
| 10K-100K | ~1-10s | ~0.1s | 10-50x |
| 100K-1M | ~10-100s | ~0.5s | 50-150x |
| 1M-10M | ~100-1000s | ~1-5s | 100-270x |
| > 10M | OOM | StreamingOptimizer | ∞ |

**Recommendation:**
- N < 10K: SciPy or NLSQ (similar)
- N ≥ 10K: NLSQ (significant speedup)
- N ≥ 1M: NLSQ (orders of magnitude faster)
- N > GPU memory: StreamingOptimizer required

## Workflow Patterns

### Pattern 1: Quick Fit

For straightforward curve fitting:

```python
from nlsq import CurveFit
import jax.numpy as jnp

# 1. Define model
def model(x, params):
    return params[0] * jnp.sin(params[1] * x + params[2])

# 2. Prepare data
x = jnp.array(x_data)
y = jnp.array(y_data)
p0 = jnp.array([1.0, 1.0, 0.0])  # Initial guess

# 3. Fit
result = CurveFit(model, x, y, p0).fit()

# 4. Use results
print(f"Fitted parameters: {result.x}")
```

### Pattern 2: Production-Ready Fit

With robustness, diagnostics, and error handling:

```python
from nlsq import CurveFit
import jax.numpy as jnp
from scripts.diagnose_optimization import diagnose_result

def robust_fit(model, x, y, p0, bounds=None):
    """Production-ready curve fitting with diagnostics."""

    # Input validation
    assert len(x) == len(y), "x and y must have same length"
    assert len(p0) > 0, "Must provide initial guess"

    # Optimize with robust settings
    optimizer = CurveFit(
        model=model,
        x=jnp.array(x),
        y=jnp.array(y),
        p0=jnp.array(p0),
        bounds=bounds,
        method='trf' if bounds else 'lm',
        loss='soft_l1',  # Mild robustness
        ftol=1e-10,      # Tight tolerances
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=500     # Allow sufficient iterations
    )

    # Fit with error handling
    try:
        result = optimizer.fit()
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None

    # Diagnose results
    diagnostics = diagnose_result(result, verbose=False)

    # Check quality
    if not result.success:
        print("⚠️  Convergence warning:")
        print(f"   {result.message}")

    if diagnostics['warnings']:
        print(f"⚠️  {len(diagnostics['warnings'])} diagnostic warnings")
        for warning in diagnostics['warnings']:
            print(f"   • {warning}")

    return result, diagnostics

# Usage
result, diag = robust_fit(model, x, y, p0, bounds=([0, 0, 0], [10, 5, 2]))
```

### Pattern 3: Large-Scale Streaming

For datasets exceeding GPU memory:

```python
from nlsq import StreamingOptimizer
import jax.numpy as jnp

def stream_from_database(model, p0, connection_string, chunk_size=500_000):
    """Stream data from database and fit."""

    import pandas as pd
    from sqlalchemy import create_engine

    # Setup optimizer
    optimizer = StreamingOptimizer(
        model=model,
        p0=p0,
        chunk_size=chunk_size,
        loss='huber',
        method='trf'
    )

    # Stream from database
    engine = create_engine(connection_string)
    query = "SELECT x, y FROM measurements ORDER BY timestamp"

    for chunk_idx, chunk_df in enumerate(
        pd.read_sql(query, engine, chunksize=chunk_size)
    ):
        # Convert to JAX arrays
        x_chunk = jnp.array(chunk_df['x'].values)
        y_chunk = jnp.array(chunk_df['y'].values)

        # Update optimizer
        convergence = optimizer.update(x_chunk, y_chunk)

        print(f"Chunk {chunk_idx}: cost={convergence.cost:.2e}")

        if convergence.converged:
            print(f"Converged after {chunk_idx + 1} chunks")
            break

    return optimizer.result()

# Usage
result = stream_from_database(
    model=my_model,
    p0=initial_guess,
    connection_string='postgresql://user:pass@host/db'
)
```

### Pattern 4: Multi-Start Optimization

For difficult optimization landscapes:

```python
def multi_start_fit(model, x, y, n_starts=10, bounds=None):
    """Try multiple initial guesses, return best result."""

    best_result = None
    best_cost = np.inf

    for i in range(n_starts):
        # Random initial guess within bounds
        if bounds:
            lower, upper = bounds
            p0 = jax.random.uniform(
                jax.random.PRNGKey(i),
                shape=(len(lower),),
                minval=lower,
                maxval=upper
            )
        else:
            p0 = jax.random.normal(jax.random.PRNGKey(i), shape=(n_params,))

        # Try this initial guess
        try:
            optimizer = CurveFit(model, x, y, p0, bounds=bounds)
            result = optimizer.fit()

            if result.success and result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except:
            continue

    return best_result

# Usage
result = multi_start_fit(model, x, y, n_starts=20, bounds=([0, 0], [10, 10]))
```

## Real-World Applications

### Physics: Exponential Decay

Fitting radioactive decay or relaxation data:

```python
def radioactive_decay(t, params):
    """N(t) = N₀ * exp(-λt)"""
    N0, lambda_ = params
    return N0 * jnp.exp(-lambda_ * t)

# Fit with physical constraints (all positive)
result = CurveFit(
    radioactive_decay,
    t_measured,
    N_measured,
    p0=[1000, 0.1],
    bounds=([0, 0], [np.inf, np.inf]),
    loss='huber'  # Robust to measurement errors
).fit()

N0, lambda_fit = result.x
half_life = jnp.log(2) / lambda_fit
print(f"Half-life: {half_life:.2f} hours")
```

### Biology: Dose-Response Curves

Fitting sigmoid dose-response (4-parameter logistic):

```python
def four_param_logistic(dose, params):
    """Sigmoid dose-response curve."""
    bottom, top, EC50, hill = params
    return bottom + (top - bottom) / (1 + (dose / EC50)**(-hill))

result = CurveFit(
    four_param_logistic,
    dose,
    response,
    p0=[0, 100, 1, 1],
    bounds=([0, 0, 0, 0.1], [50, 150, 1000, 10]),
    loss='soft_l1'
).fit()

bottom, top, EC50, hill = result.x
print(f"EC50: {EC50:.4f} μM")
print(f"Hill slope: {hill:.3f}")
```

### Spectroscopy: Multi-Gaussian Peaks

Fitting multiple overlapping peaks:

```python
def multi_gaussian(x, params):
    """Sum of N Gaussians + baseline."""
    n_peaks = (len(params) - 1) // 3
    baseline = params[-1]

    result = jnp.zeros_like(x) + baseline

    for i in range(n_peaks):
        A = params[3*i]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]
        result += A * jnp.exp(-0.5 * ((x - mu) / sigma)**2)

    return result

# Fit 3 peaks
p0 = [100, 1500, 50, 200, 2000, 60, 150, 2500, 55, 20]
result = CurveFit(
    multi_gaussian,
    energy,
    intensity,
    p0,
    method='trf',  # Many parameters, use TRF
    loss='soft_l1',
    max_nfev=500
).fit()
```

## Best Practices

### Initial Guess Quality

Good initial guesses are critical:

```python
# Strategy 1: Domain knowledge
def get_exponential_guess(x, y):
    """Intelligent guess for exponential decay."""
    c_guess = jnp.mean(y[-10:])  # Offset from tail
    A_guess = y[0] - c_guess      # Amplitude
    y_half = A_guess / 2 + c_guess
    idx_half = jnp.argmin(jnp.abs(y - y_half))
    lambda_guess = jnp.log(2) / x[idx_half]
    return jnp.array([A_guess, lambda_guess, c_guess])

# Strategy 2: Linear approximation
def get_linear_guess(x, y):
    """Use linear regression for initial guess."""
    # For y = a*x + b
    A_linear = jnp.vstack([x, jnp.ones_like(x)]).T
    params = jnp.linalg.lstsq(A_linear, y)[0]
    return params
```

### Parameter Scaling

Normalize parameters for better conditioning:

```python
# Map parameters to [0, 1] range
def create_normalized_model(model_fn, param_ranges):
    """Wrap model to use normalized parameters."""
    lower, upper = param_ranges

    def normalized_model(x, params_01):
        # Map [0, 1] to physical ranges
        params = lower + params_01 * (upper - lower)
        return model_fn(x, params)

    return normalized_model

# Usage
ranges = (jnp.array([0, 0, -10]), jnp.array([100, 2, 10]))
model_norm = create_normalized_model(model, ranges)

result = CurveFit(
    model_norm, x, y,
    p0=jnp.array([0.5, 0.5, 0.5]),  # All in [0, 1]
    bounds=([0, 0, 0], [1, 1, 1])
).fit()

# Map back to physical parameters
params_physical = ranges[0] + result.x * (ranges[1] - ranges[0])
```

### Testing Optimization Code

Always validate on synthetic data:

```python
def test_parameter_recovery():
    """Test that optimizer recovers known parameters."""

    # Generate perfect synthetic data
    true_params = jnp.array([5.0, 0.5, 1.0])
    x = jnp.linspace(0, 10, 100)
    y = model(x, true_params)

    # Add small noise
    y_noisy = y + jax.random.normal(key, y.shape) * 0.01

    # Optimize
    result = CurveFit(
        model, x, y_noisy,
        p0=jnp.array([4.0, 0.4, 0.8]),
        ftol=1e-10, xtol=1e-10
    ).fit()

    # Assert parameters recovered
    param_error = jnp.linalg.norm(result.x - true_params)
    assert param_error < 0.01, f"Parameter error too large: {param_error}"
    assert result.success, f"Optimization failed: {result.message}"
    assert result.cost < 1e-3, f"Final cost too large: {result.cost}"

    print("✓ Parameter recovery test passed")

test_parameter_recovery()
```

## Resources

This skill includes:

### scripts/

**Core Utilities**:

**`benchmark_comparison.py`**
Compare NLSQ vs SciPy performance across dataset sizes. Run to determine when NLSQ provides significant speedup.

```bash
python scripts/benchmark_comparison.py
```

**`diagnose_optimization.py`**
Comprehensive diagnostics for optimization results. Import and use on any OptimizeResult.

```python
from scripts.diagnose_optimization import diagnose_result
result = optimizer.fit()
diagnose_result(result)
```

**Example Collection** (19 production-ready scripts + README):

See `scripts/examples/README.md` for comprehensive documentation of all examples.

**`examples/gallery/`** - Domain-specific complete workflows (11 examples):
- `physics/` - Radioactive decay, damped oscillation, spectroscopy peaks
- `chemistry/` - Titration curves, reaction kinetics
- `biology/` - Dose-response, growth curves, enzyme kinetics
- `engineering/` - Sensor calibration, system identification, materials

**`examples/streaming/`** - Large-scale patterns (4 examples):
- Basic fault tolerance, checkpoint/resume, custom retry, diagnostics

**`examples/demos/`** - Feature showcases (4 examples):
- Enhanced errors, function library, result enhancements, callbacks

```bash
# Run any example
python scripts/examples/gallery/physics/radioactive_decay.py
python scripts/examples/streaming/01_basic_fault_tolerance.py
python scripts/examples/demos/callbacks_demo.py

# See full documentation
cat scripts/examples/README.md
```

### assets/

**Interactive Jupyter Notebooks** (3 notebooks + README):

**`nlsq_quickstart.ipynb`** - 10-minute introduction (beginner)
**`nlsq_interactive_tutorial.ipynb`** - 45-minute comprehensive tutorial (intermediate)
**`advanced_features_demo.ipynb`** - 30-minute advanced techniques (advanced)

See `assets/README.md` for complete notebook guide.

```bash
# Start learning
jupyter notebook assets/nlsq_quickstart.ipynb

# See notebook documentation
cat assets/README.md
```

### references/

**`convergence_diagnostics.md`**
Comprehensive guide to diagnosing convergence issues:
- Quick convergence checklist
- Common failure modes (divergence, stagnation, oscillation)
- Tolerance tuning guide
- Performance troubleshooting
- Real-time monitoring patterns

**`loss_functions.md`**
Detailed loss function reference:
- Mathematical theory for each loss function
- Decision tree for loss selection
- Impact on convergence and robustness
- Comparison examples with code
- When to switch loss functions

## Common Pitfalls

**1. Poor Initial Guess**
- Problem: Optimization diverges or converges to wrong minimum
- Solution: Use domain knowledge, linear approximation, or multi-start

**2. Ill-Conditioned Problems**
- Problem: Jacobian condition number > 1e10
- Solution: Scale parameters, remove redundant parameters, add regularization

**3. Wrong Loss Function**
- Problem: Fit dominated by outliers or underfit
- Solution: See `references/loss_functions.md` decision tree

**4. Memory Issues**
- Problem: Out of GPU memory errors
- Solution: Estimate memory, use StreamingOptimizer if needed

**5. JAX Tracer Errors**
- Problem: Python conditionals on traced values break JIT
- Solution: Use `jnp.where`, `jnp.select`, or `lax.cond` instead

**6. Slow Convergence**
- Problem: Hundreds of iterations without convergence
- Solution: Check conditioning, loosen tolerances, try different algorithm

## Quick Reference

**Basic fit:**
```python
result = CurveFit(model, x, y, p0).fit()
```

**Bounded fit:**
```python
result = CurveFit(model, x, y, p0, bounds=(lower, upper), method='trf').fit()
```

**Robust fit:**
```python
result = CurveFit(model, x, y, p0, loss='huber').fit()
```

**Streaming fit:**
```python
opt = StreamingOptimizer(model, p0, chunk_size)
for x_chunk, y_chunk in data_loader:
    opt.update(x_chunk, y_chunk)
result = opt.result()
```

**Diagnose:**
```python
from scripts.diagnose_optimization import diagnose_result
diagnose_result(result)
```

## Further Learning

- **NLSQ Documentation**: https://nlsq.readthedocs.io/
- **NLSQ Repository**: https://github.com/imewei/NLSQ
- **JAX Documentation**: https://jax.readthedocs.io/
- **Trust Region Methods**: Nocedal & Wright, "Numerical Optimization"
- **Robust Statistics**: Huber, "Robust Statistics"

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-10-28
**NLSQ Library**: JAX-based GPU/TPU-accelerated nonlinear least squares
