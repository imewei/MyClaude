---
name: nlsq-core-mastery
description: Master GPU/TPU-accelerated nonlinear least squares optimization using the NLSQ library with JAX for high-performance curve fitting and parameter estimation. This skill provides comprehensive guidance for implementing production-ready optimization workflows that are 150-270x faster than SciPy. Use this skill when writing or modifying Python files (.py) that perform curve fitting, parameter estimation, or nonlinear optimization; when working with files that import NLSQ library components (from nlsq import CurveFit, curve_fit_large, LargeDatasetFitter, StreamingOptimizer, or any nlsq.* modules); when implementing curve fitting in Jupyter notebooks (.ipynb) for data analysis or scientific computing; when writing optimization scripts for experimental data processing; when creating model fitting pipelines for physics simulations, biological assays, chemical kinetics, spectroscopy analysis, or engineering sensor calibration; when performing parameter estimation on datasets ranging from thousands to 100M+ data points; when fitting exponential decay models, dose-response curves, sigmoid functions, Gaussian peaks, multi-component mixtures, reaction kinetics, or any nonlinear mathematical model; when implementing robust fitting algorithms that handle outlier-contaminated data using Huber, Cauchy, or Arctan loss functions; when selecting appropriate loss functions (linear, soft_l1, huber, cauchy, arctan) based on data quality and outlier prevalence; when choosing between optimization algorithms (Trust-Region-Reflective TRF vs Levenberg-Marquardt LM) for bounded or unbounded problems; when handling large-scale datasets (1M-100M+ points) that require memory-efficient chunking, streaming, or epoch-based optimization; when working with HDF5 files (.h5, .hdf5) containing large scientific datasets; when implementing fault-tolerant optimization with checkpointing for multi-hour or multi-day computations; when writing JAX-compatible pure functions for automatic differentiation and JIT compilation; when enabling mixed precision fallback for automatic memory optimization (up to 50% savings); when implementing callbacks for progress monitoring, early stopping, or iteration logging; when optimizing sparse Jacobian problems with >10 parameters; when diagnosing convergence failures, checking gradient norms, analyzing residuals, or validating fit quality; when implementing bounded parameter constraints with physical or mathematical limits; when benchmarking NLSQ performance against SciPy curve_fit; when migrating existing SciPy optimization code to GPU/TPU acceleration; when creating data analysis workflows in scientific Python projects; when working with time-series data requiring exponential, logarithmic, or power-law fitting; when processing measurement data from laboratory instruments, sensors, or detectors; when analyzing spectroscopic data (NMR, IR, UV-Vis, mass spec) with multi-peak deconvolution; when fitting pharmacokinetic models, enzyme kinetics (Michaelis-Menten), or growth curves (logistic, Gompertz); when implementing real-time fitting applications that require sub-second optimization; when deploying production machine learning pipelines that include nonlinear regression components; when creating automated analysis tools for physics experiments, chemical reactions, biological assays, or materials characterization; or when any code requires high-performance nonlinear least squares optimization with GPU/TPU acceleration, robust loss functions, large-scale data handling, automatic precision management, or production-ready convergence diagnostics.
---

# NLSQ Core Mastery

## When to use this skill

**File Types & Code Contexts:**
- Writing or modifying Python files (.py) that perform curve fitting, parameter estimation, or nonlinear optimization
- Working with Jupyter notebooks (.ipynb) for data analysis, scientific computing, or exploratory fitting
- Editing files that import NLSQ library components (from nlsq import CurveFit, curve_fit_large, LargeDatasetFitter, StreamingOptimizer)
- Processing HDF5 files (.h5, .hdf5) containing large scientific datasets
- Creating optimization scripts for experimental data pipelines
- Migrating existing SciPy curve_fit code to GPU/TPU-accelerated NLSQ

**Dataset Scales & Performance Optimization:**
- Fitting datasets with 1,000 to 100M+ data points
- Large-scale optimization (1M-10M points) requiring automatic chunking with curve_fit_large
- Very large datasets (10M-100M points) needing manual memory control with LargeDatasetFitter
- Massive streaming datasets (>100M points) requiring epoch-based StreamingOptimizer
- Achieving 150-270x speedup over SciPy curve_fit with GPU/TPU acceleration
- Real-time fitting applications requiring sub-second optimization
- Memory-constrained environments needing mixed precision fallback (up to 50% memory savings)

**Mathematical Models & Applications:**
- Fitting exponential decay models (radioactive decay, relaxation, damping)
- Dose-response curves and sigmoid functions (4-parameter logistic, Hill equation)
- Multi-peak deconvolution (Gaussian, Lorentzian, Voigt profiles)
- Spectroscopy analysis (NMR, IR, UV-Vis, mass spectrometry)
- Reaction kinetics and chemical dynamics
- Pharmacokinetic models and enzyme kinetics (Michaelis-Menten)
- Growth curves (logistic, Gompertz, exponential)
- Time-series fitting (exponential, logarithmic, power-law)
- Multi-component mixture analysis
- Any nonlinear mathematical model requiring parameter estimation

**Domain-Specific Applications:**
- Physics: radioactive decay, damped oscillations, quantum mechanics fitting
- Biology: dose-response, enzyme kinetics, growth curves, fluorescence decay
- Chemistry: titration curves, reaction rates, thermodynamics
- Engineering: sensor calibration, system identification, materials characterization
- Data science: nonlinear regression in machine learning pipelines

**Robust Fitting & Data Quality:**
- Implementing robust fitting for outlier-contaminated data
- Selecting loss functions (linear, soft_l1, huber, cauchy, arctan) based on outlier prevalence
- Handling noisy experimental measurements from laboratory instruments
- Processing data with <5% outliers (soft_l1, huber), 5-20% outliers (cauchy), or >20% outliers (arctan)

**Algorithm Selection & Optimization:**
- Choosing between Trust-Region-Reflective (TRF) for bounded problems or Levenberg-Marquardt (LM) for unbounded
- Implementing bounded parameter constraints with physical or mathematical limits
- Optimizing sparse Jacobian problems with >10 parameters (2-10x speedup)
- Writing JAX-compatible pure functions for automatic differentiation and JIT compilation
- Comparing different optimization algorithms and loss functions

**Convergence & Diagnostics:**
- Diagnosing convergence failures, gradient explosion, or numerical instability
- Checking gradient norms, Jacobian conditioning, or residual analysis
- Validating fit quality and parameter recovery
- Implementing progress monitoring, early stopping, or iteration logging with callbacks
- Troubleshooting ill-conditioned problems or poor initial guesses

**Production & Deployment:**
- Deploying production curve fitting pipelines with fault tolerance and checkpointing
- Creating automated analysis tools for multi-hour or multi-day optimizations
- Benchmarking NLSQ performance against SciPy for justifying migration
- Implementing batch processing for multiple datasets with GPU parallelism
- Building scientific data analysis workflows with reproducible results

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
- Automatic mixed precision fallback (up to 50% memory savings)
- Built-in callbacks for monitoring and early stopping
- Sparse Jacobian optimization (2-10x speedup for >10 parameters)

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

### 2. Large Dataset Optimization (4M - 100M+ Points)

NLSQ provides three strategies for handling large datasets: automatic detection, manual chunking, and streaming optimization.

#### Decision Tree by Dataset Size

| Dataset Size | Strategy | API | Memory Usage | Notes |
|--------------|----------|-----|--------------|-------|
| **< 1M** | Standard | `CurveFit` | Low | No special handling needed |
| **1M - 4M** | Automatic | `curve_fit_large()` | Managed | Automatic chunking, best for most users |
| **4M - 10M** | Manual Chunking | `LargeDatasetFitter` | Configurable | Fine-grained control |
| **10M - 100M** | Advanced Chunking | `LargeDatasetFitter` | Minimal | Progress monitoring essential |
| **> 100M** | Streaming | `StreamingOptimizer` | Constant | Epoch-based optimization |

**Memory scaling (baseline):**
- 10M points, 3 params: ~1.34 GB
- Linear scaling: 100M points ≈ 13.4 GB
- Formula: `memory_gb = (n_points × n_params × 4 bytes × 4) / 1e9`

#### 2a. Automatic Large Dataset Handling (RECOMMENDED)

The `curve_fit_large()` function automatically detects dataset size and applies optimal strategies:

```python
from nlsq import curve_fit_large
import jax.numpy as jnp

# Prepare large dataset (e.g., 5 million points)
x = jnp.linspace(0, 100, 5_000_000)
y = measured_data  # Your 5M data points

# Automatic optimization - handles chunking, memory, progress
result = curve_fit_large(
    model=exponential_decay,
    x=x,
    y=y,
    p0=jnp.array([5.0, 0.5, 1.0]),
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
    loss='huber',
    show_progress=True  # Display progress bar
)

print(f"✓ Optimized {len(x):,} points")
print(f"  Parameters: {result.x}")
print(f"  Iterations: {result.nfev}")
print(f"  Final cost: {result.cost:.6e}")
```

**What curve_fit_large() does automatically:**
1. Estimates memory requirements
2. Determines optimal chunk size based on available GPU/TPU memory
3. Configures chunking strategy (typically 4 chunks for 10M points)
4. Displays progress bar during optimization
5. Handles memory management seamlessly

**When to use:**
- ✅ Dataset 1M - 100M points
- ✅ Want automatic handling without manual configuration
- ✅ Need progress monitoring
- ✅ Prefer simplicity over fine-grained control

#### 2b. Manual Large Dataset Control

For advanced users needing fine-grained control over chunking and memory:

```python
from nlsq import LargeDatasetFitter
from nlsq.config import LDMemoryConfig

# Configure memory limits
memory_config = LDMemoryConfig(
    max_memory_gb=4.0,        # Maximum memory usage (default: 4.0 GB)
    min_chunk_size=50_000,    # Minimum points per chunk (default: 50K)
    max_chunk_size=2_000_000, # Maximum points per chunk (default: 2M)
    enable_streaming=False     # Disable streaming (use chunking only)
)

# Create fitter with custom configuration
fitter = LargeDatasetFitter(
    model=model,
    x=x,
    y=y,
    p0=p0,
    memory_config=memory_config,
    bounds=bounds,
    loss='huber',
    method='trf'
)

# Estimate memory before fitting
memory_estimate = fitter.estimate_memory()
print(f"Estimated memory: {memory_estimate['total_gb']:.2f} GB")
print(f"Recommended chunks: {memory_estimate['n_chunks']}")
print(f"Chunk size: {memory_estimate['chunk_size']:,} points")

# Fit with progress monitoring
result = fitter.fit_with_progress(show_progress=True)
```

**Memory configuration options:**
- `max_memory_gb`: Hard limit on GPU memory usage
- `min_chunk_size`: Avoid creating too many small chunks
- `max_chunk_size`: Upper bound for optimal kernel sizes
- `enable_streaming`: Switch to streaming if chunks still too large

**When to use:**
- ✅ Need specific memory constraints (e.g., multi-GPU, shared environments)
- ✅ Want control over chunk sizes for performance tuning
- ✅ Running on constrained hardware
- ✅ Benchmarking different chunking strategies

#### 2c. Streaming Optimization (100M+ Points)

For datasets exceeding 100M points or unlimited streaming data:

```python
from nlsq import StreamingOptimizer
from nlsq.config import StreamingConfig

# Configure streaming optimization
streaming_config = StreamingConfig(
    batch_size=100_000,     # Points per batch (50K-100K typical)
    n_epochs=10,            # Number of passes through data (10-20)
    optimizer='adam',       # Adam or SGD (Adam recommended)
    learning_rate=0.001,    # Initial learning rate
    enable_checkpointing=True  # Fault tolerance
)

# Initialize streaming optimizer
optimizer = StreamingOptimizer(
    model=model,
    p0=initial_params,
    config=streaming_config,
    loss='huber',
    method='trf'
)

# Epoch-based streaming (multiple passes)
for epoch in range(streaming_config.n_epochs):
    print(f"\n=== Epoch {epoch + 1}/{streaming_config.n_epochs} ===")

    for batch_idx, (x_batch, y_batch) in enumerate(data_generator):
        # Update with new batch
        state = optimizer.update(x_batch, y_batch)

        # Monitor progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: cost={state.cost:.4e}, "
                  f"grad_norm={jnp.linalg.norm(state.grad):.4e}")

        # Early stopping within epoch
        if state.converged:
            print(f"  ✓ Converged at batch {batch_idx}")
            break

    # Checkpoint after each epoch
    if streaming_config.enable_checkpointing:
        optimizer.save_checkpoint(f'checkpoint_epoch_{epoch}.pkl')

# Get final result
result = optimizer.result()
print(f"\n✓ Streaming optimization complete")
print(f"  Final parameters: {result.x}")
print(f"  Final cost: {result.cost:.6e}")
```

**Key streaming parameters:**
- `batch_size`: 50K-100K points typical (balance speed vs memory)
- `n_epochs`: 10-20 passes through data (more epochs for better convergence)
- `optimizer`: Adam (adaptive learning) or SGD (classic gradient descent)
- `learning_rate`: 0.001-0.01 typical (Adam adapts automatically)

**When to use streaming:**
- ✅ Dataset > 100M points
- ✅ Data doesn't fit in memory (database, file streams)
- ✅ Continuous/infinite data streams
- ✅ Need constant memory footprint regardless of dataset size

#### 2d. HDF5 Large Dataset Support

For datasets stored in HDF5 files (common in scientific computing):

```python
import h5py
from nlsq import StreamingOptimizer

def hdf5_data_generator(filepath, dataset_name, batch_size=100_000):
    """Generator for streaming HDF5 data."""
    with h5py.File(filepath, 'r') as f:
        dataset = f[dataset_name]
        n_points = dataset.shape[0]

        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)

            # Load batch from disk
            batch_data = dataset[start_idx:end_idx]

            # Separate x and y
            x_batch = jnp.array(batch_data[:, 0])
            y_batch = jnp.array(batch_data[:, 1])

            yield x_batch, y_batch

# Use with StreamingOptimizer
optimizer = StreamingOptimizer(model, p0, batch_size=100_000)

for epoch in range(10):
    for x_batch, y_batch in hdf5_data_generator('large_data.h5', 'measurements'):
        state = optimizer.update(x_batch, y_batch)

result = optimizer.result()
```

**HDF5 benefits:**
- Datasets larger than RAM
- Efficient random access to chunks
- Compressed storage
- Standard format for scientific data

#### 2e. Fault Tolerance & Checkpointing

For long-running optimizations (hours/days), implement checkpointing:

```python
from nlsq import StreamingOptimizer
import pickle
import os

def fit_with_fault_tolerance(model, p0, data_generator, checkpoint_file='checkpoint.pkl'):
    """Fault-tolerant streaming optimization with checkpointing."""

    # Try to resume from checkpoint
    if os.path.exists(checkpoint_file):
        print("📂 Resuming from checkpoint...")
        with open(checkpoint_file, 'rb') as f:
            optimizer = pickle.load(f)
        start_epoch = optimizer.current_epoch
    else:
        print("🆕 Starting new optimization...")
        optimizer = StreamingOptimizer(model, p0, batch_size=100_000)
        start_epoch = 0

    try:
        for epoch in range(start_epoch, 20):
            print(f"\nEpoch {epoch + 1}/20")

            for batch_idx, (x_batch, y_batch) in enumerate(data_generator):
                state = optimizer.update(x_batch, y_batch)

                # Save checkpoint every 100 batches
                if batch_idx % 100 == 0:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(optimizer, f)
                    print(f"  💾 Checkpoint saved (batch {batch_idx})")

            # Checkpoint after each epoch
            optimizer.current_epoch = epoch + 1
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(optimizer, f)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted! Checkpoint saved. Resume by running again.")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(optimizer, f)
        return None

    # Clean up checkpoint on success
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return optimizer.result()

# Usage
result = fit_with_fault_tolerance(model, p0, data_gen, 'my_checkpoint.pkl')
```

**Fault tolerance benefits:**
- Resume after crashes, interruptions, or time limits
- Checkpoint at epochs or batch intervals
- Save computational time and resources
- Essential for multi-day optimizations

#### Memory Estimation for Large Datasets

**Accurate memory estimation:**
```python
def estimate_memory_requirements(n_points, n_params, dtype='float32'):
    """
    Comprehensive memory estimation for NLSQ optimization.

    Memory breakdown:
    - Input data (x, y): 2 × n_points × sizeof(dtype)
    - Jacobian: n_points × n_params × sizeof(dtype)
    - Gradient: n_params × sizeof(dtype)
    - Hessian approximation: n_params × n_params × sizeof(dtype)
    - Intermediate arrays: ~3× Jacobian size

    Returns memory in GB and recommended strategy.
    """
    dtype_size = 4 if dtype == 'float32' else 8

    # Component sizes
    data_gb = 2 * n_points * dtype_size / 1e9
    jacobian_gb = n_points * n_params * dtype_size / 1e9
    params_gb = n_params * dtype_size / 1e9
    hessian_gb = n_params * n_params * dtype_size / 1e9
    intermediate_gb = 3 * jacobian_gb

    # Total memory
    total_gb = data_gb + jacobian_gb + params_gb + hessian_gb + intermediate_gb

    # Recommend strategy
    if total_gb < 1.0:
        strategy = "CurveFit (standard)"
    elif total_gb < 4.0:
        strategy = "curve_fit_large (automatic)"
        n_chunks = max(2, int(total_gb / 1.0))
    elif total_gb < 16.0:
        strategy = "LargeDatasetFitter (manual chunking)"
        n_chunks = max(4, int(total_gb / 2.0))
    else:
        strategy = "StreamingOptimizer (streaming)"
        n_chunks = "N/A (streaming)"

    return {
        'total_gb': total_gb,
        'components': {
            'data': data_gb,
            'jacobian': jacobian_gb,
            'parameters': params_gb,
            'hessian': hessian_gb,
            'intermediate': intermediate_gb
        },
        'strategy': strategy,
        'n_chunks': n_chunks if strategy != "StreamingOptimizer (streaming)" else None,
        'chunk_size': n_points // n_chunks if isinstance(n_chunks, int) else None
    }

# Examples
print("=== 10M points, 3 params ===")
estimate = estimate_memory_requirements(10_000_000, 3)
print(f"Total memory: {estimate['total_gb']:.2f} GB")
print(f"Strategy: {estimate['strategy']}")
if estimate['n_chunks']:
    print(f"Chunks: {estimate['n_chunks']} × {estimate['chunk_size']:,} points")

print("\n=== 100M points, 5 params ===")
estimate = estimate_memory_requirements(100_000_000, 5)
print(f"Total memory: {estimate['total_gb']:.2f} GB")
print(f"Strategy: {estimate['strategy']}")
```

**Real-world estimates:**
- 4M points, 3 params: ~0.54 GB → curve_fit_large
- 10M points, 3 params: ~1.34 GB → curve_fit_large (2-4 chunks)
- 50M points, 5 params: ~8.4 GB → LargeDatasetFitter (4-6 chunks)
- 100M points, 10 params: ~33.6 GB → StreamingOptimizer

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

### 6. Mixed Precision Fallback

NLSQ automatically manages numerical precision to balance memory efficiency with accuracy:

**How it works:**
- Starts optimization in float32 for memory efficiency
- Automatically upgrades to float64 when convergence stalls
- Up to 50% memory savings when optimization completes in float32

**Enable globally:**
```python
from nlsq.config import configure_mixed_precision

# Enable mixed precision fallback
configure_mixed_precision(enable=True)

# Optional: customize thresholds
configure_mixed_precision(
    enable=True,
    max_degradation_iterations=3,  # Upgrade after 3 stalled iterations
    gradient_threshold=1e10         # Upgrade if gradient exceeds this
)
```

**Enable via environment variables:**
```bash
export NLSQ_MIXED_PRECISION_ENABLE=true
export NLSQ_MIXED_PRECISION_MAX_DEGRADATION_ITERATIONS=3
```

**Automatic fallback triggers:**
1. **Convergence stalling**: No cost improvement for N iterations (default: 5)
2. **Gradient explosion**: Gradient magnitude exceeds threshold (default: 1e10)
3. **Numerical instability**: NaN/Inf values appear in state variables
4. **Trust radius collapse**: Trust radius becomes excessively small

**When to use:**
- ✅ Memory-constrained systems (GPU/TPU)
- ✅ Large datasets exceeding 100K points
- ✅ Batch processing multiple fits
- ✅ Production deployments requiring memory efficiency

**Example:**
```python
from nlsq import CurveFit
from nlsq.config import configure_mixed_precision

# Enable mixed precision globally
configure_mixed_precision(enable=True)

# Optimization starts in float32, upgrades automatically if needed
result = CurveFit(model, x, y, p0).fit()

# Check if precision was upgraded
if result.precision_upgraded:
    print("⚠️  Upgraded to float64 for numerical stability")
else:
    print("✓ Completed in float32 (50% memory savings)")
```

**Note:** This feature eliminates the need for manual `dtype=jnp.float64` handling. The system automatically selects the optimal precision for your problem.

### 7. Callbacks & Progress Monitoring

Monitor optimization progress and implement custom stopping criteria:

```python
from nlsq import CurveFit
from nlsq.callbacks import ProgressBar, IterationLogger, EarlyStopping

# Progress bar with real-time updates
optimizer = CurveFit(
    model, x, y, p0,
    callbacks=[ProgressBar()]
)

# Log iterations to file
optimizer = CurveFit(
    model, x, y, p0,
    callbacks=[IterationLogger(filename='optimization.log')]
)

# Early stopping based on cost improvement
optimizer = CurveFit(
    model, x, y, p0,
    callbacks=[EarlyStopping(min_delta=1e-8, patience=10)]
)

# Combine multiple callbacks
optimizer = CurveFit(
    model, x, y, p0,
    callbacks=[
        ProgressBar(),
        IterationLogger('opt.log'),
        EarlyStopping(min_delta=1e-6, patience=5)
    ]
)
result = optimizer.fit()
```

**Custom callbacks:**
```python
class CustomCallback:
    def __init__(self):
        self.iteration_costs = []

    def on_iteration(self, state, iteration):
        """Called after each optimization iteration."""
        self.iteration_costs.append(state.cost)

        # Custom logic
        if iteration > 10 and state.cost > 1e3:
            return True  # Stop optimization
        return False  # Continue

    def on_complete(self, result):
        """Called when optimization completes."""
        print(f"Completed in {len(self.iteration_costs)} iterations")

# Use custom callback
callback = CustomCallback()
result = CurveFit(model, x, y, p0, callbacks=[callback]).fit()
```

### 8. Sparse Jacobian Optimization

Achieve 2-10x speedups for problems with >10 parameters by providing sparsity patterns:

```python
import jax.numpy as jnp
from nlsq import CurveFit

# Example: Multi-exponential with sparse structure
def multi_exponential(t, params):
    """Sum of independent exponentials (sparse Jacobian)."""
    n_terms = len(params) // 2
    result = jnp.zeros_like(t)

    for i in range(n_terms):
        A = params[2*i]
        lambda_ = params[2*i + 1]
        result += A * jnp.exp(-lambda_ * t)

    return result

# Define sparsity pattern (which parameters affect which outputs)
def sparsity_pattern(n_params):
    """Return boolean mask of Jacobian sparsity."""
    # For multi-exponential, each term affects all outputs
    # but parameters within terms are independent
    return jnp.ones((n_points, n_params), dtype=bool)

# Use sparse Jacobian
optimizer = CurveFit(
    model=multi_exponential,
    x=t,
    y=y,
    p0=initial_params,
    jac_sparsity=sparsity_pattern(len(initial_params))
)

result = optimizer.fit()
```

**When sparse optimization helps:**
- Models with >10 parameters
- Independent parameter groups (e.g., multi-peak fitting)
- Block-diagonal Jacobian structures
- Separable model components

### 9. Convergence Diagnostics

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

### 10. Performance Optimization

**Benchmark NLSQ vs SciPy:**
```bash
# Run provided benchmark script
python scripts/benchmark_comparison.py
```

**Real-world benchmark (NVIDIA Tesla V100):**
- 1M data points, 5 parameters: **0.15s (NLSQ) vs 40.5s (SciPy) = 270x speedup**
- Excellent scaling: 50x more data → only 1.2x slower

**Expected speedups:**

| Dataset Size | CPU (SciPy) | GPU (NLSQ) | Speedup | Recommended API |
|--------------|-------------|------------|---------|-----------------|
| < 1K | Instant | JIT overhead | 0.5x | `CurveFit` or SciPy |
| 1K-10K | ~0.1s | ~0.05s | 2-5x | `CurveFit` |
| 10K-100K | ~1-10s | ~0.1s | 10-50x | `CurveFit` |
| 100K-1M | ~10-100s | ~0.5s | 50-150x | `CurveFit` |
| 1M-4M | ~100-400s | ~1-2s | 100-200x | **`curve_fit_large`** |
| 4M-10M | ~400-1000s | ~2-5s | 150-270x | **`curve_fit_large`** |
| 10M-100M | OOM or hours | ~5-30s | ∞ | **`LargeDatasetFitter`** |
| > 100M | OOM | Constant | ∞ | **`StreamingOptimizer`** |

**Decision Guide:**
- **N < 1M**: Use `CurveFit` (standard API, simple, fast)
- **1M ≤ N < 10M**: Use `curve_fit_large` (automatic chunking, progress bar)
- **10M ≤ N < 100M**: Use `LargeDatasetFitter` (manual memory control)
- **N ≥ 100M**: Use `StreamingOptimizer` (epoch-based, constant memory)

**Memory optimization with Mixed Precision:**
```python
from nlsq.config import configure_mixed_precision

# Enable for automatic memory savings
configure_mixed_precision(enable=True)

# Reduces memory by up to 50% when optimization completes in float32
# Automatically upgrades to float64 only when needed for stability
```

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

### Pattern 3: Large-Scale Datasets (4M-100M+ Points)

**For 1M-10M points (automatic):**
```python
from nlsq import curve_fit_large
import jax.numpy as jnp

# Simple, automatic large dataset handling
result = curve_fit_large(
    model=exponential_decay,
    x=x_data,  # e.g., 5 million points
    y=y_data,
    p0=jnp.array([1.0, 0.1, 0.5]),
    bounds=([0, 0, 0], [10, 2, 5]),
    loss='huber',
    show_progress=True  # Progress bar
)

print(f"✓ Fitted {len(x_data):,} points")
print(f"  Parameters: {result.x}")
```

**For 10M-100M points (manual chunking):**
```python
from nlsq import LargeDatasetFitter
from nlsq.config import LDMemoryConfig

# Configure memory constraints
memory_config = LDMemoryConfig(
    max_memory_gb=8.0,         # Your GPU memory
    min_chunk_size=100_000,
    max_chunk_size=5_000_000
)

# Manual chunking control
fitter = LargeDatasetFitter(
    model=model,
    x=x_data,  # e.g., 50 million points
    y=y_data,
    p0=p0,
    memory_config=memory_config,
    loss='soft_l1'
)

# Estimate and fit
memory_est = fitter.estimate_memory()
print(f"Memory: {memory_est['total_gb']:.2f} GB, "
      f"Chunks: {memory_est['n_chunks']}")

result = fitter.fit_with_progress(show_progress=True)
```

**For >100M points (streaming from database):**
```python
from nlsq import StreamingOptimizer
from nlsq.config import StreamingConfig
import jax.numpy as jnp
import pandas as pd
from sqlalchemy import create_engine

def stream_fit_database(model, p0, connection_string, batch_size=100_000):
    """Streaming optimization from database with epochs."""

    # Configure streaming
    config = StreamingConfig(
        batch_size=batch_size,
        n_epochs=15,
        optimizer='adam',
        learning_rate=0.001,
        enable_checkpointing=True
    )

    optimizer = StreamingOptimizer(model, p0, config=config, loss='huber')

    # Database connection
    engine = create_engine(connection_string)
    query = "SELECT x, y FROM measurements ORDER BY id"

    # Multiple epochs for convergence
    for epoch in range(config.n_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.n_epochs} ===")

        for batch_idx, batch_df in enumerate(
            pd.read_sql(query, engine, chunksize=batch_size)
        ):
            x_batch = jnp.array(batch_df['x'].values)
            y_batch = jnp.array(batch_df['y'].values)

            state = optimizer.update(x_batch, y_batch)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: cost={state.cost:.4e}")

        # Checkpoint after epoch
        optimizer.save_checkpoint(f'epoch_{epoch}.pkl')

    return optimizer.result()

# Usage
result = stream_fit_database(
    model=my_model,
    p0=initial_guess,
    connection_string='postgresql://user:pass@host/db',
    batch_size=100_000
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
- Solution:
  - 1M-10M points: Use `curve_fit_large()` for automatic chunking
  - 10M-100M points: Use `LargeDatasetFitter` with custom memory limits
  - >100M points: Use `StreamingOptimizer` for constant memory
  - Always estimate memory first: `estimate_memory_requirements(n_points, n_params)`

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

**Large dataset fit:**
```python
# Automatic (1M-10M points)
result = curve_fit_large(model, x, y, p0, show_progress=True)

# Manual control (10M-100M points)
from nlsq import LargeDatasetFitter
from nlsq.config import LDMemoryConfig
fitter = LargeDatasetFitter(model, x, y, p0, memory_config=LDMemoryConfig(max_memory_gb=8.0))
result = fitter.fit_with_progress()

# Streaming (>100M points)
from nlsq import StreamingOptimizer
from nlsq.config import StreamingConfig
opt = StreamingOptimizer(model, p0, config=StreamingConfig(batch_size=100_000, n_epochs=15))
for epoch in range(15):
    for x_batch, y_batch in data_loader:
        opt.update(x_batch, y_batch)
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

**Skill Version**: 1.0.2
**Last Updated**: 2025-10-31
**NLSQ Library**: JAX-based GPU/TPU-accelerated nonlinear least squares (v0.2.1+)
**Key Updates**:
- **v1.0.2**: Aligned with plugin version 1.0.2
- **v1.0.1**: Enhanced skill discoverability with comprehensive use-case descriptions
- **v1.0.0**: Initial skill with core NLSQ capabilities including large dataset handling, mixed precision fallback, callbacks, and streaming optimization
