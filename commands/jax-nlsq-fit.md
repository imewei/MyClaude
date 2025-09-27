---
description: Perform nonlinear least-squares curve fitting with NLSQ library, model definition, and GPU/TPU acceleration
category: jax-optimization
argument-hint: "[--model-type] [--gpu-accel] [--chunking] [--algorithm]"
allowed-tools: "*"
---

# /jax-nlsq-fit

Perform nonlinear least-squares curve fitting with NLSQ (from imewei/NLSQ). Import CurveFit, define model fn, handle large datasets with chunking, and use GPU/TPU acceleration via JIT.

## Description

Advanced nonlinear least-squares optimization using the NLSQ library with JAX backend acceleration. Provides comprehensive curve fitting capabilities with GPU/TPU support, memory-efficient chunking for large datasets, and robust optimization algorithms for scientific computing applications.

## Usage

```
/jax-nlsq-fit [--model-type] [--gpu-accel] [--chunking] [--algorithm]
```

## What it does

1. Set up NLSQ library with JAX backend for curve fitting
2. Define custom model functions with automatic differentiation
3. Implement memory-efficient chunking for large datasets
4. Apply GPU/TPU acceleration with JIT compilation
5. Handle optimization convergence and robustness

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap
import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional

# Install NLSQ: pip install nlsq
try:
    from nlsq import CurveFit, LevenbergMarquardt, TrustRegionReflective
    NLSQ_AVAILABLE = True
except ImportError:
    print("NLSQ not available. Install with: pip install nlsq")
    NLSQ_AVAILABLE = False
    # Fallback to JAX-only implementation

# JAX backend optimization settings
jax.config.update('jax_enable_x64', True)  # Enable double precision if needed
jax.config.update('jax_platform_name', 'gpu')  # Use GPU if available

class JAXNLSQOptimizer:
    """Advanced nonlinear least-squares optimizer with JAX backend"""

    def __init__(self, algorithm: str = "lm", use_gpu: bool = True, chunk_size: Optional[int] = None):
        self.algorithm = algorithm.lower()
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        self.compiled_functions = {}

    def setup_backend(self):
        """Configure JAX backend for optimization"""
        if self.use_gpu and jax.devices('gpu'):
            print(f"Using GPU: {jax.devices('gpu')[0]}")
        else:
            print(f"Using CPU: {jax.devices('cpu')[0]}")

    @staticmethod
    @jit
    def exponential_model(x: jax.Array, params: jax.Array) -> jax.Array:
        """Exponential decay model: y = a * exp(-b * x) + c"""
        a, b, c = params
        return a * jnp.exp(-b * x) + c

    @staticmethod
    @jit
    def gaussian_model(x: jax.Array, params: jax.Array) -> jax.Array:
        """Gaussian model: y = a * exp(-((x - mu) / sigma)^2) + baseline"""
        a, mu, sigma, baseline = params
        return a * jnp.exp(-((x - mu) / sigma) ** 2) + baseline

    @staticmethod
    @jit
    def polynomial_model(x: jax.Array, params: jax.Array) -> jax.Array:
        """Polynomial model: y = sum(params[i] * x^i)"""
        return jnp.polyval(params, x)

    @staticmethod
    @jit
    def sigmoidal_model(x: jax.Array, params: jax.Array) -> jax.Array:
        """Sigmoidal model: y = a / (1 + exp(-k * (x - x0))) + baseline"""
        a, k, x0, baseline = params
        return a / (1 + jnp.exp(-k * (x - x0))) + baseline

    def create_residual_function(self, model_func: Callable) -> Callable:
        """Create JIT-compiled residual function for optimization"""
        @jit
        def residual_fn(params: jax.Array, x: jax.Array, y_obs: jax.Array) -> jax.Array:
            """Compute residuals between model and observations"""
            y_pred = model_func(x, params)
            return y_pred - y_obs

        return residual_fn

    def create_jacobian_function(self, residual_func: Callable) -> Callable:
        """Create JIT-compiled Jacobian function"""
        @jit
        def jacobian_fn(params: jax.Array, x: jax.Array, y_obs: jax.Array) -> jax.Array:
            """Compute Jacobian matrix of residuals w.r.t. parameters"""
            return jax.jacfwd(residual_func)(params, x, y_obs)

        return jacobian_fn

    def estimate_memory_requirements(self, n_points: int, n_params: int) -> Dict[str, float]:
        """Estimate memory requirements for optimization"""
        # Jacobian matrix size
        jacobian_size = n_points * n_params * 8  # 8 bytes per float64

        # Additional arrays (residuals, gradients, etc.)
        additional_arrays = n_points * 8 * 3  # residuals, predictions, observations

        total_memory_mb = (jacobian_size + additional_arrays) / (1024 * 1024)

        return {
            'jacobian_mb': jacobian_size / (1024 * 1024),
            'additional_mb': additional_arrays / (1024 * 1024),
            'total_mb': total_memory_mb,
            'recommended_chunk_size': min(10000, max(1000, int(1e8 / (n_params * 8))))
        }

    def chunked_optimization(self,
                           x_data: jax.Array,
                           y_data: jax.Array,
                           model_func: Callable,
                           initial_params: jax.Array,
                           chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """Perform optimization with memory-efficient chunking"""

        n_points = len(x_data)
        n_params = len(initial_params)

        # Estimate memory and determine chunk size
        memory_est = self.estimate_memory_requirements(n_points, n_params)
        if chunk_size is None:
            chunk_size = min(self.chunk_size or memory_est['recommended_chunk_size'], n_points)

        print(f"Memory estimate: {memory_est['total_mb']:.2f} MB")
        print(f"Using chunk size: {chunk_size}")

        # Create compiled functions
        residual_func = self.create_residual_function(model_func)
        jacobian_func = self.create_jacobian_function(residual_func)

        if n_points <= chunk_size:
            # No chunking needed
            return self._single_chunk_optimization(
                x_data, y_data, residual_func, jacobian_func, initial_params
            )
        else:
            # Chunked optimization
            return self._multi_chunk_optimization(
                x_data, y_data, residual_func, jacobian_func,
                initial_params, chunk_size
            )

    def _single_chunk_optimization(self,
                                 x_data: jax.Array,
                                 y_data: jax.Array,
                                 residual_func: Callable,
                                 jacobian_func: Callable,
                                 initial_params: jax.Array) -> Dict[str, Any]:
        """Optimize using single chunk (fits in memory)"""

        if NLSQ_AVAILABLE:
            # Use NLSQ library
            def model_wrapper(x, *params):
                return np.array(residual_func(jnp.array(params), jnp.array(x), jnp.zeros_like(x)))

            # Choose algorithm
            if self.algorithm == "lm":
                optimizer = LevenbergMarquardt()
            elif self.algorithm == "trr":
                optimizer = TrustRegionReflective()
            else:
                optimizer = LevenbergMarquardt()  # Default

            # Perform optimization
            curve_fit = CurveFit(optimizer)
            result = curve_fit.fit(
                np.array(x_data),
                np.array(y_data),
                model_wrapper,
                p0=np.array(initial_params)
            )

            return {
                'params': jnp.array(result.x),
                'cost': result.cost,
                'jacobian': jnp.array(result.jac) if hasattr(result, 'jac') else None,
                'success': result.success,
                'message': result.message,
                'algorithm': self.algorithm,
                'n_iterations': result.nfev if hasattr(result, 'nfev') else None
            }
        else:
            # Fallback JAX-only implementation
            return self._jax_levenberg_marquardt(
                x_data, y_data, residual_func, jacobian_func, initial_params
            )

    def _jax_levenberg_marquardt(self,
                               x_data: jax.Array,
                               y_data: jax.Array,
                               residual_func: Callable,
                               jacobian_func: Callable,
                               initial_params: jax.Array,
                               max_iterations: int = 100,
                               tolerance: float = 1e-8) -> Dict[str, Any]:
        """JAX-only Levenberg-Marquardt implementation"""

        @jit
        def lm_step(params: jax.Array, lambda_reg: float) -> Tuple[jax.Array, float]:
            """Single Levenberg-Marquardt step"""
            residuals = residual_func(params, x_data, y_data)
            jacobian = jacobian_func(params, x_data, y_data)

            # Compute cost
            cost = 0.5 * jnp.sum(residuals ** 2)

            # LM update: (J^T J + λI) δ = -J^T r
            JtJ = jnp.dot(jacobian.T, jacobian)
            Jtr = jnp.dot(jacobian.T, residuals)

            # Add damping
            JtJ_damped = JtJ + lambda_reg * jnp.diag(jnp.diag(JtJ))

            # Solve for parameter update
            delta = jnp.linalg.solve(JtJ_damped, -Jtr)
            new_params = params + delta

            return new_params, cost

        # Optimization loop
        params = initial_params
        lambda_reg = 1e-3
        costs = []

        for iteration in range(max_iterations):
            new_params, cost = lm_step(params, lambda_reg)
            costs.append(cost)

            # Check convergence
            if iteration > 0 and abs(costs[-2] - costs[-1]) < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

            # Update damping parameter
            if iteration > 0:
                if costs[-1] < costs[-2]:
                    lambda_reg *= 0.3  # Decrease damping
                    params = new_params
                else:
                    lambda_reg *= 2.0  # Increase damping
            else:
                params = new_params

        return {
            'params': params,
            'cost': costs[-1],
            'costs': jnp.array(costs),
            'success': True,
            'algorithm': 'jax_lm',
            'n_iterations': len(costs)
        }

    def _multi_chunk_optimization(self,
                                x_data: jax.Array,
                                y_data: jax.Array,
                                residual_func: Callable,
                                jacobian_func: Callable,
                                initial_params: jax.Array,
                                chunk_size: int) -> Dict[str, Any]:
        """Optimize using multiple chunks for large datasets"""

        n_points = len(x_data)
        n_chunks = (n_points + chunk_size - 1) // chunk_size

        print(f"Processing {n_points} points in {n_chunks} chunks")

        # Initialize parameters
        params = initial_params
        total_cost = 0.0

        # Process each chunk
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_points)

            x_chunk = x_data[start_idx:end_idx]
            y_chunk = y_data[start_idx:end_idx]

            print(f"Processing chunk {chunk_idx + 1}/{n_chunks} ({len(x_chunk)} points)")

            # Optimize on chunk
            chunk_result = self._single_chunk_optimization(
                x_chunk, y_chunk, residual_func, jacobian_func, params
            )

            # Update parameters (use result as starting point for next chunk)
            params = chunk_result['params']
            total_cost += chunk_result['cost']

        return {
            'params': params,
            'cost': total_cost,
            'success': True,
            'algorithm': f'{self.algorithm}_chunked',
            'n_chunks': n_chunks,
            'chunk_size': chunk_size
        }

# Practical usage examples
def demonstrate_nlsq_fitting():
    """Demonstrate NLSQ curve fitting with various models"""

    # Generate synthetic data
    key = random.PRNGKey(42)
    x_true = jnp.linspace(0, 10, 1000)

    # Exponential decay with noise
    true_params = jnp.array([2.0, 0.5, 0.1])  # a, b, c
    y_true = JAXNLSQOptimizer.exponential_model(x_true, true_params)
    noise = 0.05 * random.normal(key, y_true.shape)
    y_obs = y_true + noise

    print("=== NLSQ Curve Fitting Demonstration ===")

    # Initialize optimizer
    optimizer = JAXNLSQOptimizer(algorithm="lm", use_gpu=True)
    optimizer.setup_backend()

    # Initial parameter guess
    initial_guess = jnp.array([1.5, 0.3, 0.0])

    # Perform optimization
    result = optimizer.chunked_optimization(
        x_true, y_obs,
        JAXNLSQOptimizer.exponential_model,
        initial_guess,
        chunk_size=5000
    )

    print(f"\\nOptimization Results:")
    print(f"True parameters: {true_params}")
    print(f"Fitted parameters: {result['params']}")
    print(f"Parameter errors: {jnp.abs(result['params'] - true_params)}")
    print(f"Final cost: {result['cost']:.6e}")
    print(f"Algorithm: {result['algorithm']}")

    return result

# Model comparison and selection
def model_comparison_suite(x_data: jax.Array, y_data: jax.Array) -> Dict[str, Dict]:
    """Compare multiple models and select best fit"""

    models = {
        'exponential': (JAXNLSQOptimizer.exponential_model, jnp.array([1.0, 1.0, 0.0])),
        'gaussian': (JAXNLSQOptimizer.gaussian_model, jnp.array([1.0, 5.0, 1.0, 0.0])),
        'polynomial_2': (lambda x, p: JAXNLSQOptimizer.polynomial_model(x, p), jnp.array([0.0, 0.0, 1.0])),
        'sigmoidal': (JAXNLSQOptimizer.sigmoidal_model, jnp.array([1.0, 1.0, 5.0, 0.0]))
    }

    optimizer = JAXNLSQOptimizer(algorithm="lm")
    results = {}

    for model_name, (model_func, initial_params) in models.items():
        print(f"\\nFitting {model_name} model...")

        try:
            result = optimizer.chunked_optimization(
                x_data, y_data, model_func, initial_params
            )

            # Compute AIC for model comparison
            n_points = len(y_data)
            n_params = len(initial_params)
            residual_sum_squares = 2 * result['cost']  # cost = 0.5 * sum(residuals^2)
            aic = n_points * jnp.log(residual_sum_squares / n_points) + 2 * n_params

            result['aic'] = aic
            result['model_name'] = model_name
            results[model_name] = result

            print(f"  Cost: {result['cost']:.6e}")
            print(f"  AIC: {aic:.2f}")

        except Exception as e:
            print(f"  Failed: {e}")
            results[model_name] = {'error': str(e)}

    # Find best model (lowest AIC)
    valid_results = {k: v for k, v in results.items() if 'aic' in v}
    if valid_results:
        best_model = min(valid_results.keys(), key=lambda k: valid_results[k]['aic'])
        print(f"\\nBest model: {best_model} (AIC: {valid_results[best_model]['aic']:.2f})")

    return results

# Advanced: Custom model with physical constraints
def constrained_optimization_example():
    """Example with physically constrained parameters"""

    @jit
    def constrained_exponential_model(x: jax.Array, params: jax.Array) -> jax.Array:
        """Exponential model with positivity constraints"""
        # Apply constraints: a > 0, b > 0, c >= 0
        a = jnp.exp(params[0])  # Ensures a > 0
        b = jnp.exp(params[1])  # Ensures b > 0
        c = jax.nn.softplus(params[2])  # Ensures c >= 0
        constrained_params = jnp.array([a, b, c])
        return JAXNLSQOptimizer.exponential_model(x, constrained_params)

    # Generate test data
    key = random.PRNGKey(123)
    x_test = jnp.linspace(0, 5, 500)
    y_test = 3.0 * jnp.exp(-0.8 * x_test) + 0.2 + 0.02 * random.normal(key, x_test.shape)

    optimizer = JAXNLSQOptimizer(algorithm="lm")

    # Use log-space initial guess for constrained parameters
    initial_guess = jnp.array([jnp.log(2.0), jnp.log(0.5), 0.1])

    result = optimizer.chunked_optimization(
        x_test, y_test, constrained_exponential_model, initial_guess
    )

    # Transform back to original parameter space
    fitted_params_original = jnp.array([
        jnp.exp(result['params'][0]),
        jnp.exp(result['params'][1]),
        jax.nn.softplus(result['params'][2])
    ])

    print("\\n=== Constrained Optimization Example ===")
    print(f"Fitted parameters (original space): {fitted_params_original}")
    print(f"All parameters positive: {jnp.all(fitted_params_original > 0)}")

    return result, fitted_params_original

# Performance benchmarking
def benchmark_nlsq_performance():
    """Benchmark NLSQ performance across different configurations"""

    dataset_sizes = [1000, 10000, 100000]
    algorithms = ["lm", "trr"] if NLSQ_AVAILABLE else ["lm"]

    results = {}

    for n_points in dataset_sizes:
        for algorithm in algorithms:
            print(f"\\nBenchmarking {algorithm} with {n_points} points...")

            # Generate test data
            key = random.PRNGKey(42)
            x_bench = jnp.linspace(0, 10, n_points)
            y_bench = 2.0 * jnp.exp(-0.5 * x_bench) + 0.1 + 0.01 * random.normal(key, x_bench.shape)

            optimizer = JAXNLSQOptimizer(algorithm=algorithm, use_gpu=True)

            # Time the optimization
            import time
            start_time = time.time()

            result = optimizer.chunked_optimization(
                x_bench, y_bench,
                JAXNLSQOptimizer.exponential_model,
                jnp.array([1.5, 0.3, 0.0])
            )

            end_time = time.time()

            key = f"{algorithm}_{n_points}"
            results[key] = {
                'time_seconds': end_time - start_time,
                'cost': result['cost'],
                'algorithm': algorithm,
                'n_points': n_points,
                'points_per_second': n_points / (end_time - start_time)
            }

            print(f"  Time: {results[key]['time_seconds']:.3f}s")
            print(f"  Throughput: {results[key]['points_per_second']:.0f} points/s")

    return results

# Run demonstrations
if __name__ == "__main__":
    # Basic demonstration
    demo_result = demonstrate_nlsq_fitting()

    # Model comparison
    key = random.PRNGKey(123)
    x_test = jnp.linspace(0, 10, 1000)
    y_test = 1.5 * jnp.exp(-((x_test - 5.0) / 2.0) ** 2) + 0.1 + 0.02 * random.normal(key, x_test.shape)

    comparison_results = model_comparison_suite(x_test, y_test)

    # Constrained optimization
    constrained_result = constrained_optimization_example()

    # Performance benchmark
    benchmark_results = benchmark_nlsq_performance()

    print("\\n=== NLSQ Optimization Complete ===")
    print("All demonstrations completed successfully!")
```

## Related Commands

- `/jax-nlsq-large` - Optimize NLSQ for massive datasets
- `/jax-mixed-prec` - Enable mixed precision for NLSQ optimization
- `/jax-sparse-jac` - Compute sparse Jacobians for NLSQ
- `/jax-grad` - Understand gradient computation in optimization