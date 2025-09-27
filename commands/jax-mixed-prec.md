---
description: Enable mixed precision in JAX backends using bfloat16 via ml_dtypes with float64 fallback for numerical stability
category: jax-performance
argument-hint: "[--precision-level] [--fallback-ops] [--monitoring] [--optimization]"
allowed-tools: "*"
---

# /jax-mixed-prec

Enable mixed precision in JAX backends (bfloat16 via ml_dtypes). Fallback to float64 for numerical stability in optimizations like NLSQ.

## Description

Comprehensive mixed precision training and computation in JAX using bfloat16 for speed and memory efficiency while maintaining numerical stability through strategic float64 fallbacks. Includes precision monitoring, automatic fallback detection, and optimization for scientific computing workflows.

## Usage

```
/jax-mixed-prec [--precision-level] [--fallback-ops] [--monitoring] [--optimization]
```

## What it does

1. Configure JAX for mixed precision computation with bfloat16/float32/float64
2. Implement automatic fallback strategies for numerical stability
3. Monitor precision-related numerical issues during computation
4. Optimize memory usage and computational speed
5. Handle special cases in scientific computing and optimization

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap
import numpy as np
from typing import Dict, Any, Callable, Optional, Union, Tuple
import warnings

# Import ml_dtypes for extended precision support
try:
    import ml_dtypes
    ML_DTYPES_AVAILABLE = True
    print("ml_dtypes available - extended precision support enabled")
except ImportError:
    ML_DTYPES_AVAILABLE = False
    print("ml_dtypes not available. Install with: pip install ml_dtypes")

# Configure JAX for mixed precision
jax.config.update('jax_enable_x64', True)  # Enable float64 when needed

class MixedPrecisionManager:
    """Comprehensive mixed precision management for JAX computations"""

    def __init__(self,
                 default_precision: str = "float32",
                 compute_precision: str = "bfloat16",
                 fallback_precision: str = "float64",
                 stability_threshold: float = 1e-7,
                 monitor_numerics: bool = True):

        self.default_precision = default_precision
        self.compute_precision = compute_precision
        self.fallback_precision = fallback_precision
        self.stability_threshold = stability_threshold
        self.monitor_numerics = monitor_numerics

        # Precision hierarchy: lower to higher precision
        self.precision_hierarchy = {
            'bfloat16': 1,
            'float16': 1,
            'float32': 2,
            'float64': 3
        }

        self.dtype_map = self._setup_dtype_mapping()
        self.numerical_issues = []

    def _setup_dtype_mapping(self) -> Dict[str, jnp.dtype]:
        """Set up dtype mapping with ml_dtypes support"""
        dtype_map = {
            'float16': jnp.float16,
            'float32': jnp.float32,
            'float64': jnp.float64
        }

        if ML_DTYPES_AVAILABLE:
            dtype_map.update({
                'bfloat16': ml_dtypes.bfloat16,
            })
        else:
            # Fallback to float16 if bfloat16 not available
            dtype_map['bfloat16'] = jnp.float16
            warnings.warn("bfloat16 not available, using float16 as fallback")

        return dtype_map

    def get_dtype(self, precision: str) -> jnp.dtype:
        """Get JAX dtype for precision specification"""
        if precision not in self.dtype_map:
            raise ValueError(f"Unknown precision: {precision}")
        return self.dtype_map[precision]

    def cast_to_precision(self, array: jax.Array, precision: str) -> jax.Array:
        """Cast array to specified precision"""
        target_dtype = self.get_dtype(precision)
        return array.astype(target_dtype)

    def detect_numerical_instability(self, array: jax.Array, operation: str = "unknown") -> bool:
        """Detect numerical instability in arrays"""
        has_nan = jnp.any(jnp.isnan(array))
        has_inf = jnp.any(jnp.isinf(array))

        if has_nan or has_inf:
            issue = {
                'operation': operation,
                'has_nan': bool(has_nan),
                'has_inf': bool(has_inf),
                'array_shape': array.shape,
                'array_dtype': array.dtype
            }
            self.numerical_issues.append(issue)
            return True

        # Check for values close to precision limits
        if array.dtype == self.get_dtype('bfloat16'):
            # bfloat16 has limited range
            max_val = jnp.max(jnp.abs(array))
            if max_val > 1e30 or (max_val > 0 and max_val < 1e-30):
                return True

        return False

    @jit
    def mixed_precision_computation(self,
                                  x: jax.Array,
                                  computation_fn: Callable,
                                  use_mixed_precision: bool = True) -> jax.Array:
        """Perform computation with mixed precision and automatic fallback"""

        if not use_mixed_precision:
            return computation_fn(x)

        # Cast input to compute precision
        x_compute = self.cast_to_precision(x, self.compute_precision)

        # Perform computation
        result_compute = computation_fn(x_compute)

        # Cast result back to default precision
        result = self.cast_to_precision(result_compute, self.default_precision)

        return result

    def adaptive_precision_wrapper(self, computation_fn: Callable) -> Callable:
        """Create adaptive precision wrapper that falls back on numerical issues"""

        def wrapped_computation(x: jax.Array, *args, **kwargs) -> jax.Array:
            # Try with compute precision first
            try:
                x_compute = self.cast_to_precision(x, self.compute_precision)
                result = computation_fn(x_compute, *args, **kwargs)

                # Check for numerical issues
                if self.monitor_numerics and self.detect_numerical_instability(result, "mixed_precision"):
                    raise ValueError("Numerical instability detected")

                return self.cast_to_precision(result, self.default_precision)

            except (ValueError, FloatingPointError):
                # Fallback to higher precision
                print(f"Falling back to {self.fallback_precision} precision")
                x_fallback = self.cast_to_precision(x, self.fallback_precision)
                result = computation_fn(x_fallback, *args, **kwargs)
                return self.cast_to_precision(result, self.default_precision)

        return wrapped_computation

# Scientific computing applications with mixed precision
class MixedPrecisionScientificComputing:
    """Mixed precision for scientific computing applications"""

    def __init__(self, precision_manager: MixedPrecisionManager):
        self.pm = precision_manager

    @jit
    def mixed_precision_matrix_operations(self,
                                        A: jax.Array,
                                        B: jax.Array,
                                        operation: str = "multiply") -> jax.Array:
        """Matrix operations with mixed precision"""

        # Cast to compute precision
        A_compute = self.pm.cast_to_precision(A, self.pm.compute_precision)
        B_compute = self.pm.cast_to_precision(B, self.pm.compute_precision)

        if operation == "multiply":
            result = jnp.dot(A_compute, B_compute)
        elif operation == "add":
            result = A_compute + B_compute
        elif operation == "solve":
            result = jnp.linalg.solve(A_compute, B_compute)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return self.pm.cast_to_precision(result, self.pm.default_precision)

    def mixed_precision_optimization_step(self,
                                        params: Dict[str, jax.Array],
                                        gradients: Dict[str, jax.Array],
                                        learning_rate: float) -> Dict[str, jax.Array]:
        """Optimization step with mixed precision"""

        updated_params = {}

        for key in params:
            param = params[key]
            grad = gradients[key]

            # Cast to compute precision for gradient update
            param_compute = self.pm.cast_to_precision(param, self.pm.compute_precision)
            grad_compute = self.pm.cast_to_precision(grad, self.pm.compute_precision)

            # Perform update
            update_compute = param_compute - learning_rate * grad_compute

            # Check for numerical issues
            if self.pm.detect_numerical_instability(update_compute, f"param_update_{key}"):
                # Fallback to higher precision
                param_fallback = self.pm.cast_to_precision(param, self.pm.fallback_precision)
                grad_fallback = self.pm.cast_to_precision(grad, self.pm.fallback_precision)
                update_fallback = param_fallback - learning_rate * grad_fallback
                updated_params[key] = self.pm.cast_to_precision(update_fallback, self.pm.default_precision)
            else:
                updated_params[key] = self.pm.cast_to_precision(update_compute, self.pm.default_precision)

        return updated_params

    def mixed_precision_gradient_computation(self,
                                           loss_fn: Callable,
                                           params: Dict[str, jax.Array],
                                           *args) -> Dict[str, jax.Array]:
        """Gradient computation with mixed precision"""

        # Create mixed precision loss function
        def mixed_precision_loss(mixed_params, *loss_args):
            # Cast parameters to compute precision
            compute_params = {}
            for key, param in mixed_params.items():
                compute_params[key] = self.pm.cast_to_precision(param, self.pm.compute_precision)

            # Compute loss
            loss = loss_fn(compute_params, *loss_args)

            # Cast loss back to default precision for gradient computation
            return self.pm.cast_to_precision(loss, self.pm.default_precision)

        # Compute gradients
        gradients = grad(mixed_precision_loss)(params, *args)

        return gradients

# NLSQ optimization with mixed precision
class MixedPrecisionNLSQ:
    """NLSQ optimization with mixed precision support"""

    def __init__(self, precision_manager: MixedPrecisionManager):
        self.pm = precision_manager

    def mixed_precision_jacobian(self,
                                residual_fn: Callable,
                                params: jax.Array,
                                *args) -> jax.Array:
        """Compute Jacobian with mixed precision"""

        # Adaptive precision for Jacobian computation
        def compute_jacobian_at_precision(precision: str):
            params_prec = self.pm.cast_to_precision(params, precision)

            def residual_wrapper(p):
                return residual_fn(p, *args)

            jacobian = jax.jacfwd(residual_wrapper)(params_prec)
            return self.pm.cast_to_precision(jacobian, self.pm.default_precision)

        # Try compute precision first
        try:
            jacobian = compute_jacobian_at_precision(self.pm.compute_precision)

            # Check condition number
            if jacobian.ndim == 2:
                cond_num = jnp.linalg.cond(jacobian)
                if cond_num > 1e12:  # Potentially ill-conditioned
                    print("High condition number detected, using higher precision")
                    jacobian = compute_jacobian_at_precision(self.pm.fallback_precision)

            return jacobian

        except Exception:
            # Fallback to higher precision
            return compute_jacobian_at_precision(self.pm.fallback_precision)

    def mixed_precision_gauss_newton_step(self,
                                        jacobian: jax.Array,
                                        residuals: jax.Array,
                                        damping: float = 0.0) -> jax.Array:
        """Gauss-Newton step with mixed precision"""

        # Cast to compute precision
        J_compute = self.pm.cast_to_precision(jacobian, self.pm.compute_precision)
        r_compute = self.pm.cast_to_precision(residuals, self.pm.compute_precision)

        try:
            # Compute J^T J
            JtJ = jnp.dot(J_compute.T, J_compute)

            # Add damping if specified
            if damping > 0:
                JtJ += damping * jnp.eye(JtJ.shape[0])

            # Compute J^T r
            Jtr = jnp.dot(J_compute.T, r_compute)

            # Solve normal equations
            delta = jnp.linalg.solve(JtJ, -Jtr)

            # Check for numerical issues
            if self.pm.detect_numerical_instability(delta, "gauss_newton_step"):
                raise ValueError("Numerical instability in Gauss-Newton step")

            return self.pm.cast_to_precision(delta, self.pm.default_precision)

        except (ValueError, jnp.linalg.LinAlgError):
            # Fallback to higher precision
            print("Falling back to higher precision for Gauss-Newton step")
            J_fallback = self.pm.cast_to_precision(jacobian, self.pm.fallback_precision)
            r_fallback = self.pm.cast_to_precision(residuals, self.pm.fallback_precision)

            JtJ = jnp.dot(J_fallback.T, J_fallback)
            if damping > 0:
                JtJ += damping * jnp.eye(JtJ.shape[0])
            Jtr = jnp.dot(J_fallback.T, r_fallback)
            delta = jnp.linalg.solve(JtJ, -Jtr)

            return self.pm.cast_to_precision(delta, self.pm.default_precision)

# Memory and performance monitoring
class MixedPrecisionProfiler:
    """Profile memory usage and performance of mixed precision computations"""

    def __init__(self):
        self.profiles = []

    def profile_computation(self,
                          computation_fn: Callable,
                          inputs: Dict[str, jax.Array],
                          precision_configs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Profile computation across different precision configurations"""

        results = {}

        for config in precision_configs:
            config_name = f"{config.get('compute', 'float32')}_{config.get('default', 'float32')}"

            # Setup precision manager
            pm = MixedPrecisionManager(
                default_precision=config.get('default', 'float32'),
                compute_precision=config.get('compute', 'float32'),
                fallback_precision=config.get('fallback', 'float64')
            )

            # Cast inputs to appropriate precision
            cast_inputs = {}
            for key, array in inputs.items():
                cast_inputs[key] = pm.cast_to_precision(array, pm.default_precision)

            # Time the computation
            import time

            # Warm up
            for _ in range(3):
                _ = computation_fn(**cast_inputs)

            # Actual timing
            start_time = time.time()
            result = computation_fn(**cast_inputs)
            # Ensure computation completes
            result.block_until_ready()
            end_time = time.time()

            # Memory estimation (approximate)
            def estimate_memory_usage(arrays):
                total_bytes = 0
                for arr in arrays:
                    total_bytes += arr.nbytes
                return total_bytes

            input_memory = estimate_memory_usage(cast_inputs.values())
            output_memory = result.nbytes if hasattr(result, 'nbytes') else 0

            results[config_name] = {
                'execution_time': end_time - start_time,
                'input_memory_mb': input_memory / (1024 * 1024),
                'output_memory_mb': output_memory / (1024 * 1024),
                'total_memory_mb': (input_memory + output_memory) / (1024 * 1024),
                'numerical_issues': len(pm.numerical_issues),
                'result_dtype': str(result.dtype),
                'config': config
            }

        return results

# Comprehensive examples and demonstrations
def demonstrate_mixed_precision():
    """Demonstrate mixed precision capabilities"""

    print("=== JAX Mixed Precision Demonstration ===")

    # Setup precision manager
    pm = MixedPrecisionManager(
        default_precision="float32",
        compute_precision="bfloat16",
        fallback_precision="float64"
    )

    # Test matrix operations
    print("\\n1. Matrix Operations Test:")
    key = random.PRNGKey(42)
    A = random.normal(key, (1000, 1000))
    B = random.normal(random.split(key)[1], (1000, 1000))

    sci_comp = MixedPrecisionScientificComputing(pm)

    # Matrix multiplication
    result_mm = sci_comp.mixed_precision_matrix_operations(A, B, "multiply")
    print(f"Matrix multiply result dtype: {result_mm.dtype}")

    # Test optimization step
    print("\\n2. Optimization Step Test:")
    params = {'w': random.normal(key, (100, 10)), 'b': jnp.zeros(10)}
    gradients = {'w': random.normal(key, (100, 10)), 'b': random.normal(key, (10,))}

    updated_params = sci_comp.mixed_precision_optimization_step(params, gradients, 0.01)
    print(f"Updated parameter dtypes: {[updated_params[k].dtype for k in updated_params]}")

    # Test NLSQ operations
    print("\\n3. NLSQ Operations Test:")
    nlsq_mp = MixedPrecisionNLSQ(pm)

    def simple_residual(params, x, y):
        return params[0] * x + params[1] - y

    x_data = jnp.linspace(0, 10, 100)
    y_data = 2.0 * x_data + 1.0 + 0.1 * random.normal(key, x_data.shape)
    params_nlsq = jnp.array([1.5, 0.5])

    jacobian = nlsq_mp.mixed_precision_jacobian(simple_residual, params_nlsq, x_data, y_data)
    print(f"Jacobian dtype: {jacobian.dtype}, shape: {jacobian.shape}")

    # Performance profiling
    print("\\n4. Performance Profiling:")
    profiler = MixedPrecisionProfiler()

    def test_computation(A, B):
        return jnp.dot(A, B) + jnp.sum(A * B)

    precision_configs = [
        {'default': 'float32', 'compute': 'float32'},
        {'default': 'float32', 'compute': 'bfloat16'},
        {'default': 'float64', 'compute': 'float64'}
    ]

    if ML_DTYPES_AVAILABLE:
        inputs = {'A': A[:100, :100], 'B': B[:100, :100]}  # Smaller for demo
        profile_results = profiler.profile_computation(test_computation, inputs, precision_configs)

        for config_name, stats in profile_results.items():
            print(f"  {config_name}: {stats['execution_time']:.4f}s, "
                  f"{stats['total_memory_mb']:.1f}MB")

    print(f"\\nNumerical issues detected: {len(pm.numerical_issues)}")

    return pm, sci_comp, nlsq_mp

# Advanced mixed precision strategies
def advanced_mixed_precision_strategies():
    """Demonstrate advanced mixed precision strategies"""

    print("\\n=== Advanced Mixed Precision Strategies ===")

    # Precision scheduling
    class PrecisionScheduler:
        def __init__(self, initial_precision: str = "bfloat16"):
            self.current_precision = initial_precision
            self.precision_history = []

        def update_precision(self, convergence_rate: float, iteration: int):
            # Increase precision if convergence is slow
            if convergence_rate < 1e-6 and iteration > 10:
                if self.current_precision == "bfloat16":
                    self.current_precision = "float32"
                elif self.current_precision == "float32":
                    self.current_precision = "float64"

            self.precision_history.append(self.current_precision)

    # Layered precision for neural networks
    def layered_precision_model(x: jax.Array, params: Dict[str, jax.Array]) -> jax.Array:
        """Neural network with different precision for different layers"""
        pm = MixedPrecisionManager()

        # Early layers: lower precision (speed)
        h1 = jnp.dot(x, pm.cast_to_precision(params['w1'], 'bfloat16'))
        h1 = jax.nn.relu(h1)

        # Middle layers: medium precision (balance)
        h2 = jnp.dot(h1, pm.cast_to_precision(params['w2'], 'float32'))
        h2 = jax.nn.relu(h2)

        # Final layer: high precision (accuracy)
        output = jnp.dot(h2, pm.cast_to_precision(params['w3'], 'float64'))

        return output

    # Demonstrate precision scheduling
    scheduler = PrecisionScheduler()
    for i in range(20):
        # Simulate convergence rate
        conv_rate = max(1e-3 / (i + 1), 1e-8)
        scheduler.update_precision(conv_rate, i)

    print(f"Precision evolution: {scheduler.precision_history[:10]}...")

if __name__ == "__main__":
    # Run demonstrations
    pm, sci_comp, nlsq_mp = demonstrate_mixed_precision()
    advanced_mixed_precision_strategies()

    print("\\n=== Mixed Precision Optimization Complete ===")
```

## Related Commands

- `/jax-nlsq-fit` - Apply mixed precision to NLSQ curve fitting
- `/jax-nlsq-large` - Memory efficiency with mixed precision for large datasets
- `/jax-cache-opt` - Optimize caching with precision-specific compilation
- `/jax-optax-optimizer` - Mixed precision in optimization algorithms