---
description: Optimize NLSQ for massive datasets using imewei/NLSQ with memory estimation, chunking, and algorithm selection
category: jax-optimization
argument-hint: "[--dataset-size] [--memory-limit] [--algorithm] [--chunking-strategy]"
allowed-tools: "*"
---

# /jax-nlsq-large

Optimize NLSQ for massive datasets (>10M points). Estimate memory with estimate_memory_requirements, apply chunking, and select algorithms (TRR or LM) for stability.

## Description

Advanced optimization strategies for handling massive datasets in nonlinear least-squares fitting using the NLSQ library. Includes intelligent memory estimation, adaptive chunking algorithms, robust algorithm selection, and distributed computing strategies for datasets exceeding typical memory constraints.

## Usage

```
/jax-nlsq-large [--dataset-size] [--memory-limit] [--algorithm] [--chunking-strategy]
```

## What it does

1. Estimate memory requirements for massive NLSQ problems
2. Implement adaptive chunking strategies for datasets >10M points
3. Select optimal algorithms (TRR vs LM) based on problem characteristics
4. Apply distributed computing and streaming optimization
5. Monitor convergence and stability across chunks

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap, pmap
import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional, List
import psutil
import gc
import warnings

# NLSQ library for large-scale optimization
try:
    from nlsq import CurveFit, LevenbergMarquardt, TrustRegionReflective
    from nlsq.utils import estimate_memory_requirements
    NLSQ_AVAILABLE = True
except ImportError:
    print("NLSQ not available. Install with: pip install nlsq")
    NLSQ_AVAILABLE = False

# Configure JAX for large-scale computing
jax.config.update('jax_enable_x64', True)  # Higher precision for stability
jax.config.update('jax_platform_name', 'gpu')  # GPU acceleration

class MassiveNLSQOptimizer:
    """Large-scale NLSQ optimizer for datasets >10M points"""

    def __init__(self,
                 memory_limit_gb: float = 8.0,
                 algorithm: str = "auto",
                 chunking_strategy: str = "adaptive",
                 distributed: bool = False):
        self.memory_limit_gb = memory_limit_gb
        self.algorithm = algorithm.lower()
        self.chunking_strategy = chunking_strategy.lower()
        self.distributed = distributed
        self.chunk_stats = []

    def estimate_memory_requirements(self,
                                   n_points: int,
                                   n_params: int,
                                   algorithm: str = "lm") -> Dict[str, Any]:
        """Comprehensive memory estimation for large-scale problems"""

        # Base memory requirements (in bytes)
        data_arrays = n_points * 8 * 3  # x, y, weights (float64)
        residuals = n_points * 8  # residual vector
        jacobian = n_points * n_params * 8  # Jacobian matrix

        # Algorithm-specific memory
        if algorithm.lower() == "lm":
            # Levenberg-Marquardt: J^T*J, J^T*r, parameter updates
            jtj_matrix = n_params * n_params * 8
            additional = n_params * 8 * 3  # gradients, parameter updates, etc.
            algorithm_memory = jtj_matrix + additional
        elif algorithm.lower() == "trr":
            # Trust Region Reflective: additional arrays for bounds, reflections
            trr_arrays = n_points * 8 * 2 + n_params * 8 * 4
            algorithm_memory = trr_arrays
        else:
            algorithm_memory = jacobian * 0.5  # Conservative estimate

        # Memory overhead (JAX compilation, GPU transfers, etc.)
        overhead_factor = 1.5 if jax.devices('gpu') else 1.2
        total_memory = (data_arrays + residuals + jacobian + algorithm_memory) * overhead_factor

        # Convert to different units
        memory_mb = total_memory / (1024 * 1024)
        memory_gb = memory_mb / 1024

        # Chunking recommendations
        if memory_gb > self.memory_limit_gb:
            target_chunk_memory = self.memory_limit_gb * 0.8  # 80% of limit
            scale_factor = target_chunk_memory / memory_gb
            recommended_chunk_size = int(n_points * scale_factor)
            recommended_chunks = (n_points + recommended_chunk_size - 1) // recommended_chunk_size
        else:
            recommended_chunk_size = n_points
            recommended_chunks = 1

        return {
            'n_points': n_points,
            'n_params': n_params,
            'data_memory_mb': data_arrays / (1024 * 1024),
            'jacobian_memory_mb': jacobian / (1024 * 1024),
            'algorithm_memory_mb': algorithm_memory / (1024 * 1024),
            'total_memory_mb': memory_mb,
            'total_memory_gb': memory_gb,
            'exceeds_limit': memory_gb > self.memory_limit_gb,
            'recommended_chunk_size': recommended_chunk_size,
            'recommended_chunks': recommended_chunks,
            'chunking_required': recommended_chunks > 1
        }

    def select_optimal_algorithm(self,
                               n_points: int,
                               n_params: int,
                               has_bounds: bool = False,
                               condition_estimate: Optional[float] = None) -> str:
        """Select optimal algorithm based on problem characteristics"""

        if self.algorithm != "auto":
            return self.algorithm

        # Algorithm selection heuristics
        param_ratio = n_points / n_params

        if has_bounds:
            # Trust Region Reflective handles bounds naturally
            return "trr"
        elif param_ratio < 10:
            # Overdetermined problems: prefer robust methods
            return "trr"
        elif condition_estimate and condition_estimate > 1e12:
            # Ill-conditioned problems: prefer stable methods
            return "trr"
        elif n_points > 1e7 and n_params < 20:
            # Large data, few parameters: LM often faster
            return "lm"
        else:
            # Default: Levenberg-Marquardt
            return "lm"

    def adaptive_chunking_strategy(self,
                                 n_points: int,
                                 memory_estimate: Dict[str, Any],
                                 convergence_history: List[float] = None) -> Dict[str, Any]:
        """Determine optimal chunking strategy based on problem characteristics"""

        if self.chunking_strategy == "fixed":
            # Fixed chunk size based on memory
            chunk_size = memory_estimate['recommended_chunk_size']
            return {
                'strategy': 'fixed',
                'chunk_size': chunk_size,
                'n_chunks': (n_points + chunk_size - 1) // chunk_size,
                'overlap': 0
            }

        elif self.chunking_strategy == "overlap":
            # Overlapping chunks for better convergence
            base_chunk_size = memory_estimate['recommended_chunk_size']
            overlap_ratio = 0.1  # 10% overlap
            overlap_size = int(base_chunk_size * overlap_ratio)

            return {
                'strategy': 'overlap',
                'chunk_size': base_chunk_size,
                'overlap_size': overlap_size,
                'effective_chunk_size': base_chunk_size - overlap_size,
                'n_chunks': (n_points + base_chunk_size - overlap_size - 1) // (base_chunk_size - overlap_size)
            }

        elif self.chunking_strategy == "adaptive":
            # Adaptive chunk sizing based on convergence
            if convergence_history and len(convergence_history) > 3:
                # Analyze convergence rate
                recent_improvement = abs(convergence_history[-1] - convergence_history[-3])
                if recent_improvement < 1e-8:
                    # Slow convergence: use smaller chunks for better accuracy
                    chunk_size = int(memory_estimate['recommended_chunk_size'] * 0.7)
                else:
                    # Good convergence: use larger chunks for speed
                    chunk_size = int(memory_estimate['recommended_chunk_size'] * 1.3)
            else:
                chunk_size = memory_estimate['recommended_chunk_size']

            return {
                'strategy': 'adaptive',
                'chunk_size': min(chunk_size, n_points),
                'n_chunks': (n_points + chunk_size - 1) // chunk_size,
                'adaptive': True
            }

        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")

    def streaming_data_loader(self,
                            data_source: Any,
                            chunk_size: int,
                            preprocess_fn: Optional[Callable] = None) -> Any:
        """Memory-efficient streaming data loader for massive datasets"""

        @jit
        def default_preprocess(x, y):
            """Default preprocessing: ensure arrays are JAX arrays"""
            return jnp.asarray(x), jnp.asarray(y)

        if preprocess_fn is None:
            preprocess_fn = default_preprocess

        class StreamingLoader:
            def __init__(self, source, chunk_size, preprocess):
                self.source = source
                self.chunk_size = chunk_size
                self.preprocess = preprocess
                self.current_index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if hasattr(self.source, '__len__'):
                    total_size = len(self.source)
                    if self.current_index >= total_size:
                        raise StopIteration

                    # Load chunk from source
                    end_idx = min(self.current_index + self.chunk_size, total_size)

                    if hasattr(self.source, '__getitem__'):
                        # Array-like interface
                        x_chunk = self.source[0][self.current_index:end_idx]
                        y_chunk = self.source[1][self.current_index:end_idx]
                    else:
                        # Custom data source
                        x_chunk, y_chunk = self.source.get_chunk(self.current_index, end_idx)

                    # Preprocess
                    x_processed, y_processed = self.preprocess(x_chunk, y_chunk)

                    chunk_info = {
                        'x': x_processed,
                        'y': y_processed,
                        'start_idx': self.current_index,
                        'end_idx': end_idx,
                        'chunk_id': self.current_index // self.chunk_size
                    }

                    self.current_index = end_idx
                    return chunk_info
                else:
                    raise ValueError("Data source must be indexable or have get_chunk method")

        return StreamingLoader(data_source, chunk_size, preprocess_fn)

    def robust_optimization_with_nlsq(self,
                                    data_loader: Any,
                                    model_func: Callable,
                                    initial_params: jax.Array,
                                    algorithm: str = "lm") -> Dict[str, Any]:
        """Robust optimization using NLSQ library with error handling"""

        if not NLSQ_AVAILABLE:
            raise ImportError("NLSQ library required for large-scale optimization")

        # Setup NLSQ algorithm
        if algorithm == "lm":
            optimizer = LevenbergMarquardt(
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                max_nfev=1000
            )
        elif algorithm == "trr":
            optimizer = TrustRegionReflective(
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
                max_nfev=1000
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        curve_fit = CurveFit(optimizer)

        # Optimization across chunks
        params = initial_params
        chunk_results = []
        total_cost = 0.0

        for chunk_info in data_loader:
            x_chunk = chunk_info['x']
            y_chunk = chunk_info['y']
            chunk_id = chunk_info['chunk_id']

            print(f"Optimizing chunk {chunk_id} ({len(x_chunk)} points)...")

            try:
                # Convert JAX arrays to numpy for NLSQ
                x_np = np.array(x_chunk)
                y_np = np.array(y_chunk)
                params_np = np.array(params)

                # Define model wrapper for NLSQ
                def model_wrapper(x, *p):
                    return np.array(model_func(jnp.array(x), jnp.array(p)))

                # Fit chunk
                result = curve_fit.fit(x_np, y_np, model_wrapper, p0=params_np)

                # Update parameters for next chunk
                params = jnp.array(result.x)
                chunk_cost = result.cost

                chunk_results.append({
                    'chunk_id': chunk_id,
                    'success': result.success,
                    'cost': chunk_cost,
                    'params': params,
                    'n_iterations': getattr(result, 'nfev', 0),
                    'message': getattr(result, 'message', '')
                })

                total_cost += chunk_cost

                # Memory cleanup
                del x_np, y_np, params_np
                gc.collect()

            except Exception as e:
                print(f"Warning: Chunk {chunk_id} failed: {e}")
                chunk_results.append({
                    'chunk_id': chunk_id,
                    'success': False,
                    'error': str(e),
                    'params': params
                })

        return {
            'final_params': params,
            'total_cost': total_cost,
            'chunk_results': chunk_results,
            'successful_chunks': sum(1 for r in chunk_results if r.get('success', False)),
            'total_chunks': len(chunk_results),
            'algorithm': algorithm
        }

    def distributed_optimization(self,
                               data_chunks: List[Tuple[jax.Array, jax.Array]],
                               model_func: Callable,
                               initial_params: jax.Array) -> Dict[str, Any]:
        """Distributed optimization across multiple devices/processes"""

        if not self.distributed or len(jax.devices()) == 1:
            # Fall back to sequential processing
            return self._sequential_chunk_processing(data_chunks, model_func, initial_params)

        devices = jax.devices()
        n_devices = len(devices)

        print(f"Using distributed optimization across {n_devices} devices")

        # Distribute chunks across devices
        device_chunks = [[] for _ in range(n_devices)]
        for i, chunk in enumerate(data_chunks):
            device_chunks[i % n_devices].append(chunk)

        @pmap
        def parallel_chunk_optimization(device_data):
            """Optimize chunks in parallel on each device"""
            params = initial_params
            device_results = []

            for x_chunk, y_chunk in device_data:
                # Run optimization on this device
                result = self._single_chunk_nlsq_optimization(
                    x_chunk, y_chunk, model_func, params
                )
                params = result['params']
                device_results.append(result)

            return params, device_results

        # Execute parallel optimization
        device_data_arrays = [jnp.array(chunks) for chunks in device_chunks]
        final_params_per_device, results_per_device = parallel_chunk_optimization(device_data_arrays)

        # Aggregate results (simple averaging for demonstration)
        final_params = jnp.mean(final_params_per_device, axis=0)

        return {
            'final_params': final_params,
            'device_results': results_per_device,
            'n_devices': n_devices,
            'distributed': True
        }

    def _single_chunk_nlsq_optimization(self,
                                      x_data: jax.Array,
                                      y_data: jax.Array,
                                      model_func: Callable,
                                      initial_params: jax.Array) -> Dict[str, Any]:
        """Single chunk optimization using NLSQ"""

        if not NLSQ_AVAILABLE:
            # Fallback to JAX-only implementation
            return self._jax_fallback_optimization(x_data, y_data, model_func, initial_params)

        # Convert to numpy for NLSQ
        x_np = np.array(x_data)
        y_np = np.array(y_data)
        params_np = np.array(initial_params)

        # Model wrapper
        def model_wrapper(x, *p):
            return np.array(model_func(jnp.array(x), jnp.array(p)))

        # Choose algorithm
        algorithm = self.select_optimal_algorithm(len(x_data), len(initial_params))

        if algorithm == "lm":
            optimizer = LevenbergMarquardt()
        else:
            optimizer = TrustRegionReflective()

        curve_fit = CurveFit(optimizer)
        result = curve_fit.fit(x_np, y_np, model_wrapper, p0=params_np)

        return {
            'params': jnp.array(result.x),
            'cost': result.cost,
            'success': result.success,
            'algorithm': algorithm
        }

    def monitor_convergence_stability(self, chunk_results: List[Dict]) -> Dict[str, Any]:
        """Monitor convergence stability across chunks"""

        if not chunk_results:
            return {'stable': False, 'reason': 'No results'}

        # Extract parameter trajectories
        param_history = []
        cost_history = []

        for result in chunk_results:
            if result.get('success', False):
                param_history.append(result['params'])
                cost_history.append(result['cost'])

        if len(param_history) < 2:
            return {'stable': False, 'reason': 'Insufficient successful chunks'}

        param_history = jnp.array(param_history)
        cost_history = jnp.array(cost_history)

        # Check parameter stability
        param_changes = jnp.diff(param_history, axis=0)
        param_stability = jnp.mean(jnp.std(param_changes, axis=0))

        # Check cost stability
        cost_changes = jnp.diff(cost_history)
        cost_trend = jnp.mean(cost_changes)

        # Stability criteria
        param_stable = param_stability < 0.1  # Parameter changes < 10%
        cost_decreasing = cost_trend <= 0  # Cost should not increase

        return {
            'stable': param_stable and cost_decreasing,
            'param_stability': float(param_stability),
            'cost_trend': float(cost_trend),
            'successful_chunks': len(param_history),
            'total_chunks': len(chunk_results),
            'final_cost': float(cost_history[-1]) if len(cost_history) > 0 else None
        }

# Comprehensive massive dataset optimization workflow
def optimize_massive_dataset(x_data: jax.Array,
                           y_data: jax.Array,
                           model_func: Callable,
                           initial_params: jax.Array,
                           memory_limit_gb: float = 8.0) -> Dict[str, Any]:
    """Complete workflow for massive dataset optimization"""

    n_points = len(x_data)
    n_params = len(initial_params)

    print(f"=== Massive Dataset Optimization ===")
    print(f"Dataset size: {n_points:,} points")
    print(f"Parameters: {n_params}")

    # Initialize optimizer
    optimizer = MassiveNLSQOptimizer(
        memory_limit_gb=memory_limit_gb,
        algorithm="auto",
        chunking_strategy="adaptive"
    )

    # Estimate memory requirements
    memory_est = optimizer.estimate_memory_requirements(n_points, n_params)
    print(f"Memory estimate: {memory_est['total_memory_gb']:.2f} GB")
    print(f"Chunking required: {memory_est['chunking_required']}")

    if memory_est['chunking_required']:
        # Setup chunking strategy
        chunking_config = optimizer.adaptive_chunking_strategy(n_points, memory_est)
        print(f"Chunking strategy: {chunking_config['strategy']}")
        print(f"Chunk size: {chunking_config['chunk_size']:,}")
        print(f"Number of chunks: {chunking_config['n_chunks']}")

        # Create streaming data loader
        data_source = (x_data, y_data)
        data_loader = optimizer.streaming_data_loader(
            data_source, chunking_config['chunk_size']
        )

        # Optimize with NLSQ
        algorithm = optimizer.select_optimal_algorithm(n_points, n_params)
        print(f"Selected algorithm: {algorithm}")

        result = optimizer.robust_optimization_with_nlsq(
            data_loader, model_func, initial_params, algorithm
        )

        # Monitor stability
        stability = optimizer.monitor_convergence_stability(result['chunk_results'])
        result['stability'] = stability

    else:
        # Single chunk optimization
        print("Using single chunk optimization")
        result = optimizer._single_chunk_nlsq_optimization(
            x_data, y_data, model_func, initial_params
        )

    return result

# Demonstration with synthetic massive dataset
def demonstrate_massive_optimization():
    """Demonstrate optimization on a massive synthetic dataset"""

    print("Generating massive synthetic dataset...")

    # Generate large dataset (adjust size based on available memory)
    n_points = 1_000_000  # 1M points
    key = random.PRNGKey(42)

    x_massive = jnp.linspace(0, 100, n_points)
    true_params = jnp.array([5.0, 0.1, 1.0])  # a, b, c for exponential model

    @jit
    def exponential_model(x: jax.Array, params: jax.Array) -> jax.Array:
        a, b, c = params
        return a * jnp.exp(-b * x) + c

    # Generate noisy observations
    y_true = exponential_model(x_massive, true_params)
    noise = 0.1 * random.normal(key, y_true.shape)
    y_massive = y_true + noise

    print(f"Generated dataset: {len(x_massive):,} points")

    # Optimize
    initial_guess = jnp.array([4.0, 0.08, 0.8])
    result = optimize_massive_dataset(
        x_massive, y_massive, exponential_model, initial_guess, memory_limit_gb=4.0
    )

    print(f"\\nOptimization Results:")
    print(f"True parameters: {true_params}")
    print(f"Fitted parameters: {result['final_params']}")
    print(f"Parameter errors: {jnp.abs(result['final_params'] - true_params)}")

    if 'stability' in result:
        print(f"Convergence stable: {result['stability']['stable']}")

    return result

if __name__ == "__main__":
    # Run massive dataset demonstration
    demo_result = demonstrate_massive_optimization()
    print("\\nMassive dataset optimization completed!")
```

## Related Commands

- `/jax-nlsq-fit` - Basic NLSQ curve fitting setup
- `/jax-mixed-prec` - Mixed precision for memory efficiency
- `/jax-cache-opt` - JIT caching for repeated optimizations
- `/jax-sparse-jac` - Sparse Jacobians for large problems