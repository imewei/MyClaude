---
description: Implement JIT caching strategies in JAX with cache management, eviction policies, and optimization loop acceleration
category: jax-performance
argument-hint: "[--cache-policy] [--eviction-strategy] [--memory-limit] [--profiling]"
allowed-tools: "*"
---

# /jax-cache-opt

Implement JIT caching strategies in JAX. Use jax.cache with eviction policies for repeated backend calls in optimization loops.

## Description

Advanced JIT compilation caching strategies for JAX to optimize repeated function calls in optimization loops, scientific computing workflows, and iterative algorithms. Includes intelligent cache management, memory-aware eviction policies, and performance monitoring for maximum computational efficiency.

## Usage

```
/jax-cache-opt [--cache-policy] [--eviction-strategy] [--memory-limit] [--profiling]
```

## What it does

1. Implement intelligent JIT caching with custom eviction policies
2. Optimize cache performance for optimization loops and iterative algorithms
3. Monitor cache hit rates and memory usage
4. Apply cache-aware function design patterns
5. Handle dynamic shapes and compilation strategies

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap
import functools
import weakref
import gc
import time
from typing import Dict, Any, Callable, Optional, Tuple, List
from collections import OrderedDict
import threading
import hashlib

# Configure JAX compilation cache
jax.config.update('jax_compilation_cache_dir', './jax_cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)

class AdvancedJITCacheManager:
    """Advanced JIT cache management with intelligent eviction policies"""

    def __init__(self,
                 max_cache_size_mb: float = 1024.0,
                 eviction_policy: str = "lru",
                 enable_persistent_cache: bool = True,
                 cache_statistics: bool = True):

        self.max_cache_size_mb = max_cache_size_mb
        self.eviction_policy = eviction_policy.lower()
        self.enable_persistent_cache = enable_persistent_cache
        self.cache_statistics = cache_statistics

        # Cache tracking
        self.function_caches = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_mb': 0.0,
            'compilation_times': []
        }

        # Thread safety
        self._lock = threading.Lock()

        # Setup persistent cache if enabled
        if self.enable_persistent_cache:
            self._setup_persistent_cache()

    def _setup_persistent_cache(self):
        """Setup persistent compilation cache"""
        # JAX automatically uses compilation cache if directory is set
        print(f"Persistent cache enabled at: {jax.config.jax_compilation_cache_dir}")

    def create_cached_function(self,
                             func: Callable,
                             static_argnums: Optional[Tuple[int, ...]] = None,
                             donate_argnums: Optional[Tuple[int, ...]] = None,
                             cache_key_fn: Optional[Callable] = None) -> Callable:
        """Create a cached JIT function with advanced cache management"""

        # Default cache key function based on argument shapes and dtypes
        if cache_key_fn is None:
            def default_cache_key(*args, **kwargs):
                key_parts = []
                for arg in args:
                    if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                        key_parts.append(f"{arg.shape}_{arg.dtype}")
                    else:
                        key_parts.append(str(type(arg)))
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}_{type(v)}")
                return tuple(key_parts)
            cache_key_fn = default_cache_key

        # Function-specific cache
        func_cache = {}
        func_name = func.__name__

        @functools.wraps(func)
        def cached_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_key_fn(*args, **kwargs)

            with self._lock:
                # Check if function is already compiled for this signature
                if cache_key in func_cache:
                    self.cache_stats['hits'] += 1
                    compiled_func = func_cache[cache_key]
                else:
                    self.cache_stats['misses'] += 1

                    # Compile function
                    start_time = time.time()
                    compiled_func = jit(
                        func,
                        static_argnums=static_argnums,
                        donate_argnums=donate_argnums
                    )
                    compilation_time = time.time() - start_time
                    self.cache_stats['compilation_times'].append(compilation_time)

                    # Store in cache
                    func_cache[cache_key] = compiled_func

                    # Apply eviction policy if needed
                    self._apply_eviction_policy(func_cache, func_name)

                # Store function cache reference
                self.function_caches[func_name] = func_cache

            return compiled_func(*args, **kwargs)

        return cached_wrapper

    def _apply_eviction_policy(self, func_cache: Dict, func_name: str):
        """Apply cache eviction policy when memory limits are exceeded"""

        # Estimate memory usage (approximate)
        estimated_size_mb = len(func_cache) * 10  # Rough estimate: 10MB per compiled function

        if estimated_size_mb > self.max_cache_size_mb:
            entries_to_remove = len(func_cache) - int(self.max_cache_size_mb / 10)

            if self.eviction_policy == "lru":
                # Remove least recently used (approximate with insertion order)
                if isinstance(func_cache, OrderedDict):
                    for _ in range(entries_to_remove):
                        func_cache.popitem(last=False)
                else:
                    # Convert to OrderedDict for LRU behavior
                    ordered_cache = OrderedDict(func_cache)
                    for _ in range(entries_to_remove):
                        ordered_cache.popitem(last=False)
                    func_cache.clear()
                    func_cache.update(ordered_cache)

            elif self.eviction_policy == "random":
                # Remove random entries
                keys_to_remove = list(func_cache.keys())[:entries_to_remove]
                for key in keys_to_remove:
                    del func_cache[key]

            elif self.eviction_policy == "size_based":
                # Remove entries for larger input sizes first
                def size_key(item):
                    key, _ = item
                    size_estimate = sum(
                        int(part.split('_')[0].replace('(', '').replace(')', '').replace(',', ''))
                        for part in str(key).split('_')
                        if part.replace('(', '').replace(')', '').replace(',', '').isdigit()
                    )
                    return size_estimate

                sorted_items = sorted(func_cache.items(), key=size_key, reverse=True)
                for key, _ in sorted_items[:entries_to_remove]:
                    del func_cache[key]

            self.cache_stats['evictions'] += entries_to_remove

    def optimization_loop_cache(self,
                              optimization_step_fn: Callable,
                              convergence_check_fn: Callable,
                              precompile_signatures: Optional[List[Tuple]] = None) -> Callable:
        """Optimize caching for optimization loops with predictable patterns"""

        # Pre-compile for known signatures
        if precompile_signatures:
            print(f"Pre-compiling for {len(precompile_signatures)} signatures...")
            for signature in precompile_signatures:
                try:
                    # Dummy compilation to warm up cache
                    _ = jit(optimization_step_fn)(*signature)
                except Exception as e:
                    print(f"Pre-compilation failed for signature {signature}: {e}")

        @self.create_cached_function
        def cached_optimization_step(*args, **kwargs):
            return optimization_step_fn(*args, **kwargs)

        @self.create_cached_function
        def cached_convergence_check(*args, **kwargs):
            return convergence_check_fn(*args, **kwargs)

        def optimized_loop(initial_state, max_iterations: int = 1000, tolerance: float = 1e-6):
            state = initial_state
            iteration = 0

            while iteration < max_iterations:
                # Cached optimization step
                new_state = cached_optimization_step(state)

                # Cached convergence check
                converged, convergence_metric = cached_convergence_check(state, new_state)

                if converged or convergence_metric < tolerance:
                    print(f"Converged after {iteration} iterations")
                    break

                state = new_state
                iteration += 1

            return state, iteration

        return optimized_loop

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            total_functions = len(self.function_caches)
            total_cached_variants = sum(len(cache) for cache in self.function_caches.values())

            hit_rate = (
                self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses'])
                if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0
            )

            avg_compilation_time = (
                sum(self.cache_stats['compilation_times']) / len(self.cache_stats['compilation_times'])
                if self.cache_stats['compilation_times'] else 0
            )

            return {
                'total_functions': total_functions,
                'total_cached_variants': total_cached_variants,
                'cache_hits': self.cache_stats['hits'],
                'cache_misses': self.cache_stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.cache_stats['evictions'],
                'avg_compilation_time': avg_compilation_time,
                'total_compilation_time': sum(self.cache_stats['compilation_times']),
                'estimated_memory_mb': total_cached_variants * 10  # Rough estimate
            }

    def clear_cache(self, function_name: Optional[str] = None):
        """Clear cache for specific function or all functions"""
        with self._lock:
            if function_name:
                if function_name in self.function_caches:
                    self.function_caches[function_name].clear()
                    print(f"Cleared cache for function: {function_name}")
            else:
                for cache in self.function_caches.values():
                    cache.clear()
                self.function_caches.clear()
                print("Cleared all function caches")

# Specialized caching for different use cases
class SpecializedCachingStrategies:
    """Specialized caching strategies for different computational patterns"""

    def __init__(self, cache_manager: AdvancedJITCacheManager):
        self.cache_manager = cache_manager

    def iterative_solver_cache(self,
                             solver_step_fn: Callable,
                             matrix_shape: Tuple[int, int],
                             max_iterations: int = 1000) -> Callable:
        """Optimized caching for iterative solvers"""

        # Pre-compile for the specific matrix shape
        dummy_matrix = jnp.zeros(matrix_shape)
        dummy_vector = jnp.zeros(matrix_shape[1])

        @self.cache_manager.create_cached_function
        @jit
        def cached_solver_step(A, x, b, iteration):
            return solver_step_fn(A, x, b)

        # Warm up compilation
        _ = cached_solver_step(dummy_matrix, dummy_vector, dummy_vector, 0)

        def optimized_iterative_solver(A, b, x0=None, tolerance=1e-6):
            if x0 is None:
                x = jnp.zeros_like(b)
            else:
                x = x0

            for iteration in range(max_iterations):
                x_new = cached_solver_step(A, x, b, iteration)

                # Check convergence
                residual = jnp.linalg.norm(x_new - x)
                if residual < tolerance:
                    print(f"Solver converged after {iteration} iterations")
                    break

                x = x_new

            return x, iteration

        return optimized_iterative_solver

    def gradient_based_optimization_cache(self,
                                        loss_fn: Callable,
                                        grad_fn: Optional[Callable] = None) -> Tuple[Callable, Callable]:
        """Optimized caching for gradient-based optimization"""

        if grad_fn is None:
            grad_fn = grad(loss_fn)

        @self.cache_manager.create_cached_function
        @jit
        def cached_loss_and_grad(params, *args):
            loss_val = loss_fn(params, *args)
            grad_val = grad_fn(params, *args)
            return loss_val, grad_val

        @self.cache_manager.create_cached_function
        @jit
        def cached_parameter_update(params, gradients, learning_rate):
            return params - learning_rate * gradients

        return cached_loss_and_grad, cached_parameter_update

    def monte_carlo_simulation_cache(self,
                                   simulation_step_fn: Callable,
                                   n_samples: int,
                                   sample_shape: Tuple[int, ...]) -> Callable:
        """Optimized caching for Monte Carlo simulations"""

        @self.cache_manager.create_cached_function
        @jit
        def cached_simulation_step(samples, step_params):
            return simulation_step_fn(samples, step_params)

        # Vectorized simulation
        @self.cache_manager.create_cached_function
        @jit
        def vectorized_simulation(all_samples, step_params):
            return vmap(lambda samples: simulation_step_fn(samples, step_params))(all_samples)

        def optimized_monte_carlo(step_params_list, key):
            # Pre-generate all samples
            all_samples = random.normal(key, (n_samples,) + sample_shape)

            results = []
            for step_params in step_params_list:
                # Use vectorized cached simulation
                step_results = vectorized_simulation(all_samples, step_params)
                results.append(step_results)

            return jnp.array(results)

        return optimized_monte_carlo

# Dynamic shape handling with caching
class DynamicShapeCacheManager:
    """Advanced caching for functions with dynamic shapes"""

    def __init__(self, base_cache_manager: AdvancedJITCacheManager):
        self.base_cache_manager = base_cache_manager
        self.shape_bucketing = {}

    def create_bucketed_cache_function(self,
                                     func: Callable,
                                     shape_buckets: List[Tuple[int, ...]],
                                     max_bucket_deviation: float = 0.1) -> Callable:
        """Create cached function with shape bucketing for dynamic inputs"""

        def find_best_bucket(input_shape):
            best_bucket = None
            min_size_diff = float('inf')

            for bucket_shape in shape_buckets:
                if len(input_shape) == len(bucket_shape):
                    # Calculate size difference
                    input_size = jnp.prod(jnp.array(input_shape))
                    bucket_size = jnp.prod(jnp.array(bucket_shape))
                    size_diff = abs(input_size - bucket_size) / bucket_size

                    if size_diff <= max_bucket_deviation and size_diff < min_size_diff:
                        min_size_diff = size_diff
                        best_bucket = bucket_shape

            return best_bucket

        # Cache for each bucket
        bucket_caches = {}

        @functools.wraps(func)
        def bucketed_cached_function(*args, **kwargs):
            # Get input shape (assuming first argument is the main array)
            input_shape = args[0].shape if hasattr(args[0], 'shape') else None

            if input_shape:
                bucket_shape = find_best_bucket(input_shape)

                if bucket_shape:
                    # Use existing bucket cache
                    if bucket_shape not in bucket_caches:
                        bucket_caches[bucket_shape] = self.base_cache_manager.create_cached_function(func)

                    cached_func = bucket_caches[bucket_shape]

                    # Pad input to bucket size if needed
                    if input_shape != bucket_shape:
                        padded_args = list(args)
                        pad_widths = [(0, bucket_dim - input_dim)
                                    for input_dim, bucket_dim in zip(input_shape, bucket_shape)]
                        padded_args[0] = jnp.pad(args[0], pad_widths)
                        result = cached_func(*padded_args, **kwargs)

                        # Trim result back to original size
                        if hasattr(result, 'shape'):
                            slices = tuple(slice(0, dim) for dim in input_shape)
                            result = result[slices]
                        return result
                    else:
                        return cached_func(*args, **kwargs)

            # Fallback: no suitable bucket found
            return jit(func)(*args, **kwargs)

        return bucketed_cached_function

# Performance monitoring and optimization
class CachePerformanceMonitor:
    """Monitor and optimize cache performance"""

    def __init__(self, cache_manager: AdvancedJITCacheManager):
        self.cache_manager = cache_manager
        self.performance_history = []

    def benchmark_cache_performance(self,
                                  test_functions: List[Callable],
                                  test_inputs: List[List],
                                  n_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark cache performance across different scenarios"""

        results = {}

        for i, (func, inputs_list) in enumerate(zip(test_functions, test_inputs)):
            func_name = f"test_function_{i}"

            # Test without cache
            start_time = time.time()
            for _ in range(n_iterations):
                for inputs in inputs_list:
                    jit(func)(*inputs)
            no_cache_time = time.time() - start_time

            # Clear JAX cache
            jax.clear_caches()

            # Test with cache
            cached_func = self.cache_manager.create_cached_function(func)
            start_time = time.time()
            for _ in range(n_iterations):
                for inputs in inputs_list:
                    cached_func(*inputs)
            cached_time = time.time() - start_time

            results[func_name] = {
                'no_cache_time': no_cache_time,
                'cached_time': cached_time,
                'speedup': no_cache_time / cached_time if cached_time > 0 else float('inf'),
                'cache_overhead': (cached_time - no_cache_time) / no_cache_time if no_cache_time > 0 else 0
            }

        return results

    def suggest_optimizations(self) -> List[str]:
        """Suggest cache optimization strategies based on performance history"""

        suggestions = []
        stats = self.cache_manager.get_cache_statistics()

        # Analyze hit rate
        if stats['hit_rate'] < 0.5:
            suggestions.append("Low cache hit rate - consider pre-compiling common function signatures")

        # Analyze compilation time
        if stats['avg_compilation_time'] > 1.0:
            suggestions.append("High compilation time - consider using simpler functions or reducing JIT scope")

        # Analyze memory usage
        if stats['estimated_memory_mb'] > 2048:  # 2GB
            suggestions.append("High memory usage - consider more aggressive eviction policies")

        # Analyze eviction frequency
        if stats['evictions'] > stats['total_cached_variants'] * 0.5:
            suggestions.append("Frequent evictions - consider increasing cache size limit")

        return suggestions

# Comprehensive demonstration
def demonstrate_advanced_jit_caching():
    """Demonstrate advanced JIT caching strategies"""

    print("=== Advanced JAX JIT Caching Demonstration ===")

    # Initialize cache manager
    cache_manager = AdvancedJITCacheManager(
        max_cache_size_mb=512.0,
        eviction_policy="lru",
        enable_persistent_cache=True
    )

    # Test basic caching
    print("\\n1. Basic Function Caching:")

    @cache_manager.create_cached_function
    def expensive_computation(x, n_iterations=100):
        result = x
        for _ in range(n_iterations):
            result = jnp.sin(result) + jnp.cos(result)
        return result

    # Time first call (compilation + execution)
    key = random.PRNGKey(42)
    test_input = random.normal(key, (1000,))

    start_time = time.time()
    result1 = expensive_computation(test_input)
    first_call_time = time.time() - start_time

    # Time second call (cached execution)
    start_time = time.time()
    result2 = expensive_computation(test_input)
    second_call_time = time.time() - start_time

    print(f"First call time: {first_call_time:.4f}s")
    print(f"Second call time: {second_call_time:.4f}s")
    print(f"Speedup: {first_call_time / second_call_time:.2f}x")

    # Test optimization loop caching
    print("\\n2. Optimization Loop Caching:")

    def optimization_step(params, data):
        return params - 0.01 * grad(lambda p: jnp.sum((p - data) ** 2))(params)

    def convergence_check(old_params, new_params):
        diff = jnp.linalg.norm(new_params - old_params)
        return diff < 1e-6, diff

    # Create optimized loop
    optimized_loop = cache_manager.optimization_loop_cache(
        optimization_step,
        convergence_check,
        precompile_signatures=[(jnp.zeros(10), jnp.ones(10))]
    )

    # Run optimization
    initial_params = random.normal(key, (10,))
    target_data = jnp.ones(10)

    start_time = time.time()
    final_params, iterations = optimized_loop(initial_params)
    optimization_time = time.time() - start_time

    print(f"Optimization completed in {iterations} iterations ({optimization_time:.4f}s)")

    # Test specialized caching strategies
    print("\\n3. Specialized Caching Strategies:")

    specialized = SpecializedCachingStrategies(cache_manager)

    # Gradient-based optimization caching
    def simple_loss(params, x, y):
        return jnp.mean((jnp.dot(x, params) - y) ** 2)

    cached_loss_and_grad, cached_param_update = specialized.gradient_based_optimization_cache(simple_loss)

    # Test the cached optimization
    X = random.normal(key, (100, 5))
    y = random.normal(key, (100,))
    params = random.normal(key, (5,))

    for i in range(10):
        loss_val, grad_val = cached_loss_and_grad(params, X, y)
        params = cached_param_update(params, grad_val, 0.01)

    print(f"Final loss after 10 iterations: {loss_val:.6f}")

    # Show cache statistics
    print("\\n4. Cache Statistics:")
    stats = cache_manager.get_cache_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Performance monitoring
    print("\\n5. Performance Monitoring:")
    monitor = CachePerformanceMonitor(cache_manager)
    suggestions = monitor.suggest_optimizations()

    if suggestions:
        print("Optimization suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")
    else:
        print("Cache performance is optimal!")

    return cache_manager, stats

if __name__ == "__main__":
    cache_manager, final_stats = demonstrate_advanced_jit_caching()
    print("\\n=== JIT Caching Optimization Complete ===")
```

## Related Commands

- `/jax-jit` - Basic JIT compilation strategies
- `/jax-nlsq-fit` - Apply caching to NLSQ optimization loops
- `/jax-mixed-prec` - Cache management with different precision levels
- `/python-debug-prof` - Profile JIT compilation and caching performance