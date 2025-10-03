"""
Optimization Helpers - Common optimization patterns for scientific computing.

This module provides utilities for optimizing numerical code:
- Vectorization helpers
- Caching decorators
- Memory-efficient operations
- Pre-computation utilities
"""

import functools
import numpy as np
from typing import Callable, Any, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
import pickle


def memoize(maxsize: Optional[int] = 128):
    """
    Memoization decorator for expensive function calls.

    Args:
        maxsize: Maximum cache size (None for unlimited)

    Returns:
        Decorated function with caching

    Example:
        @memoize(maxsize=100)
        def expensive_computation(n):
            return sum(range(n**2))
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create hashable key
            key = (args, tuple(sorted(kwargs.items())))

            # Try to hash numpy arrays separately
            try:
                key_hash = hash(key)
            except TypeError:
                # Handle unhashable types (like numpy arrays)
                key_str = str(key)
                key_hash = hashlib.md5(key_str.encode()).hexdigest()

            if key_hash in cache:
                return cache[key_hash]

            result = func(*args, **kwargs)

            # Store in cache
            cache[key_hash] = result
            cache_order.append(key_hash)

            # Enforce maxsize
            if maxsize is not None and len(cache) > maxsize:
                oldest = cache_order.pop(0)
                del cache[oldest]

            return result

        wrapper.cache = cache
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper

    return decorator


def vectorize_operation(func: Callable) -> Callable:
    """
    Hint decorator to remind that operation should be vectorized.

    This doesn't actually vectorize - it's a documentation/linting aid.
    Use numpy's built-in vectorization instead of loops.

    Example:
        @vectorize_operation
        def compute_distances(points):
            # Use numpy broadcasting instead of loops
            return np.sqrt(np.sum(points**2, axis=1))
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_vectorized = True
    return wrapper


class LaplacianCache:
    """
    Cache for Laplacian operators to avoid recomputation.

    Stores sparse Laplacian matrices for repeated PDE solves
    on the same grid.
    """

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get_key(self, nx: int, ny: int, dx: float, dy: float, nz: Optional[int] = None, dz: Optional[float] = None) -> str:
        """Generate cache key from grid parameters."""
        if nz is None:
            return f"2d_{nx}_{ny}_{dx:.10f}_{dy:.10f}"
        else:
            return f"3d_{nx}_{ny}_{nz}_{dx:.10f}_{dy:.10f}_{dz:.10f}"

    def get(self, nx: int, ny: int, dx: float, dy: float, nz: Optional[int] = None, dz: Optional[float] = None) -> Optional[Any]:
        """Get cached Laplacian operator."""
        key = self.get_key(nx, ny, dx, dy, nz, dz)
        return self.cache.get(key)

    def set(self, operator: Any, nx: int, ny: int, dx: float, dy: float, nz: Optional[int] = None, dz: Optional[float] = None):
        """Store Laplacian operator in cache."""
        key = self.get_key(nx, ny, dx, dy, nz, dz)
        self.cache[key] = operator

    def clear(self):
        """Clear all cached operators."""
        self.cache.clear()


# Global Laplacian cache
_laplacian_cache = LaplacianCache()


def get_laplacian_cache() -> LaplacianCache:
    """Get the global Laplacian cache instance."""
    return _laplacian_cache


def inplace_operation(warn: bool = True):
    """
    Decorator to mark functions that should use in-place operations.

    Args:
        warn: Whether to print warning if not using in-place ops

    Example:
        @inplace_operation()
        def update_array(arr):
            arr += 1  # In-place
            return arr
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._inplace = True
        return wrapper

    return decorator


def preallocate_array(shape: Tuple, dtype=np.float64, fill_value: Optional[float] = None) -> np.ndarray:
    """
    Pre-allocate array with optional fill value.

    Args:
        shape: Array shape
        dtype: Data type
        fill_value: Optional fill value (None for empty array)

    Returns:
        Pre-allocated numpy array

    Example:
        result = preallocate_array((1000, 1000), fill_value=0.0)
    """
    if fill_value is None:
        return np.empty(shape, dtype=dtype)
    elif fill_value == 0:
        return np.zeros(shape, dtype=dtype)
    else:
        arr = np.empty(shape, dtype=dtype)
        arr.fill(fill_value)
        return arr


def use_views(arr: np.ndarray, slices: Tuple) -> np.ndarray:
    """
    Create memory view instead of copy.

    Args:
        arr: Source array
        slices: Slice specification

    Returns:
        View of the array (not a copy)

    Example:
        # Memory-efficient slicing
        view = use_views(large_array, (slice(10, 20), slice(None)))
    """
    return arr[slices]


@dataclass
class OptimizationStats:
    """Track optimization statistics."""
    function_name: str
    calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (f"OptimizationStats(function={self.function_name}, "
                f"calls={self.calls}, hit_rate={self.hit_rate:.1%}, "
                f"total_time={self.total_time:.3f}s)")


# Optimization patterns documentation
OPTIMIZATION_PATTERNS = {
    'vectorization': {
        'description': 'Replace loops with numpy broadcasting',
        'example': '''
        # Bad (slow)
        for i in range(n):
            result[i] = a[i] * b[i]

        # Good (fast)
        result = a * b
        ''',
        'speedup': '10-100x'
    },

    'inplace_operations': {
        'description': 'Modify arrays in-place to avoid copies',
        'example': '''
        # Bad (creates copy)
        arr = arr + 1

        # Good (in-place)
        arr += 1
        ''',
        'speedup': '2-5x'
    },

    'preallocate': {
        'description': 'Pre-allocate arrays before filling',
        'example': '''
        # Bad (grows array)
        result = []
        for i in range(n):
            result.append(compute(i))
        result = np.array(result)

        # Good (pre-allocated)
        result = np.empty(n)
        for i in range(n):
            result[i] = compute(i)
        ''',
        'speedup': '2-10x'
    },

    'views_not_copies': {
        'description': 'Use array views instead of copies',
        'example': '''
        # Bad (creates copy)
        subset = arr[10:20].copy()

        # Good (view)
        subset = arr[10:20]
        ''',
        'speedup': '2-100x (memory)'
    },

    'sparse_matrices': {
        'description': 'Use sparse matrices for sparse data',
        'example': '''
        from scipy.sparse import csr_matrix

        # For matrices with mostly zeros
        A_sparse = csr_matrix(A_dense)
        ''',
        'speedup': '10-1000x (memory and speed)'
    },

    'caching': {
        'description': 'Cache expensive computations',
        'example': '''
        @memoize(maxsize=100)
        def expensive_function(n):
            return sum(range(n**2))
        ''',
        'speedup': '100-10000x (repeated calls)'
    },

    'numexpr': {
        'description': 'Use numexpr for complex expressions',
        'example': '''
        import numexpr as ne

        # For complex array expressions
        result = ne.evaluate("a * b + c * d")
        ''',
        'speedup': '2-10x'
    },

    'parallel': {
        'description': 'Parallelize independent operations',
        'example': '''
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(func, inputs))
        ''',
        'speedup': '2-8x (CPU cores)'
    }
}


def print_optimization_guide():
    """Print optimization patterns guide."""
    print("=" * 70)
    print("Scientific Computing Optimization Patterns")
    print("=" * 70)
    print()

    for i, (name, info) in enumerate(OPTIMIZATION_PATTERNS.items(), 1):
        print(f"{i}. {name.upper().replace('_', ' ')}")
        print(f"   Description: {info['description']}")
        print(f"   Typical speedup: {info['speedup']}")
        print(f"   Example:{info['example']}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    # Demonstrate optimization helpers
    print("Optimization Helpers Demo")
    print("=" * 70)
    print()

    # Demo 1: Memoization
    print("1. Memoization")
    print("-" * 70)

    @memoize(maxsize=10)
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    import time
    start = time.perf_counter()
    result = fibonacci(30)
    elapsed = time.perf_counter() - start

    print(f"fibonacci(30) = {result}")
    print(f"Time: {elapsed:.6f}s")
    print(f"Cache size: {len(fibonacci.cache)}")

    # Demo 2: Laplacian cache
    print("\n2. Laplacian Cache")
    print("-" * 70)

    cache = get_laplacian_cache()
    print(f"Cache empty: {len(cache.cache) == 0}")

    # Simulate storing an operator
    fake_operator = "sparse_matrix_placeholder"
    cache.set(fake_operator, nx=100, ny=100, dx=0.01, dy=0.01)
    print(f"After storing: {len(cache.cache)} items")

    # Retrieve
    retrieved = cache.get(nx=100, ny=100, dx=0.01, dy=0.01)
    print(f"Retrieved: {retrieved}")

    # Demo 3: Print optimization guide
    print("\n3. Optimization Patterns Guide")
    print("-" * 70)
    print_optimization_guide()
