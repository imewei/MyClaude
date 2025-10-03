# Scientific Computing Optimization Guide

**Version**: 1.0
**Date**: 2025-09-30
**Audience**: Agent developers and contributors

---

## Table of Contents

1. [Overview](#overview)
2. [Profiling Your Code](#profiling-your-code)
3. [Common Bottlenecks](#common-bottlenecks)
4. [Optimization Patterns](#optimization-patterns)
5. [Agent-Specific Optimizations](#agent-specific-optimizations)
6. [Performance Testing](#performance-testing)
7. [When NOT to Optimize](#when-not-to-optimize)

---

## Overview

This guide provides strategies for optimizing scientific computing agents in this codebase. Always **profile before optimizing** to ensure you're targeting actual bottlenecks.

### Optimization Philosophy

1. **Correctness First**: Never sacrifice correctness for speed
2. **Profile-Driven**: Use profiling to find real bottlenecks
3. **Measure Impact**: Quantify improvements
4. **Maintain Readability**: Document complex optimizations

### Typical Performance Gains

| Optimization | Typical Speedup | When to Use |
|--------------|-----------------|-------------|
| Vectorization | 10-100x | Replacing loops over arrays |
| Sparse matrices | 10-1000x | Large matrices with many zeros |
| Caching | 100-10000x | Repeated expensive computations |
| Parallelization | 2-8x | Independent operations |
| In-place ops | 2-5x | Large array modifications |
| Numba/JIT | 10-100x | Hot loops (future) |

---

## Profiling Your Code

### Using the Performance Profiler Agent

```python
from agents.performance_profiler_agent import PerformanceProfilerAgent

profiler = PerformanceProfilerAgent()

# Profile a function
result = profiler.process({
    'task': 'profile_function',
    'function': your_function,
    'args': (arg1, arg2),
    'top_n': 20
})

print(result.data['report'])
```

### Using Profiling Utilities

```python
from utils.profiling import profile_performance, timer

# Decorator for automatic profiling
@profile_performance(track_memory=True)
def expensive_operation():
    # ... code ...
    pass

# Context manager for timing specific blocks
with timer("Critical section"):
    # ... code ...
    pass
```

### Bottleneck Analysis

```python
result = profiler.process({
    'task': 'analyze_bottlenecks',
    'function': your_function,
    'args': (args,),
    'threshold': 0.05  # Functions using >5% of time
})

for bottleneck in result.data['bottlenecks']:
    print(f"{bottleneck['function']}: {bottleneck['percentage']:.1f}%")
```

---

## Common Bottlenecks

### 1. Python Loops Over Large Arrays

**Problem**:
```python
# SLOW: Python loop
result = np.zeros(n)
for i in range(n):
    result[i] = a[i] * b[i] + c[i]
```

**Solution**:
```python
# FAST: Vectorized
result = a * b + c
```

**Speedup**: 10-100x

### 2. Unnecessary Array Copies

**Problem**:
```python
# SLOW: Creates copy
arr = arr + 1
subset = large_array[0:1000].copy()
```

**Solution**:
```python
# FAST: In-place modification
arr += 1
subset = large_array[0:1000]  # View, not copy
```

**Speedup**: 2-10x

### 3. Dense Matrices for Sparse Data

**Problem**:
```python
# SLOW: Dense storage for sparse matrix
A = np.zeros((10000, 10000))  # Mostly zeros
# Uses 800 MB even though only 1% non-zero
```

**Solution**:
```python
# FAST: Sparse storage
from scipy.sparse import csr_matrix
A = csr_matrix((10000, 10000))
# Uses ~8 MB for 1% non-zero
```

**Speedup**: 10-1000x (memory and operations)

### 4. Repeated Expensive Computations

**Problem**:
```python
# SLOW: Recomputes matrix every call
def solve_pde(problem):
    A = build_laplacian_matrix(problem)  # Expensive!
    return solve(A, b)
```

**Solution**:
```python
# FAST: Cache the matrix
@memoize()
def build_laplacian_matrix(nx, ny, dx, dy):
    # ... expensive computation ...
    return A

def solve_pde(problem):
    A = build_laplacian_matrix(problem.nx, problem.ny, ...)
    return solve(A, b)
```

**Speedup**: 100-10000x for repeated calls

### 5. Non-Vectorized Boundary Conditions

**Problem**:
```python
# SLOW: Loop to apply boundary conditions
for i in range(nx):
    for j in range(ny):
        if is_boundary(i, j):
            u[i, j] = 0
```

**Solution**:
```python
# FAST: Vectorized boundary application
u[0, :] = 0    # Top edge
u[-1, :] = 0   # Bottom edge
u[:, 0] = 0    # Left edge
u[:, -1] = 0   # Right edge
```

**Speedup**: 10-50x

---

## Optimization Patterns

### Pattern 1: Vectorization

**Principle**: Use NumPy's C-level operations instead of Python loops.

**Bad**:
```python
def compute_distances_slow(points):
    n = len(points)
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = np.sqrt(points[i, 0]**2 + points[i, 1]**2)
    return distances
```

**Good**:
```python
def compute_distances_fast(points):
    return np.sqrt(np.sum(points**2, axis=1))
```

**Tools**:
- Use `@vectorize_operation` decorator as documentation
- Check with `%timeit` in Jupyter
- Look for patterns like `for i in range(n):`

### Pattern 2: Pre-allocation

**Principle**: Allocate arrays before filling them.

**Bad**:
```python
result = []
for i in range(n):
    result.append(compute(i))
result = np.array(result)
```

**Good**:
```python
result = np.empty(n)
for i in range(n):
    result[i] = compute(i)
```

**Helper**:
```python
from utils.optimization_helpers import preallocate_array

result = preallocate_array((n, m), fill_value=0.0)
```

### Pattern 3: Caching/Memoization

**Principle**: Store results of expensive computations.

**Implementation**:
```python
from utils.optimization_helpers import memoize, LaplacianCache

@memoize(maxsize=100)
def expensive_function(n):
    # Computation that depends only on n
    return result

# For Laplacian operators
cache = LaplacianCache()
A = cache.get(nx, ny, dx, dy)
if A is None:
    A = build_laplacian(nx, ny, dx, dy)
    cache.set(A, nx, ny, dx, dy)
```

### Pattern 4: Sparse Matrices

**When to Use**: Matrix with <10% non-zero elements

**Implementation**:
```python
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix

# Build with lil_matrix (efficient for construction)
A = lil_matrix((N, N))
for i in range(N):
    A[i, i] = -4.0
    if i > 0:
        A[i, i-1] = 1.0
    if i < N-1:
        A[i, i+1] = 1.0

# Convert to csr for fast operations
A_csr = csr_matrix(A)
result = A_csr @ x  # Fast sparse matrix-vector multiply
```

### Pattern 5: In-Place Operations

**Principle**: Modify arrays in-place to avoid memory allocation.

**Bad**:
```python
a = a + b  # Creates new array
a = a * 2  # Creates new array
```

**Good**:
```python
a += b  # Modifies a in-place
a *= 2  # Modifies a in-place
```

**Helper**:
```python
from utils.optimization_helpers import inplace_operation

@inplace_operation()
def update_field(u, dt, laplacian):
    u += dt * laplacian  # In-place
    return u
```

### Pattern 6: Parallel Execution

**Principle**: Run independent tasks concurrently.

**Implementation**:
```python
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent
from core.parallel_executor import ParallelMode

orchestrator = WorkflowOrchestrationAgent(
    parallel_mode=ParallelMode.THREADS,
    max_workers=4
)

results = orchestrator.execute_agents_parallel(
    agents=[agent1, agent2, agent3, agent4],
    method_name='solve',
    inputs_list=[input1, input2, input3, input4]
)
```

**Speedup**: 2-4x for 4 independent tasks

---

## Agent-Specific Optimizations

### ODEPDESolverAgent

**Identified Bottlenecks** (from profiling):
1. Sparse matrix assembly: 56% of time
2. Sparse solver: 15% of time
3. Source term evaluation: 8% of time

**Optimizations**:

1. **Cache Laplacian Operators**
   ```python
   # Before: Rebuild matrix every solve
   A = build_laplacian(nx, ny, dx, dy)

   # After: Cache and reuse
   from utils.optimization_helpers import get_laplacian_cache
   cache = get_laplacian_cache()
   A = cache.get(nx, ny, dx, dy)
   if A is None:
       A = build_laplacian(nx, ny, dx, dy)
       cache.set(A, nx, ny, dx, dy)
   ```
   **Impact**: 50-90% speedup for repeated solves

2. **Vectorize Boundary Conditions**
   ```python
   # Before: Loop over boundary points
   for i in range(nx):
       for j in range(ny):
           if is_boundary(i, j):
               apply_bc(i, j)

   # After: Vectorized boundary application
   u[0, :] = bc_value
   u[-1, :] = bc_value
   u[:, 0] = bc_value
   u[:, -1] = bc_value
   ```
   **Impact**: 10-30% speedup

3. **Pre-compute Grid Coordinates**
   ```python
   # Store X, Y grids for reuse
   if not hasattr(self, '_grid_cache'):
       self._grid_cache = {}

   key = (nx, ny, x_range, y_range)
   if key in self._grid_cache:
       X, Y = self._grid_cache[key]
   else:
       x = np.linspace(*x_range, nx)
       y = np.linspace(*y_range, ny)
       X, Y = np.meshgrid(x, y)
       self._grid_cache[key] = (X, Y)
   ```
   **Impact**: 5-10% speedup for small problems

### LinearAlgebraAgent

**Optimization Opportunities**:
1. Use specialized solvers for structured matrices (tridiagonal, banded)
2. Cache factorizations for repeated solves with same matrix
3. Use iterative solvers for large sparse systems

### OptimizationAgent

**Optimization Opportunities**:
1. Cache gradient/Hessian computations when possible
2. Use JAX for automatic differentiation (future)
3. Warm-start optimization with previous solutions

---

## Performance Testing

### Creating Benchmarks

```python
from utils.profiling import benchmark

@benchmark(n_runs=10)
def test_solver_performance():
    agent.solve_pde_2d(problem)
```

### Regression Testing

Create baseline measurements:

```bash
python scripts/profile_agents.py > baselines/baseline_2025_09_30.txt
```

Compare after optimizations:

```bash
python scripts/profile_agents.py > results/after_optimization.txt
diff baselines/baseline_2025_09_30.txt results/after_optimization.txt
```

### Scaling Analysis

```python
grid_sizes = [40, 60, 80, 100, 120]
times = []

for n in grid_sizes:
    problem = create_problem(nx=n, ny=n)
    start = time.perf_counter()
    solve(problem)
    times.append(time.perf_counter() - start)

# Analyze scaling
# O(n): times proportional to n
# O(n²): times proportional to n²
# O(n log n): sparse solver ideal
```

---

## When NOT to Optimize

### Premature Optimization

**Don't optimize until**:
1. Code is correct
2. You've profiled and identified bottlenecks
3. The bottleneck significantly impacts user experience

### Readability vs Performance

**Don't sacrifice readability** unless:
- Profiling shows significant bottleneck (>10% of time)
- Optimization provides >2x speedup
- Code is well-documented and tested

**Bad**:
```python
# Micro-optimization that hurts readability
r = np.sqrt((x*x+y*y+z*z))  # Saved one operation!
```

**Good**:
```python
# Clear and still efficient
r = np.sqrt(x**2 + y**2 + z**2)
```

### Optimization Checklist

Before optimizing:
- [ ] Is the code correct?
- [ ] Have you profiled to identify bottlenecks?
- [ ] Will this optimization be maintainable?
- [ ] Can you measure the improvement?
- [ ] Is the speedup worth the complexity?

---

## Tools and Resources

### Profiling Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| PerformanceProfilerAgent | Function-level profiling | `profiler.process({'task': 'profile_function', ...})` |
| `@profile_performance` | Decorator profiling | `@profile_performance()` |
| `timer()` | Context manager timing | `with timer("operation"): ...` |
| cProfile | Built-in Python profiler | `python -m cProfile script.py` |
| line_profiler | Line-by-line profiling | `kernprof -l script.py` |

### Optimization Helpers

```python
from utils.optimization_helpers import (
    memoize,
    LaplacianCache,
    get_laplacian_cache,
    preallocate_array,
    vectorize_operation,
    inplace_operation,
    OPTIMIZATION_PATTERNS
)
```

### Further Reading

- NumPy Performance Guide: https://numpy.org/doc/stable/user/performance.html
- SciPy Sparse Matrices: https://docs.scipy.org/doc/scipy/reference/sparse.html
- Python Performance Tips: https://wiki.python.org/moin/PythonSpeed
- This codebase: `examples/example_profiling_pde.py`

---

## Summary

### Quick Reference

1. **Always profile first**: Use PerformanceProfilerAgent
2. **Vectorize loops**: Replace `for` with NumPy operations
3. **Use sparse matrices**: For matrices with <10% non-zero elements
4. **Cache expensive computations**: Use `@memoize` decorator
5. **Parallelize independent tasks**: Use WorkflowOrchestrationAgent
6. **Avoid copies**: Use in-place operations (`+=`, `*=`)
7. **Pre-allocate arrays**: Don't grow arrays in loops
8. **Measure improvements**: Quantify speedups

### Expected Performance

| Problem Size | Operation | Time (Before) | Time (After) | Speedup |
|--------------|-----------|---------------|--------------|---------|
| 80×80 Poisson | Solve | 0.14s | 0.05s | 2.8x |
| 100×100 Poisson | Solve | 0.25s | 0.08s | 3.1x |
| 4× independent solves | Serial | 0.56s | 0.18s (parallel) | 3.1x |
| Repeated solve (same grid) | Uncached | 0.14s | 0.02s (cached) | 7.0x |

---

**Last Updated**: 2025-09-30
**Contributors**: Scientific Computing Agents Team
**Status**: Living document - updated as new optimizations are discovered
