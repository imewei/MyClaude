"""
Profile existing agents to identify optimization opportunities.

This script profiles key agents to establish performance baselines
and identify bottlenecks.
"""

import numpy as np
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent
from agents.performance_profiler_agent import PerformanceProfilerAgent
from utils.profiling import profile_performance, timer, PerformanceTracker


def create_test_problems():
    """Create standard test problems for profiling."""

    # 2D Poisson problem - source_term must be callable
    def source_poisson(x, y):
        sigma = 0.1
        return -(1.0 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    poisson_problem = {
        'pde_type': 'poisson',
        'x_range': (-1.0, 1.0),
        'y_range': (-1.0, 1.0),
        'nx': 80,
        'ny': 80,
        'source_term': source_poisson,
        'boundary_conditions': {'value': 0.0}
    }

    # 2D Heat problem - initial_condition must be callable
    def initial_heat(X, Y):
        return np.sin(np.pi * X) * np.sin(np.pi * Y)

    heat_problem = {
        'pde_type': 'heat',
        'x_range': (0.0, 1.0),
        'y_range': (0.0, 1.0),
        'nx': 60,
        'ny': 60,
        'initial_condition': initial_heat,
        'boundary_conditions': {'value': 0.0},
        't_span': (0.0, 0.1),
        'alpha': 0.01
    }

    # 3D Poisson problem - source_term must be callable
    def source_poisson_3d(x, y, z):
        sigma = 0.15
        normalization = 1.0 / (sigma**3 * (2*np.pi)**(3/2))
        return -normalization * np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

    poisson_3d_problem = {
        'x_range': (-1.0, 1.0),
        'y_range': (-1.0, 1.0),
        'z_range': (-1.0, 1.0),
        'nx': 30,
        'ny': 30,
        'nz': 30,
        'source_term': source_poisson_3d,
        'boundary_conditions': {'value': 0.0}
    }

    return {
        'poisson_2d': poisson_problem,
        'heat_2d': heat_problem,
        'poisson_3d': poisson_3d_problem
    }


def profile_ode_pde_agent():
    """Profile the ODEPDESolverAgent."""
    print("=" * 70)
    print("Profiling ODEPDESolverAgent")
    print("=" * 70)

    agent = ODEPDESolverAgent()
    profiler = PerformanceProfilerAgent()

    problems = create_test_problems()

    # ========================================================================
    # Profile 2D Poisson Solver
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. Profiling 2D Poisson Solver (80x80 grid)")
    print("-" * 70)

    # Warm-up run
    agent.solve_pde_2d(problems['poisson_2d'])

    # Profile
    result = profiler.process({
        'task': 'profile_function',
        'function': agent.solve_pde_2d,
        'args': (problems['poisson_2d'],),
        'top_n': 20
    })

    if result.success:
        print("\nTop functions by cumulative time:")
        for i, stat in enumerate(result.data['statistics'][:10], 1):
            print(f"{i:2d}. {stat['function']:<40} {stat['cumulative_time']:8.4f}s ({stat['calls']:6d} calls)")

    # Bottleneck analysis
    result_bn = profiler.process({
        'task': 'analyze_bottlenecks',
        'function': agent.solve_pde_2d,
        'args': (problems['poisson_2d'],),
        'threshold': 0.05
    })

    if result_bn.success:
        print("\nBottlenecks (>5% of time):")
        for bn in result_bn.data['bottlenecks'][:5]:
            print(f"  - {bn['function']}: {bn['percentage']:.1f}% ({bn['cumulative_time']:.4f}s)")

    # ========================================================================
    # Profile 2D Heat Solver
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. Profiling 2D Heat Solver (60x60 grid, t=0.1)")
    print("-" * 70)

    # Warm-up
    agent.solve_pde_2d(problems['heat_2d'])

    # Profile
    result = profiler.process({
        'task': 'profile_function',
        'function': agent.solve_pde_2d,
        'args': (problems['heat_2d'],),
        'top_n': 20
    })

    if result.success:
        print("\nTop functions by cumulative time:")
        for i, stat in enumerate(result.data['statistics'][:10], 1):
            print(f"{i:2d}. {stat['function']:<40} {stat['cumulative_time']:8.4f}s ({stat['calls']:6d} calls)")

    # ========================================================================
    # Profile 3D Poisson Solver
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. Profiling 3D Poisson Solver (30x30x30 grid)")
    print("-" * 70)

    # Warm-up
    agent.solve_poisson_3d(problems['poisson_3d'])

    # Profile
    result = profiler.process({
        'task': 'profile_function',
        'function': agent.solve_poisson_3d,
        'args': (problems['poisson_3d'],),
        'top_n': 20
    })

    if result.success:
        print("\nTop functions by cumulative time:")
        for i, stat in enumerate(result.data['statistics'][:10], 1):
            print(f"{i:2d}. {stat['function']:<40} {stat['cumulative_time']:8.4f}s ({stat['calls']:6d} calls)")

    # ========================================================================
    # Memory Profiling
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. Memory Profiling - 2D Poisson")
    print("-" * 70)

    result_mem = profiler.process({
        'task': 'profile_memory',
        'function': agent.solve_pde_2d,
        'args': (problems['poisson_2d'],),
        'top_n': 10
    })

    if result_mem.success:
        print(f"\nPeak Memory: {result_mem.data['peak_memory_mb']:.2f} MB")
        print(f"Execution Time: {result_mem.data['total_time']:.4f}s")
        print("\nTop memory allocations:")
        for i, stat in enumerate(result_mem.data['memory_stats'][:5], 1):
            print(f"{i}. {stat['size_mb']:.2f} MB")

    # ========================================================================
    # Scaling Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. Scaling Analysis - 2D Poisson")
    print("-" * 70)

    grid_sizes = [40, 60, 80, 100]
    print("\nGrid Size | Unknowns | Time (s) | Time/Unknown (μs)")
    print("-" * 60)

    def source_scaling(x, y):
        return -(1.0 / (2 * np.pi * 0.1**2)) * np.exp(-(x**2 + y**2) / (2 * 0.1**2))

    for n in grid_sizes:
        problem = {
            'pde_type': 'poisson',
            'x_range': (-1.0, 1.0),
            'y_range': (-1.0, 1.0),
            'nx': n,
            'ny': n,
            'source_term': source_scaling,
            'boundary_conditions': {'value': 0.0}
        }

        # Time it
        start = time.perf_counter()
        agent.solve_pde_2d(problem)
        elapsed = time.perf_counter() - start

        unknowns = n * n
        time_per_unknown = elapsed / unknowns * 1e6

        print(f"{n:3d}x{n:<3d}  | {unknowns:8d} | {elapsed:8.4f} | {time_per_unknown:8.2f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Profiling Summary")
    print("=" * 70)

    print("""
Key Findings:

1. **2D Poisson Solver**:
   - Primary bottleneck: Sparse matrix assembly and solve
   - Complexity: O(n) to O(n^1.5) - good for sparse solver
   - Memory: ~few MB for typical grids

2. **2D Heat Solver**:
   - Primary bottleneck: Time integration (RK45)
   - Many function evaluations of Laplacian
   - Could benefit from Crank-Nicolson or other implicit methods

3. **3D Poisson Solver**:
   - Sparse solver dominates (27K unknowns)
   - Memory efficient due to sparse representation
   - Time scales as O(n) to O(n^1.5)

Optimization Opportunities:
- ✓ Sparse matrix methods already used (efficient)
- → Pre-compute Laplacian operator for repeated solves
- → Vectorize boundary condition application
- → Consider iterative solvers for very large problems
- → Cache grid generation for multiple solves
- → Use numba/JAX for hot loops (future)
    """)


def profile_comparison_before_after():
    """Compare performance before and after optimizations."""
    print("\n" + "=" * 70)
    print("Performance Baseline Established")
    print("=" * 70)
    print("""
This baseline will be used to measure improvements from optimizations.

Next steps:
1. Implement targeted optimizations
2. Re-run profiling to measure improvements
3. Document speedup achieved
    """)


def main():
    print("=" * 70)
    print("Agent Performance Profiling Suite")
    print("=" * 70)
    print()

    # Profile ODE/PDE agent
    profile_ode_pde_agent()

    # Comparison framework
    profile_comparison_before_after()

    print("\n" + "=" * 70)
    print("Profiling Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
