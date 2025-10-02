"""
Example: Profiling PDE Solver Performance

This example demonstrates using the PerformanceProfilerAgent to profile
and optimize the 2D PDE solvers. We'll identify bottlenecks and measure
performance improvements.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent
from agents.performance_profiler_agent import PerformanceProfilerAgent
from utils.profiling import profile_performance, timer, PerformanceTracker, compare_performance


def setup_2d_poisson_problem(nx=80, ny=80):
    """Create a 2D Poisson problem for profiling."""
    # Domain
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)

    # Grid
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)

    # Source term: Gaussian charge
    def source_term(X, Y):
        sigma = 0.1
        r_squared = X**2 + Y**2
        return -(1.0 / (2 * np.pi * sigma**2)) * np.exp(-r_squared / (2 * sigma**2))

    # Boundary condition: zero at boundaries
    def boundary_condition(X, Y):
        return np.zeros_like(X)

    f = source_term(X, Y)
    bc = boundary_condition(X, Y)

    return {
        'pde_type': 'poisson',
        'x_range': x_range,
        'y_range': y_range,
        'nx': nx,
        'ny': ny,
        'source_term': f,
        'boundary_condition': bc
    }


def solve_poisson_wrapper(agent, problem_data):
    """Wrapper function for profiling PDE solver."""
    return agent.solve_pde_2d(problem_data)


def main():
    print("=" * 70)
    print("Profiling 2D PDE Solver Performance")
    print("=" * 70)
    print()

    # Create agents
    pde_agent = ODEPDESolverAgent()
    profiler_agent = PerformanceProfilerAgent()

    # ========================================================================
    # 1. Profile 2D Poisson Solver - Function Level
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. Function-Level Profiling (2D Poisson)")
    print("=" * 70)

    problem = setup_2d_poisson_problem(nx=80, ny=80)

    result = profiler_agent.process({
        'task': 'profile_function',
        'function': solve_poisson_wrapper,
        'args': [pde_agent, problem],
        'top_n': 15
    })

    if result.success:
        print(result.data['report'])
    else:
        print("Profiling failed:", result.errors)

    # ========================================================================
    # 2. Memory Profiling
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. Memory Profiling (2D Poisson)")
    print("=" * 70)

    result = profiler_agent.process({
        'task': 'profile_memory',
        'function': solve_poisson_wrapper,
        'args': [pde_agent, problem],
        'top_n': 10
    })

    if result.success:
        print(result.data['report'])
    else:
        print("Memory profiling failed:", result.errors)

    # ========================================================================
    # 3. Bottleneck Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. Bottleneck Analysis (2D Poisson)")
    print("=" * 70)

    result = profiler_agent.process({
        'task': 'analyze_bottlenecks',
        'function': solve_poisson_wrapper,
        'args': [pde_agent, problem],
        'threshold': 0.03  # 3% of total time
    })

    if result.success:
        print(result.data['report'])

        # Highlight key bottlenecks
        bottlenecks = result.data['bottlenecks']
        if bottlenecks:
            print("\nKey Performance Insights:")
            print("-" * 70)
            for i, bn in enumerate(bottlenecks[:3], 1):
                print(f"{i}. {bn['function']} consumes {bn['percentage']:.1f}% of execution time")
                print(f"   → Optimization target: {bn['file']}:{bn['line']}")
    else:
        print("Bottleneck analysis failed:", result.errors)

    # ========================================================================
    # 4. Grid Size Scaling Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. Grid Size Scaling Analysis")
    print("=" * 70)

    grid_sizes = [40, 60, 80, 100]
    scaling_results = []

    for nx in grid_sizes:
        problem = setup_2d_poisson_problem(nx=nx, ny=nx)

        with timer(f"Grid {nx}x{nx}", track=True) as t:
            result = pde_agent.solve_pde_2d(problem)

        unknowns = nx * nx
        time_per_unknown = t['elapsed'] / unknowns * 1e6  # microseconds

        scaling_results.append({
            'grid_size': nx,
            'unknowns': unknowns,
            'time': t['elapsed'],
            'time_per_unknown': time_per_unknown
        })

        print(f"  {nx:3d}x{nx:3d}: {t['elapsed']:6.3f}s ({unknowns:6d} unknowns, "
              f"{time_per_unknown:.2f} μs/unknown)")

    # Analyze scaling
    print("\nScaling Analysis:")
    print("-" * 70)
    base_time = scaling_results[0]['time']
    base_unknowns = scaling_results[0]['unknowns']

    for res in scaling_results[1:]:
        ratio = res['unknowns'] / base_unknowns
        expected_linear = base_time * ratio
        expected_quadratic = base_time * ratio**2
        actual = res['time']

        print(f"Grid {res['grid_size']}x{res['grid_size']}: "
              f"{ratio:.1f}x unknowns → {actual/base_time:.2f}x time")
        print(f"  Expected (linear O(n)): {expected_linear:.3f}s")
        print(f"  Expected (O(n²)): {expected_quadratic:.3f}s")
        print(f"  Actual: {actual:.3f}s")

    # Complexity estimate
    if len(scaling_results) >= 3:
        # Estimate scaling exponent
        r1 = scaling_results[1]
        r0 = scaling_results[0]
        exponent = np.log(r1['time'] / r0['time']) / np.log(r1['unknowns'] / r0['unknowns'])
        print(f"\nEstimated complexity: O(n^{exponent:.2f})")
        print(f"  (Ideal for sparse solver: O(n) to O(n^1.5))")

    # ========================================================================
    # 5. Compare Different PDE Types
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. Comparing Different PDE Types")
    print("=" * 70)

    nx, ny = 60, 60
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)

    # Poisson problem
    @profile_performance(track_memory=False, track_global=True)
    def solve_poisson():
        f = np.sin(np.pi * X) * np.sin(np.pi * Y)
        bc = np.zeros_like(X)
        return pde_agent.solve_pde_2d({
            'pde_type': 'poisson',
            'x_range': x_range,
            'y_range': y_range,
            'nx': nx, 'ny': ny,
            'source_term': f,
            'boundary_condition': bc
        })

    # Heat equation problem
    @profile_performance(track_memory=False, track_global=True)
    def solve_heat():
        u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
        bc = np.zeros_like(X)
        return pde_agent.solve_pde_2d({
            'pde_type': 'heat',
            'x_range': x_range,
            'y_range': y_range,
            'nx': nx, 'ny': ny,
            'initial_condition': u0,
            'boundary_condition': bc,
            't_span': (0.0, 0.1),
            'alpha': 0.01
        })

    # Wave equation problem
    @profile_performance(track_memory=False, track_global=True)
    def solve_wave():
        u0 = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        v0 = np.zeros_like(X)
        bc = np.zeros_like(X)
        return pde_agent.solve_pde_2d({
            'pde_type': 'wave',
            'x_range': x_range,
            'y_range': y_range,
            'nx': nx, 'ny': ny,
            'initial_displacement': u0,
            'initial_velocity': v0,
            'boundary_condition': bc,
            't_span': (0.0, 1.0),
            'wave_speed': 1.0
        })

    # Run comparisons
    print("\nSolving Poisson equation...")
    poisson_result = solve_poisson()

    print("\nSolving Heat equation...")
    heat_result = solve_heat()

    print("\nSolving Wave equation...")
    wave_result = solve_wave()

    # ========================================================================
    # 6. Performance Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. Overall Performance Summary")
    print("=" * 70)

    print(PerformanceTracker.generate_report())

    # ========================================================================
    # 7. Recommendations
    # ========================================================================
    print("\n" + "=" * 70)
    print("Optimization Recommendations")
    print("=" * 70)

    print("""
Based on the profiling results, here are optimization opportunities:

1. **Sparse Matrix Operations** (if bottleneck):
   - Poisson solver uses sparse direct solver
   - Consider iterative solvers (CG, GMRES) for very large problems
   - Current implementation scales as O(n) to O(n^1.5) which is good

2. **Time Integration** (Heat/Wave):
   - Method of lines with RK45 is accurate but can be slow
   - Consider specialized time steppers (Crank-Nicolson, etc.)
   - Vectorization is already applied

3. **Memory Usage**:
   - Monitor for large grid sizes (>200x200)
   - Sparse matrix storage is memory-efficient
   - Consider out-of-core solvers for extreme cases

4. **Parallelization Opportunities**:
   - Multiple PDE solves can run in parallel
   - Spatial decomposition for very large grids
   - GPU acceleration for time integration (JAX)

5. **Grid Generation**:
   - Pre-compute and cache grid for multiple solves
   - Avoid repeated meshgrid calls

Next Steps:
- Implement parallel solver execution (Week 19 Phase 2)
- Add GPU acceleration with JAX (Week 19 Phase 5)
- Benchmark against other PDE libraries
    """)

    print("=" * 70)
    print("Profiling Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
