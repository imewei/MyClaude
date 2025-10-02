"""
Example: Parallel PDE Solving

This example demonstrates parallel execution of PDE solvers, showing:
1. Solving multiple PDEs in parallel
2. Performance comparison (serial vs parallel)
3. Workflow orchestration with dependencies
4. Speedup analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent
from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent, WorkflowStep
from core.parallel_executor import ParallelMode, execute_parallel
from utils.profiling import timer


def create_2d_poisson_problem(nx, ny, source_center=(0.0, 0.0), source_sigma=0.1):
    """Create a 2D Poisson problem with Gaussian source."""
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)

    # Gaussian source
    cx, cy = source_center
    r_squared = (X - cx)**2 + (Y - cy)**2
    f = -(1.0 / (2 * np.pi * source_sigma**2)) * np.exp(-r_squared / (2 * source_sigma**2))

    bc = np.zeros_like(X)

    return {
        'pde_type': 'poisson',
        'x_range': x_range,
        'y_range': y_range,
        'nx': nx,
        'ny': ny,
        'source_term': f,
        'boundary_condition': bc
    }


def create_2d_heat_problem(nx, ny, t_final=0.1):
    """Create a 2D heat equation problem."""
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)

    # Initial condition: sin wave
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    bc = np.zeros_like(X)

    return {
        'pde_type': 'heat',
        'x_range': x_range,
        'y_range': y_range,
        'nx': nx,
        'ny': ny,
        'initial_condition': u0,
        'boundary_condition': bc,
        't_span': (0.0, t_final),
        'alpha': 0.01
    }


def solve_pde_wrapper(agent, problem):
    """Wrapper function for profiling."""
    return agent.solve_pde_2d(problem)


def main():
    print("=" * 70)
    print("Parallel PDE Solving Demonstration")
    print("=" * 70)

    # Create agent
    pde_agent = ODEPDESolverAgent()

    # ========================================================================
    # 1. Solve Multiple PDEs - Serial vs Parallel
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. Multiple Poisson Problems - Serial vs Parallel")
    print("=" * 70)

    # Create 4 different Poisson problems
    problems = [
        create_2d_poisson_problem(60, 60, source_center=(0.0, 0.0), source_sigma=0.1),
        create_2d_poisson_problem(60, 60, source_center=(0.5, 0.0), source_sigma=0.15),
        create_2d_poisson_problem(60, 60, source_center=(0.0, 0.5), source_sigma=0.2),
        create_2d_poisson_problem(60, 60, source_center=(-0.5, -0.5), source_sigma=0.12),
    ]

    # Serial execution
    print("\nSerial Execution:")
    with timer("Serial execution") as t_serial:
        serial_results = []
        for i, problem in enumerate(problems):
            result = pde_agent.solve_pde_2d(problem)
            serial_results.append(result)
            print(f"  Problem {i+1}: {result.status.name}")

    # Parallel execution (threads)
    print("\nParallel Execution (Threads):")
    orchestrator = WorkflowOrchestrationAgent(
        parallel_mode=ParallelMode.THREADS,
        max_workers=4
    )

    with timer("Parallel execution (threads)") as t_parallel_threads:
        parallel_results_threads = orchestrator.execute_agents_parallel(
            agents=[pde_agent] * len(problems),
            method_name='solve_pde_2d',
            inputs_list=problems
        )

    for i, result in enumerate(parallel_results_threads):
        print(f"  Problem {i+1}: {'SUCCESS' if result.success else 'FAILED'} "
              f"(time: {result.execution_time:.3f}s)")

    # Parallel execution (processes)
    print("\nParallel Execution (Processes):")
    orchestrator_proc = WorkflowOrchestrationAgent(
        parallel_mode=ParallelMode.PROCESSES,
        max_workers=4
    )

    with timer("Parallel execution (processes)") as t_parallel_proc:
        parallel_results_proc = orchestrator_proc.execute_agents_parallel(
            agents=[pde_agent] * len(problems),
            method_name='solve_pde_2d',
            inputs_list=problems
        )

    for i, result in enumerate(parallel_results_proc):
        print(f"  Problem {i+1}: {'SUCCESS' if result.success else 'FAILED'} "
              f"(time: {result.execution_time:.3f}s)")

    # Performance comparison
    print("\nPerformance Summary:")
    print("-" * 70)
    print(f"Serial execution:              {t_serial['elapsed']:.3f}s")
    print(f"Parallel execution (threads):  {t_parallel_threads['elapsed']:.3f}s")
    print(f"Parallel execution (processes): {t_parallel_proc['elapsed']:.3f}s")
    print(f"\nSpeedup (threads):   {t_serial['elapsed'] / t_parallel_threads['elapsed']:.2f}x")
    print(f"Speedup (processes): {t_serial['elapsed'] / t_parallel_proc['elapsed']:.2f}x")

    # ========================================================================
    # 2. Parameter Sweep - Parallel Grid Search
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. Parameter Sweep - Parallel Grid Search")
    print("=" * 70)

    # Vary source sigma parameter
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.25]
    print(f"\nSweeping sigma parameter: {sigmas}")

    problems_sweep = [
        create_2d_poisson_problem(50, 50, source_center=(0.0, 0.0), source_sigma=sigma)
        for sigma in sigmas
    ]

    with timer("Parameter sweep (parallel)") as t_sweep:
        sweep_results = orchestrator.execute_agents_parallel(
            agents=[pde_agent] * len(problems_sweep),
            method_name='solve_pde_2d',
            inputs_list=problems_sweep
        )

    print("\nResults:")
    for sigma, result in zip(sigmas, sweep_results):
        if result.success:
            solution = result.result.data['solution']
            max_potential = np.max(np.abs(solution))
            print(f"  sigma={sigma:.2f}: max|u|={max_potential:.6f}, time={result.execution_time:.3f}s")

    print(f"\nTotal sweep time: {t_sweep['elapsed']:.3f}s")
    print(f"(Serial would take: ~{sum(r.execution_time for r in sweep_results):.3f}s)")

    # ========================================================================
    # 3. Mixed PDE Types - Parallel Execution
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. Mixed PDE Types - Parallel Execution")
    print("=" * 70)

    # Create different types of PDEs
    mixed_problems = [
        create_2d_poisson_problem(50, 50),
        create_2d_heat_problem(50, 50, t_final=0.05),
        create_2d_poisson_problem(50, 50, source_center=(0.5, 0.5)),
        create_2d_heat_problem(50, 50, t_final=0.1),
    ]

    problem_types = ["Poisson 1", "Heat 1", "Poisson 2", "Heat 2"]

    with timer("Mixed PDEs (parallel)") as t_mixed:
        mixed_results = orchestrator.execute_agents_parallel(
            agents=[pde_agent] * len(mixed_problems),
            method_name='solve_pde_2d',
            inputs_list=mixed_problems
        )

    print("\nResults:")
    for ptype, result in zip(problem_types, mixed_results):
        status = 'SUCCESS' if result.success else 'FAILED'
        print(f"  {ptype:12s}: {status} (time: {result.execution_time:.3f}s)")

    print(f"\nTotal time: {t_mixed['elapsed']:.3f}s")

    # ========================================================================
    # 4. Scalability Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. Scalability Analysis")
    print("=" * 70)

    n_problems_range = [1, 2, 4, 8]
    scalability_results = []

    print("\nTesting scalability with increasing problem count:")

    for n_probs in n_problems_range:
        probs = [create_2d_poisson_problem(40, 40) for _ in range(n_probs)]

        # Serial
        start = time.perf_counter()
        for prob in probs:
            pde_agent.solve_pde_2d(prob)
        t_serial_n = time.perf_counter() - start

        # Parallel
        start = time.perf_counter()
        orchestrator.execute_agents_parallel(
            agents=[pde_agent] * len(probs),
            method_name='solve_pde_2d',
            inputs_list=probs
        )
        t_parallel_n = time.perf_counter() - start

        speedup = t_serial_n / t_parallel_n
        efficiency = speedup / n_probs * 100

        scalability_results.append({
            'n_problems': n_probs,
            't_serial': t_serial_n,
            't_parallel': t_parallel_n,
            'speedup': speedup,
            'efficiency': efficiency
        })

        print(f"\n  {n_probs} problems:")
        print(f"    Serial:   {t_serial_n:.3f}s")
        print(f"    Parallel: {t_parallel_n:.3f}s")
        print(f"    Speedup:  {speedup:.2f}x")
        print(f"    Efficiency: {efficiency:.1f}%")

    # ========================================================================
    # 5. Visualization
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. Creating Visualizations")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parallel PDE Solving - Performance Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Serial vs Parallel comparison
    ax = axes[0, 0]
    execution_modes = ['Serial', 'Parallel\n(Threads)', 'Parallel\n(Processes)']
    times = [
        t_serial['elapsed'],
        t_parallel_threads['elapsed'],
        t_parallel_proc['elapsed']
    ]
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    bars = ax.bar(execution_modes, times, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Execution Time (s)', fontsize=11)
    ax.set_title('Execution Mode Comparison\n(4 Poisson Problems)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Speedup vs number of problems
    ax = axes[0, 1]
    n_probs = [r['n_problems'] for r in scalability_results]
    speedups = [r['speedup'] for r in scalability_results]
    ax.plot(n_probs, speedups, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='Actual')
    ax.plot(n_probs, n_probs, '--', linewidth=2, color='gray', alpha=0.5, label='Ideal (linear)')
    ax.set_xlabel('Number of Problems', fontsize=11)
    ax.set_ylabel('Speedup', fontsize=11)
    ax.set_title('Parallel Speedup Scaling', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Efficiency vs number of problems
    ax = axes[1, 0]
    efficiencies = [r['efficiency'] for r in scalability_results]
    ax.plot(n_probs, efficiencies, 's-', linewidth=2, markersize=8, color='#2ca02c')
    ax.set_xlabel('Number of Problems', fontsize=11)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=11)
    ax.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)

    # Plot 4: Parameter sweep results
    ax = axes[1, 1]
    max_potentials = []
    for result in sweep_results:
        if result.success:
            solution = result.result.data['solution']
            max_potentials.append(np.max(np.abs(solution)))

    ax.plot(sigmas, max_potentials, 'o-', linewidth=2, markersize=8, color='#ff7f0e')
    ax.set_xlabel('Source Width (sigma)', fontsize=11)
    ax.set_ylabel('Max |u|', fontsize=11)
    ax.set_title('Parameter Sweep Results\n(Peak Potential vs Source Width)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'example_parallel_pde_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"""
Parallel Execution Benefits:
- Speedup (4 problems, threads):   {t_serial['elapsed'] / t_parallel_threads['elapsed']:.2f}x
- Speedup (4 problems, processes): {t_serial['elapsed'] / t_parallel_proc['elapsed']:.2f}x
- Peak speedup ({max(n_probs)} problems):  {max(speedups):.2f}x
- Best efficiency: {max(efficiencies):.1f}%

When to Use Parallel Execution:
✓ Multiple independent PDE solves
✓ Parameter sweeps / grid searches
✓ Monte Carlo simulations
✓ Ensemble calculations

Performance Considerations:
- Thread pool best for I/O-bound or moderate CPU tasks
- Process pool best for heavy CPU-bound computations
- Overhead becomes negligible with 4+ independent tasks
- Parallel efficiency: {efficiencies[-1]:.1f}% for {n_probs[-1]} problems
    """)

    print("=" * 70)
    print("Parallel PDE Solving Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
