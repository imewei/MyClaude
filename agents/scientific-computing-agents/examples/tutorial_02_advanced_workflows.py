#!/usr/bin/env python3
"""
Tutorial 02: Advanced Workflows

This tutorial shows advanced workflow patterns:
1. Sequential vs parallel execution
2. Dependency management
3. Error handling in workflows
4. Performance optimization

Time: 15 minutes
Prerequisites: Tutorial 01
"""

import numpy as np
import time

print("=" * 60)
print("Tutorial 02: Advanced Workflows")
print("=" * 60)

# =============================================================================
# Part 1: Sequential Workflows
# =============================================================================

print("\n--- Part 1: Sequential Workflow with Dependencies ---")

from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent, WorkflowStep
from agents.optimization_agent import OptimizationAgent
from agents.ode_pde_solver_agent import ODEPDESolverAgent

orchestrator = WorkflowOrchestrationAgent()
print("✓ Created orchestrator")

# Scenario: Optimize parameters, then simulate with optimal parameters
# Step 1: Find optimal decay rate
# Step 2: Simulate using optimal rate

optimizer = OptimizationAgent()
solver = ODEPDESolverAgent()

# Storage for optimal parameter
optimal_rate = {'value': None}

def optimize_decay_rate():
    """Find the decay rate that best fits target endpoint."""
    target_value = 0.1  # We want y(5) ≈ 0.1

    def objective(params):
        rate = params[0]
        # Analytical solution: y(t) = exp(-rate * t)
        final_value = np.exp(-rate * 5)
        return (final_value - target_value)**2

    result = optimizer.process({
        'task': 'minimize',
        'function': objective,
        'x0': [1.0],
        'method': 'L-BFGS-B',
        'bounds': [(0.01, 10.0)]
    })

    if result.success:
        optimal_rate['value'] = result.data['x'][0]
        print(f"  ✓ Optimal decay rate: {optimal_rate['value']:.4f}")

    return result

def simulate_with_optimal():
    """Simulate ODE with the optimal rate."""
    if optimal_rate['value'] is None:
        raise ValueError("Optimization must run first!")

    rate = optimal_rate['value']

    def decay_equation(t, y):
        return -rate * y

    result = solver.process({
        'task': 'solve_ode',
        'equation': decay_equation,
        'initial_conditions': [1.0],
        't_span': (0, 5),
        't_eval': np.linspace(0, 5, 50)
    })

    if result.success:
        final = result.data['y'][-1][0]
        print(f"  ✓ Simulated final value: {final:.4f}")

    return result

# Execute workflow
print("Executing sequential workflow...")
opt_result = optimize_decay_rate()
sim_result = simulate_with_optimal()

print(f"✓ Sequential workflow complete!")

# =============================================================================
# Part 2: Parallel Workflows
# =============================================================================

print("\n--- Part 2: Parallel Execution ---")

from agents.linear_algebra_agent import LinearAlgebraAgent

# Scenario: Solve multiple independent linear systems in parallel

linalg = LinearAlgebraAgent()

def create_random_system(size, seed):
    """Create a random linear system."""
    np.random.seed(seed)
    A = np.random.randn(size, size)
    b = np.random.randn(size)
    return A, b

# Create workflow with independent steps
sizes = [50, 100, 150, 200]
steps = []

for i, size in enumerate(sizes):
    A, b = create_random_system(size, seed=i)
    agent = LinearAlgebraAgent()  # Fresh agent for each step

    steps.append(WorkflowStep(
        step_id=f'solve_{size}x{size}',
        agent=agent,
        method='process',
        inputs={
            'task': 'solve_linear_system',
            'A': A,
            'b': b
        }
    ))

# Execute in parallel
print("Executing parallel workflow...")
start = time.time()
result = orchestrator.execute_workflow(steps, parallel=True)
parallel_time = time.time() - start

if result.success:
    print(f"✓ Parallel execution completed in {parallel_time:.3f}s")
    for size in sizes:
        step_result = result.results[f'solve_{size}x{size}']
        print(f"  ✓ Solved {size}x{size} system")

# Compare with sequential execution
print("\nComparing with sequential execution...")
start = time.time()
result_seq = orchestrator.execute_workflow(steps, parallel=False)
sequential_time = time.time() - start

print(f"Sequential time: {sequential_time:.3f}s")
print(f"Parallel time: {parallel_time:.3f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")

# =============================================================================
# Part 3: Error Handling
# =============================================================================

print("\n--- Part 3: Error Handling in Workflows ---")

# Scenario: Workflow with a step that might fail

def sometimes_fails(fail_probability=0.5):
    """Function that randomly fails."""
    if np.random.rand() < fail_probability:
        raise ValueError("Random failure!")
    return 42

class DummyAgent:
    def __init__(self, name):
        self.name = name

    def process(self, data):
        from dataclasses import dataclass

        @dataclass
        class Result:
            success: bool
            data: dict = None
            errors: list = None

        try:
            result = data['function']()
            return Result(success=True, data={'result': result})
        except Exception as e:
            return Result(success=False, errors=[str(e)])

# Create workflow with potential failure
agent1 = DummyAgent('reliable')
agent2 = DummyAgent('unreliable')
agent3 = DummyAgent('backup')

workflow_steps = [
    WorkflowStep(
        step_id='step1',
        agent=agent1,
        method='process',
        inputs={'function': lambda: 10}
    ),
    WorkflowStep(
        step_id='step2_may_fail',
        agent=agent2,
        method='process',
        inputs={'function': lambda: sometimes_fails(0.7)}
    ),
    WorkflowStep(
        step_id='step3',
        agent=agent3,
        method='process',
        inputs={'function': lambda: 20}
    )
]

result = orchestrator.execute_workflow(workflow_steps)

if result.success:
    print("✓ All steps succeeded")
else:
    print("⚠ Workflow had failures:")
    for step_id, step_result in result.results.items():
        if hasattr(step_result, 'success') and not step_result.success:
            print(f"  ✗ {step_id} failed: {step_result.errors}")
        else:
            print(f"  ✓ {step_id} succeeded")

# =============================================================================
# Part 4: Complex Workflow Pattern
# =============================================================================

print("\n--- Part 4: Multi-Stage Analysis Pipeline ---")

# Scenario: Parameter sweep → Select best → Detailed analysis

from agents.integration_agent import IntegrationAgent

# Stage 1: Parameter sweep (parallel)
print("Stage 1: Parameter sweep...")

param_values = np.linspace(0.5, 2.0, 5)
sweep_results = {}

for param in param_values:
    def objective(x):
        return (x[0] - param)**2

    opt = OptimizationAgent()
    result = opt.process({
        'task': 'minimize',
        'function': objective,
        'x0': [0],
        'method': 'L-BFGS-B'
    })

    if result.success:
        sweep_results[param] = result.data['fun']

print(f"  ✓ Tested {len(param_values)} parameter values")

# Stage 2: Select best parameter
best_param = min(sweep_results, key=sweep_results.get)
print(f"\nStage 2: Best parameter = {best_param:.2f}")

# Stage 3: Detailed analysis with best parameter
print(f"\nStage 3: Detailed analysis...")

def detailed_function(t, y):
    return -best_param * y

detailed_solver = ODEPDESolverAgent()
result = detailed_solver.process({
    'task': 'solve_ode',
    'equation': detailed_function,
    'initial_conditions': [1.0],
    't_span': (0, 5),
    't_eval': np.linspace(0, 5, 100)
})

if result.success:
    print(f"  ✓ Detailed simulation complete")
    print(f"  Final state: {result.data['y'][-1][0]:.6f}")

# =============================================================================
# Part 5: Performance Optimization Tips
# =============================================================================

print("\n--- Part 5: Performance Tips ---")

from agents.performance_profiler_agent import PerformanceProfilerAgent

profiler = PerformanceProfilerAgent()

# Tip 1: Use vectorization
def slow_version(n):
    """Slow: loop-based computation"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def fast_version(n):
    """Fast: vectorized computation"""
    return np.sum(np.arange(n) ** 2)

# Profile both
n = 100000

slow_result = profiler.process({
    'task': 'profile_function',
    'function': slow_version,
    'args': [n]
})

fast_result = profiler.process({
    'task': 'profile_function',
    'function': fast_version,
    'args': [n]
})

if slow_result.success and fast_result.success:
    slow_time = slow_result.data['total_time']
    fast_time = fast_result.data['total_time']
    print(f"Slow version: {slow_time:.6f}s")
    print(f"Fast version: {fast_time:.6f}s")
    print(f"Speedup: {slow_time/fast_time:.1f}x")

# Tip 2: Batch processing
print("\nTip 2: Batch similar operations")
print("  → Group similar computations")
print("  → Use parallel execution for independent tasks")
print("  → Cache repeated calculations")

# Tip 3: Profile before optimizing
print("\nTip 3: Always profile first!")
print("  → Use PerformanceProfilerAgent to find bottlenecks")
print("  → Optimize the slowest parts first")
print("  → Measure improvements")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("Tutorial Complete! 🎉")
print("=" * 60)
print("\nWhat you learned:")
print("  ✓ Sequential workflows with dependencies")
print("  ✓ Parallel execution for speedup")
print("  ✓ Error handling in workflows")
print("  ✓ Multi-stage analysis pipelines")
print("  ✓ Performance optimization tips")
print("\nNext steps:")
print("  → Explore real examples in examples/")
print("  → Create workflows for your own problems")
print("  → Share your workflows with the community")
print("=" * 60)
