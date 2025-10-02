#!/usr/bin/env python3
"""
Tutorial 01: Quick Start with Scientific Computing Agents

This tutorial walks you through the basics of using the system.
You'll learn to:
1. Create and use individual agents
2. Handle results
3. Understand common patterns

Time: 10 minutes
Prerequisites: Basic Python, NumPy
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("Tutorial 01: Quick Start")
print("=" * 60)

# =============================================================================
# Part 1: Your First Agent - ODE Solver
# =============================================================================

print("\n--- Part 1: Solving a Simple ODE ---")

from agents.ode_pde_solver_agent import ODEPDESolverAgent

# Create the agent
solver = ODEPDESolverAgent()
print("✓ Created ODEPDESolverAgent")

# Define a simple ODE: dy/dt = -2y (exponential decay)
def exponential_decay(t, y):
    """Rate of change: dy/dt = -2y"""
    return -2 * y

# Solve the ODE
result = solver.process({
    'task': 'solve_ode',
    'equation': exponential_decay,
    'initial_conditions': [1.0],  # y(0) = 1
    't_span': (0, 3),              # Solve from t=0 to t=3
    't_eval': np.linspace(0, 3, 50)  # 50 points for smooth plot
})

# Check if successful
if result.success:
    print("✓ ODE solved successfully!")
    print(f"  Initial value: {result.data['y'][0][0]:.4f}")
    print(f"  Final value: {result.data['y'][-1][0]:.4f}")
    print(f"  Expected final: {np.exp(-2*3):.4f} (analytical solution)")
else:
    print(f"✗ Error: {result.errors}")
    exit(1)

# =============================================================================
# Part 2: Optimization Agent
# =============================================================================

print("\n--- Part 2: Finding a Minimum ---")

from agents.optimization_agent import OptimizationAgent

# Create the agent
optimizer = OptimizationAgent()
print("✓ Created OptimizationAgent")

# Define a simple quadratic function: f(x,y) = (x-3)^2 + (y+2)^2
# Minimum is at (3, -2) with value 0
def quadratic(x):
    """Quadratic function with minimum at (3, -2)"""
    return (x[0] - 3)**2 + (x[1] + 2)**2

# Find the minimum
result = optimizer.process({
    'task': 'minimize',
    'function': quadratic,
    'x0': [0, 0],  # Start at origin
    'method': 'L-BFGS-B'
})

if result.success:
    print("✓ Optimization successful!")
    print(f"  Minimum at: ({result.data['x'][0]:.4f}, {result.data['x'][1]:.4f})")
    print(f"  Function value: {result.data['fun']:.6f}")
    print(f"  Iterations: {result.data['nit']}")
else:
    print(f"✗ Error: {result.errors}")

# =============================================================================
# Part 3: Linear Algebra Agent
# =============================================================================

print("\n--- Part 3: Solving Linear Systems ---")

from agents.linear_algebra_agent import LinearAlgebraAgent

# Create the agent
linalg = LinearAlgebraAgent()
print("✓ Created LinearAlgebraAgent")

# Define a linear system: Ax = b
# Example: 2x + 3y = 8
#          1x + 4y = 10
A = np.array([[2, 3],
              [1, 4]])
b = np.array([8, 10])

result = linalg.process({
    'task': 'solve_linear_system',
    'A': A,
    'b': b
})

if result.success:
    print("✓ Linear system solved!")
    print(f"  Solution: x = {result.data['x'][0]:.4f}, y = {result.data['x'][1]:.4f}")
    # Verify: Ax should equal b
    print(f"  Verification: Ax - b = {np.linalg.norm(A @ result.data['x'] - b):.2e}")
else:
    print(f"✗ Error: {result.errors}")

# =============================================================================
# Part 4: Performance Profiling
# =============================================================================

print("\n--- Part 4: Profiling Performance ---")

from agents.performance_profiler_agent import PerformanceProfilerAgent

# Create the agent
profiler = PerformanceProfilerAgent()
print("✓ Created PerformanceProfilerAgent")

# Define a function to profile
def compute_intensive(n):
    """A somewhat expensive computation"""
    total = 0
    for i in range(n):
        total += np.sum(np.random.randn(100))
    return total

result = profiler.process({
    'task': 'profile_function',
    'function': compute_intensive,
    'args': [1000]
})

if result.success:
    print("✓ Profiling complete!")
    print(f"  Execution time: {result.data['total_time']:.4f} seconds")
    print(f"  Function returned: {result.data['result']:.2f}")
else:
    print(f"✗ Error: {result.errors}")

# =============================================================================
# Part 5: Workflow Orchestration
# =============================================================================

print("\n--- Part 5: Creating a Workflow ---")

from agents.workflow_orchestration_agent import WorkflowOrchestrationAgent, WorkflowStep

# Create the orchestrator
orchestrator = WorkflowOrchestrationAgent()
print("✓ Created WorkflowOrchestrationAgent")

# Create a workflow: Solve ODE, then optimize based on result
ode_solver = ODEPDESolverAgent()
opt_agent = OptimizationAgent()

steps = [
    WorkflowStep(
        step_id='solve_ode',
        agent=ode_solver,
        method='process',
        inputs={
            'task': 'solve_ode',
            'equation': lambda t, y: -y,
            'initial_conditions': [1.0],
            't_span': (0, 2)
        }
    ),
    WorkflowStep(
        step_id='optimize',
        agent=opt_agent,
        method='process',
        inputs={
            'task': 'minimize',
            'function': lambda x: (x[0] - 1)**2,
            'x0': [0],
            'method': 'L-BFGS-B'
        }
    )
]

result = orchestrator.execute_workflow(steps)

if result.success:
    print("✓ Workflow executed successfully!")
    print(f"  ODE solution computed: {result.results['solve_ode'].success}")
    print(f"  Optimization completed: {result.results['optimize'].success}")
else:
    print(f"✗ Workflow failed: {result.errors}")

# =============================================================================
# Visualization (Optional)
# =============================================================================

print("\n--- Visualizing Results ---")

# Plot the ODE solution from Part 1
from agents.ode_pde_solver_agent import ODEPDESolverAgent

solver = ODEPDESolverAgent()
result = solver.process({
    'task': 'solve_ode',
    'equation': exponential_decay,
    'initial_conditions': [1.0],
    't_span': (0, 3),
    't_eval': np.linspace(0, 3, 50)
})

if result.success:
    plt.figure(figsize=(10, 6))

    # Plot numerical solution
    plt.plot(result.data['t'], result.data['y'][:, 0],
             'b-', linewidth=2, label='Numerical solution')

    # Plot analytical solution
    t_analytical = result.data['t']
    y_analytical = np.exp(-2 * t_analytical)
    plt.plot(t_analytical, y_analytical,
             'r--', linewidth=2, label='Analytical solution')

    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('y(t)', fontsize=12)
    plt.title('Exponential Decay: dy/dt = -2y', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.savefig('tutorial_01_output.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved to: tutorial_01_output.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 60)
print("Tutorial Complete! 🎉")
print("=" * 60)
print("\nWhat you learned:")
print("  ✓ How to create and use agents")
print("  ✓ How to handle results")
print("  ✓ How to profile performance")
print("  ✓ How to create workflows")
print("\nNext steps:")
print("  → Try tutorial_02_advanced_workflows.py")
print("  → Explore examples/ directory")
print("  → Read docs/USER_ONBOARDING.md")
print("=" * 60)
