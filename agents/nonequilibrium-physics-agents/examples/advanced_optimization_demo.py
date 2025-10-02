"""Demonstrations of Advanced Optimization Methods.

Shows practical applications of constrained optimization, global search,
derivative-free methods, and mixed-integer optimal control.

Author: Nonequilibrium Physics Agents
Week: 25-26 of Phase 4
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.advanced_optimization import (
    SequentialQuadraticProgramming,
    AugmentedLagrangian,
    GeneticAlgorithm,
    SimulatedAnnealing,
    CMAES,
    MixedIntegerOptimization,
    ConstrainedOptimizationConfig,
    GlobalOptimizationConfig,
    MixedIntegerConfig
)


def demo_1_sqp_constrained():
    """Demo 1: Sequential Quadratic Programming for constrained control."""
    print("\n" + "="*70)
    print("DEMO 1: Sequential Quadratic Programming (SQP)")
    print("="*70)

    print("\nScenario: Minimize control effort with state constraints")
    print("  Objective: min ||u||²")
    print("  Constraints: x_final = x_target, |u| ≤ u_max")

    # Simple 1D control problem
    def objective(u):
        # Control effort over 5 time steps
        return np.sum(u**2)

    def gradient(u):
        return 2 * u

    # Constraint: final state reaches target (simplified)
    # x_final = x0 + sum(u) = 5.0
    x0 = 0.0
    x_target = 5.0

    equality_constraints = [{
        'fun': lambda u: x0 + np.sum(u) - x_target,
        'jac': lambda u: np.ones_like(u)
    }]

    # Bounds: |u_i| ≤ 2
    n_steps = 5
    bounds = [(-2.0, 2.0)] * n_steps

    config = ConstrainedOptimizationConfig()
    sqp = SequentialQuadraticProgramming(config)

    u0 = np.ones(n_steps)
    result = sqp.solve(objective, u0, gradient=gradient,
                      equality_constraints=equality_constraints,
                      bounds=bounds)

    print(f"\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Optimal control: {result['x']}")
    print(f"  Control effort: {result['fun']:.4f}")
    print(f"  Constraint violation: {result['constraint_violation']:.2e}")

    # Verify constraint
    x_final = x0 + np.sum(result['x'])
    print(f"  Final state: {x_final:.4f} (target: {x_target})")

    print("\n→ SQP handles equality and inequality constraints efficiently")
    print("→ Converges quadratically near solution")


def demo_2_augmented_lagrangian():
    """Demo 2: Augmented Lagrangian for penalty-based optimization."""
    print("\n" + "="*70)
    print("DEMO 2: Augmented Lagrangian Method")
    print("="*70)

    print("\nScenario: Optimize with difficult constraints")
    print("  Objective: min (x-2)² + (y-3)²")
    print("  Constraint: x² + y² = 4 (circle)")

    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 3)**2

    equality_constraints = [lambda x: x[0]**2 + x[1]**2 - 4]

    config = ConstrainedOptimizationConfig(penalty_initial=10.0, max_iterations=100)
    al = AugmentedLagrangian(config)

    x0 = np.array([1.0, 1.0])
    result = al.solve(objective, x0, equality_constraints=equality_constraints)

    print(f"\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Optimal point: {result['x']}")
    print(f"  Objective value: {result['fun']:.4f}")
    print(f"  Iterations: {result['iterations']}")

    # Verify constraint
    constraint_val = result['x'][0]**2 + result['x'][1]**2
    print(f"  x² + y² = {constraint_val:.4f} (target: 4.0)")

    print("\n→ Augmented Lagrangian combines penalties and multipliers")
    print("→ Better conditioning than pure penalty methods")


def demo_3_genetic_algorithm():
    """Demo 3: Genetic algorithm for global search."""
    print("\n" + "="*70)
    print("DEMO 3: Genetic Algorithm")
    print("="*70)

    print("\nScenario: Find global optimum of multimodal function")
    print("  Function: Ackley function (many local minima)")
    print("  Global minimum: (0, 0)")

    # Ackley function
    def ackley(x):
        a, b, c = 20, 0.2, 2*np.pi
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.exp(1)

    bounds = [(-5, 5), (-5, 5)]

    config = GlobalOptimizationConfig(
        population_size=50,
        num_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.2
    )

    ga = GeneticAlgorithm(config)

    print(f"\nRunning GA:")
    print(f"  Population: {config.population_size}")
    print(f"  Generations: {config.num_generations}")

    result = ga.solve(ackley, bounds)

    print(f"\nResults:")
    print(f"  Best solution: {result['x']}")
    print(f"  Best fitness: {result['fun']:.6f}")
    print(f"  Distance from global optimum: {np.linalg.norm(result['x']):.4f}")

    print("\n→ GA explores via crossover and mutation")
    print("→ Good for multimodal, non-differentiable functions")
    print("→ No gradient information needed")


def demo_4_simulated_annealing():
    """Demo 4: Simulated annealing for combinatorial optimization."""
    print("\n" + "="*70)
    print("DEMO 4: Simulated Annealing")
    print("="*70)

    print("\nScenario: Escape local minima via probabilistic acceptance")
    print("  Function: Multiple wells")

    # Function with local minima
    def objective(x):
        return (x[0]**2 - 5)**2 + (x[1]**2 - 5)**2 + 2*np.sin(3*x[0]) + 2*np.cos(3*x[1])

    x0 = np.array([1.0, 1.0])  # Start near local minimum
    bounds = [(-5, 5), (-5, 5)]

    config = GlobalOptimizationConfig(
        initial_temperature=10.0,
        cooling_rate=0.90,
        num_iterations_per_temp=100
    )

    sa = SimulatedAnnealing(config)

    print(f"\nConfiguration:")
    print(f"  Initial temp: {config.initial_temperature}")
    print(f"  Cooling rate: {config.cooling_rate}")
    print(f"  Iterations per temp: {config.num_iterations_per_temp}")

    result = sa.solve(objective, x0, bounds)

    print(f"\nResults:")
    print(f"  Final solution: {result['x']}")
    print(f"  Final value: {result['fun']:.4f}")
    print(f"  Total iterations: {result['iterations']}")

    # Compare to initial
    f_initial = objective(x0)
    print(f"  Improvement: {f_initial - result['fun']:.4f}")

    print("\n→ SA accepts worse solutions with probability exp(-ΔE/T)")
    print("→ Temperature decreases → less exploration over time")
    print("→ Good for discrete and continuous problems")


def demo_5_cma_es():
    """Demo 5: CMA-ES for derivative-free optimization."""
    print("\n" + "="*70)
    print("DEMO 5: CMA-ES (Covariance Matrix Adaptation)")
    print("="*70)

    print("\nScenario: Optimize without gradients")
    print("  Function: Rosenbrock (banana-shaped valley)")

    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    x0 = np.array([-1.0, -1.0])

    config = GlobalOptimizationConfig(
        population_size=20,
        num_generations=100,
        cma_sigma=0.5
    )

    cma = CMAES(config)

    print(f"\nConfiguration:")
    print(f"  Population: {config.population_size}")
    print(f"  Generations: {config.num_generations}")
    print(f"  Initial sigma: {config.cma_sigma}")

    result = cma.solve(rosenbrock, x0)

    print(f"\nResults:")
    print(f"  Optimal point: {result['x']}")
    print(f"  Optimal value: {result['fun']:.6f}")
    print(f"  True optimum: [1, 1]")
    print(f"  Distance: {np.linalg.norm(result['x'] - np.array([1, 1])):.4f}")

    print("\n→ CMA-ES adapts covariance matrix to function landscape")
    print("→ Very effective for ill-conditioned problems")
    print("→ State-of-the-art derivative-free method")


def demo_6_mixed_integer():
    """Demo 6: Mixed-integer optimization via branch-and-bound."""
    print("\n" + "="*70)
    print("DEMO 6: Mixed-Integer Optimization")
    print("="*70)

    print("\nScenario: Discrete control actions")
    print("  Problem: Choose gear (integer 1-5) and throttle (continuous 0-1)")
    print("  Objective: Minimize fuel consumption")

    # Simplified fuel model: fuel = gear² + 0.1*throttle² + penalty for mismatch
    def fuel_consumption(x):
        gear = x[0]
        throttle = x[1]

        # Fuel increases with gear²
        fuel = gear**2

        # Throttle cost
        fuel += 0.1 * throttle**2

        # Penalty if gear and throttle mismatch
        optimal_throttle_for_gear = gear / 5.0
        mismatch = (throttle - optimal_throttle_for_gear)**2
        fuel += 10 * mismatch

        return fuel

    x0 = np.array([3.0, 0.5])
    bounds = [(1.0, 5.0), (0.0, 1.0)]

    config = MixedIntegerConfig(
        integer_variables=[0],  # Gear is integer
        max_nodes=50
    )

    mip = MixedIntegerOptimization(config)

    print(f"\nSolving mixed-integer problem:")
    print(f"  Variables: gear (integer), throttle (continuous)")

    result = mip.solve(fuel_consumption, x0, bounds)

    if result['success']:
        print(f"\nResults:")
        print(f"  Optimal gear: {int(result['x'][0])}")
        print(f"  Optimal throttle: {result['x'][1]:.4f}")
        print(f"  Fuel consumption: {result['fun']:.4f}")
        print(f"  Nodes explored: {result['nodes_explored']}")

        print("\n→ Branch-and-bound handles discrete variables")
        print("→ Relaxation + branching on fractional values")
        print("→ Applications: mode switching, discrete actuators")
    else:
        print("  Branch-and-bound did not converge (increase max_nodes)")


def demo_7_comparison():
    """Demo 7: Compare methods on same problem."""
    print("\n" + "="*70)
    print("DEMO 7: Method Comparison")
    print("="*70)

    print("\nScenario: Minimize test function with all methods")
    print("  Function: f(x,y) = x² + 2y² - 0.3cos(3πx) - 0.4cos(4πy) + 0.7")

    def objective(x):
        return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - \
               0.4*np.cos(4*np.pi*x[1]) + 0.7

    bounds = [(-2, 2), (-2, 2)]

    methods = []

    # 1. Gradient-based (L-BFGS-B)
    from scipy.optimize import minimize
    result_lbfgs = minimize(objective, [1, 1], method='L-BFGS-B', bounds=bounds)
    methods.append(("L-BFGS-B (gradient)", result_lbfgs.x, result_lbfgs.fun))

    # 2. SQP (unconstrained)
    config_sqp = ConstrainedOptimizationConfig()
    sqp = SequentialQuadraticProgramming(config_sqp)
    result_sqp = sqp.solve(objective, np.array([1, 1]), bounds=bounds)
    methods.append(("SQP (constrained solver)", result_sqp['x'], result_sqp['fun']))

    # 3. Genetic Algorithm
    config_ga = GlobalOptimizationConfig(population_size=30, num_generations=50)
    ga = GeneticAlgorithm(config_ga)
    result_ga = ga.solve(objective, bounds)
    methods.append(("Genetic Algorithm", result_ga['x'], result_ga['fun']))

    # 4. Simulated Annealing
    config_sa = GlobalOptimizationConfig(initial_temperature=5.0, cooling_rate=0.9)
    sa = SimulatedAnnealing(config_sa)
    result_sa = sa.solve(objective, np.array([1, 1]), bounds)
    methods.append(("Simulated Annealing", result_sa['x'], result_sa['fun']))

    # 5. CMA-ES
    config_cma = GlobalOptimizationConfig(population_size=15, num_generations=40)
    cma = CMAES(config_cma)
    result_cma = cma.solve(objective, np.array([1, 1]))
    methods.append(("CMA-ES", result_cma['x'], result_cma['fun']))

    print(f"\nResults Comparison:")
    print(f"{'Method':<25} {'Solution':<20} {'Value':<10}")
    print("-" * 60)

    for method, x, f in methods:
        print(f"{method:<25} [{x[0]:>6.3f}, {x[1]:>6.3f}]      {f:<10.6f}")

    best_method = min(methods, key=lambda m: m[2])
    print(f"\nBest: {best_method[0]} (f = {best_method[2]:.6f})")

    print("\nKey Insights:")
    print("  → Gradient methods fast but may find local minimum")
    print("  → Global methods slower but more robust")
    print("  → CMA-ES often best derivative-free method")
    print("  → GA good for discrete/combinatorial problems")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*15 + "ADVANCED OPTIMIZATION DEMONSTRATIONS")
    print("="*70)

    # Run demos
    demo_1_sqp_constrained()
    demo_2_augmented_lagrangian()
    demo_3_genetic_algorithm()
    demo_4_simulated_annealing()
    demo_5_cma_es()
    demo_6_mixed_integer()
    demo_7_comparison()

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
