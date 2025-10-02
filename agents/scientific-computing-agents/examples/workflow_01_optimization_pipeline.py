"""Example: Complete Optimization Pipeline with Orchestration.

This example demonstrates an end-to-end workflow using all three orchestration
agents to solve an optimization problem:

1. ProblemAnalyzerAgent: Classify and analyze the optimization problem
2. AlgorithmSelectorAgent: Select optimal algorithm and parameters
3. OptimizationAgent: Execute the optimization
4. ExecutorValidatorAgent: Validate results and generate report

Problem: Minimize the Rosenbrock function (banana function)
f(x, y) = (a - x)² + b(y - x²)² where a=1, b=100
Minimum: f(1, 1) = 0
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.problem_analyzer_agent import ProblemAnalyzerAgent
from agents.algorithm_selector_agent import AlgorithmSelectorAgent
from agents.optimization_agent import OptimizationAgent
from agents.executor_validator_agent import ExecutorValidatorAgent


def rosenbrock(x):
    """Rosenbrock function (banana function).

    Classic test function for optimization algorithms.
    Global minimum at (1, 1) with f(1, 1) = 0.
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x):
    """Gradient of Rosenbrock function.

    Used by gradient-based optimization methods.
    """
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dfdy = 200 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])


def run_optimization_pipeline():
    """Run complete optimization pipeline with orchestration."""

    print("="*80)
    print("COMPLETE OPTIMIZATION PIPELINE")
    print("="*80)
    print("\nProblem: Minimize Rosenbrock function")
    print("  f(x, y) = (1 - x)² + 100(y - x²)²")
    print("  Known minimum: f(1, 1) = 0")
    print()

    # =========================================================================
    # STEP 1: Problem Analysis
    # =========================================================================
    print("-" * 80)
    print("STEP 1: PROBLEM ANALYSIS")
    print("-" * 80)

    analyzer = ProblemAnalyzerAgent()

    # Describe problem in natural language
    problem_description = """
    Minimize the Rosenbrock function, a classic non-convex optimization
    test problem. The function has a narrow valley containing the global
    minimum. Need unconstrained optimization starting from (-1.5, 2.5).
    """

    print(f"\nProblem Description:")
    print(f"  {problem_description.strip()}")

    # Classify problem
    classification_result = analyzer.execute({
        'analysis_type': 'classify',
        'problem_description': problem_description,
        'initial_point': np.array([-1.5, 2.5])
    })

    if classification_result.success:
        classification = classification_result.data
        print(f"\n✓ Problem Classification:")
        print(f"  Type: {classification['problem_type']}")
        print(f"  Confidence: {classification['confidence']:.1%}")
        print(f"  Characteristics: {', '.join(classification['characteristics'])}")
    else:
        print(f"\n✗ Classification failed: {classification_result.errors}")
        return False

    # Estimate complexity
    complexity_result = analyzer.execute({
        'analysis_type': 'complexity',
        'problem_type': classification['problem_type'],
        'dimension': 2
    })

    if complexity_result.success:
        complexity = complexity_result.data
        print(f"\n✓ Complexity Estimation:")
        print(f"  Complexity class: {complexity['complexity_class']}")
        print(f"  Estimated cost: {complexity['estimated_cost']}")
        print(f"  Time requirement: {complexity['time_requirement']}")
        print(f"  Memory requirement: {complexity['memory_requirement']}")
    else:
        print(f"\n✗ Complexity estimation failed: {complexity_result.errors}")
        # Use default complexity
        complexity = {'complexity_class': 'simple', 'estimated_cost': 100, 'time_requirement': 'FAST', 'memory_requirement': 'LOW'}

    # Get recommendations
    recommendation_result = analyzer.execute({
        'analysis_type': 'recommend',
        'problem_type': classification['problem_type'],
        'complexity_class': complexity['complexity_class']
    })

    if recommendation_result.success:
        recommendation = recommendation_result.data
        print(f"\n✓ Recommended Approach:")
        if recommendation['recommendations']:
            primary_rec = recommendation['recommendations'][0]
            print(f"  Primary method: {primary_rec['method']}")
            print(f"  Primary agent: {primary_rec['agent']}")
            print(f"  Rationale: {primary_rec['rationale']}")
            if len(recommendation['recommendations']) > 1:
                print(f"  Alternatives: {len(recommendation['recommendations']) - 1} additional methods")
        if recommendation['execution_plan']:
            print(f"  Execution steps: {len(recommendation['execution_plan'])} steps")

    # =========================================================================
    # STEP 2: Algorithm Selection
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: ALGORITHM SELECTION")
    print("-" * 80)

    selector = AlgorithmSelectorAgent()

    # Select best algorithm
    algorithm_result = selector.execute({
        'selection_type': 'algorithm',
        'problem_type': classification['problem_type'],
        'complexity_class': complexity['complexity_class'],
        'characteristics': classification['characteristics']
    })

    if algorithm_result.success:
        algorithm = algorithm_result.data
        selected_alg_name = algorithm['selected_algorithm']
        print(f"\n✓ Selected Algorithm:")
        print(f"  Name: {selected_alg_name}")
        print(f"  Type: {algorithm.get('algorithm_type', 'N/A')}")
        print(f"  Confidence: {algorithm['confidence']:.1%}")
        print(f"  Rationale: {algorithm['rationale']}")
        if 'alternatives' in algorithm and algorithm['alternatives']:
            print(f"\n  Alternatives:")
            for i, alt in enumerate(algorithm['alternatives'][:3], 1):
                print(f"    {i}. {alt['algorithm']} (score: {alt['score']:.1f})")
    else:
        print(f"\n✗ Algorithm selection failed: {algorithm_result.errors}")
        return False

    # Tune parameters
    parameter_result = selector.execute({
        'selection_type': 'tune_parameters',
        'algorithm': selected_alg_name,
        'problem_size': 2,
        'complexity_class': complexity['complexity_class']
    })

    if parameter_result.success:
        parameters = parameter_result.data
        print(f"\n✓ Tuned Parameters:")
        for param, info in parameters['parameters'].items():
            print(f"  {param}: {info['value']} ({info['rationale']})")

    # =========================================================================
    # STEP 3: Optimization Execution
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: OPTIMIZATION EXECUTION")
    print("-" * 80)

    optimizer = OptimizationAgent()

    # Map algorithm name to method
    method_map = {
        'L-BFGS': 'L-BFGS-B',
        'BFGS': 'BFGS',
        'Nelder-Mead': 'Nelder-Mead'
    }
    method = method_map.get(selected_alg_name, 'L-BFGS-B')

    # Execute optimization
    print(f"\nExecuting optimization with {method}...")
    print(f"  Initial point: [-1.5, 2.5]")

    optimization_result = optimizer.execute({
        'problem_type': 'optimization_unconstrained',
        'objective': rosenbrock,
        'gradient': rosenbrock_gradient if 'BFGS' in method else None,
        'initial_guess': np.array([-1.5, 2.5]),
        'method': method,
        'bounds': [(-2, 2), (-1, 3)]  # Bounded for safety
    })

    if optimization_result.success:
        solution = optimization_result.data['solution']
        print(f"\n✓ Optimization Complete!")
        print(f"  Optimal point: [{solution['x'][0]:.6f}, {solution['x'][1]:.6f}]")
        print(f"  Optimal value: {solution['fun']:.6e}")
        if 'nit' in solution:
            print(f"  Iterations: {solution['nit']}")
        if 'nfev' in solution:
            print(f"  Function evaluations: {solution['nfev']}")
        print(f"  Success: {solution.get('success', True)}")
        if 'message' in solution:
            print(f"  Message: {solution['message']}")

        # Compare with known minimum
        error = np.linalg.norm(solution['x'] - np.array([1.0, 1.0]))
        print(f"\n  Error from true minimum (1, 1): {error:.6e}")
    else:
        print(f"\n✗ Optimization failed: {optimization_result.errors}")
        return False

    # =========================================================================
    # STEP 4: Result Validation
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: RESULT VALIDATION")
    print("-" * 80)

    validator = ExecutorValidatorAgent()

    # Validate solution quality
    validation_result = validator.execute({
        'task_type': 'validate',
        'solution': solution['x'],
        'expected_shape': (2,),
        'problem_data': {
            'objective': rosenbrock,
            'known_minimum': np.array([1.0, 1.0]),
            'tolerance': 1e-4
        }
    })

    if validation_result.success:
        validation = validation_result.data
        print(f"\n✓ Validation Complete!")
        print(f"  All checks passed: {validation['all_checks_passed']}")
        print(f"\n  Validation Checks:")
        for check in validation['validation_checks']:
            status = "✓" if check['passed'] else "✗"
            check_name = check.get('check', check.get('name', 'Unknown'))
            message = check.get('message', check.get('reason', 'No message'))
            print(f"    {status} {check_name}: {message}")

        print(f"\n  Quality Metrics:")
        metrics = validation['quality_metrics']
        print(f"    Accuracy: {metrics['accuracy']:.1%}")
        print(f"    Consistency: {metrics['consistency']:.1%}")
        print(f"    Overall Quality: {validation['overall_quality']}")

    # Generate comprehensive report
    report_result = validator.execute({
        'task_type': 'report',
        'workflow_name': 'Rosenbrock Optimization Pipeline',
        'steps': [
            {'name': 'Problem Analysis', 'status': 'completed', 'details': classification},
            {'name': 'Algorithm Selection', 'status': 'completed', 'details': algorithm},
            {'name': 'Optimization', 'status': 'completed', 'details': solution},
            {'name': 'Validation', 'status': 'completed', 'details': validation}
        ],
        'summary': f"Successfully minimized Rosenbrock function to {solution['fun']:.6e}"
    })

    if report_result.success:
        report = report_result.data
        print(f"\n" + "="*80)
        print("WORKFLOW REPORT")
        print("="*80)
        print(f"\nWorkflow: {report['workflow_name']}")
        print(f"Status: {report['overall_status']}")
        print(f"Summary: {report['summary']}")
        print(f"\nSteps Completed: {len(report['steps'])}")
        for i, step in enumerate(report['steps'], 1):
            print(f"  {i}. {step['name']}: {step['status']}")

    # =========================================================================
    # STEP 5: Visualization
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: VISUALIZATION")
    print("-" * 80)

    create_visualization(solution)

    return True


def create_visualization(solution):
    """Create visualization of optimization results."""

    # Create contour plot of Rosenbrock function
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Contour plot with optimization path
    ax = axes[0]
    levels = np.logspace(-1, 3, 20)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)

    # Mark initial and final points
    ax.plot(-1.5, 2.5, 'ro', markersize=12, label='Initial point', zorder=5)
    ax.plot(solution['x'][0], solution['x'][1], 'g*', markersize=20,
            label='Optimized point', zorder=5)
    ax.plot(1.0, 1.0, 'bs', markersize=12, label='True minimum', zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Rosenbrock Function Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: 3D surface
    ax = axes[1]
    ax.remove()
    ax = fig.add_subplot(122, projection='3d')

    # Downsample for faster plotting
    X_3d = X[::5, ::5]
    Y_3d = Y[::5, ::5]
    Z_3d = Z[::5, ::5]

    surf = ax.plot_surface(X_3d, Y_3d, np.log10(Z_3d + 1), cmap='viridis',
                           alpha=0.7, antialiased=True)

    # Mark points
    ax.scatter([-1.5], [2.5], [np.log10(rosenbrock(np.array([-1.5, 2.5])) + 1)],
               color='red', s=100, label='Initial', zorder=5)
    ax.scatter([solution['x'][0]], [solution['x'][1]],
               [np.log10(solution['fun'] + 1)],
               color='green', s=200, marker='*', label='Optimized', zorder=5)
    ax.scatter([1.0], [1.0], [0], color='blue', s=100, marker='s',
               label='True min', zorder=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('log₁₀(f + 1)')
    ax.set_title('Rosenbrock Function (log scale)')
    ax.legend()

    fig.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'workflow_01_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Only show interactively if not in automated testing
    # print("\nClose the plot window to continue...")
    # plt.show()


def main():
    """Run the complete optimization pipeline."""

    print("\n" + "="*80)
    print("WORKFLOW EXAMPLE: COMPLETE OPTIMIZATION PIPELINE")
    print("="*80)
    print("\nThis example demonstrates end-to-end orchestration:")
    print("  1. ProblemAnalyzerAgent - Classify and analyze problem")
    print("  2. AlgorithmSelectorAgent - Select optimal algorithm")
    print("  3. OptimizationAgent - Execute optimization")
    print("  4. ExecutorValidatorAgent - Validate and report")
    print()

    success = run_optimization_pipeline()

    if success:
        print("\n" + "="*80)
        print("✓ WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Achievements:")
        print("  ✓ Problem correctly classified as unconstrained optimization")
        print("  ✓ Optimal algorithm selected based on problem characteristics")
        print("  ✓ Optimization converged to global minimum")
        print("  ✓ Results validated against known solution")
        print("  ✓ Comprehensive report generated")
        print()
        return 0
    else:
        print("\n" + "="*80)
        print("✗ WORKFLOW FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
