"""Complete End-to-End Workflow Example.

This example demonstrates the full power of the scientific-computing-agents
system, showcasing automatic problem classification, algorithm selection,
execution, and validation.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agents.problem_analyzer_agent import ProblemAnalyzerAgent
from agents.algorithm_selector_agent import AlgorithmSelectorAgent
from agents.linear_algebra_agent import LinearAlgebraAgent
from agents.executor_validator_agent import ExecutorValidatorAgent


def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def example_linear_system():
    """Complete workflow for solving a linear system."""
    print_section("EXAMPLE: Linear System Solving with Full Workflow")

    # Define problem
    print("\n1. Problem Definition:")
    A = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0]
    ])
    b = np.array([1.0, 2.0, 3.0])

    print(f"   Solve Ax = b where:")
    print(f"   A = \n{A}")
    print(f"   b = {b}")

    # Step 1: Classify the problem
    print("\n2. Problem Analysis (ProblemAnalyzerAgent):")
    analyzer = ProblemAnalyzerAgent()

    analysis = analyzer.execute({
        'analysis_type': 'classify',
        'problem_description': 'Solve a linear system Ax=b',
        'problem_data': {'matrix_A': A, 'vector_b': b}
    })

    print(f"   ✓ Problem Type: {analysis.data['problem_type']}")
    print(f"   ✓ Confidence: {analysis.data['confidence']:.2%}")

    # Get complexity estimate
    complexity_result = analyzer.execute({
        'analysis_type': 'complexity',
        'problem_type': analysis.data['problem_type'],
        'problem_data': {'matrix_A': A}
    })

    print(f"   ✓ Complexity: {complexity_result.data['complexity_class']}")
    print(f"   ✓ Estimated Cost: {complexity_result.data['estimated_cost']:.1f}")

    # Step 2: Select optimal algorithm
    print("\n3. Algorithm Selection (AlgorithmSelectorAgent):")
    selector = AlgorithmSelectorAgent()

    algorithm_result = selector.execute({
        'selection_type': 'algorithm',
        'problem_type': analysis.data['problem_type'],
        'complexity_class': complexity_result.data['complexity_class']
    })

    print(f"   ✓ Selected Algorithm: {algorithm_result.data['selected_algorithm']}")
    print(f"   ✓ Algorithm Type: {algorithm_result.data['algorithm_type']}")
    print(f"   ✓ Confidence: {algorithm_result.data['confidence']:.2%}")
    print(f"   ✓ Rationale: {algorithm_result.data['rationale']}")

    if algorithm_result.data['alternatives']:
        print(f"   ✓ Alternatives: {len(algorithm_result.data['alternatives'])} other options")

    # Step 3: Solve the problem
    print("\n4. Problem Solving (LinearAlgebraAgent):")
    solver = LinearAlgebraAgent()

    solution_result = solver.execute({
        'problem_type': 'linear_system_dense',
        'matrix_A': A,
        'vector_b': b,
        'method': 'lu'
    })

    x = solution_result.data['solution']['x']
    print(f"   ✓ Solution found: x = {x}")
    print(f"   ✓ Status: {solution_result.status.value}")

    # Step 4: Validate the solution
    print("\n5. Solution Validation (ExecutorValidatorAgent):")
    validator = ExecutorValidatorAgent()

    validation_result = validator.execute({
        'task_type': 'validate',
        'solution': x,
        'problem_data': {'matrix_A': A, 'vector_b': b}
    })

    print(f"   ✓ All Checks Passed: {validation_result.data['all_checks_passed']}")
    print(f"   ✓ Overall Quality: {validation_result.data['overall_quality']}")

    # Show validation details
    print("\n   Validation Checks:")
    for check in validation_result.data['validation_checks']:
        status = "✓" if check['passed'] else "✗"
        print(f"      {status} {check['check']}: {check['message']}")

    # Show quality metrics
    print("\n   Quality Metrics:")
    for metric, value in validation_result.data['quality_metrics'].items():
        print(f"      • {metric}: {value:.1f}/100")

    # Verify solution manually
    print("\n6. Manual Verification:")
    residual = np.linalg.norm(A @ x - b)
    print(f"   ✓ ||Ax - b|| = {residual:.2e}")
    print(f"   ✓ Residual < 1e-10: {residual < 1e-10}")

    return validation_result.data['all_checks_passed']


def example_workflow_design():
    """Design a multi-agent workflow."""
    print_section("EXAMPLE: Multi-Agent Workflow Design")

    print("\n1. Design Workflow for Optimization Problem:")
    selector = AlgorithmSelectorAgent()

    workflow_result = selector.execute({
        'selection_type': 'workflow',
        'problem_type': 'optimization',
        'complexity_class': 'moderate',
        'requirements': {'uncertainty_quantification': True}
    })

    print(f"   ✓ Total Steps: {workflow_result.data['total_steps']}")
    print(f"   ✓ Estimated Runtime: {workflow_result.data['estimated_runtime']}")

    print("\n   Workflow Steps:")
    for step in workflow_result.data['workflow_steps']:
        print(f"      Step {step['step']}: {step['agent']} - {step['action']}")

    print("\n   Dependencies:")
    for dep in workflow_result.data['dependencies']:
        print(f"      Step {dep['step']} depends on: {dep['depends_on']}")

    return True


def example_convergence_monitoring():
    """Monitor convergence of iterative method."""
    print_section("EXAMPLE: Convergence Monitoring")

    print("\n1. Simulate Iterative Solver Convergence:")
    # Simulate convergence with geometric decay
    residuals = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 1e-6])

    print(f"   Residual History: {residuals[:5]}...")

    validator = ExecutorValidatorAgent()
    convergence_result = validator.execute({
        'task_type': 'check_convergence',
        'residuals': residuals,
        'tolerance': 1e-5,
        'max_iterations': 100
    })

    print(f"\n   ✓ Converged: {convergence_result.data['converged']}")
    print(f"   ✓ Final Residual: {convergence_result.data['final_residual']:.2e}")
    print(f"   ✓ Iterations: {convergence_result.data['iterations']}")
    print(f"   ✓ Convergence Rate: {convergence_result.data['convergence_rate']:.3f}")
    print(f"   ✓ Quality: {convergence_result.data['convergence_quality']}")

    return convergence_result.data['converged']


def example_parameter_tuning():
    """Demonstrate automatic parameter tuning."""
    print_section("EXAMPLE: Automatic Parameter Tuning")

    print("\n1. Tune Parameters for GMRES Solver:")
    selector = AlgorithmSelectorAgent()

    param_result = selector.execute({
        'selection_type': 'parameters',
        'algorithm': 'GMRES',
        'problem_size': 5000,
        'desired_tolerance': 1e-8
    })

    print(f"   ✓ Algorithm: {param_result.data['algorithm']}")
    print(f"\n   Recommended Parameters:")
    for param, value in param_result.data['recommended_parameters'].items():
        print(f"      • {param}: {value}")

    print(f"\n   ✓ Rationale: {param_result.data['tuning_rationale']}")

    return True


def main():
    """Run all examples."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  Scientific Computing Agents - Complete Workflow Demo".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    results = []

    # Run examples
    try:
        results.append(("Linear System Solving", example_linear_system()))
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        results.append(("Linear System Solving", False))

    try:
        results.append(("Workflow Design", example_workflow_design()))
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        results.append(("Workflow Design", False))

    try:
        results.append(("Convergence Monitoring", example_convergence_monitoring()))
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        results.append(("Convergence Monitoring", False))

    try:
        results.append(("Parameter Tuning", example_parameter_tuning()))
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        results.append(("Parameter Tuning", False))

    # Summary
    print_section("SUMMARY")
    print("\nExample Results:")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"   {status}: {name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)

    print(f"\n   Overall: {passed}/{total} examples successful ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n   🎉 All examples completed successfully!")
        print("   🚀 System is ready for production use!")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
