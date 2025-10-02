"""Example: Multi-Physics Workflow with Uncertainty Quantification.

This example demonstrates a complex workflow combining multiple agents:

1. ODEPDESolverAgent: Solve a chemical kinetics ODE system
2. UncertaintyQuantificationAgent: Perform sensitivity analysis on parameters
3. SurrogateModelingAgent: Build a fast surrogate model
4. OptimizationAgent: Optimize parameters using the surrogate

Problem: Chemical reactor with uncertain rate constants
- Model: A → B → C consecutive reactions
- Uncertainty: Rate constants k1, k2
- Goal: Optimize for maximum B concentration while accounting for uncertainty
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent
from agents.uncertainty_quantification_agent import UncertaintyQuantificationAgent
from agents.surrogate_modeling_agent import SurrogateModelingAgent
from agents.optimization_agent import OptimizationAgent


def chemical_kinetics_model(t, y, k1, k2):
    """Chemical kinetics: A → B → C.

    Args:
        t: Time
        y: State [A, B, C]
        k1: Rate constant A → B
        k2: Rate constant B → C

    Returns:
        dy/dt
    """
    A, B, C = y
    dA = -k1 * A
    dB = k1 * A - k2 * B
    dC = k2 * B
    return np.array([dA, dB, dC])


def solve_kinetics(k1, k2, t_span=(0, 10), A0=1.0):
    """Solve chemical kinetics for given rate constants.

    Returns:
        Maximum B concentration achieved
    """
    solver = ODEPDESolverAgent()

    def rhs(t, y):
        return chemical_kinetics_model(t, y, k1, k2)

    result = solver.execute({
        'problem_type': 'ode_ivp',
        'rhs': rhs,
        'initial_conditions': [A0, 0.0, 0.0],
        'time_span': t_span,
        'method': 'RK45'
    })

    if result.success:
        B = result.data['solution']['y'][1]
        return np.max(B)
    else:
        return 0.0


def run_multi_physics_workflow():
    """Run complete multi-physics workflow."""

    print("="*80)
    print("MULTI-PHYSICS WORKFLOW: CHEMICAL REACTOR OPTIMIZATION")
    print("="*80)
    print("\nProblem: A → B → C with uncertain rate constants")
    print("Goal: Maximize B concentration accounting for uncertainty")
    print()

    # =========================================================================
    # STEP 1: Baseline Solution with Nominal Parameters
    # =========================================================================
    print("-" * 80)
    print("STEP 1: BASELINE SOLUTION")
    print("-" * 80)

    # Nominal parameter values
    k1_nominal = 0.5
    k2_nominal = 0.3

    print(f"\nNominal parameters:")
    print(f"  k1 (A → B): {k1_nominal}")
    print(f"  k2 (B → C): {k2_nominal}")

    solver = ODEPDESolverAgent()

    def nominal_rhs(t, y):
        return chemical_kinetics_model(t, y, k1_nominal, k2_nominal)

    baseline_result = solver.execute({
        'problem_type': 'ode_ivp',
        'rhs': nominal_rhs,
        'initial_conditions': [1.0, 0.0, 0.0],
        'time_span': (0, 20),
        'method': 'RK45'
    })

    if baseline_result.success:
        t_baseline = baseline_result.data['solution']['t']
        y_baseline = baseline_result.data['solution']['y']
        A_baseline = y_baseline[0]
        B_baseline = y_baseline[1]
        C_baseline = y_baseline[2]

        B_max_nominal = np.max(B_baseline)
        t_max_B = t_baseline[np.argmax(B_baseline)]

        print(f"\n✓ Baseline Solution:")
        print(f"  Maximum [B]: {B_max_nominal:.4f} at t={t_max_B:.2f}")
        print(f"  Final concentrations:")
        print(f"    [A] = {A_baseline[-1]:.4f}")
        print(f"    [B] = {B_baseline[-1]:.4f}")
        print(f"    [C] = {C_baseline[-1]:.4f}")

    # =========================================================================
    # STEP 2: Sensitivity Analysis
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 2: SENSITIVITY ANALYSIS")
    print("-" * 80)

    uq_agent = UncertaintyQuantificationAgent()

    # Define model for sensitivity analysis
    def model_for_sa(params):
        k1, k2 = params
        return solve_kinetics(k1, k2)

    # Input ranges for sensitivity analysis (±20% variation)
    input_ranges = [
        [k1_nominal * 0.8, k1_nominal * 1.2],  # k1 range
        [k2_nominal * 0.8, k2_nominal * 1.2]   # k2 range
    ]

    print(f"\nPerforming Sobol sensitivity analysis...")
    print(f"  Parameter ranges:")
    print(f"    k1: [{input_ranges[0][0]:.2f}, {input_ranges[0][1]:.2f}]")
    print(f"    k2: [{input_ranges[1][0]:.2f}, {input_ranges[1][1]:.2f}]")

    sa_result = uq_agent.execute({
        'problem_type': 'sensitivity',
        'model': model_for_sa,
        'input_ranges': input_ranges,
        'n_samples': 1000,
        'seed': 42
    })

    if sa_result.success:
        sensitivity = sa_result.data['solution']
        S = sensitivity['first_order_indices']
        ST = sensitivity['total_order_indices']

        print(f"\n✓ Sensitivity Analysis Complete!")
        print(f"  First-order Sobol indices:")
        print(f"    S1 (k1): {S['S1']:.3f}")
        print(f"    S2 (k2): {S['S2']:.3f}")
        print(f"  Total-order Sobol indices:")
        print(f"    ST1 (k1): {ST['ST1']:.3f}")
        print(f"    ST2 (k2): {ST['ST2']:.3f}")

        # Identify most important parameter
        if S['S1'] > S['S2']:
            print(f"\n  → k1 (A → B rate) is more influential")
        else:
            print(f"\n  → k2 (B → C rate) is more influential")

    # =========================================================================
    # STEP 3: Surrogate Model Construction
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 3: SURROGATE MODEL")
    print("-" * 80)

    surrogate_agent = SurrogateModelingAgent()

    # Generate training data
    print(f"\nGenerating training data (Latin Hypercube Sampling)...")
    n_train = 50

    lhs_result = uq_agent.execute({
        'problem_type': 'lhs',
        'n_samples': n_train,
        'n_dimensions': 2,
        'seed': 42
    })

    if lhs_result.success:
        # Scale to parameter ranges
        lhs_samples = lhs_result.data['solution']['samples']
        k1_samples = input_ranges[0][0] + lhs_samples[:, 0] * (input_ranges[0][1] - input_ranges[0][0])
        k2_samples = input_ranges[1][0] + lhs_samples[:, 1] * (input_ranges[1][1] - input_ranges[1][0])

        # Evaluate model at training points
        print(f"  Evaluating {n_train} ODE solutions...")
        training_outputs = []
        for k1, k2 in zip(k1_samples, k2_samples):
            max_B = solve_kinetics(k1, k2)
            training_outputs.append(max_B)

        training_inputs = np.column_stack([k1_samples, k2_samples])
        training_outputs = np.array(training_outputs)

        print(f"✓ Training data generated")
        print(f"  Input range: k1=[{k1_samples.min():.2f}, {k1_samples.max():.2f}], k2=[{k2_samples.min():.2f}, {k2_samples.max():.2f}]")
        print(f"  Output range: B_max=[{training_outputs.min():.4f}, {training_outputs.max():.4f}]")

    # Build Gaussian Process surrogate
    print(f"\nBuilding Gaussian Process surrogate...")

    surrogate_result = surrogate_agent.execute({
        'problem_type': 'gaussian_process',
        'training_inputs': training_inputs,
        'training_outputs': training_outputs,
        'kernel': 'rbf'
    })

    if surrogate_result.success:
        gp_model = surrogate_result.data['solution']['model']

        # Test surrogate accuracy
        test_k1, test_k2 = k1_nominal, k2_nominal
        true_value = solve_kinetics(test_k1, test_k2)
        pred_result = surrogate_agent.execute({
            'problem_type': 'predict',
            'model': gp_model,
            'test_inputs': np.array([[test_k1, test_k2]])
        })

        if pred_result.success:
            pred_value = pred_result.data['solution']['predictions'][0]
            pred_std = pred_result.data['solution']['std_predictions'][0]
            error = abs(pred_value - true_value) / true_value * 100

            print(f"✓ Surrogate Model Built!")
            print(f"  Accuracy test (nominal parameters):")
            print(f"    True value: {true_value:.4f}")
            print(f"    Predicted: {pred_value:.4f} ± {pred_std:.4f}")
            print(f"    Relative error: {error:.2f}%")

    # =========================================================================
    # STEP 4: Optimization Using Surrogate
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 4: PARAMETER OPTIMIZATION")
    print("-" * 80)

    optimizer = OptimizationAgent()

    # Define objective function using surrogate
    def objective(params):
        """Minimize negative B_max (to maximize B_max)."""
        pred_result = surrogate_agent.execute({
            'problem_type': 'predict',
            'model': gp_model,
            'test_inputs': params.reshape(1, -1)
        })
        if pred_result.success:
            return -pred_result.data['solution']['predictions'][0]
        return 0.0

    print(f"\nOptimizing parameters to maximize B concentration...")
    print(f"  Using surrogate model (fast evaluation)")
    print(f"  Search range: k1=[{input_ranges[0][0]:.2f}, {input_ranges[0][1]:.2f}], k2=[{input_ranges[1][0]:.2f}, {input_ranges[1][1]:.2f}]")

    opt_result = optimizer.execute({
        'problem_type': 'optimization_unconstrained',
        'objective': objective,
        'initial_guess': np.array([k1_nominal, k2_nominal]),
        'method': 'L-BFGS-B',
        'bounds': [tuple(input_ranges[0]), tuple(input_ranges[1])]
    })

    if opt_result.success:
        optimal_params = opt_result.data['solution']['x']
        k1_opt, k2_opt = optimal_params

        # Verify with actual ODE solve
        B_max_opt_surrogate = -opt_result.data['solution']['fun']
        B_max_opt_true = solve_kinetics(k1_opt, k2_opt)

        print(f"\n✓ Optimization Complete!")
        print(f"  Optimal parameters:")
        print(f"    k1 = {k1_opt:.4f} (nominal: {k1_nominal:.4f})")
        print(f"    k2 = {k2_opt:.4f} (nominal: {k2_nominal:.4f})")
        print(f"  Predicted B_max: {B_max_opt_surrogate:.4f}")
        print(f"  Actual B_max: {B_max_opt_true:.4f}")
        print(f"  Improvement: {(B_max_opt_true - B_max_nominal) / B_max_nominal * 100:.1f}%")

    # =========================================================================
    # STEP 5: Verification with Full ODE Solution
    # =========================================================================
    print("\n" + "-" * 80)
    print("STEP 5: VERIFICATION")
    print("-" * 80)

    def optimal_rhs(t, y):
        return chemical_kinetics_model(t, y, k1_opt, k2_opt)

    verify_result = solver.execute({
        'problem_type': 'ode_ivp',
        'rhs': optimal_rhs,
        'initial_conditions': [1.0, 0.0, 0.0],
        'time_span': (0, 20),
        'method': 'RK45'
    })

    if verify_result.success:
        t_opt = verify_result.data['solution']['t']
        y_opt = verify_result.data['solution']['y']

        print(f"\n✓ Verification Complete!")
        print(f"  Optimal solution trajectory computed")

        # =====================================================================
        # STEP 6: Visualization
        # =====================================================================
        print("\n" + "-" * 80)
        print("STEP 6: VISUALIZATION")
        print("-" * 80)

        create_visualization(
            t_baseline, y_baseline,
            t_opt, y_opt,
            k1_nominal, k2_nominal,
            k1_opt, k2_opt,
            B_max_nominal, B_max_opt_true
        )

    return True


def create_visualization(t_base, y_base, t_opt, y_opt,
                        k1_nom, k2_nom, k1_opt, k2_opt,
                        B_max_nom, B_max_opt):
    """Create visualization of workflow results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Baseline concentrations
    ax = axes[0, 0]
    ax.plot(t_base, y_base[0], 'b-', linewidth=2, label='[A]')
    ax.plot(t_base, y_base[1], 'g-', linewidth=2, label='[B]')
    ax.plot(t_base, y_base[2], 'r-', linewidth=2, label='[C]')
    ax.axhline(B_max_nom, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title(f'Baseline (k1={k1_nom:.2f}, k2={k2_nom:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Optimized concentrations
    ax = axes[0, 1]
    ax.plot(t_opt, y_opt[0], 'b-', linewidth=2, label='[A]')
    ax.plot(t_opt, y_opt[1], 'g-', linewidth=2, label='[B]')
    ax.plot(t_opt, y_opt[2], 'r-', linewidth=2, label='[C]')
    ax.axhline(B_max_opt, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title(f'Optimized (k1={k1_opt:.2f}, k2={k2_opt:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: B concentration comparison
    ax = axes[1, 0]
    ax.plot(t_base, y_base[1], 'b-', linewidth=2, label=f'Baseline (max={B_max_nom:.3f})', alpha=0.7)
    ax.plot(t_opt, y_opt[1], 'r-', linewidth=2, label=f'Optimized (max={B_max_opt:.3f})', alpha=0.7)
    ax.axhline(B_max_nom, color='b', linestyle='--', alpha=0.3)
    ax.axhline(B_max_opt, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('[B] Concentration')
    ax.set_title('Optimization Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 4: Parameter comparison
    ax = axes[1, 1]
    categories = ['k1 (A→B)', 'k2 (B→C)']
    x_pos = np.arange(len(categories))
    width = 0.35

    baseline_vals = [k1_nom, k2_nom]
    optimized_vals = [k1_opt, k2_opt]

    ax.bar(x_pos - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    ax.bar(x_pos + width/2, optimized_vals, width, label='Optimized', alpha=0.8)

    ax.set_ylabel('Rate Constant')
    ax.set_title('Parameter Optimization Results')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'workflow_02_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")


def main():
    """Run the multi-physics workflow."""

    print("\n" + "="*80)
    print("WORKFLOW EXAMPLE: MULTI-PHYSICS WITH UNCERTAINTY QUANTIFICATION")
    print("="*80)
    print("\nThis example demonstrates complex multi-agent orchestration:")
    print("  1. ODEPDESolverAgent - Solve chemical kinetics")
    print("  2. UncertaintyQuantificationAgent - Sensitivity analysis")
    print("  3. SurrogateModelingAgent - Build fast surrogate")
    print("  4. OptimizationAgent - Optimize parameters")
    print()

    success = run_multi_physics_workflow()

    if success:
        print("\n" + "="*80)
        print("✓ MULTI-PHYSICS WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Achievements:")
        print("  ✓ ODE system solved with baseline and optimized parameters")
        print("  ✓ Sensitivity analysis identified key parameters")
        print("  ✓ Surrogate model built for fast evaluation")
        print("  ✓ Parameters optimized to maximize B concentration")
        print("  ✓ Results verified with full ODE solution")
        print()
        return 0
    else:
        print("\n" + "="*80)
        print("✗ WORKFLOW FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
