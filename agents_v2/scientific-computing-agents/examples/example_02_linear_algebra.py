"""Example 2: Linear Algebra Problems.

This example demonstrates using the LinearAlgebraAgent to solve various
linear algebra problems:
1. Solving linear systems (circuit analysis)
2. Computing eigenvalues (stability analysis)
3. Matrix factorizations (QR decomposition)
4. Analyzing matrix properties
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.linear_algebra_agent import LinearAlgebraAgent


def example_1_circuit_analysis():
    """Example 1: Circuit analysis using Kirchhoff's laws.

    Simple resistor network:
    - 3 nodes with resistances
    - Solve for node voltages using linear system

    System derived from Kirchhoff's current law:
    Node equations give us a linear system Ax = b
    """
    print("\n" + "="*70)
    print("Example 1: Circuit Analysis (Linear System)")
    print("="*70)

    # Circuit with 3 nodes, resistances, and current sources
    # Conductance matrix (inverse of resistance)
    # A = conductance matrix, b = current sources
    A = np.array([
        [3.0, -1.0, -1.0],
        [-1.0, 4.0, -2.0],
        [-1.0, -2.0, 4.0]
    ])
    b = np.array([2.0, 1.0, 0.0])  # Current sources

    # Create agent
    agent = LinearAlgebraAgent(config={'tolerance': 1e-8})

    # Solve system
    print("\nSolving circuit equations: Ax = b")
    print(f"Conductance matrix A:\n{A}")
    print(f"Current sources b: {b}")
    print(f"Method: LU decomposition")

    result = agent.execute({
        'problem_type': 'linear_system_dense',
        'matrix_A': A,
        'vector_b': b,
        'method': 'lu'
    })

    if result.success:
        print(f"\n✓ Solution successful!")
        print(f"  - Execution time: {result.metadata['execution_time_sec']:.4f}s")

        # Get solution
        voltages = result.data['solution']['x']
        residual = result.data['solution']['residual']

        print(f"\n  Node voltages (V):")
        for i, v in enumerate(voltages, 1):
            print(f"    Node {i}: {v:.6f} V")

        print(f"\n  Solution quality:")
        print(f"    Residual: {residual:.2e}")

        # Verify solution
        Ax = A @ voltages
        print(f"\n  Verification:")
        print(f"    A @ x = {Ax}")
        print(f"    b     = {b}")
        print(f"    Match: {np.allclose(Ax, b)}")

        return voltages
    else:
        print(f"\n✗ Solution failed: {result.errors}")
        return None


def example_2_stability_analysis():
    """Example 2: Stability analysis via eigenvalues.

    Analyze the stability of a dynamical system by computing eigenvalues
    of the Jacobian matrix. Negative real parts indicate stability.

    Example: 3-state linear system
    """
    print("\n" + "="*70)
    print("Example 2: Stability Analysis (Eigenvalues)")
    print("="*70)

    # Jacobian matrix of a linear system dx/dt = Ax
    # Represents damped oscillator coupling
    A = np.array([
        [-1.0,  0.5,  0.0],
        [ 0.5, -2.0,  0.5],
        [ 0.0,  0.5, -1.0]
    ])

    # Create agent
    agent = LinearAlgebraAgent()

    # Compute eigenvalues
    print("\nAnalyzing system stability via eigenvalues")
    print(f"Jacobian matrix A:\n{A}")
    print(f"System: dx/dt = A @ x")

    result = agent.execute({
        'problem_type': 'eigenvalue_problem',
        'matrix_A': A
    })

    if result.success:
        print(f"\n✓ Eigenvalue computation successful!")

        # Get eigenvalues and eigenvectors
        eigenvalues = result.data['solution']['eigenvalues']
        eigenvectors = result.data['solution']['eigenvectors']

        print(f"\n  Eigenvalues:")
        for i, λ in enumerate(eigenvalues):
            real_part = np.real(λ)
            imag_part = np.imag(λ)
            if np.abs(imag_part) < 1e-10:
                print(f"    λ{i+1} = {real_part:.6f}")
            else:
                print(f"    λ{i+1} = {real_part:.6f} + {imag_part:.6f}i")

        # Stability analysis
        print(f"\n  Stability analysis:")
        max_real = np.max(np.real(eigenvalues))
        if max_real < 0:
            print(f"    ✓ STABLE: All eigenvalues have negative real parts")
            print(f"    Maximum real part: {max_real:.6f}")
        else:
            print(f"    ✗ UNSTABLE: Some eigenvalues have positive real parts")
            print(f"    Maximum real part: {max_real:.6f}")

        # Dominant time constant (slowest decay)
        if max_real < 0:
            tau = -1.0 / max_real
            print(f"    Dominant time constant: {tau:.3f}")

        return eigenvalues, eigenvectors
    else:
        print(f"\n✗ Eigenvalue computation failed: {result.errors}")
        return None, None


def example_3_qr_decomposition():
    """Example 3: QR decomposition for least squares.

    Use QR factorization to solve an overdetermined system (more equations
    than unknowns) in the least squares sense.
    """
    print("\n" + "="*70)
    print("Example 3: QR Decomposition for Least Squares")
    print("="*70)

    # Overdetermined system: fit a line y = ax + b to data points
    # Create data with some noise
    np.random.seed(42)
    x_data = np.linspace(0, 10, 20)
    y_data = 2.5 * x_data + 1.5 + np.random.normal(0, 0.5, 20)

    # Build matrix A for least squares: [x, 1] * [a, b]' = y
    A = np.column_stack([x_data, np.ones_like(x_data)])
    b = y_data

    print(f"\nFitting line y = ax + b to {len(x_data)} data points")
    print(f"Data matrix A shape: {A.shape}")
    print(f"Target vector b shape: {b.shape}")

    # Create agent
    agent = LinearAlgebraAgent()

    # Compute QR factorization
    result = agent.execute({
        'problem_type': 'matrix_factorization',
        'matrix_A': A,
        'factorization_type': 'qr'
    })

    if result.success:
        print(f"\n✓ QR factorization successful!")

        Q = result.data['solution']['Q']
        R = result.data['solution']['R']

        print(f"  Q shape: {Q.shape}, R shape: {R.shape}")

        # Solve least squares using QR: x = R^(-1) @ Q^T @ b
        from scipy.linalg import solve_triangular
        x_ls = solve_triangular(R[:2, :2], Q[:, :2].T @ b)

        a, b_intercept = x_ls
        print(f"\n  Least squares solution:")
        print(f"    Slope (a): {a:.6f}")
        print(f"    Intercept (b): {b_intercept:.6f}")
        print(f"    Line: y = {a:.3f}x + {b_intercept:.3f}")

        # Compute residual
        y_fit = a * x_data + b_intercept
        residual = np.linalg.norm(y_data - y_fit)
        print(f"\n  Fit quality:")
        print(f"    Residual norm: {residual:.4f}")

        # Verify QR decomposition
        A_reconstructed = Q @ R
        qr_error = np.linalg.norm(A - A_reconstructed)
        print(f"    QR reconstruction error: {qr_error:.2e}")

        return Q, R, a, b_intercept, x_data, y_data
    else:
        print(f"\n✗ QR factorization failed: {result.errors}")
        return None, None, None, None, None, None


def example_4_condition_number():
    """Example 4: Matrix conditioning analysis.

    Analyze the condition number of matrices to understand numerical
    stability of linear solves.
    """
    print("\n" + "="*70)
    print("Example 4: Matrix Conditioning Analysis")
    print("="*70)

    # Create agent
    agent = LinearAlgebraAgent()

    # Well-conditioned matrix (identity)
    print("\n--- Well-conditioned matrix (Identity) ---")
    A_good = np.eye(3)

    result_good = agent.execute({
        'problem_type': 'matrix_analysis',
        'matrix_A': A_good
    })

    if result_good.success:
        analysis = result_good.data['solution']
        print(f"Condition number: {analysis['condition_number']:.2e}")
        print(f"Rank: {analysis['rank']}")
        print(f"Is symmetric: {analysis['is_symmetric']}")
        print(f"Interpretation: Excellent conditioning (κ ≈ 1)")

    # Poorly conditioned matrix
    print("\n--- Poorly conditioned matrix ---")
    # Create nearly singular matrix
    A_bad = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0 + 1e-8, 1.0],
        [1.0, 1.0, 1.0 + 1e-8]
    ])

    result_bad = agent.execute({
        'problem_type': 'matrix_analysis',
        'matrix_A': A_bad
    })

    if result_bad.success:
        analysis = result_bad.data['solution']
        print(f"Condition number: {analysis['condition_number']:.2e}")
        print(f"Rank: {analysis['rank']}")
        print(f"Interpretation: Poor conditioning (κ >> 1)")
        print(f"  - Small changes in input cause large changes in output")
        print(f"  - Numerical errors amplified by factor of {analysis['condition_number']:.1e}")

    # Singular matrix
    print("\n--- Singular matrix (rank deficient) ---")
    A_singular = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ])

    result_singular = agent.execute({
        'problem_type': 'matrix_analysis',
        'matrix_A': A_singular
    })

    if result_singular.success:
        analysis = result_singular.data['solution']
        print(f"Condition number: {analysis['condition_number']:.2e} (infinite)")
        print(f"Rank: {analysis['rank']} (should be 3)")
        print(f"Determinant: {analysis['determinant']:.2e} (zero)")
        print(f"Interpretation: Singular - no unique solution exists")


def example_5_iterative_solver():
    """Example 5: Iterative solver for large sparse system.

    Demonstrate conjugate gradient for symmetric positive definite systems.
    """
    print("\n" + "="*70)
    print("Example 5: Iterative Solver (Conjugate Gradient)")
    print("="*70)

    # Create a large SPD system (Poisson-like)
    n = 100
    # Tridiagonal matrix: -1, 2, -1 pattern (discretized Laplacian)
    A = np.zeros((n, n))
    np.fill_diagonal(A, 2.0)
    np.fill_diagonal(A[1:], -1.0)
    np.fill_diagonal(A[:, 1:], -1.0)

    # Add small diagonal perturbation to ensure SPD
    A += 0.1 * np.eye(n)

    # Right-hand side
    b = np.ones(n)

    print(f"\nSolving large SPD system: {n}×{n}")
    print(f"Condition number (estimated): ~{n**2:.0f}")

    # Create agent
    agent = LinearAlgebraAgent(config={'tolerance': 1e-6, 'max_iterations': 200})

    # Solve with CG
    result = agent.execute({
        'problem_type': 'linear_system_dense',
        'matrix_A': A,
        'vector_b': b,
        'method': 'cg'
    })

    if result.success:
        print(f"\n✓ Iterative solution successful!")
        print(f"  - Method: {result.data['metadata']['method']}")
        print(f"  - Iterations: {result.data['metadata']['iterations']}")
        print(f"  - Residual: {result.data['metadata']['residual']:.2e}")
        print(f"  - Execution time: {result.metadata['execution_time_sec']:.4f}s")

        x = result.data['solution']['x']

        # Verify solution
        residual_norm = np.linalg.norm(A @ x - b)
        print(f"\n  Verification:")
        print(f"    ||Ax - b|| = {residual_norm:.2e}")
        print(f"    Converged: {residual_norm < 1e-6}")

        return x
    else:
        print(f"\n✗ Iterative solution failed: {result.errors}")
        return None


def create_plots():
    """Create visualization of all examples."""
    print("\n" + "="*70)
    print("Creating Plots...")
    print("="*70)

    fig = plt.figure(figsize=(14, 10))

    # Example 1: Circuit voltages
    print("\nRunning Example 1...")
    voltages = example_1_circuit_analysis()
    if voltages is not None:
        ax1 = plt.subplot(2, 3, 1)
        nodes = ['Node 1', 'Node 2', 'Node 3']
        ax1.bar(nodes, voltages, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Circuit Analysis: Node Voltages')
        ax1.grid(True, alpha=0.3)

    # Example 2: Eigenvalue spectrum
    print("\nRunning Example 2...")
    eigenvalues, eigenvectors = example_2_stability_analysis()
    if eigenvalues is not None:
        ax2 = plt.subplot(2, 3, 2)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        ax2.scatter(real_parts, imag_parts, s=100, c='red', marker='x', linewidths=2)
        ax2.axvline(0, color='k', linestyle='--', linewidth=0.5)
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Real Part')
        ax2.set_ylabel('Imaginary Part')
        ax2.set_title('Eigenvalue Spectrum')
        ax2.grid(True, alpha=0.3)

    # Example 3: Least squares fit
    print("\nRunning Example 3...")
    Q, R, a, b, x_data, y_data = example_3_qr_decomposition()
    if Q is not None:
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(x_data, y_data, alpha=0.6, label='Data')
        x_fit = np.linspace(0, 10, 100)
        y_fit = a * x_fit + b
        ax3.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fit: y={a:.2f}x+{b:.2f}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title('QR Decomposition: Least Squares Fit')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Example 4: Condition numbers
    print("\nRunning Example 4...")
    example_4_condition_number()
    ax4 = plt.subplot(2, 3, 4)
    matrices = ['Identity\n(well-cond.)', 'Nearly singular\n(ill-cond.)', 'Singular\n(rank def.)']
    cond_numbers = [1, 1e8, np.inf]
    colors = ['green', 'orange', 'red']
    bars = ax4.bar(matrices, [np.log10(c) if c != np.inf else 16 for c in cond_numbers], color=colors)
    ax4.set_ylabel('log₁₀(Condition Number)')
    ax4.set_title('Matrix Conditioning Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    # Example 5: Iterative convergence
    print("\nRunning Example 5...")
    x = example_5_iterative_solver()
    if x is not None:
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(x, linewidth=1)
        ax5.set_xlabel('Index')
        ax5.set_ylabel('Solution value')
        ax5.set_title('CG Solution: Large SPD System')
        ax5.grid(True, alpha=0.3)

    # Summary subplot
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = """
    Linear Algebra Agent Examples:

    1. Circuit Analysis (LU)
       - 3-node resistor network
       - Direct solve: Ax = b

    2. Stability Analysis (Eigen)
       - Jacobian eigenvalues
       - System stability check

    3. Least Squares (QR)
       - Line fitting to noisy data
       - Overdetermined system

    4. Conditioning Analysis
       - Well/ill-conditioned matrices
       - Numerical stability

    5. Iterative Solver (CG)
       - Large sparse SPD system
       - Conjugate gradient
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'example_02_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    # Show plot
    print("\nClose the plot window to continue...")
    plt.show()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Linear Algebra Agent - Usage Examples")
    print("="*70)
    print("\nThis script demonstrates solving various linear algebra problems:")
    print("  1. Circuit analysis (linear systems)")
    print("  2. Stability analysis (eigenvalues)")
    print("  3. Least squares fitting (QR decomposition)")
    print("  4. Matrix conditioning analysis")
    print("  5. Iterative solvers (conjugate gradient)")

    # Run examples and create plots
    create_plots()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
