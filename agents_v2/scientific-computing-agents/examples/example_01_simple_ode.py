"""Example 1: Solving Simple Ordinary Differential Equations.

This example demonstrates using the ODEPDESolverAgent to solve various
simple ODE problems:
1. Exponential decay (first-order linear ODE)
2. Harmonic oscillator (second-order linear ODE)
3. Chemical kinetics (first-order system)
4. Predator-prey dynamics (nonlinear system)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ode_pde_solver_agent import ODEPDESolverAgent


def example_1_exponential_decay():
    """Example 1: Exponential decay dy/dt = -k*y, y(0) = y0.

    This models radioactive decay, cooling, or first-order chemical reactions.
    Analytical solution: y(t) = y0 * exp(-k*t)
    """
    print("\n" + "="*70)
    print("Example 1: Exponential Decay")
    print("="*70)

    # Problem parameters
    k = 0.5  # Decay constant
    y0 = 10.0  # Initial value
    t_span = [0, 10]  # Time interval

    # Create agent
    agent = ODEPDESolverAgent(config={'tolerance': 1e-8})

    # Define ODE
    def decay_ode(t, y):
        return -k * y

    # Solve
    print("\nSolving: dy/dt = -0.5*y, y(0) = 10")
    print(f"Time span: {t_span}")
    print(f"Method: RK45 (Runge-Kutta)")

    result = agent.execute({
        'problem_type': 'ode_ivp',
        'rhs': decay_ode,
        'initial_conditions': [y0],
        'time_span': t_span,
        'method': 'RK45'
    })

    if result.success:
        print(f"\n✓ Solution successful!")
        print(f"  - Method: {result.data['metadata']['method']}")
        print(f"  - Time points: {result.data['metadata']['num_time_points']}")
        print(f"  - Function evaluations: {result.data['metadata']['num_function_evals']}")
        print(f"  - Execution time: {result.metadata['execution_time_sec']:.4f}s")

        # Get solution
        t = result.data['solution']['t']
        y = result.data['solution']['y'][0]

        print(f"\n  Solution at key times:")
        print(f"    t=0:   y = {y[0]:.4f}")
        print(f"    t=5:   y = {y[len(y)//2]:.4f} (analytical: {y0 * np.exp(-k * 5):.4f})")
        print(f"    t=10:  y = {y[-1]:.4f} (analytical: {y0 * np.exp(-k * 10):.4f})")

        # Calculate error vs analytical solution
        y_analytical = y0 * np.exp(-k * t)
        max_error = np.max(np.abs(y - y_analytical))
        print(f"\n  Maximum error vs. analytical: {max_error:.2e}")

        return t, y, y_analytical
    else:
        print(f"\n✗ Solution failed: {result.errors}")
        return None, None, None


def example_2_harmonic_oscillator():
    """Example 2: Harmonic oscillator d²x/dt² + ω²x = 0.

    This models springs, pendulums, LC circuits.
    Converted to first-order system: dy/dt = [v, -ω²x]
    Analytical solution: x(t) = A*cos(ωt) + B*sin(ωt)
    """
    print("\n" + "="*70)
    print("Example 2: Harmonic Oscillator")
    print("="*70)

    # Problem parameters
    omega = 2.0  # Angular frequency
    x0 = 1.0  # Initial position
    v0 = 0.0  # Initial velocity
    t_span = [0, 2*np.pi/omega]  # One period

    # Create agent
    agent = ODEPDESolverAgent()

    # Define ODE as first-order system [x, v]
    def oscillator(t, y):
        x, v = y
        return [v, -omega**2 * x]

    # Solve
    print(f"\nSolving: d²x/dt² + {omega**2}*x = 0")
    print(f"Initial conditions: x(0) = {x0}, v(0) = {v0}")
    print(f"Time span: {t_span} (one period)")

    result = agent.execute({
        'problem_type': 'ode_ivp',
        'rhs': oscillator,
        'initial_conditions': [x0, v0],
        'time_span': t_span,
        'method': 'RK45'
    })

    if result.success:
        print(f"\n✓ Solution successful!")
        print(f"  - Execution time: {result.metadata['execution_time_sec']:.4f}s")

        # Get solution
        t = result.data['solution']['t']
        x = result.data['solution']['y'][0]
        v = result.data['solution']['y'][1]

        print(f"\n  Solution verification (should be periodic):")
        print(f"    Initial: x={x[0]:.6f}, v={v[0]:.6f}")
        print(f"    Final:   x={x[-1]:.6f}, v={v[-1]:.6f}")
        print(f"    Difference: Δx={abs(x[-1]-x[0]):.2e}, Δv={abs(v[-1]-v[0]):.2e}")

        # Energy conservation (should be constant)
        energy = 0.5 * v**2 + 0.5 * omega**2 * x**2
        print(f"\n  Energy conservation:")
        print(f"    Initial energy: {energy[0]:.6f}")
        print(f"    Final energy:   {energy[-1]:.6f}")
        print(f"    Relative change: {abs(energy[-1]-energy[0])/energy[0]:.2e}")

        return t, x, v
    else:
        print(f"\n✗ Solution failed: {result.errors}")
        return None, None, None


def example_3_chemical_kinetics():
    """Example 3: First-order consecutive reactions A → B → C.

    This models chemical kinetics with rate constants k1, k2.
    System: dA/dt = -k1*A
            dB/dt = k1*A - k2*B
            dC/dt = k2*B
    """
    print("\n" + "="*70)
    print("Example 3: Chemical Kinetics (A → B → C)")
    print("="*70)

    # Problem parameters
    k1 = 0.5  # Rate constant A → B
    k2 = 0.3  # Rate constant B → C
    A0 = 1.0  # Initial concentration of A
    t_span = [0, 20]

    # Create agent
    agent = ODEPDESolverAgent()

    # Define ODE system
    def kinetics(t, y):
        A, B, C = y
        dA = -k1 * A
        dB = k1 * A - k2 * B
        dC = k2 * B
        return [dA, dB, dC]

    # Solve
    print(f"\nSolving: A → B → C")
    print(f"Rate constants: k1 = {k1}, k2 = {k2}")
    print(f"Initial: [A]={A0}, [B]=0, [C]=0")

    result = agent.execute({
        'problem_type': 'ode_ivp',
        'rhs': kinetics,
        'initial_conditions': [A0, 0.0, 0.0],
        'time_span': t_span,
        'method': 'RK45'
    })

    if result.success:
        print(f"\n✓ Solution successful!")

        # Get solution
        t = result.data['solution']['t']
        A = result.data['solution']['y'][0]
        B = result.data['solution']['y'][1]
        C = result.data['solution']['y'][2]

        # Check mass balance
        total = A + B + C
        print(f"\n  Mass balance (should be constant = {A0}):")
        print(f"    Initial: A+B+C = {total[0]:.6f}")
        print(f"    Final:   A+B+C = {total[-1]:.6f}")
        print(f"    Max deviation: {np.max(np.abs(total - A0)):.2e}")

        # Final concentrations
        print(f"\n  Final concentrations:")
        print(f"    [A] = {A[-1]:.4f}")
        print(f"    [B] = {B[-1]:.4f}")
        print(f"    [C] = {C[-1]:.4f}")

        return t, A, B, C
    else:
        print(f"\n✗ Solution failed: {result.errors}")
        return None, None, None, None


def example_4_predator_prey():
    """Example 4: Lotka-Volterra predator-prey model.

    This models population dynamics of predators and prey.
    dx/dt = αx - βxy    (prey growth - predation)
    dy/dt = δxy - γy    (predation benefit - predator death)

    This is a nonlinear system exhibiting periodic behavior.
    """
    print("\n" + "="*70)
    print("Example 4: Predator-Prey Dynamics (Lotka-Volterra)")
    print("="*70)

    # Problem parameters
    alpha = 1.0   # Prey growth rate
    beta = 0.1    # Predation rate
    gamma = 1.5   # Predator death rate
    delta = 0.075 # Predator efficiency
    x0 = 10.0     # Initial prey population
    y0 = 5.0      # Initial predator population
    t_span = [0, 30]

    # Create agent
    agent = ODEPDESolverAgent()

    # Define ODE system
    def lotka_volterra(t, z):
        x, y = z  # x=prey, y=predator
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return [dx, dy]

    # Solve
    print(f"\nSolving Lotka-Volterra equations:")
    print(f"  dx/dt = {alpha}*x - {beta}*x*y")
    print(f"  dy/dt = {delta}*x*y - {gamma}*y")
    print(f"Initial populations: prey={x0}, predator={y0}")

    result = agent.execute({
        'problem_type': 'ode_ivp',
        'rhs': lotka_volterra,
        'initial_conditions': [x0, y0],
        'time_span': t_span,
        'method': 'RK45'
    })

    if result.success:
        print(f"\n✓ Solution successful!")

        # Get solution
        t = result.data['solution']['t']
        prey = result.data['solution']['y'][0]
        predator = result.data['solution']['y'][1]

        # Find peaks (approximate period)
        prey_max = np.max(prey)
        predator_max = np.max(predator)

        print(f"\n  Population dynamics:")
        print(f"    Prey:     min={np.min(prey):.2f}, max={prey_max:.2f}")
        print(f"    Predator: min={np.min(predator):.2f}, max={predator_max:.2f}")

        return t, prey, predator
    else:
        print(f"\n✗ Solution failed: {result.errors}")
        return None, None, None


def create_plots():
    """Create visualization of all examples."""
    print("\n" + "="*70)
    print("Creating Plots...")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Example 1: Exponential decay
    t1, y1, y1_ana = example_1_exponential_decay()
    if t1 is not None:
        axes[0, 0].plot(t1, y1, 'b-', label='Numerical', linewidth=2)
        axes[0, 0].plot(t1, y1_ana, 'r--', label='Analytical', linewidth=1.5)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('y(t)')
        axes[0, 0].set_title('Exponential Decay: dy/dt = -0.5y')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Example 2: Harmonic oscillator
    t2, x2, v2 = example_2_harmonic_oscillator()
    if t2 is not None:
        axes[0, 1].plot(t2, x2, 'b-', label='Position x(t)', linewidth=2)
        axes[0, 1].plot(t2, v2, 'r-', label='Velocity v(t)', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('x(t), v(t)')
        axes[0, 1].set_title('Harmonic Oscillator: d²x/dt² + 4x = 0')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Example 3: Chemical kinetics
    t3, A3, B3, C3 = example_3_chemical_kinetics()
    if t3 is not None:
        axes[1, 0].plot(t3, A3, 'b-', label='[A]', linewidth=2)
        axes[1, 0].plot(t3, B3, 'g-', label='[B]', linewidth=2)
        axes[1, 0].plot(t3, C3, 'r-', label='[C]', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Concentration')
        axes[1, 0].set_title('Chemical Kinetics: A → B → C')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Example 4: Predator-prey
    t4, prey, predator = example_4_predator_prey()
    if t4 is not None:
        axes[1, 1].plot(t4, prey, 'b-', label='Prey', linewidth=2)
        axes[1, 1].plot(t4, predator, 'r-', label='Predator', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Population')
        axes[1, 1].set_title('Predator-Prey Dynamics (Lotka-Volterra)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'example_01_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    # Show plot
    print("\nClose the plot window to continue...")
    plt.show()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("ODE/PDE Solver Agent - Usage Examples")
    print("="*70)
    print("\nThis script demonstrates solving various ODEs:")
    print("  1. Exponential decay (first-order linear)")
    print("  2. Harmonic oscillator (second-order linear)")
    print("  3. Chemical kinetics (first-order system)")
    print("  4. Predator-prey dynamics (nonlinear system)")

    # Run examples and create plots
    create_plots()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
