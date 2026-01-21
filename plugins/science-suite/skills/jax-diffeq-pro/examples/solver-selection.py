"""
Solver Selection Patterns for Diffrax

Demonstrates choosing between explicit and implicit solvers,
detecting stiffness, and configuring step size control.
"""

import jax.numpy as jnp
import diffrax
from typing import Callable, Tuple


# =============================================================================
# Pattern 1: Basic Explicit Solver
# =============================================================================

def explicit_solver_example():
    """Non-stiff system with explicit solver (Tsit5)."""

    def harmonic_oscillator(t, y, args):
        """Simple harmonic motion: d²x/dt² = -ω²x"""
        x, v = y
        omega = args['omega']
        return jnp.array([v, -omega**2 * x])

    term = diffrax.ODETerm(harmonic_oscillator)
    solver = diffrax.Tsit5()  # 5th order explicit Runge-Kutta

    stepsize_controller = diffrax.PIDController(
        rtol=1e-5,
        atol=1e-7,
    )

    y0 = jnp.array([1.0, 0.0])  # x=1, v=0
    ts = jnp.linspace(0, 10, 100)

    solution = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=10.0, dt0=0.01,
        y0=y0,
        args={'omega': 2.0},
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=stepsize_controller,
    )

    return solution.ts, solution.ys


# =============================================================================
# Pattern 2: Stiff System with Implicit Solver
# =============================================================================

def implicit_solver_example():
    """Stiff chemical kinetics requiring implicit solver."""

    def robertson_kinetics(t, y, args):
        """Robertson problem: widely separated rate constants."""
        k1, k2, k3 = args['k1'], args['k2'], args['k3']
        y1, y2, y3 = y

        dy1 = -k1 * y1 + k3 * y2 * y3
        dy2 = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
        dy3 = k2 * y2**2

        return jnp.array([dy1, dy2, dy3])

    term = diffrax.ODETerm(robertson_kinetics)

    # Kvaerno5: 5th order implicit solver for stiff systems
    solver = diffrax.Kvaerno5()

    # Tighter tolerances for stiff problems
    stepsize_controller = diffrax.PIDController(
        rtol=1e-6,
        atol=1e-8,
        dtmin=1e-12,  # Allow very small steps
        dtmax=1.0,
    )

    y0 = jnp.array([1.0, 0.0, 0.0])

    # Log-spaced times for stiff problem
    ts = jnp.logspace(-5, 5, 100)

    solution = diffrax.diffeqsolve(
        term, solver,
        t0=ts[0], t1=ts[-1], dt0=1e-6,
        y0=y0,
        args={'k1': 0.04, 'k2': 3e7, 'k3': 1e4},
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=stepsize_controller,
        max_steps=100000,
    )

    return solution.ts, solution.ys


# =============================================================================
# Pattern 3: Stiffness Detection Heuristic
# =============================================================================

def detect_stiffness(jacobian_fn: Callable, y: jnp.ndarray) -> Tuple[float, str]:
    """Estimate stiffness from Jacobian eigenvalue spread.

    Returns stiffness ratio and recommended solver.
    """
    J = jacobian_fn(y)
    eigenvalues = jnp.linalg.eigvals(J)
    real_parts = jnp.real(eigenvalues)

    # Avoid division by zero
    min_real = jnp.min(real_parts)
    max_real = jnp.max(real_parts)

    ratio = jnp.where(
        jnp.abs(min_real) > 1e-10,
        jnp.abs(max_real / min_real),
        1.0
    )

    if ratio > 1000:
        return float(ratio), "Kvaerno5 (highly stiff)"
    elif ratio > 100:
        return float(ratio), "KenCarp4 (moderately stiff)"
    else:
        return float(ratio), "Tsit5 (non-stiff)"


def stiffness_detection_example():
    """Demonstrate stiffness detection on Robertson problem."""

    def jacobian(y):
        """Jacobian of Robertson kinetics."""
        k1, k2, k3 = 0.04, 3e7, 1e4
        y1, y2, y3 = y

        return jnp.array([
            [-k1, k3 * y3, k3 * y2],
            [k1, -2 * k2 * y2 - k3 * y3, -k3 * y2],
            [0, 2 * k2 * y2, 0]
        ])

    y0 = jnp.array([1.0, 0.0, 0.0])
    ratio, recommendation = detect_stiffness(jacobian, y0)

    print(f"Stiffness ratio: {ratio:.0f}")
    print(f"Recommendation: {recommendation}")

    return ratio, recommendation


# =============================================================================
# Pattern 4: Solver Selection Function
# =============================================================================

def get_solver(stiffness: str = "auto", problem_type: str = "ode"):
    """Return appropriate solver for problem characteristics.

    Args:
        stiffness: "non-stiff", "stiff", "mixed", or "auto"
        problem_type: "ode" or "sde"
    """
    solvers = {
        # ODE solvers
        ("ode", "non-stiff"): diffrax.Tsit5(),
        ("ode", "stiff"): diffrax.Kvaerno5(),
        ("ode", "mixed"): diffrax.KenCarp4(),  # IMEX solver

        # SDE solvers (need appropriate schemes)
        ("sde", "non-stiff"): diffrax.Euler(),
        ("sde", "stiff"): diffrax.Heun(),
    }

    key = (problem_type, stiffness)
    return solvers.get(key, diffrax.Tsit5())


# =============================================================================
# Pattern 5: Step Size Control Configurations
# =============================================================================

def stepsize_controller_configs():
    """Various step size controller configurations."""

    # Standard configuration
    standard = diffrax.PIDController(
        rtol=1e-5,
        atol=1e-7,
    )

    # Aggressive for rapidly changing systems
    aggressive = diffrax.PIDController(
        rtol=1e-4,
        atol=1e-6,
        pcoeff=0.4,  # Proportional gain
        icoeff=0.3,  # Integral gain
        dcoeff=0.0,  # Derivative gain
    )

    # Conservative for sensitive/chaotic systems
    conservative = diffrax.PIDController(
        rtol=1e-8,
        atol=1e-10,
        pcoeff=0.0,
        icoeff=0.7,  # Slower adaptation
        dcoeff=0.0,
        dtmin=1e-15,
        dtmax=0.1,
    )

    # Fixed step (for reproducibility or when comparing methods)
    fixed = diffrax.ConstantStepSize()

    return {
        'standard': standard,
        'aggressive': aggressive,
        'conservative': conservative,
        'fixed': fixed,
    }


# =============================================================================
# Pattern 6: Comparing Solvers
# =============================================================================

def compare_solvers():
    """Compare explicit vs implicit solver on same problem."""

    def vector_field(t, y, args):
        """Mildly stiff ODE."""
        return jnp.array([
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1]**2,
            3e7 * y[1]**2
        ])

    term = diffrax.ODETerm(vector_field)
    y0 = jnp.array([1.0, 0.0, 0.0])
    ts = jnp.linspace(0, 1, 50)

    controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    results = {}

    # Try explicit solver
    try:
        sol_explicit = diffrax.diffeqsolve(
            term, diffrax.Tsit5(),
            t0=0.0, t1=1.0, dt0=0.001,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller,
            max_steps=100000,
        )
        results['explicit'] = {
            'success': True,
            'num_steps': sol_explicit.stats['num_steps'],
        }
    except Exception as e:
        results['explicit'] = {'success': False, 'error': str(e)}

    # Try implicit solver
    try:
        sol_implicit = diffrax.diffeqsolve(
            term, diffrax.Kvaerno5(),
            t0=0.0, t1=1.0, dt0=0.001,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=controller,
            max_steps=100000,
        )
        results['implicit'] = {
            'success': True,
            'num_steps': sol_implicit.stats['num_steps'],
        }
    except Exception as e:
        results['implicit'] = {'success': False, 'error': str(e)}

    return results


# =============================================================================
# Pattern 7: Solver with Max Steps Protection
# =============================================================================

def solve_with_protection(vector_field, y0, t_span, args=None):
    """Solve ODE with graceful handling of solver failures."""

    term = diffrax.ODETerm(vector_field)

    # Start with explicit solver
    solver = diffrax.Tsit5()

    controller = diffrax.PIDController(
        rtol=1e-5,
        atol=1e-7,
        dtmin=1e-12,
        force_dtmin=True,  # Don't error, just use dtmin
    )

    t0, t1 = t_span

    solution = diffrax.diffeqsolve(
        term, solver,
        t0=t0, t1=t1, dt0=(t1 - t0) / 100,
        y0=y0,
        args=args,
        stepsize_controller=controller,
        max_steps=50000,
        throw=False,  # Don't throw on solver failure
    )

    # Check for solver issues
    if solution.result != diffrax.RESULTS.successful:
        print(f"Warning: Solver returned {solution.result}")

    return solution


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate solver selection patterns."""
    print("=" * 60)
    print("Diffrax Solver Selection Demo")
    print("=" * 60)

    # Explicit solver for non-stiff
    print("\n1. Explicit Solver (Harmonic Oscillator)")
    ts, ys = explicit_solver_example()
    print(f"   Solved {len(ts)} time points, final position: {ys[-1, 0]:.4f}")

    # Implicit solver for stiff
    print("\n2. Implicit Solver (Robertson Kinetics)")
    ts, ys = implicit_solver_example()
    print(f"   Solved {len(ts)} time points")
    print(f"   Final concentrations: {ys[-1]}")

    # Stiffness detection
    print("\n3. Stiffness Detection")
    stiffness_detection_example()

    # Compare solvers
    print("\n4. Solver Comparison")
    results = compare_solvers()
    for name, res in results.items():
        if res['success']:
            print(f"   {name}: {res['num_steps']} steps")
        else:
            print(f"   {name}: FAILED - {res.get('error', 'unknown')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
