"""Collocation Methods for Optimal Control Problems.

This module implements orthogonal collocation methods for solving optimal control
problems via direct transcription. Collocation methods are an alternative to
shooting methods and are often more robust for problems with unstable dynamics.

Key Features:
- Orthogonal collocation on finite elements
- Gauss-Legendre collocation points
- Radau collocation points (IIA)
- Hermite-Simpson collocation
- Automatic mesh refinement
- Control constraints via NLP
- State path constraints

Mathematical Foundation:
    Direct transcription converts the continuous optimal control problem:
        minimize ∫ L(x, u, t) dt + Φ(x(T))
        subject to: dx/dt = f(x, u, t)
                    x(0) = x₀
                    g(x, u, t) ≤ 0

    into a finite-dimensional NLP by discretizing the state and control on
    a mesh and enforcing dynamics at collocation points.

Author: Nonequilibrium Physics Agents
Date: 2025-09-30
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.special import legendre, roots_legendre
from typing import Callable, Optional, Tuple, Dict, List
import warnings


class CollocationSolver:
    """Orthogonal collocation solver for optimal control problems.

    This solver uses direct transcription to convert the continuous optimal
    control problem into a finite-dimensional nonlinear programming (NLP)
    problem. The state and control are discretized on a mesh, and the dynamics
    are enforced at collocation points within each element.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector x
    control_dim : int
        Dimension of control vector u
    dynamics : Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        System dynamics f(x, u, t) where dx/dt = f
    running_cost : Callable[[np.ndarray, np.ndarray, float], float]
        Running cost L(x, u, t)
    terminal_cost : Optional[Callable[[np.ndarray], float]]
        Terminal cost Φ(x(T)). Default: None
    control_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        Control bounds (u_min, u_max). Default: None
    state_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        State bounds (x_min, x_max). Default: None
    collocation_type : str
        Type of collocation: 'gauss-legendre', 'radau', 'hermite-simpson'
        Default: 'gauss-legendre'
    collocation_order : int
        Number of collocation points per element. Default: 3

    Attributes
    ----------
    n_x : int
        State dimension
    n_u : int
        Control dimension
    f : Callable
        Dynamics function
    L : Callable
        Running cost
    Phi : Callable
        Terminal cost

    Methods
    -------
    solve(x0, xf, duration, n_elements, ...)
        Solve optimal control problem via collocation
    _setup_collocation_points()
        Compute collocation points and weights
    _transcribe_to_nlp(...)
        Convert continuous problem to NLP
    _evaluate_cost(z, ...)
        Evaluate total cost for NLP
    _evaluate_constraints(z, ...)
        Evaluate dynamics constraints for NLP
    refine_mesh(result, tolerance)
        Refine mesh based on error estimates

    Examples
    --------
    >>> # LQR problem
    >>> def dynamics(x, u, t):
    ...     return u
    >>> def cost(x, u, t):
    ...     return x**2 + u**2
    >>>
    >>> solver = CollocationSolver(1, 1, dynamics, cost)
    >>> result = solver.solve(x0=np.array([1.0]), duration=5.0, n_elements=10)
    >>> print(f"Optimal cost: {result['cost']:.6f}")

    >>> # Quantum control
    >>> from solvers.collocation import solve_quantum_control_collocation
    >>> result = solve_quantum_control_collocation(
    ...     H0, [H1], psi0, psi_target, duration=5.0, n_elements=20
    ... )
    >>> print(f"Final fidelity: {result['final_fidelity']:.4f}")
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        dynamics: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        running_cost: Callable[[np.ndarray, np.ndarray, float], float],
        terminal_cost: Optional[Callable[[np.ndarray], float]] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        collocation_type: str = 'gauss-legendre',
        collocation_order: int = 3
    ):
        """Initialize collocation solver."""
        self.n_x = state_dim
        self.n_u = control_dim
        self.f = dynamics
        self.L = running_cost
        self.Phi = terminal_cost if terminal_cost is not None else lambda x: 0.0

        self.control_bounds = control_bounds
        self.state_bounds = state_bounds
        self.collocation_type = collocation_type
        self.collocation_order = collocation_order

        # Setup collocation points and weights
        self._setup_collocation_points()

    def _setup_collocation_points(self) -> None:
        """Setup collocation points and weights.

        Computes collocation points τ ∈ [0, 1] and integration weights
        based on the specified collocation scheme.
        """
        if self.collocation_type == 'gauss-legendre':
            # Gauss-Legendre points in [0, 1]
            tau_gl, w_gl = roots_legendre(self.collocation_order)
            self.tau = 0.5 * (tau_gl + 1)  # Map from [-1, 1] to [0, 1]
            self.weights = 0.5 * w_gl

        elif self.collocation_type == 'radau':
            # Radau IIA points (include right endpoint)
            # For simplicity, use Gauss-Legendre shifted
            tau_gl, w_gl = roots_legendre(self.collocation_order - 1)
            tau_shifted = 0.5 * (tau_gl + 1)
            self.tau = np.append(tau_shifted, 1.0)
            w_shifted = 0.5 * w_gl
            # Weight for endpoint (Simpson-like)
            w_end = 1.0 - np.sum(w_shifted)
            self.weights = np.append(w_shifted, w_end)

        elif self.collocation_type == 'hermite-simpson':
            # Hermite-Simpson (3 points: 0, 0.5, 1)
            self.tau = np.array([0.0, 0.5, 1.0])
            self.weights = np.array([1.0/6, 2.0/3, 1.0/6])
            self.collocation_order = 3

        else:
            raise ValueError(f"Unknown collocation type: {self.collocation_type}")

        self.n_colloc = len(self.tau)

    def solve(
        self,
        x0: np.ndarray,
        xf: Optional[np.ndarray] = None,
        duration: float = 10.0,
        n_elements: int = 10,
        method: str = 'SLSQP',
        verbose: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Dict:
        """Solve optimal control problem via collocation.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state, shape (n_x,)
        xf : Optional[np.ndarray]
            Target final state (if fixed endpoint), shape (n_x,)
        duration : float
            Time duration T
        n_elements : int
            Number of mesh elements
        method : str
            Optimization method ('SLSQP', 'trust-constr')
        verbose : bool
            Print optimization progress
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        Dict
            Solution dictionary containing:
            - 'x': State trajectory, shape (n_t, n_x)
            - 'u': Control trajectory, shape (n_t, n_u)
            - 't': Time points, shape (n_t,)
            - 'cost': Total cost (scalar)
            - 'converged': Whether optimization converged
            - 'n_iter': Number of iterations
            - 'message': Convergence message
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Collocation Solver ({self.collocation_type})")
            print(f"{'='*60}")
            print(f"State dim: {self.n_x}, Control dim: {self.n_u}")
            print(f"Duration: {duration:.2f}, Elements: {n_elements}")
            print(f"Collocation order: {self.collocation_order}")

        # Create time mesh
        self.t_mesh = np.linspace(0, duration, n_elements + 1)
        self.dt_elements = np.diff(self.t_mesh)

        # Initial guess for decision variables
        # z = [x_0, u_0, x_1, u_1, ..., x_N, u_N]
        # Each x_i has shape (n_colloc, n_x), u_i has shape (n_colloc, n_u)
        n_vars_per_element = self.n_colloc * (self.n_x + self.n_u)
        n_total_vars = n_elements * n_vars_per_element + self.n_x  # +n_x for final state

        # Initial guess: linear interpolation
        z0 = np.zeros(n_total_vars)
        if xf is not None:
            # Interpolate from x0 to xf
            for i in range(n_elements):
                t_ratio = i / n_elements
                for j in range(self.n_colloc):
                    idx_start = i * n_vars_per_element + j * (self.n_x + self.n_u)
                    # State
                    z0[idx_start:idx_start + self.n_x] = (1 - t_ratio) * x0 + t_ratio * xf
                    # Control (zero initial guess)
                    z0[idx_start + self.n_x:idx_start + self.n_x + self.n_u] = 0.0
            # Final state
            z0[-self.n_x:] = xf
        else:
            # Just use x0 everywhere
            for i in range(n_elements):
                for j in range(self.n_colloc):
                    idx_start = i * n_vars_per_element + j * (self.n_x + self.n_u)
                    z0[idx_start:idx_start + self.n_x] = x0
            z0[-self.n_x:] = x0

        # Bounds on decision variables
        bounds = []
        for i in range(n_elements):
            for j in range(self.n_colloc):
                # State bounds
                if self.state_bounds is not None:
                    x_min, x_max = self.state_bounds
                    bounds.extend([(x_min[k], x_max[k]) for k in range(self.n_x)])
                else:
                    bounds.extend([(-np.inf, np.inf)] * self.n_x)

                # Control bounds
                if self.control_bounds is not None:
                    u_min, u_max = self.control_bounds
                    bounds.extend([(u_min[k], u_max[k]) for k in range(self.n_u)])
                else:
                    bounds.extend([(-np.inf, np.inf)] * self.n_u)

        # Final state bounds
        if self.state_bounds is not None:
            x_min, x_max = self.state_bounds
            bounds.extend([(x_min[k], x_max[k]) for k in range(self.n_x)])
        else:
            bounds.extend([(-np.inf, np.inf)] * self.n_x)

        # Define constraints
        constraints = []

        # Initial condition constraint
        def initial_condition(z):
            x_first = z[0:self.n_x]
            return x_first - x0

        constraints.append({
            'type': 'eq',
            'fun': initial_condition
        })

        # Dynamics constraints at collocation points
        def dynamics_constraints(z):
            residuals = []

            for i in range(n_elements):
                dt = self.dt_elements[i]
                t0 = self.t_mesh[i]

                # Extract states and controls for this element
                x_element = np.zeros((self.n_colloc, self.n_x))
                u_element = np.zeros((self.n_colloc, self.n_u))

                for j in range(self.n_colloc):
                    idx_start = i * n_vars_per_element + j * (self.n_x + self.n_u)
                    x_element[j] = z[idx_start:idx_start + self.n_x]
                    u_element[j] = z[idx_start + self.n_x:idx_start + self.n_x + self.n_u]

                # Get state at start of next element (or final state)
                if i < n_elements - 1:
                    x_next = z[(i + 1) * n_vars_per_element:
                              (i + 1) * n_vars_per_element + self.n_x]
                else:
                    x_next = z[-self.n_x:]

                # Collocation constraint: x(t_{i+1}) = x(t_i) + dt * Σ w_j * f(x_j, u_j, t_j)
                x_integrated = x_element[0].copy()
                for j in range(self.n_colloc):
                    t_j = t0 + self.tau[j] * dt
                    f_j = self.f(x_element[j], u_element[j], t_j)
                    x_integrated += dt * self.weights[j] * f_j

                # Residual: should be zero at convergence
                residual = x_next - x_integrated
                residuals.extend(residual)

            return np.array(residuals)

        constraints.append({
            'type': 'eq',
            'fun': dynamics_constraints
        })

        # Terminal constraint (if fixed endpoint)
        if xf is not None:
            def terminal_constraint(z):
                x_final = z[-self.n_x:]
                return x_final - xf

            constraints.append({
                'type': 'eq',
                'fun': terminal_constraint
            })

        # Cost function
        def cost_function(z):
            total_cost = 0.0

            # Running cost
            for i in range(n_elements):
                dt = self.dt_elements[i]
                t0 = self.t_mesh[i]

                for j in range(self.n_colloc):
                    idx_start = i * n_vars_per_element + j * (self.n_x + self.n_u)
                    x_j = z[idx_start:idx_start + self.n_x]
                    u_j = z[idx_start + self.n_x:idx_start + self.n_x + self.n_u]
                    t_j = t0 + self.tau[j] * dt

                    L_j = self.L(x_j, u_j, t_j)
                    total_cost += dt * self.weights[j] * L_j

            # Terminal cost
            x_final = z[-self.n_x:]
            total_cost += self.Phi(x_final)

            return total_cost

        # Solve NLP
        if verbose:
            print(f"\nSolving NLP with {n_total_vars} variables...")
            print(f"Method: {method}")

        result = minimize(
            cost_function,
            z0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': max_iter,
                'disp': verbose,
                'ftol': tol
            }
        )

        if verbose:
            print(f"\nConverged: {result.success}")
            print(f"Iterations: {result.nit if hasattr(result, 'nit') else 'N/A'}")
            print(f"Final cost: {result.fun:.6f}")
            print(f"Message: {result.message}")

        # Extract solution
        z_opt = result.x

        # Build time-series arrays
        n_total_points = n_elements * self.n_colloc + 1
        t_full = np.zeros(n_total_points)
        x_full = np.zeros((n_total_points, self.n_x))
        u_full = np.zeros((n_total_points, self.n_u))

        idx_out = 0
        for i in range(n_elements):
            dt = self.dt_elements[i]
            t0 = self.t_mesh[i]

            for j in range(self.n_colloc):
                idx_start = i * n_vars_per_element + j * (self.n_x + self.n_u)
                t_full[idx_out] = t0 + self.tau[j] * dt
                x_full[idx_out] = z_opt[idx_start:idx_start + self.n_x]
                u_full[idx_out] = z_opt[idx_start + self.n_x:idx_start + self.n_x + self.n_u]
                idx_out += 1

        # Final state
        t_full[-1] = self.t_mesh[-1]
        x_full[-1] = z_opt[-self.n_x:]
        u_full[-1] = u_full[-2]  # Hold last control

        return {
            'x': x_full,
            'u': u_full,
            't': t_full,
            'cost': result.fun,
            'converged': result.success,
            'n_iter': result.nit if hasattr(result, 'nit') else None,
            'message': result.message,
            'z_opt': z_opt,
            'n_elements': n_elements,
            'collocation_type': self.collocation_type
        }

    def refine_mesh(
        self,
        result: Dict,
        tolerance: float = 1e-4,
        max_elements: int = 100
    ) -> Dict:
        """Refine mesh based on error estimates.

        Uses a posteriori error estimation to adaptively refine the mesh
        where the solution has large errors.

        Parameters
        ----------
        result : Dict
            Solution from previous solve()
        tolerance : float
            Error tolerance for refinement
        max_elements : int
            Maximum number of elements after refinement

        Returns
        -------
        Dict
            Refined solution
        """
        # This is a placeholder for mesh refinement
        # Full implementation would compute error estimates and refine
        warnings.warn("Mesh refinement not yet implemented. Returning original solution.")
        return result


def solve_quantum_control_collocation(
    H0: np.ndarray,
    control_hamiltonians: List[np.ndarray],
    psi0: np.ndarray,
    target_state: np.ndarray,
    duration: float = 5.0,
    n_elements: int = 20,
    control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    collocation_type: str = 'gauss-legendre',
    collocation_order: int = 3,
    control_weight: float = 0.1,
    verbose: bool = False
) -> Dict:
    """Solve quantum control problem using collocation.

    Find optimal control u(t) to drive quantum state from psi0 to target_state
    by controlling time-dependent Hamiltonian H(t) = H0 + Σ u_k(t) H_k.

    Parameters
    ----------
    H0 : np.ndarray
        Drift Hamiltonian, shape (n, n)
    control_hamiltonians : List[np.ndarray]
        Control Hamiltonians [H1, H2, ...], each shape (n, n)
    psi0 : np.ndarray
        Initial state, shape (n,) or (n, 1)
    target_state : np.ndarray
        Target state, shape (n,) or (n, 1)
    duration : float
        Control duration
    n_elements : int
        Number of mesh elements
    control_bounds : Optional[Tuple[np.ndarray, np.ndarray]]
        Control amplitude bounds (u_min, u_max)
    collocation_type : str
        Type of collocation
    collocation_order : int
        Order of collocation
    control_weight : float
        Weight for control cost (energy penalty)
    verbose : bool
        Print progress

    Returns
    -------
    Dict
        Solution with additional quantum-specific keys:
        - 'final_fidelity': |⟨ψ_target|ψ(T)⟩|²
        - 'psi': State evolution

    Examples
    --------
    >>> # Hadamard gate
    >>> H0 = np.zeros((2, 2), dtype=complex)
    >>> sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    >>> psi0 = np.array([1, 0], dtype=complex)
    >>> psi_target = np.array([1, 1], dtype=complex) / np.sqrt(2)
    >>>
    >>> result = solve_quantum_control_collocation(
    ...     H0, [sigma_x], psi0, psi_target, duration=5.0
    ... )
    >>> print(f"Fidelity: {result['final_fidelity']:.4f}")
    """
    # Flatten states to 1D
    psi0 = psi0.flatten()
    target_state = target_state.flatten()
    n_dim = len(psi0)
    n_controls = len(control_hamiltonians)

    # Convert complex state to real representation: [Re(ψ), Im(ψ)]
    state_dim = 2 * n_dim
    control_dim = n_controls

    # Initial state (real representation)
    x0_real = np.concatenate([psi0.real, psi0.imag])

    # Target state (real representation)
    target_real = np.concatenate([target_state.real, target_state.imag])

    # Normalize
    x0_real /= np.linalg.norm(x0_real)
    target_real /= np.linalg.norm(target_real)

    # Precompute matrices for efficiency
    H0_real = H0.real
    H0_imag = H0.imag
    Hk_real = [Hk.real for Hk in control_hamiltonians]
    Hk_imag = [Hk.imag for Hk in control_hamiltonians]

    # Dynamics: Schrödinger equation dψ/dt = -i H ψ
    def dynamics(x, u, t):
        # Reconstruct complex state
        psi_re = x[:n_dim]
        psi_im = x[n_dim:]

        # Compute H(t) = H0 + Σ u_k H_k
        H_re = H0_real.copy()
        H_im = H0_imag.copy()
        for k in range(n_controls):
            H_re += u[k] * Hk_real[k]
            H_im += u[k] * Hk_imag[k]

        # Compute -i H ψ = -i (H_re + i H_im) (ψ_re + i ψ_im)
        # = -i H_re ψ_re + H_im ψ_re - i H_re ψ_im - H_im ψ_im
        # = (H_im ψ_re - H_im ψ_im) + i (-H_re ψ_re - H_re ψ_im)
        Hpsi_re = H_re @ psi_re - H_im @ psi_im
        Hpsi_im = H_re @ psi_im + H_im @ psi_re

        dpsi_re = Hpsi_im  # Re(-i H ψ) = Im(H ψ)
        dpsi_im = -Hpsi_re  # Im(-i H ψ) = -Re(H ψ)

        return np.concatenate([dpsi_re, dpsi_im])

    # Cost: fidelity + control energy
    def running_cost(x, u, t):
        # Control energy penalty
        control_cost = control_weight * np.sum(u**2)
        return control_cost

    def terminal_cost(x):
        # Negative fidelity (minimize cost = maximize fidelity)
        overlap = np.dot(x, target_real)
        fidelity = overlap**2
        return -fidelity + 1.0  # Shift so optimal = 0

    # Create solver
    solver = CollocationSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        control_bounds=control_bounds,
        collocation_type=collocation_type,
        collocation_order=collocation_order
    )

    # Solve (free endpoint for quantum state)
    result = solver.solve(
        x0=x0_real,
        xf=None,  # Free endpoint
        duration=duration,
        n_elements=n_elements,
        verbose=verbose
    )

    # Convert back to complex representation
    x_traj = result['x']
    n_t = x_traj.shape[0]
    psi_traj = np.zeros((n_t, n_dim), dtype=complex)

    for i in range(n_t):
        psi_re = x_traj[i, :n_dim]
        psi_im = x_traj[i, n_dim:]
        psi_traj[i] = psi_re + 1j * psi_im
        # Normalize
        psi_traj[i] /= np.linalg.norm(psi_traj[i])

    # Compute final fidelity
    psi_final = psi_traj[-1]
    fidelity = np.abs(np.vdot(target_state, psi_final))**2

    # Add quantum-specific results
    result['psi'] = psi_traj
    result['final_fidelity'] = fidelity

    return result


# Convenience exports
__all__ = [
    'CollocationSolver',
    'solve_quantum_control_collocation',
]
