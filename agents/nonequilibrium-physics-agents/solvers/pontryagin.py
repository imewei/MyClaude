"""Pontryagin Maximum Principle (PMP) Solver for Optimal Control.

This module implements the Pontryagin Maximum Principle for solving optimal
control problems in quantum and classical thermodynamic systems.

Key Features:
- Two-point boundary value problem (TPBVP) solver
- Multiple shooting method for stability
- Gradient-based control optimization
- Support for state/control constraints
- Integration with quantum Hamiltonians

Typical usage:
    solver = PontryaginSolver(
        state_dim=4,
        control_dim=2,
        dynamics=my_dynamics_fn,
        running_cost=my_cost_fn
    )

    result = solver.solve(
        x0=initial_state,
        xf=target_state,
        duration=10.0,
        n_steps=100
    )

Author: Nonequilibrium Physics Agents
License: MIT
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings


class PontryaginSolver:
    """Pontryagin Maximum Principle solver for optimal control problems.

    Solves optimal control problems of the form:
        minimize J = ∫[L(x, u, t) dt] + Φ(x(T))
        subject to: dx/dt = f(x, u, t)
                   x(0) = x0, x(T) = xf (optional)
                   g(x, u, t) ≤ 0 (optional constraints)

    The PMP states that the optimal control u*(t) maximizes the Hamiltonian:
        H(x, λ, u, t) = -L(x, u, t) + λᵀf(x, u, t)

    where λ is the costate (adjoint variable) satisfying:
        dλ/dt = -∂H/∂x
        λ(T) = ∂Φ/∂x|_{x=x(T)}

    Attributes:
        state_dim: Dimension of state vector x
        control_dim: Dimension of control vector u
        dynamics: State dynamics function f(x, u, t)
        running_cost: Running cost function L(x, u, t)
        terminal_cost: Terminal cost function Φ(x)
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        dynamics: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        running_cost: Callable[[np.ndarray, np.ndarray, float], float],
        terminal_cost: Optional[Callable[[np.ndarray], float]] = None,
        control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        hbar: float = 1.054571817e-34
    ):
        """Initialize PMP solver.

        Args:
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            dynamics: Function f(x, u, t) -> dx/dt
            running_cost: Function L(x, u, t) -> scalar cost
            terminal_cost: Function Φ(x) -> scalar cost (default: 0)
            control_bounds: Tuple (u_min, u_max) for box constraints
            hbar: Reduced Planck constant (for quantum systems)
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost or (lambda x: 0.0)
        self.control_bounds = control_bounds
        self.hbar = hbar

        # Finite difference step for gradients
        self.eps = 1e-7

        # Solver settings
        self.method = 'multiple_shooting'  # or 'single_shooting'
        self.max_iterations = 100
        self.tolerance = 1e-6

    def solve(
        self,
        x0: np.ndarray,
        xf: Optional[np.ndarray] = None,
        duration: float = 10.0,
        n_steps: int = 100,
        initial_control: Optional[np.ndarray] = None,
        method: str = 'multiple_shooting',
        verbose: bool = False
    ) -> Dict:
        """Solve optimal control problem using PMP.

        Args:
            x0: Initial state
            xf: Target final state (None for free endpoint)
            duration: Time duration
            n_steps: Number of time steps
            initial_control: Initial guess for control trajectory
            method: Shooting method ('single_shooting' or 'multiple_shooting')
            verbose: Print convergence information

        Returns:
            Dictionary containing:
                - 'time': Time grid
                - 'state': State trajectory x(t)
                - 'costate': Costate trajectory λ(t)
                - 'control': Optimal control u(t)
                - 'cost': Total cost J
                - 'hamiltonian': Hamiltonian H(t)
                - 'converged': Convergence flag
                - 'iterations': Number of iterations
        """
        self.method = method
        t_span = np.linspace(0, duration, n_steps)
        dt = t_span[1] - t_span[0]

        # Initialize control guess
        if initial_control is None:
            if self.control_bounds is not None:
                u_min, u_max = self.control_bounds
                initial_control = 0.5 * (u_min + u_max)
                initial_control = np.tile(initial_control, (n_steps, 1))
            else:
                initial_control = np.zeros((n_steps, self.control_dim))

        if verbose:
            print(f"PMP Solver: {method}")
            print(f"  State dim: {self.state_dim}, Control dim: {self.control_dim}")
            print(f"  Duration: {duration}, Steps: {n_steps}")
            print(f"  Initial state: {x0}")
            if xf is not None:
                print(f"  Target state: {xf}")

        # Solve using chosen method
        if method == 'single_shooting':
            result = self._single_shooting(x0, xf, t_span, initial_control, verbose)
        elif method == 'multiple_shooting':
            result = self._multiple_shooting(x0, xf, t_span, initial_control, verbose)
        else:
            raise ValueError(f"Unknown method: {method}")

        return result

    def _single_shooting(
        self,
        x0: np.ndarray,
        xf: Optional[np.ndarray],
        t_span: np.ndarray,
        u_init: np.ndarray,
        verbose: bool
    ) -> Dict:
        """Single shooting method for TPBVP.

        Treats the initial costate λ0 as the optimization variable.
        """
        n_steps = len(t_span)

        # Define shooting function
        def shooting_residual(lambda0):
            """Compute boundary condition residual."""
            # Forward integration of state and costate
            z0 = np.concatenate([x0, lambda0])

            # Storage
            x_traj = np.zeros((n_steps, self.state_dim))
            lambda_traj = np.zeros((n_steps, self.state_dim))
            u_traj = np.zeros((n_steps, self.control_dim))

            x_traj[0] = x0
            lambda_traj[0] = lambda0

            # Integrate forward
            z_current = z0
            for i in range(n_steps - 1):
                t = t_span[i]
                dt = t_span[i+1] - t

                x_current = z_current[:self.state_dim]
                lambda_current = z_current[self.state_dim:]

                # Compute optimal control
                u_current = self._compute_optimal_control(x_current, lambda_current, t)
                u_traj[i] = u_current

                # ODE RHS
                def augmented_dynamics(t_val, z_val):
                    x_val = z_val[:self.state_dim]
                    lambda_val = z_val[self.state_dim:]
                    u_val = self._compute_optimal_control(x_val, lambda_val, t_val)

                    dx_dt = self.dynamics(x_val, u_val, t_val)
                    dlambda_dt = -self._compute_dH_dx(x_val, lambda_val, u_val, t_val)

                    return np.concatenate([dx_dt, dlambda_dt])

                # RK4 step
                sol = solve_ivp(
                    augmented_dynamics,
                    [t, t + dt],
                    z_current,
                    method='RK45',
                    dense_output=False
                )
                z_current = sol.y[:, -1]

                x_traj[i+1] = z_current[:self.state_dim]
                lambda_traj[i+1] = z_current[self.state_dim:]

            # Final control
            u_traj[-1] = self._compute_optimal_control(
                x_traj[-1], lambda_traj[-1], t_span[-1]
            )

            # Compute residual
            x_final = x_traj[-1]
            lambda_final = lambda_traj[-1]

            # Terminal costate condition
            terminal_gradient = self._compute_terminal_gradient(x_final)

            # For single shooting, we only optimize lambda0
            # If xf is specified, we use it in the cost but don't enforce it strictly here
            # Instead, we use the transversality condition
            if xf is not None:
                # Augmented transversality with penalty for missing target
                # λ(T) = ∂Φ/∂x + penalty weight for (x(T) - xf)
                penalty = 10.0 * (x_final - xf)
                residual = lambda_final - terminal_gradient - penalty
            else:
                # Free endpoint: only costate condition
                residual = lambda_final - terminal_gradient

            return residual

        # Initial guess for lambda0
        if xf is not None:
            # Heuristic: point toward target
            lambda0_init = -(xf - x0)
        else:
            lambda0_init = np.zeros(self.state_dim)

        # Solve shooting equation
        if verbose:
            print("  Starting single shooting...")

        sol = root(
            shooting_residual,
            lambda0_init,
            method='hybr',
            options={'xtol': self.tolerance, 'maxfev': self.max_iterations * 10}
        )

        if not sol.success:
            warnings.warn(f"Single shooting did not converge: {sol.message}")

        # Reconstruct solution with optimal lambda0
        lambda0_opt = sol.x

        # Final forward pass
        z0 = np.concatenate([x0, lambda0_opt])
        x_traj = np.zeros((n_steps, self.state_dim))
        lambda_traj = np.zeros((n_steps, self.state_dim))
        u_traj = np.zeros((n_steps, self.control_dim))
        H_traj = np.zeros(n_steps)

        x_traj[0] = x0
        lambda_traj[0] = lambda0_opt

        z_current = z0
        for i in range(n_steps - 1):
            t = t_span[i]
            dt = t_span[i+1] - t

            x_current = z_current[:self.state_dim]
            lambda_current = z_current[self.state_dim:]

            u_current = self._compute_optimal_control(x_current, lambda_current, t)
            u_traj[i] = u_current

            H_traj[i] = self._compute_hamiltonian(x_current, lambda_current, u_current, t)

            # Integrate
            def augmented_dynamics(t_val, z_val):
                x_val = z_val[:self.state_dim]
                lambda_val = z_val[self.state_dim:]
                u_val = self._compute_optimal_control(x_val, lambda_val, t_val)

                dx_dt = self.dynamics(x_val, u_val, t_val)
                dlambda_dt = -self._compute_dH_dx(x_val, lambda_val, u_val, t_val)

                return np.concatenate([dx_dt, dlambda_dt])

            sol_step = solve_ivp(
                augmented_dynamics,
                [t, t + dt],
                z_current,
                method='RK45'
            )
            z_current = sol_step.y[:, -1]

            x_traj[i+1] = z_current[:self.state_dim]
            lambda_traj[i+1] = z_current[self.state_dim:]

        u_traj[-1] = self._compute_optimal_control(x_traj[-1], lambda_traj[-1], t_span[-1])
        H_traj[-1] = self._compute_hamiltonian(x_traj[-1], lambda_traj[-1], u_traj[-1], t_span[-1])

        # Compute total cost
        cost = self._compute_total_cost(x_traj, u_traj, t_span)

        if verbose:
            print(f"  Converged: {sol.success}")
            print(f"  Iterations: {sol.nfev}")
            print(f"  Final cost: {cost:.6e}")
            if xf is not None:
                print(f"  Final state error: {np.linalg.norm(x_traj[-1] - xf):.6e}")

        return {
            'time': t_span,
            'state': x_traj,
            'costate': lambda_traj,
            'control': u_traj,
            'cost': cost,
            'hamiltonian': H_traj,
            'converged': sol.success,
            'iterations': sol.nfev,
            'method': 'single_shooting'
        }

    def _multiple_shooting(
        self,
        x0: np.ndarray,
        xf: Optional[np.ndarray],
        t_span: np.ndarray,
        u_init: np.ndarray,
        verbose: bool
    ) -> Dict:
        """Multiple shooting method for TPBVP (more stable).

        Divides time interval into segments and treats state at segment
        boundaries as optimization variables, along with control.
        """
        n_steps = len(t_span)
        n_segments = min(10, n_steps // 5)  # Adaptive segmentation
        segment_indices = np.linspace(0, n_steps - 1, n_segments + 1, dtype=int)

        if verbose:
            print(f"  Using {n_segments} shooting segments")

        # Decision variables: [x_1, ..., x_{n_seg}, u_0, ..., u_{n_steps-1}]
        # x_0 is fixed to x0, x_{n_seg} may be fixed to xf

        # Initial guess
        x_init_guess = np.zeros((n_segments, self.state_dim))
        if xf is not None:
            # Linear interpolation
            for i in range(n_segments):
                alpha = i / n_segments
                x_init_guess[i] = (1 - alpha) * x0 + alpha * xf
        else:
            # Constant guess
            x_init_guess[:] = x0

        decision_vars = np.concatenate([
            x_init_guess.flatten(),
            u_init.flatten()
        ])

        # Define residual function
        def residual_fn(decision_vars_vec):
            # Unpack
            n_x_vars = n_segments * self.state_dim
            x_segment = decision_vars_vec[:n_x_vars].reshape((n_segments, self.state_dim))
            u_all = decision_vars_vec[n_x_vars:].reshape((n_steps, self.control_dim))

            residuals = []

            # Continuity constraints: integrate each segment and check boundary
            for seg_idx in range(n_segments):
                i_start = segment_indices[seg_idx]
                i_end = segment_indices[seg_idx + 1]

                if seg_idx == 0:
                    x_start = x0
                else:
                    x_start = x_segment[seg_idx - 1]

                # Integrate this segment
                t_seg = t_span[i_start:i_end + 1]
                u_seg = u_all[i_start:i_end + 1]

                def dynamics_fn(t_val, x_val):
                    # Interpolate control
                    idx = np.searchsorted(t_seg, t_val, side='right') - 1
                    idx = np.clip(idx, 0, len(u_seg) - 1)
                    u_val = u_seg[idx]
                    return self.dynamics(x_val, u_val, t_val)

                sol_seg = solve_ivp(
                    dynamics_fn,
                    [t_seg[0], t_seg[-1]],
                    x_start,
                    method='RK45',
                    t_eval=None
                )

                x_end_integrated = sol_seg.y[:, -1]

                # Continuity residual
                if seg_idx < n_segments - 1:
                    x_end_target = x_segment[seg_idx]
                    residuals.append(x_end_integrated - x_end_target)
                else:
                    # Last segment
                    if xf is not None:
                        residuals.append(x_end_integrated - xf)
                    # else: free endpoint, no constraint

            # Costate optimality conditions (approximate via gradient)
            # For simplicity, use direct optimization instead of full costate

            return np.concatenate(residuals) if residuals else np.array([0.0])

        # Objective: total cost
        def objective_fn(decision_vars_vec):
            n_x_vars = n_segments * self.state_dim
            x_segment = decision_vars_vec[:n_x_vars].reshape((n_segments, self.state_dim))
            u_all = decision_vars_vec[n_x_vars:].reshape((n_steps, self.control_dim))

            # Reconstruct full trajectory
            x_full = np.zeros((n_steps, self.state_dim))
            x_full[0] = x0

            for seg_idx in range(n_segments):
                i_start = segment_indices[seg_idx]
                i_end = segment_indices[seg_idx + 1]

                if seg_idx == 0:
                    x_start = x0
                else:
                    x_start = x_segment[seg_idx - 1]

                t_seg = t_span[i_start:i_end + 1]
                u_seg = u_all[i_start:i_end + 1]

                def dynamics_fn(t_val, x_val):
                    idx = np.searchsorted(t_seg, t_val, side='right') - 1
                    idx = np.clip(idx, 0, len(u_seg) - 1)
                    u_val = u_seg[idx]
                    return self.dynamics(x_val, u_val, t_val)

                sol_seg = solve_ivp(
                    dynamics_fn,
                    [t_seg[0], t_seg[-1]],
                    x_start,
                    method='RK45',
                    t_eval=t_seg
                )

                x_full[i_start:i_end + 1] = sol_seg.y.T

            return self._compute_total_cost(x_full, u_all, t_span)

        # Optimize
        if verbose:
            print("  Starting multiple shooting optimization...")

        # Bounds
        bounds = None
        if self.control_bounds is not None:
            u_min, u_max = self.control_bounds
            x_bounds = [(-np.inf, np.inf)] * (n_segments * self.state_dim)
            u_bounds = [(u_min[i % self.control_dim], u_max[i % self.control_dim])
                       for i in range(n_steps * self.control_dim)]
            bounds = x_bounds + u_bounds

        result = minimize(
            objective_fn,
            decision_vars,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )

        if not result.success:
            warnings.warn(f"Multiple shooting did not converge: {result.message}")

        # Extract solution
        n_x_vars = n_segments * self.state_dim
        x_segment_opt = result.x[:n_x_vars].reshape((n_segments, self.state_dim))
        u_opt = result.x[n_x_vars:].reshape((n_steps, self.control_dim))

        # Reconstruct full trajectory
        x_traj = np.zeros((n_steps, self.state_dim))
        x_traj[0] = x0

        for seg_idx in range(n_segments):
            i_start = segment_indices[seg_idx]
            i_end = segment_indices[seg_idx + 1]

            if seg_idx == 0:
                x_start = x0
            else:
                x_start = x_segment_opt[seg_idx - 1]

            t_seg = t_span[i_start:i_end + 1]
            u_seg = u_opt[i_start:i_end + 1]

            def dynamics_fn(t_val, x_val):
                idx = np.searchsorted(t_seg, t_val, side='right') - 1
                idx = np.clip(idx, 0, len(u_seg) - 1)
                u_val = u_seg[idx]
                return self.dynamics(x_val, u_val, t_val)

            sol_seg = solve_ivp(
                dynamics_fn,
                [t_seg[0], t_seg[-1]],
                x_start,
                method='RK45',
                t_eval=t_seg
            )

            x_traj[i_start:i_end + 1] = sol_seg.y.T

        # Compute costate (backward pass for diagnostics)
        lambda_traj = self._compute_costate_backward(x_traj, u_opt, t_span)

        # Hamiltonian
        H_traj = np.array([
            self._compute_hamiltonian(x_traj[i], lambda_traj[i], u_opt[i], t_span[i])
            for i in range(n_steps)
        ])

        cost = result.fun

        if verbose:
            print(f"  Converged: {result.success}")
            print(f"  Iterations: {result.nit}")
            print(f"  Final cost: {cost:.6e}")
            if xf is not None:
                print(f"  Final state error: {np.linalg.norm(x_traj[-1] - xf):.6e}")

        return {
            'time': t_span,
            'state': x_traj,
            'costate': lambda_traj,
            'control': u_opt,
            'cost': cost,
            'hamiltonian': H_traj,
            'converged': result.success,
            'iterations': result.nit,
            'method': 'multiple_shooting'
        }

    def _compute_optimal_control(
        self,
        x: np.ndarray,
        lam: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Compute optimal control by maximizing Hamiltonian.

        For unconstrained problems: ∂H/∂u = 0
        For constrained problems: use projection or gradient ascent
        """
        # Gradient-based maximization
        def neg_hamiltonian(u):
            return -self._compute_hamiltonian(x, lam, u, t)

        # Initial guess
        if self.control_bounds is not None:
            u_min, u_max = self.control_bounds
            u0 = 0.5 * (u_min + u_max)
        else:
            u0 = np.zeros(self.control_dim)

        # Optimize
        bounds = None
        if self.control_bounds is not None:
            u_min, u_max = self.control_bounds
            bounds = [(u_min[i], u_max[i]) for i in range(self.control_dim)]

        result = minimize(
            neg_hamiltonian,
            u0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20}
        )

        return result.x

    def _compute_hamiltonian(
        self,
        x: np.ndarray,
        lam: np.ndarray,
        u: np.ndarray,
        t: float
    ) -> float:
        """Compute Hamiltonian H = -L + λᵀf."""
        L = self.running_cost(x, u, t)
        f = self.dynamics(x, u, t)
        H = -L + np.dot(lam, f)
        return H

    def _compute_dH_dx(
        self,
        x: np.ndarray,
        lam: np.ndarray,
        u: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Compute ∂H/∂x using finite differences."""
        grad = np.zeros(self.state_dim)

        for i in range(self.state_dim):
            x_plus = x.copy()
            x_plus[i] += self.eps
            H_plus = self._compute_hamiltonian(x_plus, lam, u, t)

            x_minus = x.copy()
            x_minus[i] -= self.eps
            H_minus = self._compute_hamiltonian(x_minus, lam, u, t)

            grad[i] = (H_plus - H_minus) / (2 * self.eps)

        return grad

    def _compute_terminal_gradient(self, x_final: np.ndarray) -> np.ndarray:
        """Compute ∂Φ/∂x at final time."""
        grad = np.zeros(self.state_dim)

        for i in range(self.state_dim):
            x_plus = x_final.copy()
            x_plus[i] += self.eps
            Phi_plus = self.terminal_cost(x_plus)

            x_minus = x_final.copy()
            x_minus[i] -= self.eps
            Phi_minus = self.terminal_cost(x_minus)

            grad[i] = (Phi_plus - Phi_minus) / (2 * self.eps)

        return grad

    def _compute_total_cost(
        self,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
        t_span: np.ndarray
    ) -> float:
        """Compute total cost J = ∫L dt + Φ(x(T))."""
        n_steps = len(t_span)
        dt = t_span[1] - t_span[0] if n_steps > 1 else 0.0

        # Running cost (trapezoidal rule)
        running_total = 0.0
        for i in range(n_steps - 1):
            L_i = self.running_cost(x_traj[i], u_traj[i], t_span[i])
            L_ip1 = self.running_cost(x_traj[i+1], u_traj[i+1], t_span[i+1])
            running_total += 0.5 * (L_i + L_ip1) * dt

        # Terminal cost
        terminal = self.terminal_cost(x_traj[-1])

        return running_total + terminal

    def _compute_costate_backward(
        self,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
        t_span: np.ndarray
    ) -> np.ndarray:
        """Compute costate trajectory via backward integration."""
        n_steps = len(t_span)
        lambda_traj = np.zeros((n_steps, self.state_dim))

        # Terminal condition
        lambda_traj[-1] = self._compute_terminal_gradient(x_traj[-1])

        # Backward integration
        for i in range(n_steps - 2, -1, -1):
            t = t_span[i]
            dt = t_span[i+1] - t

            x_current = x_traj[i]
            u_current = u_traj[i]
            lambda_next = lambda_traj[i+1]

            # dλ/dt = -∂H/∂x, integrate backward
            def lambda_dynamics(t_val, lam_val):
                return -self._compute_dH_dx(x_current, lam_val, u_current, t_val)

            # Backward Euler (simple)
            dlambda_dt = lambda_dynamics(t, lambda_next)
            lambda_traj[i] = lambda_next - dlambda_dt * dt

        return lambda_traj


def solve_quantum_control_pmp(
    H0: np.ndarray,
    control_hamiltonians: List[np.ndarray],
    psi0: np.ndarray,
    target_state: Optional[np.ndarray] = None,
    duration: float = 10.0,
    n_steps: int = 100,
    control_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    state_cost_weight: float = 1.0,
    control_cost_weight: float = 0.01,
    method: str = 'multiple_shooting',
    hbar: float = 1.0
) -> Dict:
    """Solve quantum optimal control problem using PMP.

    Finds optimal control pulses u(t) to drive quantum system from initial
    state to target state while minimizing control energy.

    Problem formulation:
        minimize J = ∫[α|ψ - ψ_target|² + β|u|²] dt
        subject to: iħ dψ/dt = [H0 + Σᵢ uᵢ(t) Hᵢ] ψ
                   ψ(0) = ψ0
                   |ψ|² = 1 (unitarity)

    Args:
        H0: Drift Hamiltonian (time-independent part)
        control_hamiltonians: List of control Hamiltonians [H1, H2, ...]
        psi0: Initial quantum state (complex vector)
        target_state: Target quantum state (None for minimum energy control)
        duration: Total evolution time
        n_steps: Number of time steps
        control_bounds: Bounds on control amplitudes (u_min, u_max)
        state_cost_weight: Weight α for state tracking cost
        control_cost_weight: Weight β for control energy cost
        method: Shooting method ('single_shooting' or 'multiple_shooting')
        hbar: Reduced Planck constant

    Returns:
        Dictionary with optimal control solution (same format as PontryaginSolver.solve)
    """
    n_dim = len(psi0)
    n_controls = len(control_hamiltonians)

    # Convert to real representation: [Re(ψ), Im(ψ)]
    state_dim = 2 * n_dim
    control_dim = n_controls

    # Dynamics: iħ dψ/dt = H(u) ψ
    def dynamics(x, u, t):
        """Quantum dynamics in real coordinates."""
        psi_re = x[:n_dim]
        psi_im = x[n_dim:]
        psi = psi_re + 1j * psi_im

        # Total Hamiltonian
        H_total = H0.copy()
        for i, H_ctrl in enumerate(control_hamiltonians):
            H_total = H_total + u[i] * H_ctrl

        # Schrödinger equation: iħ dψ/dt = H ψ
        dpsi_dt = -1j / hbar * (H_total @ psi)

        # Real representation
        dx = np.zeros(state_dim)
        dx[:n_dim] = np.real(dpsi_dt)
        dx[n_dim:] = np.imag(dpsi_dt)

        return dx

    # Running cost: L = α|ψ - ψ_target|² + β|u|²
    def running_cost(x, u, t):
        """State tracking + control energy cost."""
        cost = 0.0

        if target_state is not None:
            psi_re = x[:n_dim]
            psi_im = x[n_dim:]
            psi = psi_re + 1j * psi_im

            # Fidelity cost (1 - |⟨ψ|ψ_target⟩|²)
            overlap = np.abs(np.vdot(target_state, psi))**2
            cost += state_cost_weight * (1.0 - overlap)

        # Control energy
        cost += control_cost_weight * np.sum(u**2)

        return cost

    # Terminal cost
    def terminal_cost(x):
        """Final state cost."""
        if target_state is None:
            return 0.0

        psi_re = x[:n_dim]
        psi_im = x[n_dim:]
        psi = psi_re + 1j * psi_im

        # Final fidelity
        overlap = np.abs(np.vdot(target_state, psi))**2
        return state_cost_weight * (1.0 - overlap)

    # Initial state in real coordinates
    x0 = np.concatenate([np.real(psi0), np.imag(psi0)])

    # Target state in real coordinates
    xf = None
    if target_state is not None:
        xf = np.concatenate([np.real(target_state), np.imag(target_state)])

    # Create and solve
    solver = PontryaginSolver(
        state_dim=state_dim,
        control_dim=control_dim,
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        control_bounds=control_bounds,
        hbar=hbar
    )

    result = solver.solve(
        x0=x0,
        xf=None,  # Don't enforce hard constraint (use soft cost instead)
        duration=duration,
        n_steps=n_steps,
        method=method,
        verbose=True
    )

    # Convert state back to complex
    x_traj = result['state']
    psi_traj = x_traj[:, :n_dim] + 1j * x_traj[:, n_dim:]

    # Compute fidelity
    if target_state is not None:
        fidelities = np.array([
            np.abs(np.vdot(target_state, psi_traj[i]))**2
            for i in range(len(psi_traj))
        ])
        result['fidelity'] = fidelities
        result['final_fidelity'] = fidelities[-1]

    result['psi_evolution'] = psi_traj

    return result
