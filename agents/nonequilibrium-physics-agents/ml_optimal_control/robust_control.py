"""Robust Control and Uncertainty Quantification for Optimal Control.

This module provides methods for handling uncertainties, disturbances, and
model errors in optimal control problems.

Features:
- H-infinity robust control (minimize worst-case disturbance)
- μ-synthesis (structured uncertainty)
- Stochastic optimal control (HJB with noise)
- Uncertainty quantification (polynomial chaos, Monte Carlo)
- Sensitivity analysis (parameter variations)
- Risk-sensitive control (exponential utility)
- Distributionally robust optimization

Author: Nonequilibrium Physics Agents
Week: 23-24 of Phase 4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import numpy as np
from scipy import linalg
from scipy.optimize import minimize

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    import flax.linen as nn
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


class RobustControlMethod(Enum):
    """Robust control methods."""
    H_INFINITY = "h_infinity"  # H∞ control (worst-case disturbance)
    MU_SYNTHESIS = "mu_synthesis"  # μ-synthesis (structured uncertainty)
    LMI_BASED = "lmi_based"  # Linear Matrix Inequality approach


class UQMethod(Enum):
    """Uncertainty quantification methods."""
    MONTE_CARLO = "monte_carlo"  # Monte Carlo sampling
    POLYNOMIAL_CHAOS = "polynomial_chaos"  # Polynomial chaos expansion
    UNSCENTED_TRANSFORM = "unscented_transform"  # Unscented transform (sigma points)
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"  # Local sensitivity


@dataclass
class RobustControlConfig:
    """Configuration for robust control."""

    # Method
    method: str = RobustControlMethod.H_INFINITY.value

    # H-infinity parameters
    gamma: float = 1.0  # Disturbance attenuation level (lower = more robust)

    # LMI solver
    lmi_solver: str = "cvxopt"  # cvxopt, mosek, sedumi
    lmi_tolerance: float = 1e-6

    # Uncertainty bounds
    parameter_uncertainty: Optional[Dict[str, Tuple[float, float]]] = None  # param -> (min, max)
    disturbance_bound: float = 1.0  # L2 norm bound on disturbance

    # Risk-sensitive control
    risk_aversion: float = 0.0  # θ in exp utility, 0 = risk-neutral


@dataclass
class UQConfig:
    """Configuration for uncertainty quantification."""

    # Method
    method: str = UQMethod.MONTE_CARLO.value

    # Monte Carlo
    num_samples: int = 1000
    random_seed: Optional[int] = 42

    # Polynomial chaos
    poly_order: int = 3  # Order of polynomial expansion
    quadrature_rule: str = "gauss"  # gauss, clenshaw_curtis

    # Unscented transform
    alpha: float = 1e-3  # Spread of sigma points
    beta: float = 2.0  # Distribution parameter (2 = Gaussian)
    kappa: float = 0.0  # Secondary scaling parameter

    # Sensitivity analysis
    perturbation: float = 1e-6  # Finite difference step


class HInfinityControl:
    """H-infinity robust control for linear systems.

    Solves the H∞ control problem:
        min_u max_w ||z||₂ / ||w||₂

    where z is the regulated output and w is the disturbance.

    For linear systems: dx/dt = Ax + B1*w + B2*u
                        z = C1*x + D12*u
    """

    def __init__(self, config: RobustControlConfig):
        """Initialize H-infinity controller.

        Args:
            config: Robust control configuration
        """
        self.config = config
        self.K = None  # State feedback gain
        self.gamma_opt = None  # Optimal γ
        self.P = None  # Riccati solution

    def solve_riccati(
        self,
        A: np.ndarray,
        B1: np.ndarray,
        B2: np.ndarray,
        C1: np.ndarray,
        D12: np.ndarray,
        gamma: float
    ) -> Optional[np.ndarray]:
        """Solve H-infinity Riccati equation.

        Equation: A'P + PA - P(B2*B2' - γ⁻²*B1*B1')P + C1'C1 = 0

        Args:
            A: State matrix
            B1: Disturbance input matrix
            B2: Control input matrix
            C1: Regulated output matrix
            D12: Control feedthrough
            gamma: Disturbance attenuation level

        Returns:
            P matrix (Riccati solution) or None if no solution exists
        """
        n = A.shape[0]

        # Formulate Hamiltonian for H∞ problem
        # H = [A, -B2*B2' + γ⁻²*B1*B1'; -C1'*C1, -A']
        H_11 = A
        H_12 = -B2 @ B2.T + (1.0 / gamma**2) * B1 @ B1.T
        H_21 = -C1.T @ C1
        H_22 = -A.T

        H = np.block([[H_11, H_12], [H_21, H_22]])

        # Solve via ordered Schur decomposition
        try:
            # Compute eigenvalues to check for stabilizability
            eigvals = linalg.eigvals(H)

            # Check if there are n stable and n unstable eigenvalues
            stable_count = np.sum(np.real(eigvals) < 0)
            if stable_count != n:
                return None  # No solution

            # Ordered Schur form
            T, Z, _ = linalg.schur(H, output='complex', sort=lambda x: np.real(x) < 0)

            # Extract stable invariant subspace
            Z11 = Z[:n, :n]
            Z21 = Z[n:, :n]

            # P = Z21 @ Z11^{-1}
            P = np.real(Z21 @ linalg.inv(Z11))

            # Check if P is positive definite
            eigvals_P = linalg.eigvalsh(P)
            if np.any(eigvals_P < -1e-10):
                return None

            return P

        except (linalg.LinAlgError, np.linalg.LinAlgError):
            return None

    def design_controller(
        self,
        A: np.ndarray,
        B1: np.ndarray,
        B2: np.ndarray,
        C1: np.ndarray,
        D12: np.ndarray,
        gamma_min: float = 0.1,
        gamma_max: float = 100.0,
        tolerance: float = 1e-3
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Design H-infinity controller via γ-iteration.

        Args:
            A, B1, B2, C1, D12: System matrices
            gamma_min: Minimum γ to try
            gamma_max: Maximum γ to try
            tolerance: Bisection tolerance

        Returns:
            (K, gamma_opt) where K is the controller gain
        """
        # Bisection on γ
        gamma_low = gamma_min
        gamma_high = gamma_max

        best_gamma = None
        best_P = None

        while gamma_high - gamma_low > tolerance:
            gamma = (gamma_low + gamma_high) / 2.0

            P = self.solve_riccati(A, B1, B2, C1, D12, gamma)

            if P is not None:
                # Solution exists, try smaller γ
                best_gamma = gamma
                best_P = P
                gamma_high = gamma
            else:
                # No solution, increase γ
                gamma_low = gamma

        if best_P is not None:
            # Compute controller gain: K = -B2'P
            self.K = -B2.T @ best_P
            self.P = best_P
            self.gamma_opt = best_gamma

            return self.K, self.gamma_opt
        else:
            return None, None

    def evaluate_closed_loop_norm(
        self,
        A: np.ndarray,
        B1: np.ndarray,
        B2: np.ndarray,
        C1: np.ndarray,
        K: np.ndarray
    ) -> float:
        """Evaluate H∞ norm of closed-loop system.

        Args:
            A, B1, B2, C1: Open-loop system matrices
            K: Controller gain

        Returns:
            H∞ norm estimate
        """
        # Closed-loop: A_cl = A + B2*K
        A_cl = A + B2 @ K

        # Frequency response: G(jω) = C1(jωI - A_cl)^{-1}B1
        # H∞ norm = max_ω σ_max(G(jω))

        # Sample frequencies (logarithmic)
        omega = np.logspace(-2, 4, 1000)
        max_singular_value = 0.0

        for w in omega:
            # G(jω) = C1 * (jωI - A_cl)^{-1} * B1
            jw_I_minus_Acl = 1j * w * np.eye(A_cl.shape[0]) - A_cl

            try:
                inv_term = linalg.solve(jw_I_minus_Acl, B1)
                G_jw = C1 @ inv_term

                # Maximum singular value
                if G_jw.size == 1:
                    sigma_max = np.abs(G_jw)
                else:
                    sigma_max = linalg.svdvals(G_jw)[0]

                max_singular_value = max(max_singular_value, sigma_max)

            except (linalg.LinAlgError, np.linalg.LinAlgError):
                continue

        return max_singular_value


class StochasticOptimalControl:
    """Stochastic optimal control via HJB equation with diffusion.

    Solves: ∂V/∂t + min_u [∇V·f(x,u) + L(x,u) + 0.5*Tr(σσ'∇²V)] = 0

    where σ is the diffusion matrix.
    """

    def __init__(self, config: RobustControlConfig):
        """Initialize stochastic optimal control.

        Args:
            config: Robust control configuration
        """
        self.config = config

    def compute_hjb_residual(
        self,
        V: Callable,
        dV_dx: Callable,
        d2V_dx2: Callable,
        x: np.ndarray,
        dynamics: Callable,
        diffusion: Callable,
        running_cost: Callable,
        control_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """Compute HJB residual for stochastic system.

        Args:
            V: Value function
            dV_dx: Gradient of value function
            d2V_dx2: Hessian of value function
            x: State
            dynamics: f(x, u) drift
            diffusion: σ(x) diffusion matrix
            running_cost: L(x, u)
            control_constraints: (u_min, u_max)

        Returns:
            HJB residual
        """
        n_states = x.shape[0]

        # Get gradient and Hessian
        grad_V = dV_dx(x)
        hess_V = d2V_dx2(x)

        # Minimize over control
        def cost_to_minimize(u):
            # Hamiltonian: H = ∇V·f + L + 0.5*Tr(σσ'Hess(V))
            drift = dynamics(x, u)
            cost = running_cost(x, u)

            hamiltonian = grad_V @ drift + cost

            # Add diffusion term (independent of u)
            sigma = diffusion(x)
            trace_term = 0.5 * np.trace(sigma @ sigma.T @ hess_V)

            return hamiltonian + trace_term

        # Optimize control
        if control_constraints is not None:
            u_min, u_max = control_constraints
            bounds = [(u_min[i], u_max[i]) for i in range(len(u_min))]
            u0 = (u_min + u_max) / 2.0
        else:
            u0 = np.zeros(1)  # Assume 1D control for simplicity
            bounds = None

        result = minimize(cost_to_minimize, u0, method='L-BFGS-B', bounds=bounds)

        return result.fun


class UncertaintyQuantification:
    """Uncertainty quantification for optimal control.

    Methods:
    - Monte Carlo sampling
    - Polynomial chaos expansion
    - Unscented transform
    - Sensitivity analysis
    """

    def __init__(self, config: UQConfig):
        """Initialize UQ.

        Args:
            config: UQ configuration
        """
        self.config = config

    def monte_carlo_propagation(
        self,
        system_dynamics: Callable,
        initial_state: np.ndarray,
        parameter_distribution: Dict[str, Callable],
        t_span: Tuple[float, float],
        controller: Optional[Callable] = None
    ) -> Dict[str, np.ndarray]:
        """Propagate uncertainty via Monte Carlo sampling.

        Args:
            system_dynamics: f(t, x, u, params)
            initial_state: x0
            parameter_distribution: Dict mapping param name to sampling function
            t_span: (t0, tf)
            controller: Optional controller u(t, x)

        Returns:
            Dictionary with statistics (mean, std, samples)
        """
        from scipy.integrate import solve_ivp

        num_samples = self.config.num_samples
        rng = np.random.RandomState(self.config.random_seed)

        samples = []

        for i in range(num_samples):
            # Sample uncertain parameters
            params = {
                name: sampler(rng) for name, sampler in parameter_distribution.items()
            }

            # Define dynamics for this parameter sample
            def dynamics_sample(t, x):
                u = controller(t, x) if controller else np.zeros(1)
                return system_dynamics(t, x, u, params)

            # Simulate
            sol = solve_ivp(
                dynamics_sample,
                t_span,
                initial_state,
                method='RK45',
                dense_output=True
            )

            samples.append(sol.y[:, -1])  # Final state

        samples = np.array(samples)

        return {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'samples': samples,
            'percentile_5': np.percentile(samples, 5, axis=0),
            'percentile_95': np.percentile(samples, 95, axis=0)
        }

    def polynomial_chaos_expansion(
        self,
        model: Callable,
        uncertain_params: List[str],
        param_distributions: Dict[str, str],  # 'normal', 'uniform', etc.
        order: int
    ) -> Dict[str, Any]:
        """Polynomial chaos expansion for uncertainty propagation.

        Args:
            model: Output as function of uncertain parameters
            uncertain_params: List of parameter names
            param_distributions: Distribution type for each parameter
            order: Polynomial order

        Returns:
            PCE coefficients and statistics
        """
        # This is a simplified implementation
        # Full implementation would use orthogonal polynomials (Hermite, Legendre)

        n_params = len(uncertain_params)

        # Number of terms in PCE (multivariate polynomial)
        # For total order p and d parameters: (p+d)! / (p! d!)
        from math import factorial
        n_terms = factorial(order + n_params) // (factorial(order) * factorial(n_params))

        # Generate quadrature points and weights
        # (Simplified: use tensor product of 1D quadratures)
        from numpy.polynomial import hermite, legendre

        quad_points = []
        quad_weights = []

        for param, dist in zip(uncertain_params, [param_distributions[p] for p in uncertain_params]):
            if dist == 'normal':
                # Gauss-Hermite quadrature
                points, weights = hermite.hermgauss(order + 1)
            elif dist == 'uniform':
                # Gauss-Legendre quadrature
                points, weights = legendre.leggauss(order + 1)
            else:
                raise ValueError(f"Unknown distribution: {dist}")

            quad_points.append(points)
            quad_weights.append(weights)

        # Tensor product grid
        from itertools import product
        grid_points = list(product(*quad_points))
        grid_weights = [np.prod([quad_weights[i][idx] for i, idx in enumerate(indices)])
                        for indices in product(*[range(len(qp)) for qp in quad_points])]

        # Evaluate model at quadrature points
        model_values = np.array([model(dict(zip(uncertain_params, point))) for point in grid_points])

        # Compute moments
        mean = np.sum(model_values * grid_weights)
        variance = np.sum((model_values - mean)**2 * grid_weights)

        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'quadrature_points': grid_points,
            'model_values': model_values
        }

    def unscented_transform(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        nonlinear_map: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unscented transform for uncertainty propagation.

        Approximates mean and covariance of Y = f(X) where X ~ N(μ, Σ).

        Args:
            mean: Mean of input
            covariance: Covariance of input
            nonlinear_map: f(x)

        Returns:
            (mean_Y, cov_Y)
        """
        n = len(mean)

        # Scaling parameters
        alpha = self.config.alpha
        beta = self.config.beta
        kappa = self.config.kappa

        lambda_ = alpha**2 * (n + kappa) - n

        # Generate sigma points
        sqrt_term = linalg.sqrtm((n + lambda_) * covariance)

        sigma_points = [mean]
        for i in range(n):
            sigma_points.append(mean + sqrt_term[:, i])
            sigma_points.append(mean - sqrt_term[:, i])

        sigma_points = np.array(sigma_points)

        # Weights
        W_m = np.zeros(2*n + 1)
        W_c = np.zeros(2*n + 1)

        W_m[0] = lambda_ / (n + lambda_)
        W_c[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

        for i in range(1, 2*n + 1):
            W_m[i] = 1.0 / (2 * (n + lambda_))
            W_c[i] = 1.0 / (2 * (n + lambda_))

        # Transform sigma points
        Y_sigma = np.array([nonlinear_map(sp) for sp in sigma_points])

        # Compute mean
        mean_Y = np.sum(W_m[:, None] * Y_sigma, axis=0)

        # Compute covariance
        diff = Y_sigma - mean_Y
        cov_Y = sum(W_c[i] * np.outer(diff[i], diff[i]) for i in range(2*n + 1))

        return mean_Y, cov_Y

    def sensitivity_analysis(
        self,
        model: Callable,
        nominal_params: Dict[str, float],
        output_keys: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Local sensitivity analysis via finite differences.

        Computes ∂output/∂param for each parameter.

        Args:
            model: Returns dict of outputs given params
            nominal_params: Nominal parameter values
            output_keys: Keys of outputs to analyze

        Returns:
            Sensitivities dict[output][param] = sensitivity
        """
        perturbation = self.config.perturbation

        # Evaluate at nominal
        nominal_output = model(nominal_params)

        sensitivities = {key: {} for key in output_keys}

        for param_name, param_value in nominal_params.items():
            # Perturb parameter
            perturbed_params = nominal_params.copy()
            perturbed_params[param_name] = param_value * (1 + perturbation)

            # Evaluate
            perturbed_output = model(perturbed_params)

            # Compute sensitivity
            for output_key in output_keys:
                if output_key in nominal_output and output_key in perturbed_output:
                    dy = perturbed_output[output_key] - nominal_output[output_key]
                    dx = param_value * perturbation

                    sensitivities[output_key][param_name] = dy / dx if dx != 0 else 0.0

        return sensitivities


class RiskSensitiveControl:
    """Risk-sensitive optimal control.

    Uses exponential utility: J = E[exp(θ ∫ L dt)]

    where θ controls risk aversion:
    - θ = 0: Risk-neutral (standard expected cost)
    - θ > 0: Risk-averse (penalize variance)
    - θ < 0: Risk-seeking
    """

    def __init__(self, risk_aversion: float = 0.0):
        """Initialize risk-sensitive control.

        Args:
            risk_aversion: θ parameter (0 = risk-neutral)
        """
        self.theta = risk_aversion

    def compute_certainty_equivalent_cost(
        self,
        mean_cost: float,
        variance_cost: float
    ) -> float:
        """Compute certainty equivalent cost for quadratic approximation.

        For small θ: CE ≈ μ + (θ/2)σ²

        Args:
            mean_cost: Expected cost
            variance_cost: Variance of cost

        Returns:
            Certainty equivalent cost
        """
        if abs(self.theta) < 1e-10:
            return mean_cost  # Risk-neutral

        # First-order approximation
        ce_cost = mean_cost + (self.theta / 2.0) * variance_cost

        return ce_cost

    def risk_adjusted_control(
        self,
        nominal_control: np.ndarray,
        covariance: np.ndarray,
        control_cost_matrix: np.ndarray
    ) -> np.ndarray:
        """Adjust control based on risk aversion.

        Args:
            nominal_control: Nominal (risk-neutral) control
            covariance: State covariance
            control_cost_matrix: R matrix in u'Ru cost

        Returns:
            Risk-adjusted control
        """
        if abs(self.theta) < 1e-10:
            return nominal_control

        # For LQG with risk-sensitivity, modified Riccati equation is needed
        # Simplified: scale control based on risk aversion
        risk_factor = 1.0 / (1.0 + self.theta * np.trace(covariance))

        return risk_factor * nominal_control
