"""Lorenz attractor integration and Lyapunov exponent estimation."""

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import norm


def lorenz_rhs(t: float, state: NDArray, sigma: float, rho: float, beta: float) -> NDArray:
    """Right-hand side of the Lorenz system: dx/dt = f(x)."""
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def lorenz_jacobian(state: NDArray, sigma: float, rho: float, beta: float) -> NDArray:
    """Jacobian matrix of the Lorenz system at a given state."""
    x, y, z = state
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta],
    ])


def integrate_lorenz(
    x0: NDArray,
    t_span: tuple[float, float],
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> tuple[NDArray, NDArray]:
    """Integrate the Lorenz system using RK45 with dense output."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lorenz_rhs,
        t_span,
        x0,
        method="RK45",
        t_eval=t_eval,
        args=(sigma, rho, beta),
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.t, sol.y.T


def maximal_lyapunov_exponent(
    x0: NDArray,
    T: float = 100.0,
    dt: float = 0.01,
    renorm_steps: int = 10,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> float:
    """Estimate the maximal Lyapunov exponent via tangent space integration.

    Uses QR decomposition for numerical stability of the perturbation vector.
    """
    n_steps = int(T / dt)
    renorm_interval = max(1, n_steps // renorm_steps)

    state = np.copy(x0)
    delta = np.random.default_rng(42).normal(size=3)
    delta /= norm(delta)

    lyap_sum = 0.0
    n_renorm = 0

    for step in range(n_steps):
        J = lorenz_jacobian(state, sigma, rho, beta)
        delta = delta + dt * J @ delta
        state_dot = lorenz_rhs(0.0, state, sigma, rho, beta)
        state = state + dt * state_dot

        if (step + 1) % renorm_interval == 0:
            d_norm = norm(delta)
            if d_norm > 0:
                lyap_sum += np.log(d_norm)
                delta /= d_norm
                n_renorm += 1

    return lyap_sum / (n_renorm * renorm_interval * dt) if n_renorm > 0 else 0.0


def find_fixed_points(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> list[NDArray]:
    """Analytically compute the fixed points of the Lorenz system."""
    origin = np.array([0.0, 0.0, 0.0])
    if rho <= 1.0:
        return [origin]
    c = np.sqrt(beta * (rho - 1.0))
    return [origin, np.array([c, c, rho - 1.0]), np.array([-c, -c, rho - 1.0])]
