"""ODE solver numerical kernels.

Implementations of ODE/DAE solving methods:
- Explicit Runge-Kutta (RK45, Dormand-Prince)
- Implicit methods (BDF, Radau)
- Adaptive time stepping
- Stability analysis
"""

import numpy as np
from typing import Callable, Tuple, Optional


def rk45_step(
    f: Callable,
    t: float,
    y: np.ndarray,
    h: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Single step of Dormand-Prince RK45 method.

    Args:
        f: Right-hand side function dy/dt = f(t, y)
        t: Current time
        y: Current state
        h: Time step

    Returns:
        Tuple of (y_next, error_estimate, optimal_h)
    """
    # Butcher tableau for Dormand-Prince
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

    # Compute stages
    k = np.zeros((7, len(y)))
    k[0] = f(t, y)
    for i in range(1, 7):
        y_temp = y + h * np.dot(a[i, :i], k[:i])
        k[i] = f(t + c[i] * h, y_temp)

    # 5th order solution
    y_next = y + h * np.dot(b, k[:6])

    # 4th order solution for error estimate
    y_star = y + h * np.dot(b_star, k)

    # Error estimate
    error = np.linalg.norm(y_next - y_star)

    # Optimal step size (with safety factor)
    tol = 1e-6
    optimal_h = 0.9 * h * (tol / max(error, 1e-10)) ** 0.2

    return y_next, error, optimal_h


def bdf_step(
    f: Callable,
    t: float,
    y_history: list,
    h: float,
    order: int = 2
) -> np.ndarray:
    """Single step of BDF (Backward Differentiation Formula) method.

    Args:
        f: Right-hand side function dy/dt = f(t, y)
        t: Current time
        y_history: History of previous solutions [y_n, y_(n-1), ...]
        h: Time step
        order: BDF order (1-5)

    Returns:
        y_next: Solution at t + h
    """
    # BDF coefficients
    if order == 1:
        alpha = np.array([1, -1])
        beta = 1
    elif order == 2:
        alpha = np.array([3/2, -2, 1/2])
        beta = 1
    else:
        raise NotImplementedError(f"BDF order {order} not implemented")

    # Solve implicit equation: alpha[0]*y_next + sum(alpha[i]*y_(n-i+1)) = h*beta*f(t+h, y_next)
    # Using fixed-point iteration (simplified)
    y_next = y_history[0]  # Initial guess
    for _ in range(10):  # Fixed-point iterations
        rhs = -np.dot(alpha[1:], y_history[:len(alpha)-1]) + h * beta * f(t + h, y_next)
        y_next = rhs / alpha[0]

    return y_next


def adaptive_step_size(
    error: float,
    h_current: float,
    tolerance: float,
    order: int
) -> float:
    """Compute adaptive step size based on error estimate.

    Args:
        error: Estimated error
        h_current: Current step size
        tolerance: Error tolerance
        order: Method order

    Returns:
        Optimal step size
    """
    safety_factor = 0.9
    min_factor = 0.2
    max_factor = 5.0

    if error < tolerance:
        # Increase step size
        factor = min(max_factor, safety_factor * (tolerance / max(error, 1e-10)) ** (1 / (order + 1)))
    else:
        # Decrease step size
        factor = max(min_factor, safety_factor * (tolerance / error) ** (1 / (order + 1)))

    return h_current * factor


def check_stability(
    jacobian: np.ndarray,
    h: float
) -> bool:
    """Check stability of explicit method using eigenvalues.

    Args:
        jacobian: Jacobian matrix df/dy
        h: Time step

    Returns:
        True if stable, False otherwise
    """
    eigenvalues = np.linalg.eigvals(jacobian)
    # For explicit RK, stability region is approximate circle
    max_real = np.max(np.real(eigenvalues))
    return h * max_real < 2.8  # Approximate stability limit for RK4
