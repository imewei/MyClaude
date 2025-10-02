"""Optimization numerical kernels.

Implementations of optimization and root-finding methods:
- Gradient-based optimization (BFGS, Newton)
- Derivative-free methods (Nelder-Mead)
- Root finding (Newton, bisection, Brent)
- Line search algorithms
"""

import numpy as np
from typing import Callable, Tuple, Optional


def minimize_bfgs(
    f: Callable,
    x0: np.ndarray,
    grad: Optional[Callable] = None,
    tol: float = 1e-6,
    maxiter: int = 1000
) -> Tuple[np.ndarray, dict]:
    """BFGS quasi-Newton optimization.

    Args:
        f: Objective function
        x0: Initial point
        grad: Gradient function (finite differences if None)
        tol: Convergence tolerance
        maxiter: Maximum iterations

    Returns:
        Tuple of (optimal x, info dict)
    """
    from scipy.optimize import minimize as scipy_minimize

    result = scipy_minimize(f, x0, method='BFGS', jac=grad, tol=tol,
                           options={'maxiter': maxiter})

    return result.x, {
        'iterations': result.nit,
        'function_value': result.fun,
        'converged': result.success,
        'message': result.message
    }


def find_root_newton(
    f: Callable,
    x0: float,
    fprime: Optional[Callable] = None,
    tol: float = 1e-6,
    maxiter: int = 100
) -> Tuple[float, dict]:
    """Newton's method for root finding.

    Args:
        f: Function to find root of
        x0: Initial guess
        fprime: Derivative (finite difference if None)
        tol: Convergence tolerance
        maxiter: Maximum iterations

    Returns:
        Tuple of (root, info dict)
    """
    x = x0
    history = [x]

    for i in range(maxiter):
        fx = f(x)

        if abs(fx) < tol:
            return x, {'iterations': i + 1, 'residual': abs(fx), 'history': history}

        # Compute derivative
        if fprime is not None:
            dfx = fprime(x)
        else:
            h = 1e-8
            dfx = (f(x + h) - fx) / h

        if abs(dfx) < 1e-12:
            return x, {'iterations': i + 1, 'converged': False, 'message': 'Derivative too small'}

        # Newton step
        x = x - fx / dfx
        history.append(x)

    return x, {'iterations': maxiter, 'residual': abs(f(x)), 'converged': False, 'history': history}


def line_search_backtracking(
    f: Callable,
    x: np.ndarray,
    direction: np.ndarray,
    grad: np.ndarray,
    alpha: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    maxiter: int = 50
) -> float:
    """Backtracking line search.

    Args:
        f: Objective function
        x: Current point
        direction: Search direction
        grad: Gradient at x
        alpha: Initial step size
        rho: Backtracking parameter
        c: Armijo condition parameter
        maxiter: Maximum iterations

    Returns:
        Step size alpha
    """
    fx = f(x)
    slope = grad @ direction

    for _ in range(maxiter):
        if f(x + alpha * direction) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho

    return alpha


def golden_section_search(
    f: Callable,
    a: float,
    b: float,
    tol: float = 1e-5
) -> Tuple[float, float]:
    """Golden section search for 1D optimization.

    Args:
        f: Objective function
        a: Lower bound
        b: Upper bound
        tol: Tolerance

    Returns:
        Tuple of (optimal x, optimal f(x))
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    resphi = 2 - phi

    # Initial points
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)

    return (a + b) / 2, f((a + b) / 2)
