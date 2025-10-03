"""Integration numerical kernels.

Implementations of numerical integration methods:
- Adaptive quadrature (Gauss-Kronrod)
- Multi-dimensional integration
- Monte Carlo integration
- Special integration techniques
"""

import numpy as np
from typing import Callable, Tuple, Optional


def adaptive_quadrature(
    f: Callable,
    a: float,
    b: float,
    tol: float = 1e-6,
    maxiter: int = 1000
) -> Tuple[float, float]:
    """Adaptive quadrature using Simpson's rule.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        tol: Error tolerance
        maxiter: Maximum subdivisions

    Returns:
        Tuple of (integral value, error estimate)
    """
    from scipy.integrate import quad

    result, error = quad(f, a, b, epsabs=tol, limit=maxiter)
    return result, error


def simpson_rule(
    f: Callable,
    a: float,
    b: float,
    n: int = 100
) -> float:
    """Simpson's 1/3 rule for integration.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of intervals (must be even)

    Returns:
        Integral approximation
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's rule weights
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    weights[0] = 1
    weights[-1] = 1

    integral = h / 3 * np.sum(weights * y)
    return integral


def monte_carlo_integrate(
    f: Callable,
    bounds: list,
    n_samples: int = 10000,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """Monte Carlo integration for multi-dimensional integrals.

    Args:
        f: Function to integrate
        bounds: List of (lower, upper) bounds for each dimension
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (integral estimate, standard error)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random samples
    ndim = len(bounds)
    samples = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_samples, ndim)
    )

    # Evaluate function at samples
    values = np.array([f(s) for s in samples])

    # Volume of integration domain
    volume = np.prod([b[1] - b[0] for b in bounds])

    # Monte Carlo estimate
    integral = volume * np.mean(values)
    std_error = volume * np.std(values) / np.sqrt(n_samples)

    return integral, std_error


def gaussian_quadrature(
    f: Callable,
    a: float,
    b: float,
    n: int = 5
) -> float:
    """Gaussian quadrature integration.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of quadrature points

    Returns:
        Integral approximation
    """
    from numpy.polynomial.legendre import leggauss

    # Get Gauss-Legendre nodes and weights on [-1, 1]
    nodes, weights = leggauss(n)

    # Transform to [a, b]
    nodes_transformed = 0.5 * (b - a) * nodes + 0.5 * (b + a)

    # Evaluate integral
    integral = 0.5 * (b - a) * np.sum(weights * f(nodes_transformed))
    return integral


def trapezoidal_rule(
    f: Callable,
    a: float,
    b: float,
    n: int = 100
) -> float:
    """Trapezoidal rule for integration.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of intervals

    Returns:
        Integral approximation
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral


def romberg_integration(
    f: Callable,
    a: float,
    b: float,
    tol: float = 1e-6,
    maxiter: int = 20
) -> Tuple[float, int]:
    """Romberg integration using Richardson extrapolation.

    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        tol: Error tolerance
        maxiter: Maximum iterations

    Returns:
        Tuple of (integral, number of iterations)
    """
    from scipy.integrate import romberg

    result = romberg(f, a, b, tol=tol, rtol=tol, divmax=maxiter, show=False)
    return result, maxiter  # scipy doesn't return iteration count
