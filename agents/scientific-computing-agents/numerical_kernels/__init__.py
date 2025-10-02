"""Numerical kernels library.

Reusable numerical implementations for computational method agents.
"""

from .ode_solvers import *
from .linear_algebra import *
from .optimization import *
from .integration import *

__all__ = [
    # ODE solvers (to be implemented)
    'rk45_step',
    'bdf_step',

    # Linear algebra (to be implemented)
    'solve_linear_system',
    'compute_eigenvalues',

    # Optimization (to be implemented)
    'minimize_bfgs',
    'find_root_newton',

    # Integration (to be implemented)
    'adaptive_quadrature',
    'monte_carlo_integrate'
]
