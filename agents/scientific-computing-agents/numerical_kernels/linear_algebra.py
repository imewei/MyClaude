"""Linear algebra numerical kernels.

Implementations of linear system solvers and eigenvalue methods:
- Direct solvers (LU, Cholesky, QR)
- Iterative solvers (CG, GMRES, BiCGSTAB)
- Eigenvalue solvers
- Matrix factorizations
"""

import numpy as np
from typing import Tuple, Optional
from scipy import linalg, sparse


def solve_linear_system(
    A: np.ndarray,
    b: np.ndarray,
    method: str = 'auto',
    tol: float = 1e-6,
    maxiter: int = 1000
) -> Tuple[np.ndarray, dict]:
    """Solve linear system Ax = b.

    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        method: Solution method ('lu', 'qr', 'cholesky', 'cg', 'gmres', 'auto')
        tol: Tolerance for iterative methods
        maxiter: Maximum iterations for iterative methods

    Returns:
        Tuple of (solution x, info dict)
    """
    n = A.shape[0]
    info = {'method': method, 'iterations': 0, 'residual': 0.0}

    # Auto-select method
    if method == 'auto':
        if n < 1000:
            method = 'lu'
        elif sparse.issparse(A):
            method = 'gmres'
        else:
            method = 'cg' if is_symmetric(A) else 'gmres'
        info['method'] = method

    # Direct methods
    if method == 'lu':
        x = linalg.solve(A, b)
    elif method == 'qr':
        Q, R = linalg.qr(A)
        x = linalg.solve_triangular(R, Q.T @ b)
    elif method == 'cholesky':
        L = linalg.cholesky(A, lower=True)
        y = linalg.solve_triangular(L, b, lower=True)
        x = linalg.solve_triangular(L.T, y, lower=False)

    # Iterative methods (simplified implementations)
    elif method == 'cg':
        x, info_cg = conjugate_gradient(A, b, tol=tol, maxiter=maxiter)
        info.update(info_cg)
    elif method == 'gmres':
        x, info_gmres = gmres_solver(A, b, tol=tol, maxiter=maxiter)
        info.update(info_gmres)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute residual
    residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    info['residual'] = residual

    return x, info


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    maxiter: int = 1000
) -> Tuple[np.ndarray, dict]:
    """Conjugate gradient method for symmetric positive definite systems.

    Args:
        A: SPD coefficient matrix
        b: Right-hand side
        x0: Initial guess
        tol: Convergence tolerance
        maxiter: Maximum iterations

    Returns:
        Tuple of (solution, info dict)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()
    rsold = r @ r

    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @ r

        if np.sqrt(rsnew) < tol:
            return x, {'iterations': i + 1, 'residual': np.sqrt(rsnew)}

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, {'iterations': maxiter, 'residual': np.sqrt(rsold), 'converged': False}


def gmres_solver(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    maxiter: int = 100,
    restart: int = 20
) -> Tuple[np.ndarray, dict]:
    """GMRES (Generalized Minimal Residual) method.

    Simplified implementation using scipy.

    Args:
        A: Coefficient matrix
        b: Right-hand side
        x0: Initial guess
        tol: Convergence tolerance
        maxiter: Maximum iterations
        restart: Restart parameter

    Returns:
        Tuple of (solution, info dict)
    """
    from scipy.sparse.linalg import gmres as scipy_gmres

    counter = {'niter': 0}

    def callback(rk):
        counter['niter'] += 1

    x, info_code = scipy_gmres(A, b, x0=x0, tol=tol, restart=restart,
                                maxiter=maxiter, callback=callback)

    return x, {'iterations': counter['niter'], 'info_code': info_code}


def compute_eigenvalues(
    A: np.ndarray,
    k: Optional[int] = None,
    which: str = 'largest'
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors.

    Args:
        A: Matrix
        k: Number of eigenvalues (None for all)
        which: Which eigenvalues ('largest', 'smallest', 'all')

    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    if k is None or which == 'all':
        eigenvalues, eigenvectors = linalg.eig(A)
    else:
        from scipy.sparse.linalg import eigs
        eigenvalues, eigenvectors = eigs(A, k=k, which='LM' if which == 'largest' else 'SM')

    # Sort
    idx = np.argsort(np.abs(eigenvalues))[::-1] if which == 'largest' else np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is symmetric.

    Args:
        A: Matrix
        tol: Tolerance

    Returns:
        True if symmetric
    """
    return np.allclose(A, A.T, atol=tol)


def condition_number(A: np.ndarray) -> float:
    """Compute condition number of matrix.

    Args:
        A: Matrix

    Returns:
        Condition number
    """
    return np.linalg.cond(A)
