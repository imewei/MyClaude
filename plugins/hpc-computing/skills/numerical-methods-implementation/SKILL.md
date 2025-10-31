---
name: numerical-methods-implementation
description: Implement robust numerical algorithms for differential equations, optimization, and linear algebra in scientific computing applications. Use this skill when selecting and implementing ODE solvers (Runge-Kutta, BDF, Rosenbrock methods) with adaptive stepping and error control using scipy.integrate.solve_ivp or Julia DifferentialEquations.jl, implementing PDE solvers using finite difference, finite element, or spectral methods with stability analysis and CFL condition checking, choosing optimization algorithms (L-BFGS, BFGS, Newton-CG, Nelder-Mead) for minimizing objective functions with scipy.optimize.minimize or Julia Optim.jl, performing matrix decompositions (LU, QR, SVD, Cholesky) for solving linear systems and least squares problems with numpy.linalg or Julia LinearAlgebra, implementing iterative solvers (Conjugate Gradient, GMRES, BiCGSTAB) for large sparse systems using scipy.sparse.linalg or Julia IterativeSolvers.jl, handling stiff differential equations that require implicit methods and Jacobian computation, designing gradient-based optimization with automatic differentiation or finite difference gradients, implementing derivative-free optimization (Nelder-Mead, Powell, genetic algorithms) for non-smooth objective functions, selecting between dense and sparse linear algebra based on matrix structure and sparsity patterns, implementing eigenvalue solvers (eig, eigh, eigs) for spectral analysis and stability computation, ensuring numerical stability through condition number analysis and error bounds verification, implementing adaptive time-stepping with tolerance control for ODE/PDE integration, handling constrained optimization problems with linear or nonlinear constraints using SLSQP or trust-region methods, performing convergence analysis and Richardson extrapolation for verifying numerical accuracy, working with Python scripts using SciPy/NumPy or Julia .jl files using DifferentialEquations.jl/Optim.jl/LinearAlgebra.jl, or validating numerical implementations against analytical solutions and benchmark problems.
---

# Numerical Methods Implementation

## When to use this skill

- When implementing ODE solvers (RK4, RK45, BDF, Rosenbrock) with adaptive stepping
- When selecting PDE discretization methods (finite difference, finite element, spectral)
- When using scipy.integrate.solve_ivp or Julia DifferentialEquations.jl for differential equations
- When implementing optimization algorithms with scipy.optimize or Julia Optim.jl
- When choosing between gradient-based (L-BFGS, BFGS, Newton) and derivative-free (Nelder-Mead) methods
- When performing matrix decompositions (LU, QR, SVD, Cholesky) for linear systems
- When working with sparse matrices and iterative solvers (CG, GMRES, BiCGSTAB)
- When implementing eigenvalue solvers for spectral analysis or stability computation
- When handling stiff differential equations requiring implicit methods
- When ensuring numerical stability through condition number analysis
- When implementing adaptive time-stepping with CFL condition checking for PDEs
- When solving constrained optimization problems with linear or nonlinear constraints
- When performing convergence analysis and error bound verification
- When working with Python .py files using SciPy/NumPy for numerical algorithms
- When writing Julia .jl scripts with DifferentialEquations.jl, Optim.jl, or LinearAlgebra.jl
- When validating numerical accuracy against analytical solutions or benchmark problems
- When implementing custom numerical methods for domain-specific scientific applications
- When selecting between direct (LU, Cholesky) and iterative (CG, GMRES) linear solvers
- When analyzing numerical stability of algorithms with Lyapunov analysis or CFL conditions

## Overview

Implement robust numerical algorithms for scientific computing workflows, covering differential equation solvers, optimization techniques, and linear algebra operations. This skill provides guidance for selecting appropriate methods, implementing algorithms efficiently, and ensuring numerical stability and accuracy across Python and Julia ecosystems.

## Core Capabilities

### 1. ODE/PDE Solver Selection and Implementation

#### ODE Solvers

Select and implement ordinary differential equation solvers based on problem characteristics:

**Explicit Methods:**
- **Runge-Kutta Methods**: For non-stiff systems with moderate accuracy requirements
  - RK4 (4th order): General purpose, good accuracy-to-cost ratio
  - Dormand-Prince (RK45): Adaptive stepping with error control
  - Use SciPy's `solve_ivp` with method='RK45' or Julia's `ODEProblem` with `Tsit5()`

**Implicit Methods:**
- **BDF (Backward Differentiation Formulas)**: For stiff systems
  - Use SciPy's `solve_ivp` with method='BDF'
  - Julia: `ODEProblem` with `QNDF()` or `FBDF()`
- **Rosenbrock Methods**: For moderately stiff systems with discontinuities
  - Julia: `Rodas4()` or `Rodas5()` from DifferentialEquations.jl

**Adaptive Stepping Strategy:**
```python
# Python (SciPy)
from scipy.integrate import solve_ivp

def adaptive_ode_solve(f, t_span, y0, rtol=1e-6, atol=1e-9):
    """
    Solve ODE with adaptive error control.

    Parameters:
    - f: derivative function dy/dt = f(t, y)
    - t_span: (t0, tf) integration interval
    - y0: initial conditions
    - rtol/atol: relative/absolute tolerances
    """
    sol = solve_ivp(
        f, t_span, y0,
        method='RK45',  # or 'BDF' for stiff systems
        rtol=rtol,
        atol=atol,
        dense_output=True  # for continuous solution
    )
    return sol
```

```julia
# Julia (DifferentialEquations.jl)
using DifferentialEquations

function adaptive_ode_solve(f, u0, tspan; reltol=1e-6, abstol=1e-9)
    """
    Solve ODE with adaptive error control in Julia.

    Parameters:
    - f: derivative function du/dt = f(u, p, t)
    - u0: initial conditions
    - tspan: (t0, tf) integration interval
    """
    prob = ODEProblem(f, u0, tspan)
    sol = solve(prob, Tsit5(); reltol=reltol, abstol=abstol)
    return sol
end
```

#### PDE Solvers

**Finite Difference Methods:**
- **Explicit schemes**: Forward Euler, FTCS - for parabolic/hyperbolic PDEs with stability constraints
- **Implicit schemes**: Backward Euler, Crank-Nicolson - for better stability, requires linear system solve
- **Considerations**: Grid spacing (Δx, Δt), CFL condition, boundary conditions

**Finite Element Methods (FEM):**
- Variational formulation, weak forms, basis functions
- Libraries: FEniCS (Python), Gridap.jl (Julia)
- Suitable for complex geometries and adaptive mesh refinement

**Spectral Methods:**
- Fourier or Chebyshev basis functions for smooth solutions
- FFT-based for periodic problems
- Libraries: numpy.fft, FFTW.jl

**Stability Analysis:**
- Check CFL condition for explicit methods: `c * Δt / Δx ≤ 1`
- Von Neumann stability analysis for linear PDEs
- Monitor solution growth and energy conservation

### 2. Optimization Techniques

#### Gradient-Based Methods

**First-Order Methods:**
- **Gradient Descent**: Simple but slow convergence
- **Momentum Methods**: SGD with momentum, Nesterov accelerated gradient
- **Adaptive Methods**: Adam, RMSprop, AdaGrad - adaptive learning rates

**Second-Order Methods:**
- **Newton's Method**: Quadratic convergence but expensive Hessian computation
- **Quasi-Newton Methods**: BFGS, L-BFGS - approximate Hessian updates
  - L-BFGS: Memory-efficient for high-dimensional problems
- **Trust Region Methods**: More robust than line search for ill-conditioned problems

**Implementation Example:**
```python
# Python (SciPy)
from scipy.optimize import minimize

def optimize_function(f, x0, gradient=None, hessian=None, method='L-BFGS-B'):
    """
    Optimize objective function with gradient-based methods.

    Parameters:
    - f: objective function
    - x0: initial guess
    - gradient: gradient function (optional, uses finite differences if None)
    - method: 'L-BFGS-B', 'BFGS', 'Newton-CG', 'trust-ncg'
    """
    result = minimize(
        f, x0,
        method=method,
        jac=gradient,
        hess=hessian,
        options={'maxiter': 1000, 'disp': True}
    )
    return result
```

```julia
# Julia (Optim.jl)
using Optim

function optimize_function(f, x0; gradient=nothing, hessian=nothing)
    """
    Optimize objective function with gradient-based methods in Julia.
    """
    if gradient !== nothing && hessian !== nothing
        result = optimize(f, gradient, hessian, x0, Newton())
    elseif gradient !== nothing
        result = optimize(f, gradient, x0, LBFGS())
    else
        result = optimize(f, x0, NelderMead())
    end
    return result
end
```

#### Derivative-Free Methods

**When to Use:**
- Objective function is non-smooth or discontinuous
- Gradients are unavailable or expensive to compute
- Black-box optimization scenarios

**Methods:**
- **Nelder-Mead Simplex**: Robust but slow convergence
- **Powell's Method**: Conjugate direction search
- **Genetic Algorithms**: Global optimization for multimodal functions
- **Bayesian Optimization**: Sample-efficient for expensive evaluations

**Constrained Optimization:**
- **Linear Constraints**: Simplex method, interior point methods
- **Nonlinear Constraints**: SLSQP, trust-constr, Augmented Lagrangian
- **Bound Constraints**: L-BFGS-B, TNC (truncated Newton)

#### Handling Non-Convex Problems

**Strategies:**
- Multi-start optimization with different initial points
- Simulated annealing or genetic algorithms for global search
- Basin-hopping for escaping local minima
- Gradient descent with momentum for better exploration

**Convergence Monitoring:**
- Track objective function value and gradient norm
- Check for oscillations or divergence
- Validate final solution with constraints and optimality conditions

### 3. Linear Algebra Operations

#### Matrix Decompositions

**LU Decomposition:**
- For solving linear systems Ax = b
- Efficient for multiple right-hand sides
- Use `scipy.linalg.lu_factor` / `scipy.linalg.lu_solve` or Julia's `lu()`

**QR Decomposition:**
- For least squares problems and orthogonalization
- Stable for ill-conditioned matrices
- Use `numpy.linalg.qr` or Julia's `qr()`

**SVD (Singular Value Decomposition):**
- For low-rank approximations, pseudoinverse, matrix analysis
- Reveals rank, condition number, null space
- Use `numpy.linalg.svd` or Julia's `svd()`

**Cholesky Decomposition:**
- For symmetric positive definite matrices
- Faster than LU for SPD systems
- Use `scipy.linalg.cholesky` or Julia's `cholesky()`

**Implementation Pattern:**
```python
# Python
import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve

def efficient_linear_solve(A, b, symmetric=False, sparse=False):
    """
    Solve Ax = b efficiently based on matrix properties.
    """
    if sparse:
        from scipy.sparse.linalg import spsolve
        return spsolve(A, b)
    elif symmetric:
        from scipy.linalg import cho_factor, cho_solve
        c, low = cho_factor(A)
        return cho_solve((c, low), b)
    else:
        # General dense solver
        return solve(A, b)
```

```julia
# Julia
using LinearAlgebra

function efficient_linear_solve(A, b; symmetric=false, sparse=false)
    """
    Solve Ax = b efficiently based on matrix properties in Julia.
    """
    if sparse
        # Use sparse solver
        return A \ b  # Julia automatically detects sparsity
    elseif symmetric
        return cholesky(A) \ b
    else
        return lu(A) \ b
    end
end
```

#### Iterative Solvers for Sparse Systems

**Krylov Subspace Methods:**
- **Conjugate Gradient (CG)**: For symmetric positive definite systems
- **GMRES**: For general non-symmetric systems
- **BiCGSTAB**: For non-symmetric systems with better convergence
- **MINRES**: For symmetric indefinite systems

**When to Use Iterative Solvers:**
- Large sparse matrices where direct methods are expensive
- Systems with special structure (Toeplitz, circulant)
- Approximate solutions are acceptable

**Preconditioning:**
- Incomplete LU (ILU) factorization
- Jacobi or Gauss-Seidel preconditioners
- Multigrid methods for structured problems

**Example:**
```python
# Python (SciPy sparse)
from scipy.sparse.linalg import cg, gmres, bicgstab

def iterative_sparse_solve(A, b, method='cg', preconditioner=None):
    """
    Solve sparse linear system iteratively.
    """
    if method == 'cg':
        x, info = cg(A, b, M=preconditioner, tol=1e-8)
    elif method == 'gmres':
        x, info = gmres(A, b, M=preconditioner, restart=50)
    elif method == 'bicgstab':
        x, info = bicgstab(A, b, M=preconditioner)

    if info > 0:
        print(f"Convergence not achieved after {info} iterations")
    return x
```

#### Eigenvalue Problems

**Dense Eigenvalue Solvers:**
- Full spectrum: `numpy.linalg.eig`, Julia's `eigen()`
- Symmetric/Hermitian: `numpy.linalg.eigh`, Julia's `eigen(Symmetric(A))`
- Generalized eigenvalue: `scipy.linalg.eig(A, B)`

**Sparse Eigenvalue Solvers:**
- Few eigenvalues: `scipy.sparse.linalg.eigs`, `scipy.sparse.linalg.eigsh`
- Largest/smallest eigenvalues with iterative methods (Lanczos, Arnoldi)
- Julia: `eigs()` from Arpack.jl

**Applications:**
- Principal Component Analysis (PCA)
- Stability analysis of dynamical systems
- Quantum mechanics (Hamiltonian diagonalization)

## Numerical Accuracy and Stability

### Error Analysis

**Sources of Error:**
1. **Truncation Error**: From finite precision approximations (e.g., Taylor series)
2. **Roundoff Error**: From floating-point arithmetic
3. **Discretization Error**: From spatial/temporal discretization in PDEs

**Error Control Strategies:**
- Use adaptive methods with error estimators
- Monitor relative/absolute errors
- Perform Richardson extrapolation for grid refinement
- Check conservation laws (energy, momentum, mass)

### Stability Assessment

**Numerical Stability Criteria:**
- **Lyapunov Stability**: For ODE systems
- **CFL Condition**: For explicit PDE solvers
- **Condition Number**: For linear systems (check with `numpy.linalg.cond`)

**Ill-Conditioned Problems:**
- High condition number (≫ 1/machine precision) indicates sensitivity
- Use regularization (Tikhonov, truncated SVD)
- Apply preconditioning for iterative methods
- Consider alternative formulations

### Validation Techniques

**Testing Numerical Implementations:**
1. **Method of Manufactured Solutions**: Known exact solutions
2. **Convergence Tests**: Verify order of accuracy with grid refinement
3. **Conservation Tests**: Check physical invariants
4. **Benchmark Problems**: Compare with established solutions

## Library Selection Guide

### Python (SciPy/NumPy)

**When to Use:**
- Rapid prototyping and development
- Integration with ML frameworks (TensorFlow, PyTorch)
- Rich ecosystem of scientific libraries
- Existing Python codebase

**Key Libraries:**
- `scipy.integrate`: ODE/PDE solvers
- `scipy.optimize`: Optimization algorithms
- `scipy.linalg`: Dense linear algebra
- `scipy.sparse`: Sparse matrix operations
- `numpy`: Array operations, BLAS/LAPACK wrappers

### Julia (DifferentialEquations.jl, Optim.jl)

**When to Use:**
- Performance-critical computations (10-100x speedups over Python)
- Stiff differential equations requiring adaptive methods
- Symbolic computation and automatic differentiation
- Complex scientific workflows

**Key Libraries:**
- `DifferentialEquations.jl`: Comprehensive ODE/SDE/DAE/DDE solvers
- `Optim.jl`: Optimization algorithms
- `LinearAlgebra.jl`: Built-in linear algebra
- `SparseArrays.jl`: Sparse matrix support
- `Symbolics.jl`: Symbolic mathematics

**Performance Tips:**
- Write type-stable code (check with `@code_warntype`)
- Use in-place operations (`A .= B .+ C` instead of `A = B + C`)
- Preallocate arrays where possible
- Leverage multiple dispatch for algorithm selection

## Best Practices

### Algorithm Selection

1. **Understand Problem Characteristics**: Stiffness, smoothness, dimensionality, sparsity
2. **Match Method to Problem**: Implicit for stiff, explicit for non-stiff
3. **Consider Computational Cost**: Operation count, memory requirements
4. **Validate Results**: Check accuracy, stability, physical plausibility

### Implementation Checklist

- [ ] Verify input dimensions and types
- [ ] Handle edge cases (singular matrices, empty arrays)
- [ ] Implement error checking and informative messages
- [ ] Add convergence monitoring and logging
- [ ] Document algorithm parameters and assumptions
- [ ] Include unit tests with known solutions
- [ ] Profile performance and identify bottlenecks

### Performance Optimization

1. **Vectorization**: Use NumPy/Julia array operations instead of loops
2. **Memory Layout**: Contiguous arrays, cache-friendly access patterns
3. **In-Place Operations**: Reduce memory allocations
4. **Parallel Computing**: Use multi-threading or GPU acceleration for large problems
5. **Sparse Operations**: Exploit sparsity structure when applicable

## Resources

### references/

- `ode_solver_comparison.md`: Detailed comparison of ODE solver methods with convergence properties
- `optimization_algorithm_guide.md`: Decision tree for selecting optimization algorithms
- `linear_algebra_patterns.md`: Common patterns for matrix operations and decompositions
- `numerical_stability_guide.md`: In-depth analysis of numerical stability and error control

### scripts/

- `convergence_test.py`: Script to test numerical convergence rates
- `benchmark_solvers.jl`: Benchmark ODE/PDE solvers across problem types
- `validate_implementation.py`: Validation framework for numerical algorithms

Load references as needed to understand implementation details, algorithm trade-offs, and numerical considerations for specific problems.
