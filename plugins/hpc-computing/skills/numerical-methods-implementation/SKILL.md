---
name: numerical-methods-implementation
version: "1.0.7"
description: Implement robust numerical algorithms for ODEs (RK45, BDF), PDEs (finite difference, FEM), optimization (L-BFGS, Newton), and linear algebra (LU, QR, SVD, iterative solvers). Use when selecting solvers, implementing adaptive stepping, handling stiff systems, or ensuring numerical stability.
---

# Numerical Methods Implementation

## Solver Selection

### ODE
| Method | Stiffness | Library |
|--------|-----------|---------|
| RK45 | No | `solve_ivp`, `Tsit5()` |
| BDF | Yes | `solve_ivp`, `QNDF()` |
| Rosenbrock | Moderate | `Rodas4()` |
| Radau | Very stiff, DAEs | `solve_ivp` |

### PDE
| Method | Stability |
|--------|-----------|
| Explicit FD | Conditional (CFL) |
| Implicit FD (Crank-Nicolson) | Unconditional |
| FEM | Problem-dependent |
| Spectral | Exponential |

### Optimization
| Method | Best For |
|--------|----------|
| L-BFGS | Large-scale, smooth |
| BFGS | Moderate dimensions |
| Newton-CG | Well-conditioned |
| Nelder-Mead | Black-box, noisy |

## ODE Solving

```python
from scipy.integrate import solve_ivp
def solve_ode(f, t_span, y0, stiff=False):
    method = 'BDF' if stiff else 'RK45'
    return solve_ivp(f, t_span, y0, method=method, rtol=1e-6, atol=1e-9)
```

```julia
using DifferentialEquations
prob = ODEProblem(f, u0, tspan)
alg = stiff ? QNDF() : Tsit5()
solve(prob, alg; reltol=1e-6, abstol=1e-9)
```

## Linear Algebra

| Decomposition | Use |
|---------------|-----|
| LU | General systems |
| Cholesky | SPD (faster) |
| QR | Least squares |
| SVD | Rank, pseudoinverse |

**Iterative**: CG (SPD), GMRES (general), BiCGSTAB (memory-limited)

```python
from scipy.linalg import lu_solve, cho_solve
from scipy.sparse.linalg import cg, gmres

def solve_linear(A, b, sparse=False, symmetric=False):
    if sparse:
        return (cg if symmetric else gmres)(A, b, tol=1e-8)[0]
    elif symmetric:
        return cho_solve(cho_factor(A), b)
    else:
        return lu_solve(lu_factor(A), b)
```

## Optimization

```python
from scipy.optimize import minimize
optimize(f, x0, method='L-BFGS-B', jac=gradient, bounds=bounds)
```

```julia
using Optim
gradient !== nothing ? optimize(f, gradient, x0, LBFGS()) : optimize(f, x0, NelderMead())
```

## Numerical Stability

| Issue | Solution |
|-------|----------|
| Ill-conditioning (cond >>1/eps) | Regularization, preconditioning |
| Stiffness | Implicit methods (BDF) |
| CFL violation | Reduce Δt |
| Roundoff | Higher precision, Kahan sum |

```python
cond = np.linalg.cond(A)
if cond > 1e12: print(f"Ill-conditioned: ~{-np.log10(cond/1e16):.0f} accurate digits")
```

## Validation

- Manufactured solutions
- Richardson extrapolation
- Conservation laws
- Benchmark problems

```python
def convergence_test(solver, h_values, exact):
    errors = [np.max(np.abs(solver(h) - exact)) for h in h_values]
    return np.polyfit(np.log(h_values), np.log(errors), 1)[0]  # Order
```

**Outcome**: Appropriate solvers, adaptive stepping, stability checks, convergence verification
