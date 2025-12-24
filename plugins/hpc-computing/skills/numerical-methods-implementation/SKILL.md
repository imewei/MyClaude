---
name: numerical-methods-implementation
version: "1.0.5"
maturity: "5-Expert"
specialization: Scientific Computing Algorithms
description: Implement robust numerical algorithms for ODEs (RK45, BDF), PDEs (finite difference, FEM), optimization (L-BFGS, Newton), and linear algebra (LU, QR, SVD, iterative solvers). Use when selecting solvers, implementing adaptive stepping, handling stiff systems, or ensuring numerical stability.
---

# Numerical Methods Implementation

Robust algorithms for differential equations, optimization, and linear algebra.

---

## Method Selection

### ODE Solvers

| Method | Use Case | Stiffness | Library |
|--------|----------|-----------|---------|
| RK45 (Dormand-Prince) | General non-stiff | No | `solve_ivp`, `Tsit5()` |
| BDF | Stiff systems | Yes | `solve_ivp`, `QNDF()` |
| Rosenbrock | Moderately stiff | Yes | `Rodas4()` |
| Radau | Very stiff, DAEs | Yes | `solve_ivp` |

### PDE Methods

| Method | Best For | Stability |
|--------|----------|-----------|
| Explicit FD (FTCS) | Simple, CFL limited | Conditional |
| Implicit FD (Crank-Nicolson) | Stability required | Unconditional |
| Finite Element (FEM) | Complex geometry | Problem-dependent |
| Spectral | Smooth periodic | Exponential |

### Optimization Algorithms

| Method | Gradient | Best For |
|--------|----------|----------|
| L-BFGS | Required | Large-scale, smooth |
| BFGS | Required | Moderate dimensions |
| Newton-CG | Required + Hessian | Well-conditioned |
| Nelder-Mead | Not needed | Black-box, noisy |
| SLSQP | Required | Constrained |

---

## ODE Solving

```python
from scipy.integrate import solve_ivp

def solve_ode(f, t_span, y0, stiff=False, rtol=1e-6, atol=1e-9):
    """Adaptive ODE solver with stiffness detection."""
    method = 'BDF' if stiff else 'RK45'
    return solve_ivp(f, t_span, y0, method=method,
                     rtol=rtol, atol=atol, dense_output=True)
```

```julia
using DifferentialEquations

function solve_ode(f, u0, tspan; stiff=false, reltol=1e-6, abstol=1e-9)
    prob = ODEProblem(f, u0, tspan)
    alg = stiff ? QNDF() : Tsit5()
    solve(prob, alg; reltol=reltol, abstol=abstol)
end
```

---

## Linear Algebra

### Decompositions

| Decomposition | Use Case | Complexity |
|---------------|----------|------------|
| LU | General linear systems | O(n³) |
| Cholesky | SPD systems (faster) | O(n³/3) |
| QR | Least squares | O(2n²m) |
| SVD | Rank, pseudoinverse | O(min(mn², m²n)) |

### Iterative Solvers

| Method | Matrix Type | When to Use |
|--------|-------------|-------------|
| CG | Symmetric positive definite | Large sparse SPD |
| GMRES | General | Large non-symmetric |
| BiCGSTAB | General | Memory-limited |

```python
from scipy.linalg import lu_factor, lu_solve, cho_factor, cho_solve
from scipy.sparse.linalg import cg, gmres

def solve_linear(A, b, sparse=False, symmetric=False):
    """Select appropriate solver based on matrix properties."""
    if sparse:
        method = cg if symmetric else gmres
        x, info = method(A, b, tol=1e-8)
        return x
    elif symmetric:
        c, low = cho_factor(A)
        return cho_solve((c, low), b)
    else:
        lu, piv = lu_factor(A)
        return lu_solve((lu, piv), b)
```

---

## Optimization

```python
from scipy.optimize import minimize

def optimize(f, x0, gradient=None, method='L-BFGS-B', bounds=None):
    """Gradient-based optimization with optional bounds."""
    return minimize(f, x0, method=method, jac=gradient, bounds=bounds,
                    options={'maxiter': 1000, 'disp': True})
```

```julia
using Optim

function optimize(f, x0; gradient=nothing)
    if gradient !== nothing
        optimize(f, gradient, x0, LBFGS())
    else
        optimize(f, x0, NelderMead())
    end
end
```

---

## Numerical Stability

| Issue | Detection | Solution |
|-------|-----------|----------|
| Ill-conditioning | `cond(A) >> 1/eps` | Regularization, preconditioning |
| Stiffness | Eigenvalue ratio large | Implicit methods (BDF) |
| CFL violation | Δt > CFL·Δx/c | Reduce time step |
| Roundoff accumulation | Monitor residuals | Higher precision, Kahan sum |

### Condition Number Check

```python
import numpy as np

def check_stability(A):
    """Assess numerical stability of linear system."""
    cond = np.linalg.cond(A)
    if cond > 1e12:
        return f"Ill-conditioned: cond={cond:.2e}, expect ~{-np.log10(cond/1e16):.0f} accurate digits"
    return f"Well-conditioned: cond={cond:.2e}"
```

---

## Validation

| Technique | Purpose |
|-----------|---------|
| Manufactured solutions | Verify order of accuracy |
| Richardson extrapolation | Estimate true solution |
| Conservation laws | Physical plausibility |
| Benchmark problems | Compare with known results |

```python
def convergence_test(solver, h_values, exact):
    """Verify convergence order with grid refinement."""
    errors = [np.max(np.abs(solver(h) - exact)) for h in h_values]
    order = np.polyfit(np.log(h_values), np.log(errors), 1)[0]
    return order, errors
```

---

## Library Selection

| Task | Python | Julia |
|------|--------|-------|
| ODE/PDE | scipy.integrate | DifferentialEquations.jl |
| Optimization | scipy.optimize | Optim.jl |
| Linear algebra | numpy.linalg, scipy.linalg | LinearAlgebra |
| Sparse | scipy.sparse | SparseArrays.jl |
| FFT | numpy.fft | FFTW.jl |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Adaptive stepping | Use built-in error control |
| Vectorization | NumPy/Julia array ops over loops |
| In-place operations | Reduce allocations |
| Preconditioning | For iterative solvers |
| Profile first | Identify actual bottlenecks |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Using explicit for stiff | Switch to BDF/Radau |
| Ignoring condition number | Check before solving |
| Fixed step size | Use adaptive methods |
| Dense for large sparse | Use iterative solvers |
| No convergence check | Monitor residuals |

---

## Checklist

- [ ] Problem characteristics analyzed (stiffness, sparsity)
- [ ] Appropriate method selected
- [ ] Tolerances set appropriately
- [ ] Convergence verified
- [ ] Results validated against known solutions
- [ ] Performance profiled

---

**Version**: 1.0.5
