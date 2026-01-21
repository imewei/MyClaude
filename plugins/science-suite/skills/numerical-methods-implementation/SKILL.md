---
name: numerical-methods-implementation
version: "2.1.0"
description: Implement robust numerical algorithms for ODEs (RK45, BDF), PDEs (finite difference, FEM), optimization (L-BFGS, Newton), and molecular simulations. Master solver selection, stability analysis, and differentiable physics using Python and Julia.
---

# Numerical Methods Implementation

Expert guide for implementing numerical solvers and physical simulations with high precision and performance.

## Expert Agent

For advanced numerical methods, solver selection, and stability analysis, delegate to:

- **`simulation-expert`**: For physical simulations, HPC solvers, and Molecular Dynamics.
  - *Location*: `plugins/science-suite/agents/simulation-expert.md`
- **`julia-pro`**: For DifferentialEquations.jl solvers, stiff systems, and performance tuning.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
- **`jax-pro`**: For Diffrax solvers, autodiff-compatible methods, and GPU-accelerated linear algebra.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`

## Solver Selection

### ODE
| Problem Type | Stiffness | Recommended Solver (Julia) | Recommended Solver (Python) |
|--------------|-----------|----------------------------|-----------------------------|
| **General ODE** | Non-stiff | `Tsit5()`                 | `solve_ivp(method='RK45')` |
| **Stiff ODE**   | Stiff     | `QNDF()` or `Rodas4()`    | `solve_ivp(method='BDF')`  |
| **DAEs**        | Stiff     | `IDA()`                   | `solve_ivp(method='Radau')`|

### PDE
| Method | Stability | Library |
|--------|-----------|---------|
| Explicit FD | Conditional (CFL) | Custom, `FIPY` |
| Implicit FD (Crank-Nicolson) | Unconditional | Custom, `Dedalus` |
| FEM | Problem-dependent | `FEniCS`, `Gridap.jl` |
| Spectral | Exponential | `Dedalus`, `ApproxFun.jl` |
| Method of Lines | Varies | `MethodOfLines.jl` |

### Optimization
| Method | Best For |
|--------|----------|
| L-BFGS | Large-scale, smooth |
| BFGS | Moderate dimensions |
| Newton-CG | Well-conditioned |
| Nelder-Mead | Black-box, noisy |

## Parallelization Strategies

| Method | Technique | Implementation |
|--------|-----------|----------------|
| Domain Decomposition | Split spatial grid | MPI (mpi4py, MPI.jl) |
| Vectorization | SIMD instructions | NumPy, Julia `@simd` |
| Multi-threading | Shared memory | OpenMP, `Threads.@threads` |
| GPU Acceleration | Massive parallelism | CUDA.jl, CuPy, JAX |

## Molecular Dynamics (MD)

### Traditional Engines
- **LAMMPS**: Best for materials, polymers, and large-scale inorganic systems.
- **GROMACS**: Optimized for biomolecules and solvation.
- **HOOMD-blue**: Native GPU acceleration for soft matter.

### Differentiable MD (JAX-MD)
```python
from jax_md import space, energy, simulate
displacement_fn, shift_fn = space.periodic(box_size=10.0)
energy_fn = energy.lennard_jones_pair(displacement_fn)
init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, dt=0.005)
```

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

## Differentiable Physics (JAX)

Use JAX for gradient-based optimization of physical parameters.

```python
import jax.numpy as jnp
from jax import grad, jit

@jit
def potential_energy(params, positions):
    # Differentiable energy function
    return jnp.sum(params * positions**2)

force_fn = grad(potential_energy, argnums=1)
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

## Validation Checklist

- [ ] **Condition Number**: Check `np.linalg.cond(A)` to identify ill-conditioned systems.
- [ ] **CFL Condition**: Ensure $\Delta t \leq \Delta x / v$ for explicit PDE stability.
- [ ] **Conservation Laws**: Verify energy, mass, and momentum conservation in simulations.
- [ ] **Convergence**: Perform Richardson extrapolation or manufacturing of solutions to verify order of accuracy.
- [ ] **Manufactured Solutions**: Test against known analytical solutions.

**Outcome**: Appropriate solvers, adaptive stepping, stability checks, convergence verification
