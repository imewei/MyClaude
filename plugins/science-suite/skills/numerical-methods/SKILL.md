---
name: numerical-methods
version: "1.0.0"
description: Implement robust numerical algorithms for ODEs, PDEs, optimization, and molecular simulations. Master solver selection, stability analysis, and differentiable physics using Python and Julia.
---

# Numerical Methods & Simulation

Expert guide for implementing numerical solvers and physical simulations with high precision and performance.

## 1. Differential Equations (ODE/PDE)

### Solver Selection Matrix

| Problem Type | Stiffness | Recommended Solver (Julia) | Recommended Solver (Python) |
|--------------|-----------|----------------------------|-----------------------------|
| **General ODE** | Non-stiff | `Tsit5()`                 | `solve_ivp(method='RK45')` |
| **Stiff ODE**   | Stiff     | `QNDF()` or `Rodas4()`    | `solve_ivp(method='BDF')`  |
| **DAEs**        | Stiff     | `IDA()`                   | `solve_ivp(method='Radau')`|
| **PDEs**        | Varies    | `MethodOfLines.jl`        | `FIPY` or `Dedalus`        |

### Differentiable Physics (JAX)
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

## 2. Molecular Dynamics (MD)

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

## 3. Optimization & Linear Algebra

### Optimization Methods
- **L-BFGS**: Large-scale smooth optimization.
- **Newton-CG**: When second-order information is available.
- **Nelder-Mead**: Derivative-free/noisy functions.

### Linear Systems
- **Direct Solvers**: LU, Cholesky (SPD), QR (Least Squares).
- **Iterative Solvers**: CG (Symmetric), GMRES (General), BiCGSTAB.

## 4. Stability & Validation Checklist

- [ ] **Condition Number**: Check `np.linalg.cond(A)` to identify ill-conditioned systems.
- [ ] **CFL Condition**: Ensure $\Delta t \leq \Delta x / v$ for explicit PDE stability.
- [ ] **Conservation Laws**: Verify energy, mass, and momentum conservation in simulations.
- [ ] **Convergence**: Perform Richardson extrapolation or manufacturing of solutions to verify order of accuracy.
