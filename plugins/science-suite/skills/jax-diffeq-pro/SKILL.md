---
name: jax-diffeq-pro
version: "1.0.1"
description: This skill should be used when the user asks to "solve ODEs in JAX", "use Diffrax", "implement stiff solvers", "choose implicit vs explicit solvers", "backpropagate through ODEs", "use adjoint methods", "implement RecursiveCheckpointAdjoint", "solve SDEs", "use VirtualBrownianTree", "handle ODE events", "find steady states with Optimistix", "use Lineax for linear solves", or needs expert guidance on differentiable physics, neural ODEs, rheology simulations, or soft matter dynamics.
---

# JAX DiffEq Pro: The Differentiable Physicist

A JAX-first differential equation expert builds **differentiable physics engines**. The standard stack is **Diffrax** (ODE/SDE solvers), **Lineax** (linear algebra), and **Optimistix** (root finding). An expert architects solvers that are robust to stiff soft matter systems and efficient enough for training loops.

## Expert Agent

For complex differentiable physics, neural ODEs, and stiff systems simulation, delegate to the expert agent:

- **`jax-pro`**: Unified specialist for Diffrax ODE/SDE solvers, adjoint methods, and physics integration.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`
  - *Capabilities*: Stiff solver selection (Kvaerno/KenCarp), adjoint optimization (RecursiveCheckpointAdjoint), and SDE simulation.

## The Differentiable Physicist Mindset

### Solver as Hyperparameter

The solver (Runge-Kutta, Dormand-Prince, Kvaerno) is a tunable model component. Stiff rheological models require implicit solvers; non-stiff particle systems use explicit ones.

```python
import diffrax

# Non-stiff: explicit solver
solver_explicit = diffrax.Tsit5()

# Stiff (widely varying timescales): implicit solver
solver_implicit = diffrax.Kvaerno5()
```

### Gradient-First Thinking

Always ask: "How will gradients propagate through time evolution?" Choose solver settings (checkpointing, adjoint methods) to prevent O(N) memory growth.

```python
# Memory-efficient backprop through long simulations
adjoint = diffrax.RecursiveCheckpointAdjoint()

solution = diffrax.diffeqsolve(
    term, solver, t0, t1, dt0, y0,
    adjoint=adjoint,  # Trade compute for memory
)
```

### Composability

Mix physics and ML freely. The vector field is often hybrid: part first-principles physics, part neural network (Equinox module).

```python
import equinox as eqx

class HybridVectorField(eqx.Module):
    physics_params: dict
    neural_correction: eqx.nn.MLP

    def __call__(self, t, y, args):
        # First-principles physics
        physics_term = self.constitutive_equation(y, self.physics_params)
        # Learned correction
        neural_term = self.neural_correction(y)
        return physics_term + neural_term
```

## The Diffrax Stack

| Library | Purpose | Use Case |
|---------|---------|----------|
| **Diffrax** | ODE/SDE solvers | Time evolution, dynamics |
| **Lineax** | Linear solvers | Implicit solver Jacobian solves |
| **Optimistix** | Root finding | Steady states, equilibria |

### Basic Diffrax Pattern

```python
import diffrax
import jax.numpy as jnp

def vector_field(t, y, args):
    """dy/dt = f(t, y)"""
    return -args['k'] * y

term = diffrax.ODETerm(vector_field)
solver = diffrax.Tsit5()
stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

solution = diffrax.diffeqsolve(
    term,
    solver,
    t0=0.0,
    t1=10.0,
    dt0=0.01,
    y0=jnp.array([1.0]),
    args={'k': 0.5},
    saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10, 100)),
    stepsize_controller=stepsize_controller,
)
```

## Solver Selection Guide

| System Type | Solver | Why |
|-------------|--------|-----|
| Non-stiff, smooth | `Tsit5`, `Dopri5` | Fast, accurate |
| Stiff (multi-scale) | `Kvaerno5`, `KenCarp4` | Implicit, stable |
| High accuracy | `Dopri8` | 8th order |
| SDE (additive noise) | `Euler`, `Heun` | Simple SDE schemes |
| SDE (multiplicative) | `SPaRK`, `GeneralShARK` | Strong/weak convergence |

## Skills Matrix: Junior vs Expert

| Category | Junior | Expert/Pro |
|----------|--------|------------|
| **Solver Choice** | Default `Tsit5` for everything | Implicit `Kvaerno5` for stiff rheology |
| **Backprop** | `jax.grad(solve)` (OOMs on long tasks) | `RecursiveCheckpointAdjoint` |
| **Events** | `if` statements (breaks gradients) | `diffrax.Event` with root-finding |
| **Brownian Motion** | `+ random.normal()` at fixed steps | `VirtualBrownianTree` for adaptive SDE |
| **Linear Algebra** | Default `jnp.linalg.solve` | `Lineax` with preconditioners |

## Gradient Propagation Strategies

### Discretize-then-Optimize (Checkpointing)

Backprop through exact numerical operations. Use `RecursiveCheckpointAdjoint` to trade compute for memory.

```python
# For long simulations - prevents OOM
adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=16)
```

### Optimize-then-Discretize (Backsolve Adjoint)

Solve adjoint ODE backwards in time. **Caution**: Unstable for chaotic systems.

```python
# Only for non-chaotic systems
adjoint = diffrax.BacksolveAdjoint()
```

## Additional Resources

### Reference Files

- **`references/diffrax-deep-dive.md`** - Solver internals, step size control, implicit solver Jacobians, Lineax integration, Optimistix root finding
- **`references/sde-soft-matter.md`** - Stochastic differential equations, VirtualBrownianTree, Itō vs Stratonovich, event handling, neural ODEs, rheology applications

### Example Files

- **`examples/solver-selection.py`** - Explicit vs implicit, stiffness detection, step size control
- **`examples/adjoint-methods.py`** - Gradient propagation, checkpointing, memory optimization
- **`examples/sde-brownian.py`** - Stochastic ODEs, reproducible Brownian motion, soft matter applications

**Outcome**: Build differentiable physics engines, choose appropriate solvers for stiff systems, efficiently backprop through long simulations, handle events and SDEs.
