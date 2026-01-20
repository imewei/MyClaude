# Diffrax Deep Dive

Advanced solver engineering, step size control, implicit methods, Lineax integration, and Optimistix root finding.

## Solver Internals

### Solver Categories

| Category | Solvers | Characteristics |
|----------|---------|-----------------|
| **Explicit Runge-Kutta** | `Euler`, `Heun`, `Midpoint`, `Ralston` | Fixed-order, non-stiff |
| **Adaptive Explicit** | `Tsit5`, `Dopri5`, `Dopri8` | Embedded error estimation |
| **Implicit** | `ImplicitEuler`, `Kvaerno3/4/5` | Stiff-stable, requires Jacobian |
| **IMEX** | `KenCarp3/4/5` | Mixed implicit-explicit |
| **Symplectic** | `Leapfrog`, `SemiImplicitEuler` | Energy-preserving (Hamiltonian) |

### Choosing Based on Stiffness

```python
import diffrax
import jax.numpy as jnp

def detect_stiffness_heuristic(jacobian_fn, y0):
    """Estimate stiffness from eigenvalue spread."""
    J = jacobian_fn(y0)
    eigenvalues = jnp.linalg.eigvals(J)
    real_parts = jnp.real(eigenvalues)

    # Stiffness ratio: spread of eigenvalues
    ratio = jnp.abs(jnp.max(real_parts) / jnp.min(real_parts))

    if ratio > 1000:
        print(f"Highly stiff (ratio={ratio:.0f}) → Use Kvaerno5")
        return diffrax.Kvaerno5()
    elif ratio > 100:
        print(f"Moderately stiff (ratio={ratio:.0f}) → Use KenCarp4")
        return diffrax.KenCarp4()
    else:
        print(f"Non-stiff (ratio={ratio:.0f}) → Use Tsit5")
        return diffrax.Tsit5()
```

## Step Size Control

### PID Controller

The default `PIDController` uses proportional-integral-derivative control to adapt step size based on local error.

```python
# Standard settings
controller = diffrax.PIDController(
    rtol=1e-5,          # Relative tolerance
    atol=1e-7,          # Absolute tolerance
    pcoeff=0.0,         # Proportional coefficient
    icoeff=1.0,         # Integral coefficient
    dcoeff=0.0,         # Derivative coefficient
    dtmin=1e-10,        # Minimum step size
    dtmax=1.0,          # Maximum step size
    force_dtmin=False,  # Raise error if dt < dtmin
)

# Aggressive adaptation for rapidly changing systems
controller_aggressive = diffrax.PIDController(
    rtol=1e-4,
    atol=1e-6,
    pcoeff=0.4,    # Add proportional term
    icoeff=0.3,
    dcoeff=0.0,
)

# Conservative for near-chaotic systems
controller_conservative = diffrax.PIDController(
    rtol=1e-8,
    atol=1e-10,
    pcoeff=0.0,
    icoeff=0.7,    # Slower adaptation
    dcoeff=0.0,
)
```

### Constant Step Size

For reproducibility or fixed-step requirements:

```python
controller = diffrax.ConstantStepSize()

# Or specify directly
solution = diffrax.diffeqsolve(
    term, solver, t0, t1,
    dt0=0.001,  # Fixed step
    stepsize_controller=diffrax.ConstantStepSize(),
)
```

## Implicit Solvers and Lineax

Implicit solvers require solving nonlinear systems at each step. Lineax provides the linear algebra backend.

### How Implicit Solvers Work

```
At each step, solve: F(y_{n+1}) = 0
Using Newton iteration:
  1. Linearize: J * Δy = -F(y_k)
  2. Solve linear system (Lineax)
  3. Update: y_{k+1} = y_k + Δy
  4. Repeat until convergence
```

### Configuring Newton Solver

```python
import lineax as lx

# Custom linear solver for Newton steps
linear_solver = lx.LU()  # LU decomposition (default)

# For large sparse systems
linear_solver = lx.CG(rtol=1e-6, atol=1e-8)  # Conjugate gradient

# For ill-conditioned systems
linear_solver = lx.QR()  # QR decomposition

# Create implicit solver with custom linear solver
solver = diffrax.Kvaerno5(
    root_finder=diffrax.VeryChord(
        rtol=1e-4,
        atol=1e-6,
        linear_solver=linear_solver,
    )
)
```

### Jacobian Computation

```python
# Automatic (default): JAX autodiff computes Jacobian
solver = diffrax.Kvaerno5()

# Provide analytical Jacobian for speed
def vector_field(t, y, args):
    return -args['k'] * y

def jacobian(t, y, args):
    return -args['k'] * jnp.eye(len(y))

term = diffrax.ODETerm(vector_field)
# Jacobian provided through VJP structure
```

## Optimistix: Root Finding & Steady States

Find steady states without time integration.

### Basic Root Finding

```python
import optimistix as optx
import jax.numpy as jnp

def residual(y, args):
    """Steady state: dy/dt = 0"""
    k, target = args['k'], args['target']
    return k * (target - y)  # Should equal zero at steady state

# Find root
solver = optx.Newton(rtol=1e-8, atol=1e-10)
y0 = jnp.array([1.0])
solution = optx.root_find(residual, solver, y0, args={'k': 0.5, 'target': 5.0})

print(f"Steady state: {solution.value}")  # Should be ~5.0
```

### Steady State of ODE System

```python
def find_steady_state(vector_field, y0, args, solver=None):
    """Find steady state: f(y) = 0."""
    if solver is None:
        solver = optx.Newton(rtol=1e-8, atol=1e-10)

    def residual(y, args):
        return vector_field(0.0, y, args)  # dy/dt = 0

    result = optx.root_find(residual, solver, y0, args=args)
    return result.value

# For rheology: find equilibrium stress state
def stress_evolution(t, stress, args):
    """Oldroyd-B model: dσ/dt = ..."""
    shear_rate = args['shear_rate']
    relaxation_time = args['lambda']
    viscosity = args['eta']

    return (viscosity * shear_rate - stress) / relaxation_time

equilibrium_stress = find_steady_state(
    stress_evolution,
    jnp.array([0.0]),
    {'shear_rate': 1.0, 'lambda': 1.0, 'eta': 1.0}
)
```

### Least Squares (Gauss-Newton)

```python
def model(params, x):
    return params['a'] * jnp.exp(-params['b'] * x)

def residual(params, args):
    x, y_obs = args['x'], args['y']
    return model(params, x) - y_obs

# Fit model to data
solver = optx.GaussNewton(rtol=1e-6, atol=1e-8)
initial_params = {'a': 1.0, 'b': 0.1}
result = optx.least_squares(residual, solver, initial_params, args={'x': x_data, 'y': y_data})
```

## Adjoint Methods In Depth

### RecursiveCheckpointAdjoint

Trade compute for memory by recomputing forward pass during backward pass.

```python
# Number of checkpoints controls memory/compute tradeoff
# More checkpoints = less recomputation but more memory

# Memory-constrained
adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=8)

# Speed-constrained
adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=64)

# Let Diffrax choose
adjoint = diffrax.RecursiveCheckpointAdjoint()  # Auto

solution = diffrax.diffeqsolve(
    term, solver, t0, t1, dt0, y0,
    adjoint=adjoint,
)
```

### BacksolveAdjoint

Solve adjoint ODE backwards. Memory-efficient but can be unstable.

```python
# Use for stable, non-chaotic systems
adjoint = diffrax.BacksolveAdjoint(
    solver=diffrax.Tsit5(),  # Can use different solver for adjoint
)

# Not recommended for:
# - Chaotic systems (Lyapunov exponent > 0)
# - Stiff systems (adjoint can be stiffer than forward)
# - Long integration times
```

### DirectAdjoint

Store full trajectory. Most accurate but memory-intensive.

```python
# Only for short simulations or when gradients must be exact
adjoint = diffrax.DirectAdjoint()
```

### Choosing Adjoint Method

| Scenario | Recommended Adjoint |
|----------|---------------------|
| Long simulation, memory-limited | `RecursiveCheckpointAdjoint` |
| Short simulation, accuracy critical | `DirectAdjoint` |
| Non-chaotic, compute-limited | `BacksolveAdjoint` |
| Stiff system | `RecursiveCheckpointAdjoint` |
| Neural ODE training | `RecursiveCheckpointAdjoint` |

## Event Handling

Detect and respond to discontinuities while maintaining differentiability.

### Basic Event Detection

```python
def event_fn(t, y, args, **kwargs):
    """Return 0 when event occurs (e.g., stress exceeds yield)."""
    yield_stress = args['yield_stress']
    current_stress = y[0]
    return current_stress - yield_stress  # Zero crossing = event

event = diffrax.Event(event_fn)

solution = diffrax.diffeqsolve(
    term, solver, t0, t1, dt0, y0,
    args=args,
    event=event,
)

if solution.event_mask:
    print(f"Yield occurred at t={solution.ts[-1]}")
```

### Event with State Modification

```python
def solve_with_yield(vector_field, y0, args):
    """Solve ODE with yield stress events."""

    def event_fn(t, y, args, **kwargs):
        return y[0] - args['yield_stress']

    # Solve until yield
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()

    sol = diffrax.diffeqsolve(
        term, solver, 0.0, 100.0, 0.01, y0,
        args=args,
        event=diffrax.Event(event_fn),
    )

    if sol.event_mask:
        # Modify state at yield point
        y_at_yield = sol.ys[-1]
        y_post_yield = apply_plastic_deformation(y_at_yield, args)

        # Continue from modified state
        sol2 = diffrax.diffeqsolve(
            term, solver, sol.ts[-1], 100.0, 0.01, y_post_yield,
            args=args,
        )

        return sol, sol2

    return sol, None
```

## Performance Optimization

### JIT Compilation

```python
import jax

@jax.jit
def simulate(y0, args):
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()

    return diffrax.diffeqsolve(
        term, solver, 0.0, 10.0, 0.01, y0,
        args=args,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10, 100)),
    )

# First call compiles, subsequent calls are fast
result = simulate(y0, args)
```

### Batched Simulations

```python
# Vectorize over initial conditions
@jax.vmap
def batch_simulate(y0):
    return diffrax.diffeqsolve(
        term, solver, t0, t1, dt0, y0,
        args=args,
    )

# Shape: (n_samples, state_dim)
y0_batch = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
results = batch_simulate(y0_batch)
```

### Parallelization

```python
# Across devices with pmap
@jax.pmap
def parallel_simulate(y0, rng_key):
    return diffrax.diffeqsolve(...)

# Distribute initial conditions across devices
n_devices = jax.device_count()
y0_sharded = y0_batch.reshape(n_devices, -1, state_dim)
```
