---
name: jax-julia-interop
description: Bridge JAX (Python) and Julia SciML ecosystems using PythonCall.jl and juliacall. Use when combining BifurcationKit with JAX vmap, calling Julia solvers from Python, or exchanging arrays between ecosystems.
---

# JAX-Julia Interop

Bridge JAX Python and Julia SciML ecosystems for hybrid scientific computing workflows.

---

## Expert Agents

- **jax-pro** (`plugins/science-suite/agents/jax-pro.md`): JAX optimization, JIT compilation, vmap/pmap transformations
- **julia-pro** (`plugins/science-suite/agents/julia-pro.md`): Julia SciML, DifferentialEquations.jl, BifurcationKit

---

## Setup

### Python to Julia (juliacall)

```python
pip install juliacall
```

```python
from juliacall import Main as jl

# Load Julia packages
jl.seval("using BifurcationKit")
jl.seval("using DifferentialEquations")

# Call Julia functions from Python
result = jl.seval("solve(ODEProblem(f, u0, tspan, p), Tsit5())")
```

### Julia to Python (PythonCall.jl)

```julia
using PythonCall

jax = pyimport("jax")
jnp = pyimport("jax.numpy")
np  = pyimport("numpy")

# Call JAX functions from Julia
x = jnp.linspace(0, 1, 1000)
y = jnp.sin(x)
```

### Environment Management

Use **CondaPkg.jl** for reproducible Python dependencies from Julia:

```julia
using CondaPkg
CondaPkg.add("jax")
CondaPkg.add("jaxlib")
CondaPkg.add_pip("diffrax")
```

This creates a `CondaPkg.toml` locked to exact versions, ensuring reproducibility.

---

## Array Exchange

### Zero-Copy Sharing

NumPy arrays share memory with Julia via `PythonCall.PyArray`:

```julia
using PythonCall
np = pyimport("numpy")

# Zero-copy: Julia and Python see the same memory
py_arr = np.zeros(1000)
jl_arr = pyconvert(Vector{Float64}, py_arr)  # shared memory
```

### JAX Arrays Require Copy

JAX arrays live on device (GPU/TPU) and must be transferred to host first:

```python
import numpy as np
import jax.numpy as jnp
from juliacall import Main as jl

# JAX -> NumPy -> Julia (requires copy)
x_jax = jnp.ones((100, 100))
x_np = np.asarray(x_jax)          # device-to-host transfer
jl_call("f", x_np)                # pass to Julia

# Julia -> JAX (requires copy)
result_np = np.array(jl_call("g"))
result_jax = jnp.array(result_np)  # host-to-device transfer
```

### Type Mapping

| Python | Julia | Notes |
|--------|-------|-------|
| `float` | `Float64` | Direct conversion |
| `np.float32` | `Float32` | Direct conversion |
| `np.ndarray` | `Matrix{Float64}` | Zero-copy via PyArray |
| `jnp.array` | Requires `np.asarray` | Device-to-host copy needed |
| `dict` | `Dict` | Recursive conversion |

---

## Workflow Patterns

### Pattern 1: Julia Bifurcation + JAX Dense Sweep

Julia BifurcationKit finds critical points, then JAX vmap sweeps densely around each.

```python
from juliacall import Main as jl
import jax
import jax.numpy as jnp
import numpy as np

# Step 1: Julia finds bifurcation points
jl.seval("""
using BifurcationKit

function rhs(u, p, t)
    x, y = u
    mu = p[1]
    return [mu*x - x^3, -y]
end

prob = BifurcationProblem(rhs, [0.0, 0.0], [0.0],
    (@lens _[1]); record_from_solution = (x, p) -> x[1])
opts = ContinuationPar(p_min=-1.0, p_max=2.0, ds=0.01)
br = continuation(prob, PALC(), opts)
""")

# Extract critical parameter values
critical_params = np.array(jl.seval("[bp.param for bp in br.specialpoint]"))

# Step 2: JAX dense sweep around each critical point
@jax.jit
def simulate(mu):
    x = jnp.zeros(2)
    for _ in range(1000):
        dx = jnp.array([mu * x[0] - x[0]**3, -x[1]])
        x = x + 0.01 * dx
    return x

# Dense sweep: 1000 points around each bifurcation
for mu_c in critical_params:
    mu_range = jnp.linspace(mu_c - 0.1, mu_c + 0.1, 1000)
    results = jax.vmap(simulate)(mu_range)  # GPU-parallel
```

### Pattern 2: JAX GPU Simulation + Julia Dynamical Analysis

JAX runs large-scale GPU simulation, then Julia DynamicalSystems.jl analyzes trajectories.

```python
import jax
import jax.numpy as jnp
import numpy as np
from juliacall import Main as jl

# Step 1: JAX GPU simulation (coupled oscillator network)
@jax.jit
def simulate_network(key, N=1000, T=10000):
    x = jax.random.normal(key, (N,))
    trajectory = []
    for t in range(T):
        coupling = jnp.mean(jnp.sin(x[:, None] - x[None, :]), axis=1)
        x = x + 0.01 * (-jnp.sin(x) + 0.5 * coupling)
        trajectory.append(x)
    return jnp.stack(trajectory)

traj = simulate_network(jax.random.PRNGKey(42))
traj_np = np.asarray(traj)  # device-to-host

# Step 2: Julia dynamical analysis
jl.seval("using DynamicalSystems")
jl.traj = traj_np

lyapunov_exp = float(jl.seval("""
ds = DeterministicIteratedMap(lorenz_rule, traj[1, :])
lyapunov(ds, 10000)
"""))

corr_dim = float(jl.seval("""
correlationdimension(StateSpaceSet(traj'), 2:12)
"""))
```

### Pattern 3: Julia Symbolic Model + JAX GPU Training

Julia ModelingToolkit defines a symbolic model, exports equations, and JAX trains on GPU.

```julia
# Step 1: Julia - symbolic model definition
using ModelingToolkit

@variables t x(t) y(t)
@parameters alpha beta gamma delta
D = Differential(t)

eqs = [
    D(x) ~ alpha*x - beta*x*y,
    D(y) ~ delta*x*y - gamma*y
]

# Export as callable for Python side
# ModelingToolkit can generate optimized numerical functions
```

```python
# Step 2: JAX - GPU training with exported equations
import jax
import jax.numpy as jnp
from jax import grad, jit
import optax

def lotka_volterra(state, t, params):
    x, y = state
    alpha, beta, gamma, delta = params
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return jnp.array([dxdt, dydt])

@jit
def loss_fn(params, data_t, data_obs):
    pred = solve_ode(lotka_volterra, data_obs[0], data_t, params)
    return jnp.mean((pred - data_obs) ** 2)

grad_fn = jit(grad(loss_fn))
optimizer = optax.adam(1e-3)

# Train on GPU
params = jnp.array([1.0, 0.1, 0.1, 0.1])
opt_state = optimizer.init(params)
for step in range(5000):
    grads = grad_fn(params, t_data, observations)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

---

## Performance Gotchas

| Issue | Cause | Mitigation |
|-------|-------|------------|
| Slow first call | Julia JIT warmup (~30-60s) | Precompile with `PackageCompiler.jl`, or call once before timing |
| Memory doubling | JAX arrays require copy to host | Batch transfers; avoid per-element copies |
| GIL blocking | Julia calls hold Python GIL | Use `@spawn` in Julia for long computations |
| Float32 vs Float64 mismatch | JAX defaults to float32, Julia to Float64 | Explicitly set dtypes at boundaries: `jnp.array(x, dtype=jnp.float64)` |
| CondaPkg vs pip conflicts | Mixed package managers | Use CondaPkg exclusively from Julia side, pip exclusively from Python side |

---

## Ecosystem Selection Guide

| Task | Best Ecosystem | Reason |
|------|---------------|--------|
| Bifurcation continuation | Julia (BifurcationKit) | Mature continuation algorithms, deflation |
| Lyapunov spectrum | Julia (DynamicalSystems.jl) | Purpose-built, validated implementations |
| Parameter sweep >10K points | JAX (vmap) | GPU-parallel vectorization |
| Network simulation >1K nodes | JAX (jit + vmap) | GPU memory and SIMD parallelism |
| Equation discovery | Julia (DataDrivenDiffEq.jl) | SINDy, symbolic regression integration |
| Neural ODE | Both | Diffrax (JAX) or DiffEqFlux.jl (Julia) |
| Symbolic manipulation | Julia (ModelingToolkit.jl) | Symbolic-numeric compilation pipeline |
