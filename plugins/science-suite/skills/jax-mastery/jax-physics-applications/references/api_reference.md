# JAX Physics Applications API Quick Reference

Quick reference for key functions and modules in JAX-MD, JAX-CFD, and quantum computing libraries.

## JAX-MD API

### Core Modules
```python
from jax_md import space, energy, simulate, quantity, partition
```

### space - Spatial Functions

**Displacement and Shifting:**
```python
# Periodic boundary conditions
displacement_fn, shift_fn = space.periodic(box_size=10.0)

# Free boundary conditions
displacement_fn, shift_fn = space.free()

# Distance calculation
r = space.distance(displacement_fn(R1, R2))

# Random position initialization
R = space.random_position(key, shape=(N, 3), box_size=10.0)
```

### energy - Potential Energy Functions

**Pair Potentials:**
```python
# Lennard-Jones potential
energy_fn = energy.lennard_jones_pair(
    displacement_fn,
    species=None,      # Optional species array
    sigma=1.0,         # Distance parameter
    epsilon=1.0        # Energy parameter
)

# Soft sphere potential
energy_fn = energy.soft_sphere_pair(
    displacement_fn,
    sigma=1.0,
    epsilon=1.0,
    alpha=2.0          # Exponent
)
```

**Neighbor Lists:**
```python
from jax_md import partition

neighbor_fn = partition.neighbor_list(
    displacement_fn,
    box_size=10.0,
    r_cutoff=3.0,              # Interaction cutoff
    dr_threshold=0.5,           # Rebuild threshold
    capacity_multiplier=1.25    # Extra capacity
)

# Allocate and update neighbor list
neighbors = neighbor_fn.allocate(R)
neighbors = neighbors.update(R)

# Energy with neighbor list
energy_fn = energy.lennard_jones_neighbor_list(
    displacement_fn,
    box_size=10.0,
    sigma=1.0,
    epsilon=1.0
)
energy_val = energy_fn(R, neighbor=neighbors)
```

### simulate - Time Integration

**NVE Ensemble (Microcanonical):**
```python
init_fn, apply_fn = simulate.nve(
    energy_fn,
    shift_fn,
    dt=0.005           # Timestep
)

state = init_fn(key, R, mass=1.0)
state = apply_fn(state)  # Single step
```

**NVT Ensemble (Nose-Hoover Thermostat):**
```python
init_fn, apply_fn = simulate.nvt_nose_hoover(
    energy_fn,
    shift_fn,
    dt=0.005,
    kT=1.0,            # Target temperature
    tau=0.5            # Thermostat time constant
)
```

**NVT Ensemble (Langevin Dynamics):**
```python
init_fn, apply_fn = simulate.nvt_langevin(
    energy_fn,
    shift_fn,
    dt=0.005,
    kT=1.0,
    gamma=0.1          # Friction coefficient
)
```

**NPT Ensemble (Nose-Hoover Thermostat + Parrinello-Rahman Barostat):**
```python
init_fn, apply_fn = simulate.npt_nose_hoover(
    energy_fn,
    shift_fn,
    dt=0.005,
    kT=1.0,
    pressure=1.0,
    tau_T=0.5,         # Thermostat time constant
    tau_P=5.0          # Barostat time constant
)
```

### quantity - Observable Calculations

```python
# Kinetic energy
KE = quantity.kinetic_energy(state)

# Temperature
T = quantity.temperature(state, kB=1.0)

# Pressure
P = quantity.pressure(state, energy_fn)

# Stress tensor
stress = quantity.stress(state, energy_fn)
```

---

## JAX-CFD API

### Core Modules
```python
from jax_cfd import grids, equations, spectral, finite_differences
```

### grids - Computational Grid

```python
# 2D grid
grid = grids.Grid(
    shape=(256, 256),
    domain=((0, 2*jnp.pi), (0, 2*jnp.pi))
)

# 3D grid
grid = grids.Grid(
    shape=(128, 128, 128),
    domain=((0, 1), (0, 1), (0, 1))
)

# Access mesh coordinates
x, y = grid.mesh()        # 2D
x, y, z = grid.mesh()     # 3D

# Grid spacing
dx, dy = grid.step        # Tuple of step sizes
```

### equations - CFD Equations

**Advection:**
```python
# Advect scalar: u·∇φ
advection = equations.advect(
    scalar,      # Scalar field to advect
    velocity,    # Velocity field (tuple for 2D/3D)
    grid
)
```

**Divergence:**
```python
# Compute ∇·u
div = equations.divergence(velocity, grid)
```

**Gradient:**
```python
# Compute ∇p
grad_p = equations.gradient(pressure, grid)
```

**Laplacian:**
```python
# Compute ∇²u
lap = equations.laplacian(velocity, grid)
```

### spectral - Spectral Methods

**Poisson Solver:**
```python
# Solve ∇²p = rhs using FFT
pressure = spectral.solve_cg(rhs, grid)

# Alternative: direct FFT solver
pressure = spectral.solve_fft(rhs, grid)
```

**Wavenumbers:**
```python
# Get wavenumber arrays for spectral methods
kx, ky = spectral.wavenumbers(grid)
```

---

## Flax NNX API (for PINNs)

### Module Definition

```python
import flax.nnx as nnx

class MyModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.dense1(x))
        x = self.dense2(x)
        return x

# Instantiate
model = MyModel(rngs=nnx.Rngs(0))
```

### Common Layers

```python
# Linear layer
layer = nnx.Linear(in_features, out_features, rngs=rngs)

# Convolution
layer = nnx.Conv(in_features, out_features, kernel_size=(3, 3), rngs=rngs)

# Dropout
layer = nnx.Dropout(rate=0.1, rngs=rngs)

# Batch normalization
layer = nnx.BatchNorm(num_features, rngs=rngs)
```

### Activations

```python
# Common activations
x = nnx.relu(x)
x = nnx.tanh(x)
x = nnx.sigmoid(x)
x = nnx.softmax(x)
x = nnx.gelu(x)
```

### Optimization

```python
import optax

# Create optimizer
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3))

# Training step
loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, targets)
optimizer.update(grads)
```

---

## Optax API (Optimizers)

### Common Optimizers

```python
import optax

# Adam optimizer
optimizer = optax.adam(learning_rate=1e-3)

# SGD with momentum
optimizer = optax.sgd(learning_rate=0.01, momentum=0.9)

# AdamW (Adam with weight decay)
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)

# Lion optimizer
optimizer = optax.lion(learning_rate=1e-4)
```

### Learning Rate Schedules

```python
# Exponential decay
schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=1000,
    decay_rate=0.96
)

# Cosine decay
schedule = optax.cosine_decay_schedule(
    init_value=1e-3,
    decay_steps=10000
)

# Warmup + cosine decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=10000
)

# Use schedule with optimizer
optimizer = optax.adam(learning_rate=schedule)
```

### Gradient Transformations

```python
# Gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3)
)

# Gradient accumulation
optimizer = optax.MultiSteps(
    optax.adam(1e-3),
    every_k_schedule=4
)
```

---

## Diffrax API (Differential Equations)

### ODE Solvers

```python
import diffrax

# Define ODE: dy/dt = f(t, y, args)
def vector_field(t, y, args):
    return -y  # Exponential decay

# Solve ODE
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(vector_field),
    solver=diffrax.Dopri5(),      # Runge-Kutta 4(5)
    t0=0.0,
    t1=10.0,
    dt0=0.1,
    y0=jnp.array([1.0]),
    saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10, 100))
)

# Access results
times = solution.ts
values = solution.ys
```

### Common Solvers

```python
# Explicit methods
solver = diffrax.Euler()           # Forward Euler (1st order)
solver = diffrax.Heun()            # Heun's method (2nd order)
solver = diffrax.Dopri5()          # Dormand-Prince 4(5)
solver = diffrax.Tsit5()           # Tsitouras 4(5)

# Implicit methods (for stiff problems)
solver = diffrax.ImplicitEuler()
solver = diffrax.Kvaerno5()

# SDE solvers (stochastic differential equations)
solver = diffrax.EulerHeun()       # Euler-Heun for SDEs
```

---

## Quantum Computing APIs

### Cirq (Google)

```python
import cirq

# Create qubits
qubits = cirq.LineQubit.range(4)

# Build circuit
circuit = cirq.Circuit()
circuit.append(cirq.H(qubits[0]))                    # Hadamard
circuit.append(cirq.CNOT(qubits[0], qubits[1]))     # CNOT
circuit.append(cirq.ry(0.5)(qubits[2]))             # Ry rotation
circuit.append(cirq.rz(0.3)(qubits[3]))             # Rz rotation

# Simulate
simulator = cirq.Simulator()
result = simulator.simulate(circuit)
state_vector = result.final_state_vector
```

### PennyLane (Xanadu)

```python
import pennylane as qml

# Define device
dev = qml.device('default.qubit', wires=4)

# Define quantum function
@qml.qnode(dev, interface='jax')
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Compute expectation value (differentiable)
params = jnp.array([0.5, 0.3])
expectation = circuit(params)

# Compute gradient with JAX
gradient = jax.grad(circuit)(params)
```

---

## Common Patterns

### JAX Transformations for Physics

```python
import jax

# JIT compilation
@jax.jit
def md_step(state):
    return apply_fn(state)

# Automatic differentiation
energy_fn = lambda R: compute_energy(R)
force_fn = jax.grad(energy_fn)  # Forces = -∇E

# Vectorization
distances = jax.vmap(lambda R1, R2: space.distance(displacement_fn(R1, R2)))

# Parallelization
@jax.pmap
def parallel_md_step(state_per_device):
    return md_step(state_per_device)
```

### Device Sharding

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Create device mesh
devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, axis_names=('data',))

# Shard data
sharding = NamedSharding(mesh, P('data'))
R_sharded = jax.device_put(R, sharding)
```

---

## Further Documentation

For complete API documentation, see:
- **JAX-MD**: https://github.com/google/jax-md
- **JAX-CFD**: https://github.com/google/jax-cfd
- **Flax**: https://flax.readthedocs.io/
- **Optax**: https://optax.readthedocs.io/
- **Diffrax**: https://docs.kidger.site/diffrax/
- **Cirq**: https://quantumai.google/cirq
- **PennyLane**: https://pennylane.ai/
