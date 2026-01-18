---
name: jax-physics-applications
version: "2.1.0"
maturity: "5-Expert"
specialization: JAX Physics Simulations
description: Physics simulations using JAX-based libraries (JAX-MD, JAX-CFD, PINNs). Use when implementing molecular dynamics, computational fluid dynamics, physics-informed neural networks with PDE constraints, quantum computing algorithms (VQE, QAOA), multi-physics coupling, or differentiable physics for gradient-based optimization.
---

# JAX Physics Applications

Differentiable physics simulations with JAX-MD, JAX-CFD, and PINNs.

---

## Domain Selection

| Domain | Library | Use Case |
|--------|---------|----------|
| Molecular Dynamics | JAX-MD | Particles, polymers, materials |
| Computational Fluids | JAX-CFD | Navier-Stokes, turbulence |
| PDE Solving | PINNs (Flax) | Heat, wave, diffusion |
| Quantum Computing | Cirq/PennyLane | VQE, QAOA |

---

## Molecular Dynamics (JAX-MD)

```python
import jax
import jax.numpy as jnp
from jax_md import space, energy, simulate, quantity

# 1. Define simulation space
displacement_fn, shift_fn = space.periodic(box_size=10.0)

# 2. Define energy function
energy_fn = energy.lennard_jones_pair(displacement_fn, sigma=1.0, epsilon=1.0)

# 3. Initialize system
key = jax.random.PRNGKey(0)
N = 500
R = space.random_position(key, (N, 3), box_size=10.0)

# 4. Create integrator (NVE ensemble)
init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, dt=0.005)
state = init_fn(key, R, mass=1.0)

# 5. Run simulation
@jax.jit
def step_fn(state):
    return apply_fn(state)

for _ in range(10000):
    state = step_fn(state)

# 6. Compute observables
kinetic_energy = quantity.kinetic_energy(state)
temperature = quantity.temperature(state, kB=1.0)
```

**Validation**: Energy drift < 0.01%, momentum ≈ 0, RDF matches expected structure.

---

## Computational Fluids (JAX-CFD)

```python
from jax_cfd import grids, equations, spectral

# 1. Define grid
grid = grids.Grid(shape=(256, 256), domain=((0, 2*jnp.pi), (0, 2*jnp.pi)))

# 2. Initial conditions (Taylor-Green vortex)
x, y = grid.mesh()
u = jnp.sin(x) * jnp.cos(y)
v = -jnp.cos(x) * jnp.sin(y)
velocity = (u, v)

# 3. Navier-Stokes step
viscosity, dt = 1e-3, 0.001

@jax.jit
def ns_step(velocity):
    advection = equations.advect(velocity, velocity, grid)
    pressure = spectral.solve_cg(equations.divergence(velocity, grid), grid)
    diffusion = viscosity * equations.laplacian(velocity, grid)
    return velocity + dt * (-advection - pressure + diffusion)

# 4. Time integration
for step in range(10000):
    velocity = ns_step(velocity)
```

**Validation**: ∇·u = 0 (mass conservation), energy decay matches analytical solution.

---

## Physics-Informed Neural Networks

```python
import flax.nnx as nnx
import optax

# Heat equation: ∂u/∂t = α ∇²u
class HeatPINN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(2, 64, rngs=rngs)
        self.dense2 = nnx.Linear(64, 64, rngs=rngs)
        self.dense3 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x, t):
        xt = jnp.stack([x, t], axis=-1)
        h = nnx.tanh(self.dense1(xt))
        h = nnx.tanh(self.dense2(h))
        return self.dense3(h)

def pinn_loss(model, x_pde, t_pde, x_bc, t_bc, x_ic, u_ic, alpha=0.01):
    def u_fn(x, t): return model(x, t)

    # PDE residual: ∂u/∂t - α ∂²u/∂x² = 0
    u_t = jax.vmap(jax.grad(u_fn, argnums=1))(x_pde, t_pde)
    u_xx = jax.vmap(jax.grad(jax.grad(u_fn, argnums=0), argnums=0))(x_pde, t_pde)
    loss_pde = jnp.mean((u_t - alpha * u_xx) ** 2)

    # Boundary conditions: u(0,t) = u(L,t) = 0
    loss_bc = jnp.mean(jax.vmap(u_fn)(x_bc, t_bc) ** 2)

    # Initial condition
    loss_ic = jnp.mean((jax.vmap(u_fn)(x_ic, jnp.zeros_like(x_ic)) - u_ic) ** 2)

    return loss_pde + 100 * loss_bc + 100 * loss_ic

# Training
model = HeatPINN(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

for epoch in range(10000):
    loss, grads = nnx.value_and_grad(pinn_loss)(model, x_pde, t_pde, x_bc, t_bc, x_ic, u_ic)
    optimizer.update(grads)
```

**Validation**: PDE residual < 1e-4, BC error < 1e-5.

---

## Variational Quantum Eigensolver

```python
import cirq
import optax

# H2 molecule ground state
def create_vqe_circuit(qubits, params):
    circuit = cirq.Circuit()
    for qubit in qubits:
        circuit.append(cirq.H(qubit))

    param_idx = 0
    for _ in range(3):  # Layers
        for qubit in qubits:
            circuit.append(cirq.ry(params[param_idx])(qubit))
            param_idx += 1
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

    return circuit

@jax.jit
def compute_energy(params):
    qubits = cirq.LineQubit.range(4)
    circuit = create_vqe_circuit(qubits, params)
    simulator = cirq.Simulator()
    state = simulator.simulate(circuit).final_state_vector
    # Compute <ψ|H|ψ> for H2 Hamiltonian
    return expectation_value(state, h2_hamiltonian())

# Optimize
params = jax.random.normal(jax.random.PRNGKey(0), (24,)) * 0.1
optimizer = optax.adam(0.01)
opt_state = optimizer.init(params)

for step in range(1000):
    energy, grads = jax.value_and_grad(compute_energy)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

**Validation**: Energy < exact diagonalization, chemical accuracy < 1 mHa.

---

## Multi-GPU Scaling

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Setup device mesh
devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, axis_names=('data',))

# Shard particles across devices
sharding = NamedSharding(mesh, P('data'))
R_sharded = jax.device_put(R, sharding)

@jax.jit
def parallel_md_step(R):
    return md_step_fn(R)

for step in range(100000):
    R_sharded = parallel_md_step(R_sharded)
```

---

## Validation Checklist

| Domain | Checks |
|--------|--------|
| MD | Energy drift < 0.01%, momentum ≈ 0, RDF correct |
| CFD | ∇·u = 0, CFL < 1, energy decay matches |
| PINNs | PDE residual < 1e-4, BC satisfied |
| Quantum | Energy < exact, no barren plateaus |

---

## Delegation

| Delegate To | When |
|-------------|------|
| jax-pro | Core JAX transforms, memory optimization |
| neural-architecture-engineer | Novel PINN architectures |
| simulation-expert | LAMMPS/GROMACS integration |

---

## Checklist

- [ ] Select appropriate physics library (JAX-MD, JAX-CFD)
- [ ] Validate conservation laws (energy, mass, momentum)
- [ ] Use gradient checkpointing for long trajectories
- [ ] Enable multi-device sharding for large systems
- [ ] Verify PINN satisfies PDE and boundary conditions
- [ ] Profile with JAX profiler before scaling

---

**Version**: 1.0.5
