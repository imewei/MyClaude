---
name: jax-physics-applications
description: Comprehensive workflows for physics simulations using JAX-based libraries (JAX-MD, JAX-CFD, PINNs). Use this skill when writing or modifying Python files that import jax_md, jax_cfd, cirq, or physics simulation libraries, when implementing molecular dynamics simulations (.py files with energy potentials, integrators, neighbor lists), when building computational fluid dynamics solvers (Navier-Stokes, turbulence models, incompressible/compressible flows), when designing physics-informed neural networks with PDE constraints (.py files combining Flax NNX with physics residuals), when developing quantum computing algorithms (VQE, QAOA, quantum circuits), when coupling multiple physics domains (multi-physics simulations, MD-CFD coupling), when validating physics correctness (energy conservation, mass conservation, momentum conservation), when implementing differentiable physics for gradient-based optimization, when scaling physics simulations to GPU/TPU hardware, when working with Lennard-Jones potentials, FENE bonds, or custom force fields, when solving PDEs with automatic differentiation, when implementing coarse-graining or multiscale methods, or when building data assimilation workflows combining physics models with observations.
---

# JAX Physics Applications

## Overview

Enable physics simulations and scientific computing using JAX's differentiable programming capabilities. This skill provides end-to-end workflows for computational fluid dynamics (CFD), molecular dynamics (MD), physics-informed neural networks (PINNs), and quantum computing applications using JAX-MD, JAX-CFD, Cirq/PennyLane, and related libraries.

**When to use this skill:**
- Implementing molecular dynamics simulations with JAX-MD
- Building CFD solvers with JAX-CFD or hybrid ML-CFD models
- Designing physics-informed neural networks (PINNs) with PDE constraints
- Developing quantum computing algorithms with JAX integration
- Coupling multiple physics domains (multi-physics simulations)
- Validating physics correctness of differentiable simulations
- Scaling physics simulations to multi-GPU/TPU systems

## Core Capabilities

### 1. Molecular Dynamics (MD) with JAX-MD

Implement differentiable molecular dynamics simulations for materials science, biophysics, and soft matter physics.

**Setup:**
```python
import jax
import jax.numpy as jnp
from jax_md import space, energy, simulate, quantity
```

**Workflow: Lennard-Jones Liquid Simulation**

```python
# 1. Define simulation space (periodic boundary conditions)
displacement_fn, shift_fn = space.periodic(box_size=10.0)

# 2. Define energy function (Lennard-Jones potential)
energy_fn = energy.lennard_jones_pair(
    displacement_fn,
    species=None,
    sigma=1.0,
    epsilon=1.0
)

# 3. Initialize system
key = jax.random.PRNGKey(0)
N = 500  # Number of particles
R = space.random_position(key, (N, 3), box_size=10.0)

# 4. Create integrator (NVE ensemble)
init_fn, apply_fn = simulate.nve(energy_fn, shift_fn, dt=0.005)
state = init_fn(key, R, mass=1.0)

# 5. Run simulation
@jax.jit
def step_fn(state):
    return apply_fn(state)

# Equilibration
for _ in range(1000):
    state = step_fn(state)

# Production run with observables
positions_trajectory = []
energies = []

for step in range(10000):
    state = step_fn(state)

    if step % 100 == 0:
        positions_trajectory.append(state.position)
        E = energy_fn(state.position)
        energies.append(E)

# 6. Analysis
kinetic_energy = quantity.kinetic_energy(state)
temperature = quantity.temperature(state, kB=1.0)
rdf = compute_radial_distribution(state.position, displacement_fn)
```

**Validation checklist:**
- ✓ Energy conservation (NVE: drift < 0.01%)
- ✓ Temperature stability (NVT: fluctuations < 5%)
- ✓ Radial distribution function matches expected structure
- ✓ Momentum conservation (total momentum ≈ 0)
- ✓ Translational invariance (energy unchanged by global translation)

**Advanced: Polymer Simulations with FENE Bonds**

```python
from jax_md import bond

# Define FENE potential for polymer chains
def fene_bond_energy(dr, k=30.0, r0=1.5):
    """Finitely Extensible Nonlinear Elastic potential"""
    return -0.5 * k * r0**2 * jnp.log(1 - (dr / r0)**2)

# Bond connectivity for polymer chain
bonds = [(i, i+1) for i in range(N-1)]  # Linear chain

bond_energy_fn = bond.simple_bond(
    displacement_fn,
    bond_fn=fene_bond_energy,
    bonds=bonds
)

# Total energy = LJ non-bonded + FENE bonded
total_energy_fn = lambda R: (
    energy.lennard_jones_pair(displacement_fn)(R) +
    bond_energy_fn(R)
)
```

See `references/jax_md_advanced.md` for:
- Neighbor lists for large systems
- Custom potentials (EAM, Stillinger-Weber)
- Thermostats and barostats (Nosé-Hoover, Parrinello-Rahman)
- Coarse-grained models

### 2. Computational Fluid Dynamics (CFD) with JAX-CFD

Implement differentiable CFD solvers for incompressible flows, turbulence, and ML-augmented fluid dynamics.

**Setup:**
```python
from jax_cfd import grids, equations, spectral
import jax.numpy as jnp
```

**Workflow: 2D Incompressible Navier-Stokes**

```python
# 1. Define computational grid
grid = grids.Grid(
    shape=(256, 256),
    domain=((0, 2*jnp.pi), (0, 2*jnp.pi))
)

# 2. Initial conditions (Taylor-Green vortex)
def taylor_green_ic(grid):
    x, y = grid.mesh()
    u = jnp.sin(x) * jnp.cos(y)
    v = -jnp.cos(x) * jnp.sin(y)
    return (u, v)

velocity = taylor_green_ic(grid)

# 3. Define equations (incompressible NS)
viscosity = 1e-3
dt = 0.001

@jax.jit
def navier_stokes_step(velocity):
    # Advection
    advection = equations.advect(velocity, velocity, grid)

    # Pressure projection (incompressibility)
    pressure = spectral.solve_cg(
        equations.divergence(velocity, grid),
        grid
    )

    # Diffusion
    diffusion = viscosity * equations.laplacian(velocity, grid)

    # Update
    velocity_new = velocity + dt * (-advection - pressure + diffusion)

    return velocity_new

# 4. Time integration
for step in range(10000):
    velocity = navier_stokes_step(velocity)

    if step % 100 == 0:
        kinetic_energy = jnp.sum(velocity[0]**2 + velocity[1]**2)
        enstrophy = compute_enstrophy(velocity, grid)
        print(f"Step {step}: KE={kinetic_energy:.4f}, Enstrophy={enstrophy:.4f}")
```

**Validation checklist:**
- ✓ Mass conservation (∇·u = 0 to machine precision)
- ✓ Energy decay rate matches analytical solution
- ✓ Enstrophy cascade for turbulent flows
- ✓ No-slip boundary conditions enforced correctly
- ✓ CFL condition satisfied (Courant number < 1)

**Advanced: ML-Augmented Turbulence Closure**

```python
import flax.nnx as nnx

# Neural network for subgrid-scale stress
class TurbulenceClosure(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=2, out_features=16, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(in_features=16, out_features=2, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, velocity):
        x = nnx.relu(self.conv1(velocity))
        sgs_stress = self.conv2(x)
        return sgs_stress

model = TurbulenceClosure(rngs=nnx.Rngs(0))

@jax.jit
def ml_cfd_step(velocity, model):
    # Standard NS terms
    advection = equations.advect(velocity, velocity, grid)
    pressure = spectral.solve_cg(equations.divergence(velocity, grid), grid)
    diffusion = viscosity * equations.laplacian(velocity, grid)

    # Learned turbulence closure
    sgs_stress = model(velocity)

    # Update with ML correction
    velocity_new = velocity + dt * (
        -advection - pressure + diffusion + sgs_stress
    )

    return velocity_new
```

See `references/jax_cfd_advanced.md` for:
- Compressible flow solvers
- Free surface flows
- Multiphase flow models
- Adaptive mesh refinement

### 3. Physics-Informed Neural Networks (PINNs)

Design neural networks that embed physical laws as soft constraints through PDE residuals.

**Setup:**
```python
import flax.nnx as nnx
import optax
```

**Workflow: Heat Equation PINN**

Problem: ∂u/∂t = α ∇²u with boundary conditions u(0,t) = u(L,t) = 0

```python
# 1. Define neural network architecture
class HeatPINN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(2, 64, rngs=rngs)  # Input: (x, t)
        self.dense2 = nnx.Linear(64, 64, rngs=rngs)
        self.dense3 = nnx.Linear(64, 64, rngs=rngs)
        self.dense4 = nnx.Linear(64, 1, rngs=rngs)   # Output: u(x,t)

    def __call__(self, x, t):
        xt = jnp.stack([x, t], axis=-1)
        h = nnx.tanh(self.dense1(xt))
        h = nnx.tanh(self.dense2(h))
        h = nnx.tanh(self.dense3(h))
        u = self.dense4(h)
        return u

model = HeatPINN(rngs=nnx.Rngs(0))

# 2. Define PINN loss (physics + boundary conditions + initial condition)
def pinn_loss(model, x_pde, t_pde, x_bc, t_bc, x_ic, t_ic, u_ic):
    """
    x_pde, t_pde: Collocation points for PDE residual
    x_bc, t_bc: Boundary condition points
    x_ic, t_ic, u_ic: Initial condition data
    """
    alpha = 0.01  # Thermal diffusivity

    # PDE residual: ∂u/∂t - α ∇²u = 0
    def u_fn(x, t):
        return model(x, t)

    # Automatic differentiation for derivatives
    u_t = jax.vmap(jax.grad(u_fn, argnums=1))(x_pde, t_pde)
    u_x = jax.vmap(jax.grad(u_fn, argnums=0))(x_pde, t_pde)
    u_xx = jax.vmap(jax.grad(jax.grad(u_fn, argnums=0), argnums=0))(x_pde, t_pde)

    pde_residual = u_t - alpha * u_xx
    loss_pde = jnp.mean(pde_residual ** 2)

    # Boundary conditions: u(0,t) = u(L,t) = 0
    u_bc = jax.vmap(u_fn)(x_bc, t_bc)
    loss_bc = jnp.mean(u_bc ** 2)

    # Initial condition: u(x,0) = u_ic(x)
    u_ic_pred = jax.vmap(u_fn)(x_ic, t_ic)
    loss_ic = jnp.mean((u_ic_pred - u_ic) ** 2)

    # Combined loss with weighting
    total_loss = loss_pde + 100 * loss_bc + 100 * loss_ic

    return total_loss, {
        'pde': loss_pde,
        'bc': loss_bc,
        'ic': loss_ic
    }

# 3. Training loop
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

for epoch in range(10000):
    # Sample collocation points
    key = jax.random.PRNGKey(epoch)
    x_pde = jax.random.uniform(key, (1000,), minval=0, maxval=1)
    t_pde = jax.random.uniform(key, (1000,), minval=0, maxval=1)

    # Compute gradients
    loss_grad_fn = nnx.value_and_grad(pinn_loss, has_aux=True)
    (loss, metrics), grads = loss_grad_fn(
        model, x_pde, t_pde, x_bc, t_bc, x_ic, t_ic, u_ic
    )

    # Update
    optimizer.update(grads)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss={loss:.6f}, "
              f"PDE={metrics['pde']:.6f}, BC={metrics['bc']:.6f}, IC={metrics['ic']:.6f}")
```

**Validation checklist:**
- ✓ PDE residual < 1e-4 (physics satisfied)
- ✓ Boundary conditions satisfied (error < 1e-5)
- ✓ Solution matches analytical solution (if available)
- ✓ Energy conservation (if applicable)
- ✓ Stability under perturbations

**Advanced: Inverse Problems and Parameter Estimation**

```python
# Learn unknown parameters from observations
class HeatPINNInverse(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        # Neural network for u(x,t)
        self.dense1 = nnx.Linear(2, 64, rngs=rngs)
        self.dense2 = nnx.Linear(64, 64, rngs=rngs)
        self.dense3 = nnx.Linear(64, 1, rngs=rngs)

        # Learnable thermal diffusivity (unknown parameter)
        self.alpha = nnx.Param(jnp.array([0.01]))

    def __call__(self, x, t):
        xt = jnp.stack([x, t], axis=-1)
        h = nnx.tanh(self.dense1(xt))
        h = nnx.tanh(self.dense2(h))
        u = self.dense3(h)
        return u

def inverse_loss(model, observations):
    """Learn alpha from sparse observations"""
    x_obs, t_obs, u_obs = observations

    # Data fitting loss
    u_pred = jax.vmap(model)(x_obs, t_obs)
    loss_data = jnp.mean((u_pred - u_obs) ** 2)

    # PDE residual with learnable alpha
    # (PDE loss code here using model.alpha)

    return loss_data + loss_pde
```

See `references/pinns_advanced.md` for:
- Multi-physics coupling (Navier-Stokes + heat transfer)
- Stiff PDEs and adaptive weighting
- Uncertainty quantification with dropout
- Transfer learning across geometries

### 4. Quantum Computing with JAX

Implement variational quantum algorithms with JAX-compatible quantum frameworks.

**Setup:**
```python
import cirq
import jax.numpy as jnp
```

**Workflow: Variational Quantum Eigensolver (VQE)**

```python
import cirq
import jax
import jax.numpy as jnp

# 1. Define quantum circuit (ansatz)
def create_vqe_circuit(qubits, params):
    """Variational circuit for molecular Hamiltonian"""
    circuit = cirq.Circuit()

    # Initial layer
    for qubit in qubits:
        circuit.append(cirq.H(qubit))

    # Variational layers
    n_layers = len(params) // (2 * len(qubits))
    param_idx = 0

    for layer in range(n_layers):
        # Rotation layer
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(params[param_idx])(qubit))
            param_idx += 1

        # Entangling layer
        for i in range(len(qubits) - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

        # Second rotation layer
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rz(params[param_idx])(qubit))
            param_idx += 1

    return circuit

# 2. Define Hamiltonian (molecular hydrogen H2)
def h2_hamiltonian():
    """Hamiltonian for H2 molecule in minimal basis"""
    # Simplified example - real H2 has more terms
    pauli_sum = {
        'IIII': -0.8,
        'ZIII': 0.2,
        'IZII': 0.2,
        'IIZI': -0.2,
        'IIIZ': -0.2,
        'ZZII': 0.1,
        'XXII': 0.05,
    }
    return pauli_sum

# 3. Expectation value computation (JAX-differentiable)
@jax.jit
def compute_energy(params):
    """Compute <ψ(θ)|H|ψ(θ)> using quantum circuit"""
    qubits = cirq.LineQubit.range(4)
    circuit = create_vqe_circuit(qubits, params)

    # Simulate circuit
    simulator = cirq.Simulator()
    state_vector = simulator.simulate(circuit).final_state_vector

    # Compute expectation value for each Pauli term
    hamiltonian = h2_hamiltonian()
    energy = 0.0

    for pauli_string, coeff in hamiltonian.items():
        # Convert Pauli string to matrix and compute <ψ|P|ψ>
        expectation = compute_pauli_expectation(state_vector, pauli_string)
        energy += coeff * expectation

    return energy

# 4. Optimize variational parameters
import optax

n_qubits = 4
n_layers = 3
n_params = n_layers * 2 * n_qubits
params = jax.random.normal(jax.random.PRNGKey(0), (n_params,)) * 0.1

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

for step in range(1000):
    # Compute gradient
    energy, grads = jax.value_and_grad(compute_energy)(params)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    if step % 100 == 0:
        print(f"Step {step}: Energy = {energy:.6f} Ha")

print(f"Final ground state energy: {energy:.6f} Ha")
```

**Validation checklist:**
- ✓ Energy below exact diagonalization result
- ✓ Convergence to chemical accuracy (< 1 mHa)
- ✓ Gradient flow not vanishing (barren plateaus check)
- ✓ Parameterized circuit expressiveness sufficient

See `references/quantum_jax.md` for:
- QAOA for optimization problems
- Quantum machine learning models
- Hybrid quantum-classical architectures
- Error mitigation techniques

### 5. Multi-Physics Coupling and Advanced Workflows

**Workflow: Coupled MD + PINN for Coarse-Graining**

```python
# Use MD for atomistic region, PINN for continuum region

def coupled_md_pinn_simulation():
    """Adaptive multiscale simulation"""

    # Atomistic region (JAX-MD)
    R_atoms = initialize_atomic_positions()
    energy_fn_atoms = energy.lennard_jones_pair(displacement_fn)

    # Continuum region (PINN)
    continuum_model = ContinuumPINN(rngs=nnx.Rngs(0))

    for step in range(10000):
        # 1. MD step in atomistic region
        R_atoms = md_step(R_atoms, energy_fn_atoms)

        # 2. Extract boundary conditions from MD
        boundary_stress = compute_stress_tensor(R_atoms)

        # 3. PINN step in continuum region with MD BC
        continuum_field = pinn_step(continuum_model, boundary_stress)

        # 4. Apply continuum forces back to MD boundary
        boundary_forces = extract_forces_from_continuum(continuum_field)
        apply_forces_to_md_boundary(R_atoms, boundary_forces)
```

**Workflow: Data Assimilation with Physics Models**

```python
# Combine sparse observations with physics model

def physics_data_assimilation(observations, physics_model):
    """4D-Var data assimilation with physics constraints"""

    def cost_function(initial_state):
        # Forward physics simulation
        trajectory = run_physics_model(initial_state, physics_model)

        # Data fitting term
        data_misfit = jnp.sum((trajectory[obs_times] - observations) ** 2)

        # Background term (prior)
        background_misfit = jnp.sum((initial_state - prior_state) ** 2)

        # Physics regularization (PDE residual)
        physics_residual = compute_pde_residual(trajectory)

        return data_misfit + background_misfit + physics_residual

    # Optimize initial state
    grad_fn = jax.grad(cost_function)
    optimal_initial_state = adam_optimize(grad_fn, initial_guess)

    # Posterior uncertainty via Laplace approximation
    hessian = jax.hessian(cost_function)(optimal_initial_state)
    posterior_covariance = jnp.linalg.inv(hessian)

    return optimal_initial_state, posterior_covariance
```

## Performance and Scaling

**Multi-Device Sharding for Large Simulations:**

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Setup device mesh
devices = mesh_utils.create_device_mesh((8,))  # 8 GPUs
mesh = Mesh(devices, axis_names=('data',))

# Shard MD simulation across devices
sharding = NamedSharding(mesh, P('data'))
R_sharded = jax.device_put(R, sharding)

@jax.jit
def parallel_md_step(R_sharded):
    """Each device handles portion of particles"""
    return md_step_fn(R_sharded)

# Run with automatic parallelization
for step in range(100000):
    R_sharded = parallel_md_step(R_sharded)
```

**Memory Optimization for Large Systems:**

```python
# Use gradient checkpointing for long trajectories
from jax.checkpoint import checkpoint

@checkpoint
def expensive_physics_step(state):
    """Checkpoint intermediate activations"""
    return physics_forward_pass(state)

# Reduces memory by 5-10x at cost of 30% more computation
```

## Validation and Diagnostics

**Comprehensive Physics Validation Suite:**

```python
def validate_simulation(trajectory, energy_fn, physics_type='md'):
    """Multi-level validation checklist"""

    validation_results = {}

    if physics_type == 'md':
        # Energy conservation
        energies = [energy_fn(R) for R in trajectory]
        energy_drift = jnp.std(energies) / jnp.mean(energies)
        validation_results['energy_drift'] = energy_drift
        assert energy_drift < 1e-4, f"Energy drift too large: {energy_drift}"

        # Momentum conservation
        momenta = [jnp.sum(v, axis=0) for v in velocities]
        momentum_drift = jnp.std(momenta, axis=0)
        validation_results['momentum_drift'] = momentum_drift

        # Structural properties
        rdf = compute_radial_distribution_function(trajectory)
        validation_results['rdf'] = rdf

    elif physics_type == 'cfd':
        # Mass conservation (divergence-free)
        divergence = [compute_divergence(v) for v in trajectory]
        max_divergence = jnp.max(jnp.abs(divergence))
        validation_results['max_divergence'] = max_divergence
        assert max_divergence < 1e-10, f"Not divergence-free: {max_divergence}"

        # Energy decay (turbulence)
        kinetic_energies = [compute_kinetic_energy(v) for v in trajectory]
        validation_results['energy_decay'] = kinetic_energies

    elif physics_type == 'pinn':
        # PDE residual
        pde_residuals = compute_pde_residual_on_grid(trajectory)
        max_residual = jnp.max(jnp.abs(pde_residuals))
        validation_results['max_pde_residual'] = max_residual
        assert max_residual < 1e-3, f"PDE not satisfied: {max_residual}"

        # Boundary conditions
        bc_error = compute_boundary_error(trajectory)
        validation_results['bc_error'] = bc_error

    return validation_results
```

## Resources

### scripts/
Complete, production-ready example scripts demonstrating end-to-end physics workflows:

#### `scripts/md_lennard_jones.py`
Complete molecular dynamics simulation using JAX-MD:
- Lennard-Jones liquid with 500 particles
- Periodic boundary conditions
- NVE ensemble (constant energy)
- Equilibration + production run with observable sampling
- Radial distribution function (RDF) analysis
- Energy conservation validation
- Comprehensive visualization

**Usage**: `python scripts/md_lennard_jones.py`

#### `scripts/cfd_taylor_green.py`
Complete CFD simulation using JAX-CFD:
- 2D incompressible Navier-Stokes solver
- Taylor-Green vortex benchmark
- Pressure projection for divergence-free velocity
- Energy decay validation against analytical solution
- Mass conservation checks (∇·u = 0)
- Enstrophy evolution tracking
- Multi-panel visualization

**Usage**: `python scripts/cfd_taylor_green.py`

#### `scripts/pinn_heat_equation.py`
Complete Physics-Informed Neural Network using Flax NNX:
- 1D heat equation: ∂u/∂t = α∂²u/∂x²
- Physics-informed loss (PDE residual + BC + IC)
- Automatic differentiation for PDE derivatives
- Validation against analytical solution
- PDE residual validation on test points
- Solution accuracy metrics (max/mean error)
- Comprehensive heatmaps and error analysis

**Usage**: `python scripts/pinn_heat_equation.py`

#### `scripts/vqe_hydrogen.py`
Complete Variational Quantum Eigensolver using JAX:
- H2 molecule ground state calculation
- Parameterized quantum circuit (ansatz)
- Hamiltonian in Pauli basis
- JAX-based gradient optimization
- Chemical accuracy validation (< 1 mHa)
- Convergence analysis and diagnostics
- Statevector simulation (no external quantum backend required)

**Usage**: `python scripts/vqe_hydrogen.py`

**Note**: Each script is self-contained with comprehensive comments, validation checks, and visualization. All scripts generate publication-quality plots saved as PNG files.

### references/
Detailed technical references for advanced implementations:

- `references/api_reference.md` - **Quick API reference** for JAX-MD, JAX-CFD, Flax NNX, Optax, Diffrax, and quantum libraries with function signatures and common patterns
- `references/jax_md_advanced.md` - Advanced JAX-MD patterns (neighbor lists, custom potentials, ensembles)
- `references/jax_cfd_advanced.md` - Advanced CFD solvers (compressible flow, multiphase, AMR)
- `references/pinns_advanced.md` - Advanced PINN techniques (adaptive weighting, stiff PDEs, transfer learning)
- `references/quantum_jax.md` - Quantum computing with JAX (VQE, QAOA, QML)
- `references/physics_validation.md` - Comprehensive validation checklists and diagnostics

### assets/
Visual references and architecture diagrams:

- `assets/physics_workflow_diagrams.txt` - ASCII diagrams of MD/CFD/PINN workflows
- `assets/multiphysics_coupling.txt` - Multi-physics coupling patterns
- `assets/validation_checklists.txt` - Domain-specific validation procedures

## When to Delegate

**Delegate to jax-pro:**
- Core JAX transformation optimization (jit, vmap, pmap)
- Memory efficiency patterns (gradient checkpointing, mixed precision)
- Complex device sharding strategies

**Delegate to neural-architecture-engineer:**
- Novel neural network architectures for PINNs
- Hyperparameter tuning for physics-informed models
- Advanced training strategies

**Delegate to simulation-expert:**
- Legacy MD code (LAMMPS, GROMACS) integration
- Traditional CFD benchmarks and validation
- Non-JAX simulation workflows

**Keep within this skill:**
- Physics-specific JAX-MD/JAX-CFD implementations
- PINN design with PDE constraints
- Physics validation and correctness checks
- Domain-specific library usage (JAX-MD, JAX-CFD, quantum frameworks)
