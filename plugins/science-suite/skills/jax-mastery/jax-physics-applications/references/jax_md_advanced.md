# Advanced JAX-MD Patterns

## Table of Contents
1. [Neighbor Lists for Large Systems](#neighbor-lists)
2. [Custom Potentials](#custom-potentials)
3. [Thermostats and Barostats](#thermostats-barostats)
4. [Coarse-Grained Models](#coarse-grained-models)

## Neighbor Lists for Large Systems {#neighbor-lists}

For systems with many particles (N > 1000), computing all pairwise interactions becomes O(N²), which is prohibitively expensive. Neighbor lists reduce this to O(N) by only considering particles within a cutoff distance.

### Verlet Neighbor List

```python
from jax_md import space, partition

# Define displacement and neighbor list
displacement_fn, shift_fn = space.periodic(box_size=10.0)

# Create neighbor list (update every 20 steps)
neighbor_fn = partition.neighbor_list(
    displacement_fn,
    box_size=10.0,
    r_cutoff=3.0,  # Interaction cutoff
    dr_threshold=0.5,  # Rebuild when particles move > 0.5
    capacity_multiplier=1.25  # Allocate extra space
)

# Initialize neighbor list
neighbors = neighbor_fn.allocate(R)

# Energy function using neighbor list
energy_fn = energy.lennard_jones_neighbor_list(
    displacement_fn,
    box_size=10.0,
    sigma=1.0,
    epsilon=1.0
)

# Simulation loop with neighbor list updates
for step in range(10000):
    # Update neighbor list if needed
    neighbors = neighbors.update(R)

    # Compute forces using neighbor list
    force_fn = jax.grad(lambda r: energy_fn(r, neighbor=neighbors))
    forces = force_fn(R)

    # Integrate equations of motion
    R, V = velocity_verlet_step(R, V, forces, dt=0.005)
```

### Cell List Optimization

For extremely large systems (N > 100,000), cell lists provide additional speedup:

```python
from jax_md import partition

# Create cell list (divides space into cells)
cell_size = 3.0  # Size of each cell
cell_list_fn = partition.cell_list(
    box_size=10.0,
    r_cutoff=3.0,
    cell_size=cell_size
)

# Energy function with cell list
energy_fn = energy.lennard_jones_cell_list(
    displacement_fn,
    box_size=10.0,
    sigma=1.0,
    epsilon=1.0,
    cell_size=cell_size
)
```

## Custom Potentials {#custom-potentials}

### Embedded Atom Method (EAM)

Used for metallic systems where many-body effects are important:

```python
def eam_energy(R, rho_fn, phi_fn, F_fn, displacement_fn):
    """
    EAM potential: E = Σ F(ρᵢ) + ½ Σᵢⱼ φ(rᵢⱼ)

    rho_fn: Electron density function
    phi_fn: Pair potential
    F_fn: Embedding function
    """
    N = R.shape[0]

    # Compute electron density at each atom
    def compute_rho_i(i):
        rho = 0.0
        for j in range(N):
            if i != j:
                dr = space.distance(displacement_fn(R[i], R[j]))
                rho += rho_fn(dr)
        return rho

    rho = jax.vmap(compute_rho_i)(jnp.arange(N))

    # Embedding energy
    embedding_energy = jnp.sum(jax.vmap(F_fn)(rho))

    # Pair energy
    def pair_energy_ij(i, j):
        dr = space.distance(displacement_fn(R[i], R[j]))
        return jnp.where(i < j, phi_fn(dr), 0.0)

    i_indices, j_indices = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    pair_energy = jnp.sum(jax.vmap(jax.vmap(pair_energy_ij))(i_indices, j_indices))

    return embedding_energy + pair_energy

# Example: Aluminum EAM potential
def al_rho(r):
    """Electron density for Al"""
    return jnp.exp(-3.0 * r) * (1.0 + 0.5 * r)

def al_phi(r):
    """Pair potential for Al"""
    return 1.0 / r - 2.0 * jnp.exp(-2.0 * r)

def al_F(rho):
    """Embedding function for Al"""
    return -jnp.sqrt(rho)

energy_fn = lambda R: eam_energy(R, al_rho, al_phi, al_F, displacement_fn)
```

### Stillinger-Weber Potential

Three-body potential for silicon and other tetrahedral systems:

```python
def stillinger_weber_energy(R, displacement_fn):
    """
    SW potential: E = Σᵢⱼ v₂(rᵢⱼ) + Σᵢⱼₖ v₃(rᵢⱼ, rᵢₖ, θⱼᵢₖ)
    """
    N = R.shape[0]

    # Two-body term
    def v2(r, A=7.05, B=0.602, p=4, q=0, sigma=2.095, epsilon=2.17):
        """SW two-body potential"""
        x = r / sigma
        return epsilon * A * (B * x**(-p) - x**(-q)) * jnp.exp(1.0 / (x - 1.0))

    # Three-body term
    def v3(rij, rik, theta, lambda_=21.0, gamma=1.2, sigma=2.095, epsilon=2.17):
        """SW three-body potential"""
        xij = rij / sigma
        xik = rik / sigma
        cos_theta0 = -1.0 / 3.0  # Tetrahedral angle

        h = (jnp.cos(theta) - cos_theta0)**2
        cutoff_ij = jnp.exp(gamma / (xij - 1.0))
        cutoff_ik = jnp.exp(gamma / (xik - 1.0))

        return epsilon * lambda_ * h * cutoff_ij * cutoff_ik

    # Compute total energy
    total_energy = 0.0

    # Two-body contributions
    for i in range(N):
        for j in range(i+1, N):
            rij = space.distance(displacement_fn(R[i], R[j]))
            total_energy += v2(rij)

    # Three-body contributions
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for k in range(j+1, N):
                if i == k:
                    continue

                rij = space.distance(displacement_fn(R[i], R[j]))
                rik = space.distance(displacement_fn(R[i], R[k]))

                # Compute angle θⱼᵢₖ
                Rij = displacement_fn(R[i], R[j])
                Rik = displacement_fn(R[i], R[k])
                cos_theta = jnp.dot(Rij, Rik) / (rij * rik)
                theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))

                total_energy += v3(rij, rik, theta)

    return total_energy
```

## Thermostats and Barostats {#thermostats-barostats}

### Nosé-Hoover Thermostat (NVT Ensemble)

Maintains constant temperature through a dynamic friction term:

```python
from jax_md import simulate

# Create NVT integrator with Nosé-Hoover thermostat
init_fn, apply_fn = simulate.nvt_nose_hoover(
    energy_fn,
    shift_fn,
    dt=0.005,
    kT=1.0,  # Target temperature
    tau=0.5  # Thermostat relaxation time
)

# Initialize with thermostat variables
state = init_fn(key, R, mass=1.0)

# Run NVT simulation
for step in range(10000):
    state = apply_fn(state)

    # Monitor instantaneous temperature
    T_inst = quantity.temperature(state, kB=1.0)
    print(f"Step {step}: T = {T_inst:.3f}")
```

### Langevin Thermostat (Stochastic)

Couples system to heat bath through friction and random forces:

```python
# Langevin dynamics with friction
init_fn, apply_fn = simulate.nvt_langevin(
    energy_fn,
    shift_fn,
    dt=0.005,
    kT=1.0,
    gamma=0.1  # Friction coefficient
)

state = init_fn(key, R, mass=1.0)

# Run with stochastic thermostat
for step in range(10000):
    key, subkey = jax.random.split(key)
    state = apply_fn(state, key=subkey)
```

### Parrinello-Rahman Barostat (NPT Ensemble)

Allows box size to fluctuate to maintain constant pressure:

```python
from jax_md import simulate

# NPT integrator with Parrinello-Rahman barostat
init_fn, apply_fn = simulate.npt_nose_hoover(
    energy_fn,
    shift_fn,
    dt=0.005,
    kT=1.0,        # Target temperature
    pressure=1.0,  # Target pressure
    tau_T=0.5,     # Thermostat time constant
    tau_P=5.0      # Barostat time constant
)

# Initialize with barostat variables
state = init_fn(key, R, mass=1.0, box=box)

# Run NPT simulation with variable box
for step in range(10000):
    state = apply_fn(state)

    # Monitor pressure and volume
    P_inst = quantity.pressure(state, energy_fn)
    volume = jnp.prod(state.box)
    print(f"Step {step}: P = {P_inst:.3f}, V = {volume:.3f}")
```

## Coarse-Grained Models {#coarse-grained-models}

### Dissipative Particle Dynamics (DPD)

Mesoscale simulation method for complex fluids:

```python
def dpd_force(R, V, displacement_fn, rc=1.0, a=25.0, gamma=4.5, sigma=3.0, dt=0.01):
    """
    DPD force: F = F_conservative + F_dissipative + F_random
    """
    N = R.shape[0]

    def force_on_i(i):
        F_total = jnp.zeros(3)

        for j in range(N):
            if i == j:
                continue

            # Distance and unit vector
            Rij = displacement_fn(R[i], R[j])
            rij = space.distance(Rij)
            eij = Rij / rij

            # Weight function
            w = jnp.where(rij < rc, 1.0 - rij / rc, 0.0)

            # Conservative force (soft repulsion)
            F_C = a * w * eij

            # Dissipative force (friction)
            Vij = V[i] - V[j]
            F_D = -gamma * w**2 * jnp.dot(Vij, eij) * eij

            # Random force (thermal fluctuations)
            # Note: Proper implementation requires correlated random numbers
            # sigma = sqrt(2 * gamma * kT / dt)
            xi = jax.random.normal(key, ())
            F_R = sigma * w * xi * eij

            F_total += F_C + F_D + F_R

        return F_total

    return jax.vmap(force_on_i)(jnp.arange(N))

# DPD simulation loop
for step in range(10000):
    forces = dpd_force(R, V, displacement_fn)

    # Velocity Verlet integration
    R = R + V * dt + 0.5 * forces * dt**2
    V_half = V + 0.5 * forces * dt

    forces_new = dpd_force(R, V_half, displacement_fn)
    V = V_half + 0.5 * forces_new * dt
```

### MARTINI Coarse-Grained Force Field

Widely used for lipid bilayers and proteins:

```python
def martini_energy(R, bonds, angles, dihedrals, displacement_fn):
    """
    MARTINI force field with bonded and non-bonded interactions
    """

    # Non-bonded: Lennard-Jones with types
    def lj_typed(ri, rj, type_i, type_j):
        sigma, epsilon = get_martini_params(type_i, type_j)
        dr = space.distance(displacement_fn(ri, rj))
        sigma_over_r = sigma / dr
        return 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)

    # Bonded: Harmonic bonds
    def bond_energy(i, j, r0=0.47, kb=1250):
        dr = space.distance(displacement_fn(R[i], R[j]))
        return 0.5 * kb * (dr - r0)**2

    # Angle: Cosine angle potential
    def angle_energy(i, j, k, theta0=2.0, ka=25.0):
        Rij = displacement_fn(R[i], R[j])
        Rkj = displacement_fn(R[k], R[j])
        rij = space.distance(Rij)
        rkj = space.distance(Rkj)

        cos_theta = jnp.dot(Rij, Rkj) / (rij * rkj)
        theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))

        return 0.5 * ka * (jnp.cos(theta) - jnp.cos(theta0))**2

    # Dihedral: Proper dihedral angle
    def dihedral_energy(i, j, k, l, phi0=jnp.pi, kd=50.0):
        # Compute dihedral angle φ
        # (implementation details omitted for brevity)
        return kd * (1 + jnp.cos(phi - phi0))

    # Total energy
    E_nonbonded = compute_nonbonded_energy(R, lj_typed)
    E_bonds = sum(bond_energy(i, j) for i, j in bonds)
    E_angles = sum(angle_energy(i, j, k) for i, j, k in angles)
    E_dihedrals = sum(dihedral_energy(i, j, k, l) for i, j, k, l in dihedrals)

    return E_nonbonded + E_bonds + E_angles + E_dihedrals
```

### Adaptive Resolution Scheme (AdResS)

Couples atomistic and coarse-grained regions:

```python
def adress_energy(R, R_cg, atomistic_region, cg_region, transition_region):
    """
    AdResS: E = w(r) * E_atomistic + (1 - w(r)) * E_coarse_grained

    w(r): Weighting function (1 in atomistic, 0 in CG, smooth transition)
    """

    def weighting_function(r, r_center, r_inner, r_outer):
        """Smooth transition from atomistic (1) to CG (0)"""
        dist = jnp.linalg.norm(r - r_center)

        if dist < r_inner:
            return 1.0
        elif dist > r_outer:
            return 0.0
        else:
            # Smooth interpolation
            x = (dist - r_inner) / (r_outer - r_inner)
            return 1.0 - 3*x**2 + 2*x**3

    # Compute weights for each particle
    weights = jax.vmap(lambda r: weighting_function(r, atomistic_region.center,
                                                     atomistic_region.r_inner,
                                                     atomistic_region.r_outer))(R)

    # Atomistic energy
    E_atomistic = compute_atomistic_energy(R)

    # Coarse-grained energy
    E_cg = compute_cg_energy(R_cg)

    # Hybrid energy
    E_total = jnp.sum(weights * E_atomistic) + jnp.sum((1 - weights) * E_cg)

    return E_total
```

## Performance Tips

1. **Use neighbor lists** for systems with N > 1000 particles
2. **Compile with JIT** for 10-100x speedup
3. **Use vmap** for vectorized operations over particles
4. **Enable XLA optimizations** with `jax.config.update('jax_enable_x64', True)` for precision
5. **Profile memory** using `jax.profiler.save_device_memory_profile()` for large systems
6. **Shard across devices** with `jax.pmap` for multi-GPU scaling

## Common Pitfalls

- **Neighbor list overflow**: Set `capacity_multiplier > 1.25` for dense systems
- **Energy drift**: Reduce timestep or use better integrators (e.g., velocity Verlet)
- **Thermostat oscillations**: Increase `tau` (relaxation time) for stability
- **Precision issues**: Use `jnp.float64` for long simulations
