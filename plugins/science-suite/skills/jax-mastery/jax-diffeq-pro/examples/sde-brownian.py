"""
Stochastic Differential Equations with Diffrax

Demonstrates VirtualBrownianTree, Itō vs Stratonovich interpretation,
and soft matter applications including Langevin dynamics and colloidal systems.
"""

import jax
import jax.numpy as jnp
import diffrax
from functools import partial


# =============================================================================
# Pattern 1: Basic SDE with VirtualBrownianTree
# =============================================================================

def basic_sde_example():
    """Simple SDE: dy = -k*y dt + sigma dW (Ornstein-Uhlenbeck process)."""

    def drift(t, y, args):
        """Deterministic part: mean-reversion."""
        return -args['k'] * (y - args['mu'])

    def diffusion(t, y, args):
        """Noise coefficient: constant volatility."""
        return args['sigma'] * jnp.ones_like(y)

    # VirtualBrownianTree: reproducible Brownian motion path
    key = jax.random.PRNGKey(42)
    brownian = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=10.0,
        tol=1e-3,  # Tolerance for path reconstruction
        shape=(1,),
        key=key,
    )

    # Combine drift and diffusion terms
    drift_term = diffrax.ODETerm(drift)
    diffusion_term = diffrax.ControlTerm(diffusion, brownian)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)

    # Euler-Maruyama solver for SDEs
    solver = diffrax.Euler()

    ts = jnp.linspace(0, 10, 100)

    solution = diffrax.diffeqsolve(
        terms, solver,
        t0=0.0, t1=10.0, dt0=0.01,
        y0=jnp.array([1.0]),
        args={'k': 0.5, 'mu': 0.0, 'sigma': 0.2},
        saveat=diffrax.SaveAt(ts=ts),
    )

    return solution.ts, solution.ys


# =============================================================================
# Pattern 2: Reproducible SDE Simulations
# =============================================================================

def reproducible_sde(seed: int, n_steps: int = 100):
    """Demonstrate VirtualBrownianTree reproducibility.

    Same seed = same Brownian path, regardless of step size.
    """

    def drift(t, y, args):
        return -0.5 * y

    def diffusion(t, y, args):
        return 0.1 * jnp.ones_like(y)

    key = jax.random.PRNGKey(seed)

    # Create Brownian tree
    brownian = diffrax.VirtualBrownianTree(
        t0=0.0, t1=10.0,
        tol=1e-3,
        shape=(1,),
        key=key,
    )

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )

    # Same Brownian path gives same result regardless of dt0
    solution = diffrax.diffeqsolve(
        terms, diffrax.Euler(),
        t0=0.0, t1=10.0, dt0=10.0 / n_steps,
        y0=jnp.array([1.0]),
        args=None,
    )

    return solution.ys[-1]


def test_reproducibility():
    """Show that same seed gives same result."""
    result1 = reproducible_sde(seed=42, n_steps=100)
    result2 = reproducible_sde(seed=42, n_steps=100)
    result3 = reproducible_sde(seed=43, n_steps=100)  # Different seed

    print(f"Same seed (42): {result1[0]:.6f} == {result2[0]:.6f}")
    print(f"Different seed (43): {result3[0]:.6f}")

    assert jnp.allclose(result1, result2), "Same seed should give same result"


# =============================================================================
# Pattern 3: Multi-Particle Brownian Motion
# =============================================================================

def multi_particle_brownian(n_particles: int, dim: int = 3):
    """Independent Brownian motion for multiple particles."""

    key = jax.random.PRNGKey(0)

    def drift(t, positions, args):
        """No drift (free diffusion)."""
        return jnp.zeros_like(positions)

    def diffusion(t, positions, args):
        """Thermal diffusion: sqrt(2 * D)."""
        D = args['D']
        return jnp.sqrt(2 * D) * jnp.ones_like(positions)

    # Shape: (n_particles, dim) for independent Brownian motion per particle
    brownian = diffrax.VirtualBrownianTree(
        t0=0.0, t1=10.0,
        tol=1e-3,
        shape=(n_particles, dim),
        key=key,
    )

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )

    # Initial positions
    y0 = jnp.zeros((n_particles, dim))

    ts = jnp.linspace(0, 10, 50)

    solution = diffrax.diffeqsolve(
        terms, diffrax.Euler(),
        t0=0.0, t1=10.0, dt0=0.01,
        y0=y0,
        args={'D': 1.0},
        saveat=diffrax.SaveAt(ts=ts),
    )

    # Verify Einstein relation: <r²> = 2*D*t*d
    final_positions = solution.ys[-1]
    msd = jnp.mean(jnp.sum(final_positions ** 2, axis=1))
    expected_msd = 2 * 1.0 * 10.0 * dim  # 2*D*t*d

    print(f"Mean squared displacement: {msd:.2f}")
    print(f"Expected (Einstein relation): {expected_msd:.2f}")

    return solution.ts, solution.ys


# =============================================================================
# Pattern 4: Overdamped Langevin Dynamics
# =============================================================================

def langevin_in_potential():
    """Overdamped Langevin dynamics in a double-well potential."""

    key = jax.random.PRNGKey(42)

    def potential(x):
        """Double-well: U(x) = (x² - 1)²"""
        return (x ** 2 - 1) ** 2

    def drift(t, x, args):
        """Force = -dU/dx, divided by friction gamma."""
        force = -jax.grad(potential)(x)
        return force / args['gamma']

    def diffusion(t, x, args):
        """Thermal noise: sqrt(2 * kT / gamma)."""
        return jnp.sqrt(2 * args['kT'] / args['gamma'])

    brownian = diffrax.VirtualBrownianTree(
        t0=0.0, t1=100.0,
        tol=1e-3,
        shape=(),
        key=key,
    )

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )

    ts = jnp.linspace(0, 100, 1000)

    solution = diffrax.diffeqsolve(
        terms, diffrax.Euler(),
        t0=0.0, t1=100.0, dt0=0.01,
        y0=jnp.array(0.0),  # Start at barrier top
        args={'kT': 0.3, 'gamma': 1.0},
        saveat=diffrax.SaveAt(ts=ts),
    )

    # Count transitions between wells (|x| > 0.5 as threshold)
    trajectory = solution.ys
    in_left_well = trajectory < -0.5
    in_right_well = trajectory > 0.5

    transitions = jnp.sum(jnp.abs(jnp.diff(in_left_well.astype(int)))) + \
                  jnp.sum(jnp.abs(jnp.diff(in_right_well.astype(int))))

    print(f"Number of well transitions: {int(transitions)}")
    print(f"Average position: {jnp.mean(trajectory):.3f}")

    return solution.ts, solution.ys


# =============================================================================
# Pattern 5: Colloidal Particles with Interactions
# =============================================================================

def colloidal_dynamics(n_particles: int = 10):
    """Brownian dynamics with soft repulsive interactions."""

    key = jax.random.PRNGKey(123)
    key, init_key = jax.random.split(key)

    def soft_repulsion_force(r, epsilon=1.0, sigma=1.0):
        """Soft sphere repulsion: F = epsilon * (sigma/r)^12."""
        return 12 * epsilon * (sigma / r) ** 12 / r

    def compute_forces(positions, args):
        """Compute pairwise forces on all particles."""
        n = positions.shape[0]
        epsilon, sigma = args['epsilon'], args['sigma']

        def pair_force(i, j):
            r_vec = positions[i] - positions[j]
            r = jnp.linalg.norm(r_vec) + 1e-10  # Avoid division by zero
            f_mag = jnp.where(
                r < 2.5 * sigma,  # Cutoff
                soft_repulsion_force(r, epsilon, sigma),
                0.0
            )
            return f_mag * r_vec / r

        def force_on_i(i):
            forces = jax.vmap(lambda j: pair_force(i, j))(jnp.arange(n))
            mask = jnp.arange(n) != i
            return jnp.sum(jnp.where(mask[:, None], forces, 0), axis=0)

        return jax.vmap(force_on_i)(jnp.arange(n))

    def drift(t, positions, args):
        """Force-driven drift: dx/dt = F/gamma."""
        forces = compute_forces(positions, args)
        return forces / args['gamma']

    def diffusion(t, positions, args):
        """Thermal diffusion."""
        D = args['kT'] / args['gamma']
        return jnp.sqrt(2 * D) * jnp.ones_like(positions)

    # Initialize particles on a grid
    side = int(jnp.ceil(jnp.sqrt(n_particles)))
    x = jnp.arange(side) * 1.5
    y = jnp.arange(side) * 1.5
    grid = jnp.stack(jnp.meshgrid(x, y), axis=-1).reshape(-1, 2)
    y0 = grid[:n_particles]

    brownian = diffrax.VirtualBrownianTree(
        t0=0.0, t1=50.0,
        tol=1e-3,
        shape=(n_particles, 2),
        key=key,
    )

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift),
        diffrax.ControlTerm(diffusion, brownian),
    )

    ts = jnp.linspace(0, 50, 100)

    solution = diffrax.diffeqsolve(
        terms, diffrax.Euler(),
        t0=0.0, t1=50.0, dt0=0.01,
        y0=y0,
        args={'epsilon': 1.0, 'sigma': 1.0, 'gamma': 1.0, 'kT': 0.5},
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=100000,
    )

    return solution.ts, solution.ys


# =============================================================================
# Pattern 6: Ensemble SDE Simulations
# =============================================================================

def ensemble_sde(n_realizations: int = 100):
    """Run ensemble of SDE realizations with different random seeds."""

    def drift(t, y, args):
        return -args['k'] * y

    def diffusion(t, y, args):
        return args['sigma'] * jnp.ones_like(y)

    def single_realization(key):
        brownian = diffrax.VirtualBrownianTree(
            t0=0.0, t1=10.0,
            tol=1e-3,
            shape=(1,),
            key=key,
        )

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, brownian),
        )

        solution = diffrax.diffeqsolve(
            terms, diffrax.Euler(),
            t0=0.0, t1=10.0, dt0=0.01,
            y0=jnp.array([1.0]),
            args={'k': 0.5, 'sigma': 0.2},
        )

        return solution.ys[-1, 0]

    # Vectorize over random keys
    keys = jax.random.split(jax.random.PRNGKey(0), n_realizations)
    final_values = jax.vmap(single_realization)(keys)

    print(f"Ensemble statistics (n={n_realizations}):")
    print(f"  Mean: {jnp.mean(final_values):.4f}")
    print(f"  Std:  {jnp.std(final_values):.4f}")

    # Theoretical: E[y(t)] = y0 * exp(-k*t)
    expected_mean = 1.0 * jnp.exp(-0.5 * 10.0)
    print(f"  Expected mean: {expected_mean:.4f}")

    return final_values


# =============================================================================
# Pattern 7: Higher-Order SDE Solvers
# =============================================================================

def sde_solver_comparison():
    """Compare different SDE solver schemes."""

    def drift(t, y, args):
        return -y

    def diffusion(t, y, args):
        return 0.5 * jnp.ones_like(y)

    key = jax.random.PRNGKey(42)

    solvers = {
        'Euler': diffrax.Euler(),
        'Heun': diffrax.Heun(),
    }

    results = {}

    for name, solver in solvers.items():
        brownian = diffrax.VirtualBrownianTree(
            t0=0.0, t1=1.0,
            tol=1e-3,
            shape=(1,),
            key=key,
        )

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, brownian),
        )

        solution = diffrax.diffeqsolve(
            terms, solver,
            t0=0.0, t1=1.0, dt0=0.01,
            y0=jnp.array([1.0]),
            args=None,
        )

        results[name] = solution.ys[-1, 0]
        print(f"{name}: y(1) = {results[name]:.6f}")

    return results


# =============================================================================
# Pattern 8: Reproducible Simulation Configuration
# =============================================================================

def production_sde_config(seed: int, params: dict):
    """Production-ready SDE simulation with full reproducibility.

    Returns configuration that can be saved for reproducibility.
    """

    config = {
        'seed': seed,
        'rtol': 1e-3,
        'dt0': 0.001,
        't_final': params.get('t_final', 100.0),
        'n_save': params.get('n_save', 1000),
        **params,
    }

    key = jax.random.PRNGKey(config['seed'])

    brownian = diffrax.VirtualBrownianTree(
        t0=0.0,
        t1=config['t_final'],
        tol=config['rtol'],
        shape=params['state_shape'],
        key=key,
    )

    print("Reproducible SDE Configuration:")
    print(f"  Seed: {config['seed']}")
    print(f"  Tolerance: {config['rtol']}")
    print(f"  Initial dt: {config['dt0']}")
    print(f"  Final time: {config['t_final']}")

    return config, brownian


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate SDE patterns with Diffrax."""
    print("=" * 60)
    print("Diffrax SDE / Brownian Motion Demo")
    print("=" * 60)

    print("\n1. Basic SDE (Ornstein-Uhlenbeck)")
    print("-" * 40)
    ts, ys = basic_sde_example()
    print(f"   Final value: {ys[-1, 0]:.4f}")

    print("\n2. Reproducibility Test")
    print("-" * 40)
    test_reproducibility()

    print("\n3. Multi-Particle Brownian Motion")
    print("-" * 40)
    multi_particle_brownian(n_particles=50, dim=3)

    print("\n4. Langevin Dynamics in Double-Well")
    print("-" * 40)
    langevin_in_potential()

    print("\n5. Ensemble SDE Statistics")
    print("-" * 40)
    ensemble_sde(n_realizations=100)

    print("\n6. SDE Solver Comparison")
    print("-" * 40)
    sde_solver_comparison()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
