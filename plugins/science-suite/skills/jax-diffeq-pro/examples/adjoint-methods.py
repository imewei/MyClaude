"""
Adjoint Methods for Gradient Propagation in Diffrax

Demonstrates memory-efficient backpropagation through ODE solvers
using checkpointing and adjoint methods.
"""

import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import optax


# =============================================================================
# Pattern 1: RecursiveCheckpointAdjoint (Memory-Efficient)
# =============================================================================

def checkpoint_adjoint_example():
    """Use checkpointing to trade compute for memory.

    This is the recommended approach for most cases.
    """

    def vector_field(t, y, args):
        """Lotka-Volterra dynamics."""
        x, pred = y
        alpha, beta, delta, gamma = args['alpha'], args['beta'], args['delta'], args['gamma']

        dx = alpha * x - beta * x * pred
        dpred = delta * x * pred - gamma * pred

        return jnp.array([dx, dpred])

    def simulate(params, y0, ts):
        """Simulate with checkpointing for memory-efficient gradients."""
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()

        # RecursiveCheckpointAdjoint: memory-efficient backprop
        # checkpoints=None lets Diffrax choose optimal number
        adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=None)

        solution = diffrax.diffeqsolve(
            term, solver,
            t0=ts[0], t1=ts[-1], dt0=0.01,
            y0=y0,
            args=params,
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=adjoint,
        )

        return solution.ys

    # Compute gradient with respect to parameters
    def loss(params, y0, ts, target):
        predicted = simulate(params, y0, ts)
        return jnp.mean((predicted - target) ** 2)

    params = {'alpha': 1.0, 'beta': 0.4, 'delta': 0.1, 'gamma': 0.4}
    y0 = jnp.array([10.0, 5.0])
    ts = jnp.linspace(0, 20, 100)

    # Create synthetic target
    target = simulate(params, y0, ts)

    # Compute gradient
    grad_params = jax.grad(loss)(params, y0, ts, target)

    print("Gradient of loss with respect to parameters:")
    for k, v in grad_params.items():
        print(f"  d(loss)/d({k}) = {v:.6f}")

    return grad_params


# =============================================================================
# Pattern 2: Controlling Checkpoint Count
# =============================================================================

def checkpoint_memory_tradeoff():
    """Demonstrate memory/compute tradeoff with checkpoint count."""

    def vector_field(t, y, args):
        return -args['k'] * y

    def simulate(y0, k, num_checkpoints=None):
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()

        # More checkpoints = less recomputation but more memory
        adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=num_checkpoints)

        solution = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=100.0, dt0=0.1,  # Long simulation
            y0=y0,
            args={'k': k},
            adjoint=adjoint,
        )

        return solution.ys[-1]

    y0 = jnp.array([1.0])

    # Compare different checkpoint counts
    checkpoint_configs = [4, 8, 16, 32, None]  # None = auto

    for n in checkpoint_configs:
        grad_fn = jax.grad(lambda k: jnp.sum(simulate(y0, k, n)))
        grad = grad_fn(0.5)
        print(f"Checkpoints={n}: gradient = {grad:.6f}")


# =============================================================================
# Pattern 3: BacksolveAdjoint (Continuous Adjoint)
# =============================================================================

def backsolve_adjoint_example():
    """Use continuous adjoint ODE for gradients.

    Warning: Only stable for non-chaotic systems!
    """

    def vector_field(t, y, args):
        """Simple damped oscillator (stable, non-chaotic)."""
        x, v = y
        k, gamma = args['k'], args['gamma']
        return jnp.array([v, -k * x - gamma * v])

    def simulate(params, y0, t1):
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()

        # BacksolveAdjoint: solves adjoint ODE backwards
        # Memory-efficient but can be unstable for chaotic systems
        adjoint = diffrax.BacksolveAdjoint(
            solver=diffrax.Tsit5()  # Can use different solver for adjoint
        )

        solution = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=t1, dt0=0.01,
            y0=y0,
            args=params,
            adjoint=adjoint,
        )

        return solution.ys[-1]

    def loss(params, y0):
        final_state = simulate(params, y0, 10.0)
        return jnp.sum(final_state ** 2)

    params = {'k': 1.0, 'gamma': 0.1}
    y0 = jnp.array([1.0, 0.0])

    grad_params = jax.grad(loss)(params, y0)

    print("BacksolveAdjoint gradients:")
    for k, v in grad_params.items():
        print(f"  d(loss)/d({k}) = {v:.6f}")

    return grad_params


# =============================================================================
# Pattern 4: DirectAdjoint (Exact Gradients)
# =============================================================================

def direct_adjoint_example():
    """Use direct adjoint for short simulations requiring exact gradients.

    Stores full trajectory - use only for short simulations!
    """

    def vector_field(t, y, args):
        return -args['k'] * y

    def simulate(k, y0):
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()

        # DirectAdjoint: stores full trajectory
        # Most accurate but most memory-intensive
        adjoint = diffrax.DirectAdjoint()

        solution = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=0.01,  # Short simulation only!
            y0=y0,
            args={'k': k},
            adjoint=adjoint,
        )

        return solution.ys[-1]

    y0 = jnp.array([1.0])

    grad_fn = jax.grad(lambda k: jnp.sum(simulate(k, y0)))
    grad = grad_fn(0.5)

    print(f"DirectAdjoint gradient: {grad:.6f}")
    return grad


# =============================================================================
# Pattern 5: Neural ODE Training Loop
# =============================================================================

class NeuralODE(eqx.Module):
    """Neural network defining ODE dynamics."""
    layers: list

    def __init__(self, hidden_dim: int, state_dim: int, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(state_dim + 1, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, state_dim, key=keys[2]),
        ]

    def __call__(self, t, y, args):
        x = jnp.concatenate([jnp.array([t]), y])
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)


def train_neural_ode():
    """Train Neural ODE with memory-efficient gradients."""

    key = jax.random.PRNGKey(42)
    key, model_key, data_key = jax.random.split(key, 3)

    # Create model
    model = NeuralODE(hidden_dim=32, state_dim=2, key=model_key)

    # Generate training data (spiral trajectory)
    def true_dynamics(t, y, args):
        A = jnp.array([[0.1, 1.0], [-1.0, 0.1]])
        return A @ y

    ts = jnp.linspace(0, 5, 50)
    y0 = jnp.array([2.0, 0.0])

    true_sol = diffrax.diffeqsolve(
        diffrax.ODETerm(true_dynamics),
        diffrax.Tsit5(),
        t0=0.0, t1=5.0, dt0=0.01,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
    )
    target = true_sol.ys

    def solve(model, y0, ts):
        term = diffrax.ODETerm(model)
        solver = diffrax.Tsit5()

        # Use checkpointing for memory efficiency during training
        adjoint = diffrax.RecursiveCheckpointAdjoint()

        solution = diffrax.diffeqsolve(
            term, solver,
            t0=ts[0], t1=ts[-1], dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=adjoint,
        )

        return solution.ys

    @eqx.filter_jit
    def loss_fn(model, y0, ts, target):
        predicted = solve(model, y0, ts)
        return jnp.mean((predicted - target) ** 2)

    @eqx.filter_jit
    def step(model, opt_state, y0, ts, target, optimizer):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, y0, ts, target)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training loop
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print("Training Neural ODE...")
    for epoch in range(100):
        model, opt_state, loss = step(model, opt_state, y0, ts, target, optimizer)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}")

    return model


# =============================================================================
# Pattern 6: Implicit Adjoint for Stiff Systems
# =============================================================================

def implicit_adjoint_example():
    """Use implicit adjoint for stiff ODE systems."""

    def stiff_vector_field(t, y, args):
        """Stiff chemical kinetics."""
        k1, k2, k3 = args['k1'], args['k2'], args['k3']
        y1, y2, y3 = y

        dy1 = -k1 * y1 + k3 * y2 * y3
        dy2 = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
        dy3 = k2 * y2**2

        return jnp.array([dy1, dy2, dy3])

    def simulate(params, y0):
        term = diffrax.ODETerm(stiff_vector_field)
        solver = diffrax.Kvaerno5()  # Implicit solver for stiff ODE

        # ImplicitAdjoint: use implicit solver for adjoint too
        adjoint = diffrax.ImplicitAdjoint()

        stepsize_controller = diffrax.PIDController(
            rtol=1e-6,
            atol=1e-8,
        )

        solution = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=100.0, dt0=0.001,
            y0=y0,
            args=params,
            adjoint=adjoint,
            stepsize_controller=stepsize_controller,
        )

        return solution.ys[-1]

    def loss(params, y0):
        final = simulate(params, y0)
        return jnp.sum(final ** 2)

    params = {'k1': 0.04, 'k2': 3e7, 'k3': 1e4}
    y0 = jnp.array([1.0, 0.0, 0.0])

    grad_params = jax.grad(loss)(params, y0)

    print("ImplicitAdjoint gradients (stiff system):")
    for k, v in grad_params.items():
        print(f"  d(loss)/d({k}) = {v:.6e}")

    return grad_params


# =============================================================================
# Pattern 7: Adjoint Method Selection Guide
# =============================================================================

def get_adjoint_method(
    simulation_length: str = "medium",
    system_type: str = "non-stiff",
    accuracy_requirement: str = "standard"
):
    """Select appropriate adjoint method based on problem characteristics.

    Args:
        simulation_length: "short" (<10 steps), "medium", "long" (>1000 steps)
        system_type: "non-stiff", "stiff", or "chaotic"
        accuracy_requirement: "standard" or "high"
    """

    # Short simulations: DirectAdjoint is fine
    if simulation_length == "short":
        return diffrax.DirectAdjoint()

    # Chaotic systems: must use checkpointing
    if system_type == "chaotic":
        return diffrax.RecursiveCheckpointAdjoint()

    # Stiff systems: use implicit adjoint
    if system_type == "stiff":
        return diffrax.ImplicitAdjoint()

    # Long, non-chaotic, non-stiff: can try BacksolveAdjoint
    if simulation_length == "long" and accuracy_requirement == "standard":
        return diffrax.BacksolveAdjoint()

    # Default: checkpointing (safe and memory-efficient)
    return diffrax.RecursiveCheckpointAdjoint()


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate adjoint methods for ODE gradients."""
    print("=" * 60)
    print("Diffrax Adjoint Methods Demo")
    print("=" * 60)

    print("\n1. RecursiveCheckpointAdjoint (Recommended)")
    print("-" * 40)
    checkpoint_adjoint_example()

    print("\n2. DirectAdjoint (Short Simulations)")
    print("-" * 40)
    direct_adjoint_example()

    print("\n3. BacksolveAdjoint (Stable Systems)")
    print("-" * 40)
    backsolve_adjoint_example()

    print("\n4. Neural ODE Training")
    print("-" * 40)
    train_neural_ode()

    print("\n5. ImplicitAdjoint (Stiff Systems)")
    print("-" * 40)
    implicit_adjoint_example()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
