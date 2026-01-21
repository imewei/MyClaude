"""
Bayesian Parameter Estimation with Differentiable ODE Solvers

Demonstrates integrating Diffrax with Bayesian inference to fit
physical models to experimental data.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats
import diffrax
import blackjax


# =============================================================================
# Pattern 1: Simple ODE Parameter Estimation
# =============================================================================

def exponential_decay_model():
    """Estimate decay rate from noisy observations.

    True model: dy/dt = -k * y
    Solution: y(t) = y0 * exp(-k * t)
    """

    # Generate synthetic data
    true_k = 0.5
    true_y0 = 10.0
    true_sigma = 0.5

    rng = jax.random.PRNGKey(42)
    ts = jnp.linspace(0, 5, 20)
    true_trajectory = true_y0 * jnp.exp(-true_k * ts)
    observations = true_trajectory + jax.random.normal(rng, ts.shape) * true_sigma

    def ode_fn(t, y, args):
        k = args['k']
        return -k * y

    def solve_ode(params, ts):
        """Solve ODE and return trajectory at observation times."""
        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()

        solution = diffrax.diffeqsolve(
            term, solver,
            t0=ts[0], t1=ts[-1], dt0=0.01,
            y0=params['y0'],
            args={'k': params['k']},
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),  # For gradient
        )

        return solution.ys

    def log_prob(params):
        """Log probability: prior + likelihood."""
        k = params['k']
        y0 = params['y0']
        log_sigma = params['log_sigma']
        sigma = jnp.exp(log_sigma)

        # Priors
        log_prior = (
            stats.lognorm.logpdf(k, s=1.0, scale=1.0) +  # k > 0
            stats.norm.logpdf(y0, 10, 5) +
            stats.norm.logpdf(log_sigma, 0, 1)
        )

        # Solve ODE
        predicted = solve_ode(params, ts)

        # Likelihood
        log_lik = jnp.sum(stats.norm.logpdf(observations, predicted, sigma))

        return log_prior + log_lik

    return log_prob, ts, observations, {'true_k': true_k, 'true_y0': true_y0}


# =============================================================================
# Pattern 2: Lotka-Volterra (Predator-Prey) Model
# =============================================================================

def lotka_volterra_model():
    """Estimate predator-prey dynamics parameters.

    dx/dt = alpha * x - beta * x * y   (prey)
    dy/dt = delta * x * y - gamma * y  (predator)
    """

    # True parameters
    true_params = {
        'alpha': 1.1,   # Prey birth rate
        'beta': 0.4,    # Predation rate
        'delta': 0.1,   # Predator growth from predation
        'gamma': 0.4,   # Predator death rate
    }

    # Generate synthetic data
    def true_ode(t, state, args):
        x, y = state
        alpha, beta, delta, gamma = args['alpha'], args['beta'], args['delta'], args['gamma']
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return jnp.array([dx, dy])

    ts = jnp.linspace(0, 20, 50)
    y0 = jnp.array([10.0, 5.0])

    term = diffrax.ODETerm(true_ode)
    solver = diffrax.Tsit5()
    solution = diffrax.diffeqsolve(
        term, solver,
        t0=0, t1=20, dt0=0.01,
        y0=y0,
        args=true_params,
        saveat=diffrax.SaveAt(ts=ts),
    )

    rng = jax.random.PRNGKey(0)
    observations = solution.ys + jax.random.normal(rng, solution.ys.shape) * 0.5

    def log_prob(params):
        """Log-prob for Lotka-Volterra inference."""

        # Transform from unconstrained to constrained
        alpha = jnp.exp(params['log_alpha'])
        beta = jnp.exp(params['log_beta'])
        delta = jnp.exp(params['log_delta'])
        gamma = jnp.exp(params['log_gamma'])
        sigma = jnp.exp(params['log_sigma'])

        # Priors (log-normal for positive parameters)
        log_prior = (
            stats.norm.logpdf(params['log_alpha'], 0, 1) +
            stats.norm.logpdf(params['log_beta'], -1, 1) +
            stats.norm.logpdf(params['log_delta'], -2, 1) +
            stats.norm.logpdf(params['log_gamma'], -1, 1) +
            stats.norm.logpdf(params['log_sigma'], 0, 1)
        )

        # Solve ODE
        def ode(t, state, args):
            x, y = state
            dx = alpha * x - beta * x * y
            dy = delta * x * y - gamma * y
            return jnp.array([dx, dy])

        term = diffrax.ODETerm(ode)
        solver = diffrax.Tsit5()

        try:
            sol = diffrax.diffeqsolve(
                term, solver,
                t0=ts[0], t1=ts[-1], dt0=0.01,
                y0=y0,
                args=None,
                saveat=diffrax.SaveAt(ts=ts),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
                max_steps=10000,
            )
            predicted = sol.ys
        except Exception:
            return -jnp.inf  # Failed integration

        # Check for NaN/Inf
        if jnp.any(~jnp.isfinite(predicted)):
            return -jnp.inf

        # Likelihood
        log_lik = jnp.sum(stats.norm.logpdf(observations, predicted, sigma))

        return log_prior + log_lik

    return log_prob, ts, observations, true_params


# =============================================================================
# Pattern 3: Stiff ODE with Implicit Solver
# =============================================================================

def stiff_chemical_kinetics():
    """Chemical reaction network with stiff dynamics.

    Robertson problem: A -> B -> C with very different rates.
    """

    def robertson_ode(t, y, args):
        """Robertson chemical kinetics (stiff)."""
        k1, k2, k3 = args['k1'], args['k2'], args['k3']
        y1, y2, y3 = y

        dy1 = -k1 * y1 + k3 * y2 * y3
        dy2 = k1 * y1 - k2 * y2**2 - k3 * y2 * y3
        dy3 = k2 * y2**2

        return jnp.array([dy1, dy2, dy3])

    def log_prob(params):
        """Log-prob for stiff ODE system."""
        k1 = jnp.exp(params['log_k1'])
        k2 = jnp.exp(params['log_k2'])
        k3 = jnp.exp(params['log_k3'])
        jnp.exp(params['log_sigma'])

        # Prior
        log_prior = (
            stats.norm.logpdf(params['log_k1'], -2, 2) +
            stats.norm.logpdf(params['log_k2'], 7, 2) +
            stats.norm.logpdf(params['log_k3'], 4, 2) +
            stats.norm.logpdf(params['log_sigma'], -2, 1)
        )

        # Solve with implicit solver for stiff ODE
        term = diffrax.ODETerm(robertson_ode)
        solver = diffrax.Kvaerno5()  # Implicit solver for stiff systems

        ts = jnp.logspace(-5, 5, 50)  # Log-spaced for wide time range
        y0 = jnp.array([1.0, 0.0, 0.0])

        diffrax.diffeqsolve(
            term, solver,
            t0=ts[0], t1=ts[-1], dt0=1e-6,
            y0=y0,
            args={'k1': k1, 'k2': k2, 'k3': k3},
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=diffrax.ImplicitAdjoint(),  # Implicit adjoint for stiff
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        )

        # ... likelihood computation would go here

        return log_prior  # Simplified

    return log_prob


# =============================================================================
# Pattern 4: Neural ODE as Surrogate
# =============================================================================

def neural_ode_surrogate():
    """Use neural network as surrogate for expensive ODE simulation.

    Train on simulation outputs, then run inference on the fast surrogate.
    """
    import equinox as eqx
    import optax

    class NeuralODE(eqx.Module):
        """Neural network defining the ODE dynamics."""
        layers: list

        def __init__(self, key, hidden_dim=32):
            keys = jax.random.split(key, 3)
            self.layers = [
                eqx.nn.Linear(2, hidden_dim, key=keys[0]),
                eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
                eqx.nn.Linear(hidden_dim, 2, key=keys[2]),
            ]

        def __call__(self, t, y, args):
            for layer in self.layers[:-1]:
                y = jax.nn.tanh(layer(y))
            return self.layers[-1](y)

    def solve_neural_ode(model, y0, ts):
        """Solve the neural ODE."""
        term = diffrax.ODETerm(model)
        solver = diffrax.Tsit5()

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=ts[0], t1=ts[-1], dt0=0.01,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    def train_surrogate(model, training_data):
        """Train neural ODE on simulation data."""
        ts, trajectories = training_data  # (n_sims, n_times, dim)

        optimizer = optax.adam(1e-3)
        optimizer.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def loss_fn(model, y0, target_trajectory, ts):
            predicted = solve_neural_ode(model, y0, ts)
            return jnp.mean((predicted - target_trajectory) ** 2)

        @eqx.filter_jit
        def step(model, opt_state, batch):
            y0s, targets, ts = batch
            loss, grads = eqx.filter_value_and_grad(
                lambda m: jnp.mean(jax.vmap(loss_fn, (None, 0, 0, None))(m, y0s, targets, ts))
            )(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        # Training loop would go here
        return model

    def surrogate_log_prob(model, observations, ts):
        """Log-prob using trained neural surrogate."""

        def log_prob(params):
            y0 = params['y0']
            sigma = jnp.exp(params['log_sigma'])

            # Prior
            log_prior = (
                jnp.sum(stats.norm.logpdf(y0, 0, 5)) +
                stats.norm.logpdf(params['log_sigma'], 0, 1)
            )

            # Fast surrogate prediction
            predicted = solve_neural_ode(model, y0, ts)

            # Likelihood
            log_lik = jnp.sum(stats.norm.logpdf(observations, predicted, sigma))

            return log_prior + log_lik

        return log_prob

    return NeuralODE, train_surrogate, surrogate_log_prob


# =============================================================================
# Pattern 5: Full Bayesian Workflow
# =============================================================================

def run_bayesian_ode_inference():
    """Complete workflow: define model, run inference, analyze results."""

    print("=" * 60)
    print("Bayesian ODE Parameter Estimation")
    print("=" * 60)

    # Get model
    log_prob, ts, observations, true_params = exponential_decay_model()

    print(f"\nTrue parameters: k={true_params['true_k']}, y0={true_params['true_y0']}")
    print(f"Observations: {len(ts)} time points")

    # Pack/unpack for Blackjax
    def pack(params):
        return jnp.array([
            jnp.log(params['k']),  # log-transform for positivity
            params['y0'],
            params['log_sigma']
        ])

    def unpack(arr):
        return {
            'k': jnp.exp(arr[0]),
            'y0': arr[1],
            'log_sigma': arr[2]
        }

    def packed_log_prob(arr):
        params = unpack(arr)
        # Add Jacobian for log-transform of k
        return log_prob(params) + arr[0]  # Jacobian: d(k)/d(log_k) = k = exp(log_k)

    # Initial guess
    initial = pack({'k': 1.0, 'y0': 8.0, 'log_sigma': 0.0})

    print("\nRunning NUTS inference...")

    # Warmup
    rng_key = jax.random.PRNGKey(0)
    warmup = blackjax.window_adaptation(blackjax.nuts, packed_log_prob)
    (state, params), _ = warmup.run(rng_key, initial, num_steps=500)

    print(f"Adapted step size: {params['step_size']:.4f}")

    # Sampling
    kernel = blackjax.nuts(packed_log_prob, **params).step

    def step(state, key):
        state, info = kernel(key, state)
        return state, (state.position, info.is_divergent)

    keys = jax.random.split(rng_key, 1000)
    _, (samples, divergent) = jax.lax.scan(step, state, keys)

    print(f"Divergences: {jnp.sum(divergent)}")

    # Unpack samples
    k_samples = jnp.exp(samples[:, 0])
    y0_samples = samples[:, 1]
    sigma_samples = jnp.exp(samples[:, 2])

    print("\nPosterior Summary:")
    print(f"  k: {jnp.mean(k_samples):.3f} ± {jnp.std(k_samples):.3f} (true: {true_params['true_k']})")
    print(f"  y0: {jnp.mean(y0_samples):.3f} ± {jnp.std(y0_samples):.3f} (true: {true_params['true_y0']})")
    print(f"  sigma: {jnp.mean(sigma_samples):.3f} ± {jnp.std(sigma_samples):.3f}")

    return samples


if __name__ == "__main__":
    run_bayesian_ode_inference()
