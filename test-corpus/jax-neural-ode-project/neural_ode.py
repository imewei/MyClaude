
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

def neural_ode_dynamics(state, t, params):
    """Neural ODE dynamics function."""
    x = state
    for W, b in params:
        x = jnp.tanh(x @ W + b)
    return x

def solve_ode(params, initial_state, t_span):
    """Solve ODE using Euler method."""
    dt = 0.01
    trajectory = [initial_state]
    state = initial_state

    for t in jnp.arange(t_span[0], t_span[1], dt):
        dx = neural_ode_dynamics(state, t, params)
        state = state + dx * dt
        trajectory.append(state)

    return jnp.array(trajectory)
