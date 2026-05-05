"""Fit Lorenz-system parameters by JAX-differentiable ODE rollout."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
import optax


def lorenz_rhs(state: jnp.ndarray, sigma: float, rho: float, beta: float) -> jnp.ndarray:
    x, y, z = state
    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def rk4_step(state, params, dt):
    sigma, rho, beta = params
    k1 = lorenz_rhs(state, sigma, rho, beta)
    k2 = lorenz_rhs(state + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_rhs(state + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_rhs(state + dt * k3, sigma, rho, beta)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@partial(jit, static_argnums=(2,))
def rollout(initial_state, params, n_steps, dt=0.01):
    def body(state, _):
        return rk4_step(state, params, dt), state
    _, trajectory = jax.lax.scan(body, initial_state, None, length=n_steps)
    return trajectory


def loss_fn(params, initial_state, observed):
    trajectory = rollout(initial_state, params, observed.shape[0])
    return jnp.mean((trajectory - observed) ** 2)


@jit
def update_step(params, opt_state, initial_state, observed, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, initial_state, observed)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def fit(initial_state, observed, n_steps=2000, lr=1e-2):
    params = jnp.array([8.0, 25.0, 2.0])
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    for step in range(n_steps):
        params, opt_state, loss = update_step(
            params, opt_state, initial_state, observed, optimizer
        )
        if step % 100 == 0:
            print(f"step={step} loss={loss:.6f} params={params}")
    return params
