"""
prototype_skeleton.py

Starting scaffold for Stage 6 numerical prototypes. Copy into src/<package>/core.py
and fill in the physics for your specific formalism.

Conventions (see _research-commons/code_architecture/jax_first_rules.md):
- State and Params are pytree-registered dataclasses
- `step` is pure, jit-able, takes (state, key, params) -> new_state
- No Python loops in the physics core; vmap for particle batching, scan for time
- PRNGKey discipline: every stochastic function takes `key` explicitly
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


# --- Dtype policy ------------------------------------------------------------
# Pick one and document the reason. Mixed-precision requires an explicit policy
# in a separate module-level comment.
DTYPE = jnp.float64
# Reason: the observable is a small number (O(1e-3)) accumulated over many steps,
# so float32 accumulated error exceeds the claimed precision of the prediction.


# --- Params -----------------------------------------------------------------
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Params:
    """Physical parameters. Do not vary during integration."""
    diffusion: float = 1.0
    interaction_strength: float = 0.0  # the "new physics" term; zero recovers known limit
    dt: float = 1e-3

    @classmethod
    def default(cls) -> "Params":
        return cls()


# --- State ------------------------------------------------------------------
@jax.tree_util.register_dataclass
@dataclass
class State:
    """Simulation state. Dtype-consistent per module policy."""
    positions: Array        # shape (N, D)
    velocities: Array       # shape (N, D)
    t: Array                # scalar

    @classmethod
    def initial(cls, n_particles: int, dim: int, key: Array) -> "State":
        k1, k2 = jax.random.split(key)
        positions = jax.random.normal(k1, (n_particles, dim), dtype=DTYPE)
        velocities = jax.random.normal(k2, (n_particles, dim), dtype=DTYPE)
        return cls(positions=positions, velocities=velocities, t=jnp.asarray(0.0, dtype=DTYPE))


# --- Physics core -----------------------------------------------------------
def _single_particle_update(pos: Array, vel: Array, key: Array, params: Params) -> tuple[Array, Array]:
    """
    Per-particle update. Called via vmap from step().
    Fill in the physics from Eq. (X) of the formalism here.
    """
    # Placeholder: overdamped Brownian dynamics with an interaction stub.
    # Replace with the physics from 05_formalism.tex.
    noise = jax.random.normal(key, pos.shape, dtype=DTYPE)
    drift = -params.interaction_strength * pos  # stub interaction
    new_pos = pos + drift * params.dt + jnp.sqrt(2 * params.diffusion * params.dt) * noise
    new_vel = vel  # unused in overdamped limit; keep for state consistency
    return new_pos, new_vel


@jax.jit
def step(state: State, key: Array, params: Params) -> State:
    """
    One integration step. Pure, jit-compiled. No Python loops.
    """
    n_particles = state.positions.shape[0]
    keys = jax.random.split(key, n_particles)
    new_pos, new_vel = jax.vmap(_single_particle_update, in_axes=(0, 0, 0, None))(
        state.positions, state.velocities, keys, params
    )
    return State(
        positions=new_pos,
        velocities=new_vel,
        t=state.t + params.dt,
    )


# --- Time integration -------------------------------------------------------
def integrate(
    initial_state: State,
    key: Array,
    params: Params,
    n_steps: int,
) -> State:
    """
    Roll out the integrator for n_steps. Returns the full trajectory stacked on
    the first axis.
    """
    keys = jax.random.split(key, n_steps)

    def body(state, step_key):
        new_state = step(state, step_key, params)
        return new_state, new_state

    _, trajectory = jax.lax.scan(body, initial_state, keys)
    return trajectory


# --- Observable extractor ---------------------------------------------------
def extract_observable(trajectory: State, params: Params) -> dict:
    """
    Convert the trajectory into the predicted observable that Stage 7 will
    design a measurement for. Replace this with the problem-specific observable
    from 04_theory.md.

    Returned dict follows templates/predicted_observable.md.
    """
    # Placeholder: mean-square displacement vs time.
    msd = jnp.mean(jnp.sum(trajectory.positions ** 2, axis=-1), axis=-1)
    t = trajectory.t
    return {
        "name": "mean_square_displacement",
        "t": t,
        "values": msd,
        "units": {"t": "time", "values": "length^2"},
    }


# --- Minimal self-test ------------------------------------------------------
def _self_test() -> None:
    """Quick sanity check that the skeleton runs."""
    key = jax.random.PRNGKey(0)
    params = Params.default()
    initial = State.initial(n_particles=100, dim=3, key=key)
    trajectory = integrate(initial, jax.random.PRNGKey(1), params, n_steps=1000)
    obs = extract_observable(trajectory, params)
    print(f"trajectory shape: {trajectory.positions.shape}")
    print(f"observable '{obs['name']}' final value: {obs['values'][-1]:.4f}")


if __name__ == "__main__":
    _self_test()
