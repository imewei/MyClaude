# JAX-first code architecture rules

Every skill that emits Python code (theory-scaffold for symbolic work, numerical-prototype for simulations, experiment-designer for analysis scripts) must follow these conventions. Load this file before writing code.

## Core rules

1. **No Python loops in the physics core.** The inner computational kernel must be pure JAX. Use `jax.vmap` for batching over particles, samples, or trajectories. Use `jax.lax.scan` for time integration when state must carry across steps.

2. **`jax.jit` the integration step.** The unit of work passed to `jit` should be one timestep or one Monte Carlo move, not the entire simulation. This keeps compile times reasonable and makes partial results inspectable.

3. **Functional style.** All physics functions are pure: they take state and parameters in, return new state out. No mutation, no global state, no hidden caches. Side effects (logging, checkpointing) live in the outer Python layer.

4. **PRNGkey discipline.** Every stochastic function takes a `key` as explicit input and splits it before use. Never reuse a key. Never rely on an implicit global RNG.

5. **Explicit dtype policy.** Decide at the top of each module whether it is `float32` or `float64`. Document the choice and the reason. Mixed-precision work requires an explicit policy document.

6. **Static vs. dynamic shapes.** Shapes that affect control flow must be `static_argnums` on the jitted function. Shapes that vary at runtime must be padded to a fixed maximum and masked.

## Patterns to prefer

**Batched particle update:**
```python
@jax.jit
def step(state: State, key: jax.Array, params: Params) -> State:
    # single-particle update
    def update_one(s, k):
        ...
        return new_s
    keys = jax.random.split(key, state.N)
    new_positions = jax.vmap(update_one)(state.positions, keys)
    return state.replace(positions=new_positions)
```

**Time integration with scan:**
```python
def integrate(initial_state, keys, params, n_steps):
    def body(state, key):
        new_state = step(state, key, params)
        return new_state, new_state  # carry, output
    final, trajectory = jax.lax.scan(body, initial_state, keys)
    return trajectory
```

## Patterns to avoid

- Python `for` loops over particles inside a jitted function (use `vmap`)
- `np.random` alongside `jax.random` (pick one, usually jax)
- `.item()` or `.tolist()` calls inside the hot path (forces host transfer)
- Using `jax.device_put` everywhere; place data once at the boundary
- NamedTuples for state when the state has many fields (use a registered pytree dataclass)

## Testing implication

Every JAX module has a matching pytest file that includes:
- A property test for invariants (conservation of mass/momentum/energy, symmetry)
- A limit-recovery test against an analytic case
- A shape-and-dtype regression test

See `testing_conventions.md` for structure.

## Why these rules

The rules reflect the performance and correctness patterns already adopted in `homodyne`, `heterodyne`, and `RheoJax`. Deviations fragment the ecosystem and make code from the research-spark stack harder to integrate into existing pipelines. When a rule feels wrong for a specific case, document the exception in a module-level comment rather than silently breaking the pattern.
