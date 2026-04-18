# Testing conventions

Every code module emitted by the research-spark stack has a matching test file. The test file structure is fixed so that the reconciliation scripts and the Stage 6 validation passes can locate tests automatically.

## Test file layout

```
src/mymod/core.py
src/mymod/observables.py
tests/test_core.py
tests/test_observables.py
tests/test_invariants.py       # shared property-based tests
tests/test_limits.py            # analytic-limit recovery tests
tests/benchmarks/               # synthetic-benchmark cases
    bench_ornstein_uhlenbeck.py
    bench_rouse_chain.py
```

## Mandatory tests for physics code

Every physics module must have:

1. **Property test for invariants.** Use `hypothesis` to generate inputs and assert conservation laws (mass, momentum, energy as applicable), symmetries (rotation, translation, parity as applicable), and units.

2. **Limit-recovery test.** Set the new-physics parameter to zero or its reference value; run the solver; assert that output matches the analytic reference within documented tolerance.

3. **Shape-and-dtype regression test.** Fast to run. Just asserts that the outputs have the expected shapes and dtypes. Catches accidental broadcasting bugs.

4. **Convergence test.** A short run at two grid resolutions; assert that the finer grid result falls within expected error bound of the Richardson extrapolate.

## Template for a test module

```python
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, strategies as st
from mymod.core import step, integrate


@pytest.fixture
def default_params():
    return {...}


# 1. Property / invariant test
@given(st.integers(min_value=10, max_value=1000))
def test_mass_conservation(n_particles):
    key = jax.random.PRNGKey(0)
    state = initial_state(n_particles, key)
    params = default_params_dict()
    evolved = integrate(state, jax.random.split(key, 100), params, 100)
    total_mass_initial = jnp.sum(state.mass)
    total_mass_final = jnp.sum(evolved[-1].mass)
    assert jnp.allclose(total_mass_initial, total_mass_final, rtol=1e-10)


# 2. Limit recovery
def test_recovers_free_diffusion(default_params):
    # Turn off the new physics
    params = {**default_params, "interaction_strength": 0.0}
    result = run(params)
    analytic = free_diffusion_reference(params)
    assert jnp.allclose(result, analytic, rtol=1e-3)


# 3. Shape and dtype regression
def test_output_shapes(default_params):
    result = run(default_params)
    assert result.shape == (100, 10, 3)
    assert result.dtype == jnp.float64


# 4. Convergence
def test_convergence(default_params):
    coarse = run({**default_params, "dt": 1e-2})
    fine = run({**default_params, "dt": 1e-3})
    assert jnp.allclose(coarse, fine, rtol=1e-2)
```

## What not to test

- Random outputs without seeding (flaky tests)
- Exact floating-point equality for stochastic simulations (use `rtol`/`atol` with documented tolerance)
- Wall-clock performance (put those in `benchmarks/`, not `tests/`, and mark with `@pytest.mark.benchmark`)

## How this integrates with Stage 6

The `numerical-prototype` skill runs these tests as its validation pass. If the property tests or limit-recovery tests fail, the prototype does not pass validation and the Stage 6 artifact is flagged incomplete.
