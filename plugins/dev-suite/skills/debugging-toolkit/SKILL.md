---
name: debugging-toolkit
description: Scientific computing debugging patterns — JAX TracerBoolConversionError/ConcretizationTypeError, Julia @code_warntype type-stability, NaN/inf hunting, numerical reproducibility isolation, and memory profiling for JAX/Julia. For the structured pre-fix debugging workflow, use superpowers:systematic-debugging. Provides domain knowledge for /smart-debug.
---

# Debugging Toolkit

> **SEE ALSO:** For a structured pre-fix debugging workflow (hypothesize before changing anything), use `superpowers:systematic-debugging`. For interactive scientific debugging sessions, use `dev-suite:smart-debug`.

## Expert Agents

| Failure Type | Agent |
|---|---|
| NaN/inf, JAX JIT errors, Julia dispatch, MCMC divergence | `debugger-pro` + `dev-suite:smart-debug` |
| Distributed system, microservice, K8s, production incident | `debugger-pro` |

---

## JAX Debugging

### Common JIT Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `TracerBoolConversionError` | Python bool inside `jit` | `jax.lax.cond` / `jnp.where` |
| `ConcretizationTypeError` | Dynamic index inside `jit` | `jnp.take` or static_argnums |
| `UnexpectedTracerError` | Tracer leaked out of `jit` | Check closures, use `jax.closure_convert` |
| Silent NaN gradient | Exploding activations | Add `jax.debug.print` checkpoints |
| JIT recompilation | Dynamic shapes per call | Static shapes or `static_argnums` |

```python
# Instrument forward pass for NaN detection
from jax import debug
def model(x):
    debug.print("x: min={a} max={b} has_nan={c}",
                a=x.min(), b=x.max(), c=jnp.isnan(x).any())
    return ...

# Disable JIT to get Python-level stack traces
import os; os.environ["JAX_DISABLE_JIT"] = "1"

# Gradient check: analytical vs finite difference
from jax.test_util import check_grads
check_grads(f, (x,), order=2, rtol=1e-3)
```

---

## Julia Debugging

```julia
# Type stability — look for Any / Union in return type
@code_warntype f(x)

# Allocation profiling
@allocated f(x)       # bytes per call; 0 = allocation-free
@btime f($x)          # time + allocs via BenchmarkTools

# Interactive debugger
using Debugger; @enter f(x)

# Catch type instability in test suite
@inferred f(x)        # fails if return type is not inferred
@test_nowarn f(x)     # fails if dispatch warning emitted
```

---

## NaN / Inf Hunting

| Strategy | How |
|----------|-----|
| Binary search | Disable ops one by one; print intermediate tensors |
| Log barrier | Clip before `log`: `jnp.clip(x, 1e-7, None)` |
| Large LR | Reduce LR first — NaN usually appears in step 1-3 |
| Float overflow | `jnp.finfo(jnp.float32).max ≈ 3.4e38`; use float64 to confirm |
| Gradient explosion | Clip gradients: `optax.clip_by_global_norm(1.0)` |

---

## Numerical Reproducibility

```python
# JAX: explicit seeds, always split
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)   # never reuse a key

# Verify reproducibility
import numpy as np
result_a = run_experiment(seed=0)
result_b = run_experiment(seed=0)
np.testing.assert_allclose(result_a, result_b, rtol=1e-5)
```

---

## Universal Strategies

### Systematic Process

| Phase | Action |
|-------|--------|
| Reproduce | Minimal repro, fixed seed, exact environment |
| Isolate | Binary search — remove components until bug disappears |
| Hypothesize | What changed? Where does output first diverge? |
| Fix | Root cause, not symptoms |

### Python Profiling

```python
breakpoint()                                          # drop into pdb
import cProfile; cProfile.run('fn()', 'stats')
import pstats; pstats.Stats('stats').sort_stats('cumulative').print_stats(10)
```

### Git Bisect

```bash
git bisect start && git bisect bad && git bisect good v1.0.0
# test, then: git bisect good|bad  — repeat until done
git bisect reset
```

### Parallelism Checklist

- [ ] Fix all random seeds before comparing runs
- [ ] Reproduce with single-thread mode to isolate concurrency bugs
- [ ] Check for race conditions with `pytest-repeat` + `pytest-xdist`
- [ ] In JAX: `jax.effects_barrier()` before timing measurements
- [ ] In Julia: avoid `SequentialMonteCarlo` (threading race on Julia 1.12)

### Quick Checklist

- [ ] Reproduce with minimal example and fixed seed
- [ ] Print intermediate tensor values / shapes
- [ ] Verify input dtypes (float32 vs float64)
- [ ] Test in float64 mode if float32 suspected (`jax.config.update("jax_enable_x64", True)`)
- [ ] Check recent changes (git bisect)
- [ ] Profile before optimizing
