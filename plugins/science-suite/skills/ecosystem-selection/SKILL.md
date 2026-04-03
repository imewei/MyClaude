---
name: ecosystem-selection
description: Select optimal scientific computing ecosystems and manage multi-language workflows. Use when evaluating Python vs Julia for performance-critical numerical computing, implementing hybrid PyJulia/PyCall.jl interoperability, or setting up reproducible toolchains with Conda or Pkg.jl.
---

# Python/Julia Ecosystem Selection

## Expert Agent

For ecosystem evaluation and multi-language workflow selection, delegate to:

- **`julia-pro`**: Julia and Python ecosystem comparison, interoperability, and toolchain selection.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

## Comparison

| Aspect | Python | Julia |
|--------|--------|-------|
| Performance | 10-100x slower loops | C-like speed (JIT) |
| ML/DL | Excellent (PyTorch, TF) | Limited |
| ODEs/PDEs | SciPy (basic) | SciML (10-100x faster) |
| Community | Massive | Growing |

## When to Choose

| Python | Julia | Hybrid |
|--------|-------|--------|
| ML/DL integration | Performance-critical sims | ML + numerics |
| Rapid prototyping | Complex ODE/PDE | Gradual migration |
| Existing codebase | From-scratch scientific | Mixed team |

## Hybrid Integration

### Python → Julia (PyJulia)

```python
from julia import Julia, Main
jl = Julia(compiled_modules=False)
Main.eval('using DifferentialEquations')
Main.eval('fast_ode_solve(u0, tspan) = solve(ODEProblem((u,p,t)->-u, u0, tspan), Tsit5())')
sol = Main.fast_ode_solve([1.0], (0.0, 10.0))  # 10-100x faster
```

### Julia → Python (PyCall.jl)

```julia
using PyCall
np = pyimport("numpy")
sklearn = pyimport("sklearn.linear_model")
model = sklearn.LinearRegression().fit(X_train, y_train)
```

## Toolchain

| Tool | Python | Julia |
|------|--------|-------|
| Environment | `conda`/`venv` | `Pkg.activate()` |
| Dependencies | `requirements.txt` | `Project.toml` |
| Lock | `poetry.lock` | `Manifest.toml` |

## Performance

| Language | Technique |
|----------|-----------|
| Python | Numba JIT, vectorize |
| Julia | Type stability, in-place ops |

## Parallelization Capabilities

| Feature | Python | Julia |
|---------|--------|-------|
| **Multi-threading** | GIL limitations (CPython) | Native composable threading |
| **Multi-processing** | `multiprocessing` (Heavy) | `Distributed` (Lightweight) |
| **GPU** | CuPy / PyTorch (Excellent) | CUDA.jl / KernelAbstractions.jl (Native) |
| **Distributed** | Dask / Ray / MPI4Py | Distributed.jl / MPI.jl |

## Migration Strategy

1. Profile Python → identify bottlenecks
2. Prototype critical functions in Julia
3. Benchmark (aim >10x speedup)
4. Interface via PyJulia
5. Iterate gradual migration

**Outcome**: Optimal ecosystem choice, hybrid integration, reproducible environments

## Checklist

- [ ] Profile Python bottlenecks before deciding to migrate to Julia
- [ ] Verify Julia prototype achieves >10x speedup over Python baseline on the target workload
- [ ] Confirm PyJulia bridge uses `compiled_modules=False` to avoid precompilation issues
- [ ] Ensure Python and Julia environments are locked (`uv.lock` / `Manifest.toml`) for reproducibility
- [ ] Check dtype consistency at language boundaries (Python float32 vs Julia Float64)
- [ ] Validate that hybrid workflows produce bit-identical results to single-language reference
- [ ] Ensure GIL-sensitive operations are offloaded to Julia or use multiprocessing
- [ ] Confirm package manager isolation (no mixing pip and CondaPkg in the same environment)
