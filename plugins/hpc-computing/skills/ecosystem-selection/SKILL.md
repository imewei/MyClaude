---
name: ecosystem-selection
version: "1.0.6"
maturity: "5-Expert"
specialization: Python/Julia Scientific Computing
description: Select optimal scientific computing ecosystems and manage multi-language workflows. Use when evaluating Python vs Julia for performance-critical numerical computing, implementing hybrid PyJulia/PyCall.jl interoperability, or setting up reproducible toolchains with Conda or Pkg.jl.
---

# Python/Julia Ecosystem Selection

Select optimal scientific computing ecosystems for performance and productivity.

---

## Ecosystem Comparison

| Aspect | Python | Julia |
|--------|--------|-------|
| Performance | 10-100x slower loops | C-like speed (JIT) |
| ML/DL | Excellent (PyTorch, TF) | Limited |
| ODEs/PDEs | SciPy (basic) | SciML (10-100x faster) |
| Community | Massive | Growing |
| Tooling | Mature | Developing |

---

## When to Choose

| Choose Python | Choose Julia | Use Hybrid |
|---------------|--------------|------------|
| ML/DL integration | Performance-critical sims | Need both ML + numerics |
| Rapid prototyping | Complex ODE/PDE systems | Gradual migration |
| Existing Python codebase | From-scratch scientific | Mixed team expertise |
| Data wrangling focus | Type-stable algorithms | Best-of-both-worlds |

---

## Hybrid Integration

### Python → Julia (PyJulia)

```python
from julia import Julia, Main
jl = Julia(compiled_modules=False)
Main.eval('using DifferentialEquations')

Main.eval('''
function fast_ode_solve(u0, tspan)
    prob = ODEProblem((u,p,t) -> -u, u0, tspan)
    solve(prob, Tsit5())
end
''')
sol = Main.fast_ode_solve([1.0], (0.0, 10.0))  # 10-100x faster
```

### Julia → Python (PyCall.jl)

```julia
using PyCall
np = pyimport("numpy")
sklearn = pyimport("sklearn.linear_model")
model = sklearn.LinearRegression()
model.fit(X_train, y_train)
```

---

## Toolchain Management

| Tool | Python | Julia |
|------|--------|-------|
| Environment | `conda`/`venv` | `Pkg.activate()` |
| Dependencies | `requirements.txt` | `Project.toml` |
| Lock file | `poetry.lock` | `Manifest.toml` |
| Install | `pip install`/`conda install` | `Pkg.add()` |

---

## Performance Optimization

| Language | Technique | Example |
|----------|-----------|---------|
| Python | Numba JIT | `@jit(nopython=True)` |
| Python | Vectorize | `np.arange(n)**2` |
| Julia | Type stability | `@code_warntype f(x)` |
| Julia | In-place ops | `A .= B .+ C` |

---

## Migration Strategy

1. **Profile** Python → identify bottlenecks
2. **Prototype** critical functions in Julia
3. **Benchmark** (aim >10x speedup)
4. **Interface** via PyJulia
5. **Iterate** gradual migration

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Start simple | Python first, Julia for bottlenecks |
| Benchmark both | Realistic workloads, not microbenchmarks |
| Type stability | Julia functions return consistent types |
| Vectorize Python | Avoid loops, use NumPy ops |
| Lock dependencies | Pin versions in both ecosystems |

---

## Checklist

- [ ] Performance requirements assessed (>10x speedup = Julia)
- [ ] Team expertise evaluated (Python/Julia balance)
- [ ] Hybrid integration tested if needed
- [ ] Reproducible environments configured
- [ ] Migration path documented

---

**Version**: 1.0.5
