---
name: chaos-attractors
description: Chaos characterization and attractor analysis with DynamicalSystems.jl (Julia) and JAX. Covers Lyapunov exponents (variational, QR), attractor reconstruction (Takens embedding), fractal dimension (correlation, Kaplan-Yorke), recurrence analysis, and Poincare sections. Use JAX vmap for parallel Lyapunov computation over parameter grids. Use when computing Lyapunov spectra, reconstructing attractors from time series, or quantifying chaotic behavior.
---

# Chaos Characterization & Attractor Analysis

Characterize chaotic dynamics using Lyapunov exponents, attractor reconstruction, fractal dimensions, recurrence quantification, and Poincare sections.

## Expert Agents

For chaos analysis, attractor reconstruction, and nonlinear dynamics, delegate to:

- **`nonlinear-dynamics-expert`**: Specialist for dynamical systems, bifurcation analysis, and chaos quantification.
  - *Location*: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
  - *Capabilities*: Lyapunov spectra, attractor reconstruction, recurrence analysis, fractal dimension estimation.

- **`julia-pro`**: Julia language specialist for DynamicalSystems.jl ecosystem.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: High-performance dynamical systems computation, type-stable implementations, SciML integration.

## System Construction (DynamicalSystems.jl)

### Continuous Systems

```julia
using DynamicalSystems, StaticArrays

function lorenz_rule(u, p, t)
    σ, ρ, β = p
    x, y, z = u
    return SVector(σ*(y - x), x*(ρ - z) - y, x*y - β*z)
end

u0 = SVector(1.0, 0.0, 0.0)
p = [10.0, 28.0, 8/3]
ds = CoupledODEs(lorenz_rule, u0, p)
```

### Discrete Systems

```julia
function henon_rule(u, p, n)
    a, b = p
    x, y = u
    return SVector(1.0 - a*x^2 + y, b*x)
end

u0 = SVector(0.1, 0.1)
p = [1.4, 0.3]
ds = DeterministicIteratedMap(henon_rule, u0, p)
```

**Key**: Always use `SVector` for state vectors -- enables stack allocation and automatic Jacobian computation.

## Lyapunov Exponents

### Maximum Lyapunov Exponent

```julia
lambda_max = lyapunov(ds, 10000)   # largest exponent only
```

### Full Lyapunov Spectrum

```julia
lambdas = lyapunovspectrum(ds, 10000)  # all exponents
```

### Convergence Tracking

```julia
lambdas = lyapunovspectrum(ds, 10000; show_convergence=true)
# Monitor convergence: values should stabilize within 1% over last 20% of steps
```

### Kaplan-Yorke Dimension

```julia
D_KY = kaplanyorke_dim(lambdas)
# D_KY = j + (1/|λ_{j+1}|) * Σ_{i=1}^{j} λ_i
# where j is largest index with Σ_{i=1}^{j} λ_i >= 0
```

## Attractor Reconstruction (Takens Embedding)

Reconstruct attractor geometry from scalar time series using delay coordinates.

### Step 1: Estimate Delay (tau)

```julia
using DelayEmbeddings

tau = estimate_delay(x, "mi_min")   # first minimum of mutual information
# Alternatives: "ac_zero" (autocorrelation zero crossing), "ac_min"
```

### Step 2: Estimate Embedding Dimension

```julia
d = estimate_dimension(x, tau)   # Cao's method (false nearest neighbors)
# Returns optimal embedding dimension d
```

### Step 3: Embed

```julia
Y = embed(x, d, tau)   # returns StateSpaceSet of delay vectors
# Y[i] = (x[i], x[i+tau], x[i+2*tau], ..., x[i+(d-1)*tau])
```

**Validation**: Compare reconstructed attractor dimension with known system dimension. Takens theorem guarantees faithful reconstruction when d >= 2*D_box + 1.

## Fractal Dimension

### Correlation Dimension (Grassberger-Procaccia)

```julia
radii = 10 .^ range(-3, 0, length=30)
Dc = correlationdimension(data, radii)
# C(r) ~ r^{D_c} in the scaling region
# Plot log(C(r)) vs log(r) to identify linear scaling region
```

### Generalized Dimensions

| Order q | Name | Measures |
|---------|------|----------|
| 0 | Box-counting D_0 | Geometric coverage |
| 1 | Information D_1 | Shannon entropy scaling |
| 2 | Correlation D_2 | Pair correlation scaling |

For strange attractors: D_0 > D_1 > D_2 (multifractal hierarchy).

## Recurrence Analysis

### Recurrence Matrix

```julia
using RecurrenceAnalysis

R = RecurrenceMatrix(data, epsilon)   # epsilon = recurrence threshold
# R[i,j] = 1 if ||x_i - x_j|| < epsilon
```

### Recurrence Quantification Analysis (RQA)

| Measure | Symbol | Chaos Signature |
|---------|--------|-----------------|
| Recurrence Rate | RR | Low (~1-5%) for chaos |
| Determinism | DET | Moderate (60-80%) |
| Max diagonal length | L_max | Finite (related to 1/lambda_max) |
| Entropy of diagonals | ENTR | High for complex dynamics |
| Laminarity | LAM | Low for chaos, high for intermittency |

```julia
rqa = rqa(R)
# Access: rqa.RR, rqa.DET, rqa.Lmax, rqa.ENTR, rqa.LAM
```

**Threshold selection**: Choose epsilon so that RR is between 1-5%. Too large masks structure; too small yields sparse matrix.

## Poincare Sections

```julia
plane = (3, 27.0)   # intersect z = 27.0 (variable index, value)
psos = poincaresos(ds, plane, 10000)
# Returns StateSpaceSet of intersection points
# Reveals: fixed points (period-1), closed curves (quasiperiodic), fractal dust (chaos)
```

## JAX Parallel Lyapunov Computation

Compute maximum Lyapunov exponent across a parameter grid using JAX vmap.

```python
import jax
import jax.numpy as jnp
from functools import partial

# Lorenz system
def lorenz(state, t, sigma, rho, beta):
    x, y, z = state
    return jnp.array([sigma*(y - x), x*(rho - z) - y, x*y - beta*z])

# Lorenz Jacobian
def lorenz_jacobian(state, sigma, rho, beta):
    x, y, z = state
    return jnp.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta]
    ])

# RK4 integrator step
def rk4_step(f, y, t, dt, *args):
    k1 = f(y, t, *args)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt, *args)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt, *args)
    k4 = f(y + dt*k3, t + dt, *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Variational equation for max Lyapunov exponent
@jax.jit
def max_lyapunov(rho_val, n_steps=10000, dt=0.01):
    sigma, beta = 10.0, 8.0/3.0
    state = jnp.array([1.0, 0.0, 0.0])
    w = jnp.array([1.0, 0.0, 0.0])  # perturbation vector

    def step(carry, _):
        state, w, lyap_sum = carry
        # Advance state
        state = rk4_step(lorenz, state, 0.0, dt, sigma, rho_val, beta)
        # Advance perturbation via Jacobian
        J = lorenz_jacobian(state, sigma, rho_val, beta)
        w = w + dt * J @ w
        # QR renormalization
        w_norm = jnp.linalg.norm(w)
        w = w / w_norm
        lyap_sum = lyap_sum + jnp.log(w_norm)
        return (state, w, lyap_sum), None

    (_, _, lyap_sum), _ = jax.lax.scan(step, (state, w, 0.0), None, length=n_steps)
    return lyap_sum / (n_steps * dt)

# Sweep over 10000 rho values in parallel
rho_values = jnp.linspace(0.0, 200.0, 10000)
lyap_spectrum = jax.vmap(max_lyapunov)(rho_values)
# Shape: (10000,) -- one lambda_max per rho value
```

**Performance**: vmap compiles a single fused kernel; 10000 parameter values run in one GPU dispatch.

## Python Counterparts

Julia's `DynamicalSystems.jl` umbrella has no single Python equivalent — the ecosystem is a set of focused, NumPy-based libraries. Pick by task:

| Task | Python package | Key API |
|------|---------------|---------|
| Lyapunov (Rosenstein / Eckmann), Hurst, DFA | **`nolds`** | `nolds.lyap_r`, `nolds.lyap_e`, `nolds.hurst_rs`, `nolds.dfa`, `nolds.sampen`, `nolds.corr_dim` |
| Recurrence plots + RQA, recurrence networks, visibility graphs | **`pyunicorn`** | `RecurrencePlot`, `RecurrenceNetwork`, `VisibilityGraph`, `ClimateNetwork`, `EventSynchronization` |
| Permutation / sample / SVD / spectral entropy, Higuchi / Katz / Petrosian fractal dim | **`antropy`** | `perm_entropy`, `sample_entropy`, `higuchi_fd`, `detrended_fluctuation`, `lziv_complexity` |
| Multiscale / multivariate / bidimensional entropy | **`EntropyHub`** | `SampEn`, `PermEn`, `DispEn`, `MvMSEn`, `hXMSEn` |
| Topological signal analysis (persistent homology on time series) | **`teaspoon`** (repo: `TeaspoonTDA/teaspoon`) | `SP`, `TDA`, `ML`, `DAF`, ordinal-partition networks |
| Simplex / S-map / convergent cross mapping | **`pyEDM`** | `Simplex`, `SMap`, `CCM`, `Multiview`, `EmbedDimension` |

```python
# nolds: Lyapunov + DFA on a scalar time series
import nolds
lambda_max = nolds.lyap_r(x, emb_dim=5, lag=1)       # Rosenstein method
alpha = nolds.dfa(x)                                  # DFA alpha exponent
H = nolds.hurst_rs(x)                                 # Hurst exponent

# pyunicorn: recurrence plot + RQA (PyRQA is unmaintained; use pyunicorn)
from pyunicorn.timeseries import RecurrencePlot
rp = RecurrencePlot(x, recurrence_rate=0.05)
rqa = {"DET": rp.determinism(), "LAM": rp.laminarity(), "Lmax": rp.max_diaglength()}
```

> **Gaps and caveats**:
> - **`pyRQA`** is effectively abandoned — use `pyunicorn.timeseries.RecurrencePlot` instead.
> - **`PyDSTool`** (continuation/bifurcation) is unmaintained — for production bifurcation work call `BifurcationKit.jl` from Python via `juliacall`.
> - **`EntropyHub`** PyPI wheels lag the repo; install from source for current features.
> - None of these packages are JAX-native. For GPU-parallel Lyapunov sweeps use the JAX `vmap` pattern shown above.

## Calling Python nonlinear-dynamics tools from Julia via PythonCall.jl

When a Julia-first codebase needs computations that only live in `nolds` / `antropy` / `IDTxl` / `pyEDM` / `pyunicorn` / `teaspoon`, use `PythonCall.jl` — the same package whose Python-facing half (`juliacall`) handles the Python → Julia direction in `bifurcation-analysis`.

### Canonical import pattern

```julia
using PythonCall

# One-time setup: CondaPkg.jl manages the Python environment
# pkg> add CondaPkg
# julia> CondaPkg.add("nolds")
# julia> CondaPkg.add("antropy")

nolds   = pyimport("nolds")
antropy = pyimport("antropy")
idtxl   = pyimport("idtxl")           # via IDTxl's top-level package
```

### Concrete example — compute `lyap_r` from Julia

```julia
using PythonCall
nolds = pyimport("nolds")

# Julia-side time series — logistic map in the chaotic regime
N  = 10_000
x  = Vector{Float64}(undef, N)
x[1] = 0.1
for t in 2:N
    x[t] = 3.9 * x[t-1] * (1 - x[t-1])
end

# Marshal Julia vector → Python list → call nolds; convert result back
py_x  = Py(x)
lyap  = pyconvert(Float64, nolds.lyap_r(py_x, emb_dim=5, lag=1))
println("Largest Lyapunov exponent ≈ $(lyap)")
```

The same `Py(x)` / `pyconvert` pattern works for `antropy.perm_entropy`, `antropy.higuchi_fd`, `idtxl` transfer-entropy, `pyEDM.Simplex`, `pyunicorn.RecurrencePlot`, and the rest of the packages in the counterparts table above.

### Caveats

- **Marshaling + GIL** — each `Py(x)` copies the Julia array to Python memory, and PythonCall.jl holds the GIL during every Python call, so `@threads` serializes. Call Python once over the full series (not per-window), and dispatch parallel workloads via `Distributed.jl` workers (one interpreter per worker).
- **Prefer `PythonCall.jl` over legacy `PyCall.jl`** — it ships `CondaPkg.jl` for deterministic environments and pairs with `juliacall` for the reciprocal Python → Julia path (see `bifurcation-analysis`, "Python escape hatch").

## Chaos Classification

| lambda_max | D_KY | Behavior |
|------------|------|----------|
| < 0 | Integer | Fixed point / limit cycle |
| = 0 | Integer | Quasiperiodic (torus) |
| > 0 | Non-integer | Chaotic (strange attractor) |
| >> 0 | High | Hyperchaos (multiple positive exponents) |

**Rule**: Count the number of positive Lyapunov exponents -- equals the number of expanding directions (chaos dimension).

## Common Pitfalls

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Short integration time | Lyapunov exponents not converged | Use >= 10000 steps; check convergence plot |
| Wrong embedding delay (tau) | Attractor reconstruction artifacts | Use mutual information minimum, not autocorrelation |
| Epsilon too large in RQA | Recurrence matrix saturated | Target RR = 1-5%; use fixed-mass neighbors |
| Ignoring transients | Initial conditions bias spectra | Discard first 10-20% of trajectory as transient |
| Float32 in long integrations | Accumulated numerical drift | Use Float64 for Lyapunov; reserve Float32 for vmap sweeps only |

## Checklist

- [ ] System defined with SVector for performance
- [ ] Lyapunov exponents converged (check last 20%)
- [ ] Transients discarded before analysis
- [ ] Embedding parameters (tau, d) estimated from data
- [ ] Fractal dimension consistent with Kaplan-Yorke
- [ ] Recurrence threshold yields RR in 1-5%
- [ ] Float64 used for single-trajectory analysis
