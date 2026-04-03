---
name: bifurcation-analysis
description: Numerical continuation and bifurcation analysis with BifurcationKit.jl. Covers codimension-1 (saddle-node, Hopf, pitchfork, period-doubling) and codimension-2 (Bogdanov-Takens, cusp, Bautin) bifurcations, normal forms, branch switching, and periodic orbit continuation. Julia-first; use JAX vmap for parameter sweeps around critical points. Use when computing bifurcation diagrams, tracking steady-state branches, or identifying critical transitions in dynamical systems.
---

# Bifurcation Analysis

Numerical continuation and bifurcation detection using BifurcationKit.jl. Julia-first workflow with JAX vmap for parameter sweeps around identified critical points.

---

## Expert Agents

For complex bifurcation problems requiring deep domain expertise, delegate to:

- **`nonlinear-dynamics-expert`**: Dynamical systems, bifurcation theory, normal forms, and codimension-2 unfoldings.
  - *Location*: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
- **`julia-pro`**: Julia performance, type stability, and SciML ecosystem integration.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

---

## BifurcationKit.jl Quick Start

```julia
using BifurcationKit, LinearAlgebra, Plots

# Define the vector field F(u, p) and its Jacobian
F(u, p) = [p.mu - u[1]^2 + p.alpha * u[2];
            u[1] - u[2]]

J(u, p) = [[-2u[1] p.alpha]; [1.0 -1.0]]

# Initial guess and parameters
u0 = [0.0, 0.0]
params = (mu = -1.0, alpha = 0.5)

# Build the bifurcation problem with parameter lens
prob = BifurcationProblem(F, u0, params, (@optic _.mu); J = J)

# Continuation options
opts = ContinuationPar(
    p_min = -2.0, p_max = 2.0,
    ds = 0.01, dsmax = 0.05,
    max_steps = 1000,
    newton_options = NewtonPar(tol = 1e-10, max_iterations = 20),
    detect_bifurcation = 3,  # 0=off, 1=fold, 2=+Hopf, 3=+period-doubling
)

# Run continuation with pseudo-arclength (PALC)
br = continuation(prob, PALC(), opts; bothside = true)

# Plot the bifurcation diagram
plot(br; plotfold = true, markersize = 4, legend = :topleft)
```

---

## Codimension-1 Bifurcations

| Type | Normal Form | Eigenvalue Signature | Detection Flag |
|------|-------------|---------------------|----------------|
| **Saddle-Node (Fold)** | `du/dt = mu - u^2` | Real eigenvalue crosses 0 | `detect_bifurcation >= 1` |
| **Transcritical** | `du/dt = mu*u - u^2` | Real eigenvalue crosses 0 (exchange of stability) | `detect_bifurcation >= 1` |
| **Pitchfork** | `du/dt = mu*u - u^3` (Z2 symmetry) | Real eigenvalue crosses 0 (symmetric) | `detect_bifurcation >= 1` |
| **Hopf** | `dz/dt = (mu + i*omega)*z - l1*|z|^2*z` | Complex conjugate pair crosses imaginary axis | `detect_bifurcation >= 2` |
| **Period-Doubling** | Floquet multiplier = -1 | Floquet multiplier crosses -1 on unit circle | `detect_bifurcation >= 3` |

---

## Detecting Bifurcation Points

After continuation, iterate over detected special points:

```julia
for sp in br.specialpoint
    println("Type: $(sp.type), Parameter: $(sp.param), Index: $(sp.idx)")
    println("  Eigenvalues at point: ", sp.x)
end

# Filter by type
hopf_points = filter(sp -> sp.type == :hopf, br.specialpoint)
fold_points = filter(sp -> sp.type == :bp, br.specialpoint)
```

---

## Normal Form Computation

Extract normal form coefficients for analytical classification:

```julia
# Get normal form at the i-th special point
nf = get_normal_form(br, i)

# For Hopf bifurcation: first Lyapunov coefficient l1
# l1 < 0 => supercritical (stable limit cycle)
# l1 > 0 => subcritical (unstable limit cycle)
println("First Lyapunov coefficient l1 = ", real(nf.nf.l1))

# For fold: quadratic coefficient a
# a != 0 confirms non-degenerate saddle-node
println("Fold coefficient a = ", nf.nf.a)
```

---

## Codimension-2 Bifurcations

| Type | Unfolding | Local Codim-1 Curves |
|------|-----------|---------------------|
| **Bogdanov-Takens** | `du/dt = u2, du2/dt = mu1 + mu2*u - u^2 - u*u2` | Fold + Hopf + Homoclinic |
| **Cusp** | `du/dt = mu1 + mu2*u - u^3` | Two fold curves meeting tangentially |
| **Bautin** | `dz/dt = (mu1 + i*omega)*z + mu2*|z|^2*z - |z|^4*z` | Hopf + fold of limit cycles (l1 = 0) |
| **Zero-Hopf** | Two-dimensional center manifold (0, +/- i*omega) | Fold + Hopf + Neimark-Sacker (torus) |

---

## Two-Parameter Continuation

Track codimension-1 bifurcation curves in two-parameter space:

```julia
# Continue a Hopf point in (mu, alpha) space
# `index` is the index of the Hopf point in br.specialpoint
br_hopf = continuation(br, index, (@optic _.alpha), opts;
    detect_codim2_bifurcation = 2,  # detect codim-2 on the curve
    update_minaug_every_step = 1,
    bdlinsolver = MatrixBLS(),
)

# Continue a fold point in two parameters
br_fold = continuation(br, fold_index, (@optic _.alpha), opts;
    detect_codim2_bifurcation = 2,
)

# Plot both curves together
plot(br_hopf; vars = (:mu, :alpha), label = "Hopf")
plot!(br_fold; vars = (:mu, :alpha), label = "Fold")
```

---

## Branch Switching at Hopf to Periodic Orbits

```julia
# Branch switching from Hopf point to periodic orbit branch
br_po = continuation(br, hopf_index, opts;
    usedeflation = true,
    # Periodic orbit method
    PeriodicOrbitTrapProblem(M = 50),  # M mesh points
)

# Plot period and amplitude along the branch
plot(br_po; vars = (:param, :period), ylabel = "Period")
plot(br_po; vars = (:param, :max), ylabel = "Amplitude")
```

---

## Deflation for Multiple Equilibria

Find distinct equilibria on the same branch using deflation:

```julia
using BifurcationKit: DeflationOperator

# Start from a known solution
deflation_op = DeflationOperator(2.0, dot, 1.0, [u_known])

# Find a new equilibrium
u_new, _, converged = newton(
    prob, deflation_op,
    NewtonPar(tol = 1e-10, max_iterations = 50),
)

if converged
    push!(deflation_op, u_new)  # add to deflation set
    # Repeat to find more solutions
end
```

---

## Periodic Orbit Continuation

Direct continuation of periodic orbits using the trapezoidal method:

```julia
# Set up periodic orbit problem
po_prob = PeriodicOrbitTrapProblem(
    M = 100,            # number of mesh points
    jacobian = :Dense,   # or :FullSparseInplace for large systems
    update_section_every_step = 2,
)

# Continuation options for periodic orbits
po_opts = ContinuationPar(
    p_min = 0.0, p_max = 5.0,
    ds = 0.01, dsmax = 0.1,
    max_steps = 500,
    detect_bifurcation = 3,  # detect period-doubling on PO branch
)

# Continue from initial periodic orbit guess
br_po = continuation(prob, u0_po, po_prob, PALC(), po_opts)
```

---

## JAX Integration: Parameter Sweeps Around Critical Points

Once bifurcation points are identified in Julia, use JAX vmap for GPU-accelerated parameter sweeps:

```python
import jax
import jax.numpy as jnp
from functools import partial

# System from Julia analysis (vectorized for JAX)
def F(u, mu, alpha=0.5):
    return jnp.array([mu - u[0]**2 + alpha * u[1],
                       u[0] - u[1]])

def jacobian_eigs(mu, u0_guess, alpha=0.5):
    """Compute eigenvalues at equilibrium for given mu."""
    J = jax.jacobian(lambda u: F(u, mu, alpha))(u0_guess)
    return jnp.linalg.eigvals(J)

# Sweep mu around the Hopf point identified by BifurcationKit
mu_critical = 0.25  # from Julia analysis
mu_sweep = jnp.linspace(mu_critical - 0.5, mu_critical + 0.5, 1000)
u0_guesses = jnp.zeros((1000, 2))  # initial guesses

# vmap over parameter values
eigs_sweep = jax.vmap(jacobian_eigs)(mu_sweep, u0_guesses)

# Identify sign changes in real parts (bifurcation crossings)
real_parts = jnp.real(eigs_sweep)
crossings = jnp.where(jnp.diff(jnp.sign(real_parts[:, 0])) != 0)[0]
```

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Missing Jacobian | Slow continuation, inaccurate detection | Always supply analytical `J` or use `AutoDiff()` |
| `ds` too large | Missed bifurcation points, branch jumping | Start with `ds = 0.01`, reduce `dsmax` near critical points |
| No `bothside = true` | Only half the branch computed | Pass `bothside = true` to `continuation()` |
| `detect_bifurcation = 0` | Bifurcations not flagged | Set `detect_bifurcation >= 2` for Hopf detection |
| Singular Jacobian at fold | Newton diverges near fold | Use PALC (pseudo-arclength) not natural parameter continuation |
| Incorrect lens for codim-2 | Wrong parameter varied in two-parameter continuation | Double-check `(@optic _.param_name)` matches the second parameter |
