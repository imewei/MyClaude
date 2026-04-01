---
name: pattern-formation
description: Spatiotemporal pattern formation in reaction-diffusion systems. Covers Turing instability analysis, dispersion relations, reaction-diffusion on graphs, Swift-Hohenberg equation, spiral waves, and amplitude equations (Ginzburg-Landau). Julia (MethodOfLines.jl, ModelingToolkit.jl) for symbolic PDE and JAX for GPU pseudo-spectral methods.
---

# Pattern Formation

Spatiotemporal pattern formation in reaction-diffusion and related PDE systems. Combines Julia symbolic PDE tools with JAX GPU-accelerated pseudo-spectral solvers.

## Expert Agents

For complex pattern formation problems, delegate to the appropriate expert agent:

- **`nonlinear-dynamics-expert`**: Turing instability analysis, bifurcation of spatial modes, amplitude equation derivation.
  - *Location*: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
- **`julia-pro`**: Symbolic PDE construction (ModelingToolkit.jl), method-of-lines discretization, stiff ODE solvers.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
- **`jax-pro`**: GPU pseudo-spectral time-stepping, vmap over parameter sweeps, JIT-compiled spatial simulations.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`

## Turing Instability Analysis

A homogeneous steady state that is **stable without diffusion** can become unstable when diffusion is introduced, provided the diffusion ratio D_v / D_u is sufficiently large (short-range activator, long-range inhibitor).

### Dispersion Relation

Linearise the reaction-diffusion system about the homogeneous steady state. For wavenumber k, the growth rate satisfies:

```
omega(k) = max eigenvalue of [ J - D * k^2 ]
```

where J is the Jacobian of the reaction kinetics and D = diag(D_u, D_v).

**Instability criteria:** (1) tr(J) < 0, det(J) > 0 (stable without diffusion); (2) D_v * J_11 + D_u * J_22 > 0; (3) (D_v * J_11 + D_u * J_22)^2 > 4 * D_u * D_v * det(J).

### JAX: Dispersion Relation Sweep

```python
import jax.numpy as jnp
from jax import vmap, jit

def dispersion(k, J, D):
    """Growth rate omega(k) for 2-component RD system."""
    M = J - D * k**2
    eigvals = jnp.linalg.eigvalsh(M)
    return jnp.max(eigvals)

# Sweep over wavenumbers
k_vals = jnp.linspace(0.0, 10.0, 500)
D = jnp.diag(jnp.array([1.0, 40.0]))  # D_v / D_u = 40
J = jnp.array([[0.5, -1.0], [1.0, -1.5]])

omega = vmap(dispersion, in_axes=(0, None, None))(k_vals, J, D)
k_crit = k_vals[jnp.argmax(omega)]  # most unstable wavenumber
```

### Julia: Symbolic Turing Analysis

```julia
using Symbolics
@variables k D_u D_v a b c d
J = [a b; c d]
D = Diagonal([D_u, D_v])
M = J - D * k^2
det_M = Symbolics.det(M)
# Solve det_M == 0 for critical k
```

## Reaction-Diffusion Systems

### Gray-Scott Model (JAX)

Two-species (U, V) with feed rate F and kill rate k on a 2D periodic grid:

```python
import jax.numpy as jnp
from jax import jit

def laplacian_2d(u, dx):
    """Periodic 2D Laplacian via nearest-neighbour finite differences."""
    return (jnp.roll(u, 1, 0) + jnp.roll(u, -1, 0)
          + jnp.roll(u, 1, 1) + jnp.roll(u, -1, 1)
          - 4 * u) / dx**2

@jit
def gray_scott_step(state, dt, dx, D_u, D_v, F, k):
    U, V = state
    lap_U = laplacian_2d(U, dx)
    lap_V = laplacian_2d(V, dx)
    reaction = U * V**2
    dU = D_u * lap_U - reaction + F * (1.0 - U)
    dV = D_v * lap_V + reaction - (F + k) * V
    return (U + dt * dU, V + dt * dV)
```

### FitzHugh-Nagumo (Julia + MethodOfLines.jl)

```julia
using ModelingToolkit, MethodOfLines, DomainSets, OrdinaryDiffEq

@parameters t x y D_u D_v a b epsilon
@variables u(..) v(..)
Dt = Differential(t); Dx = Differential(x); Dy = Differential(y)

eqs = [
    Dt(u(t,x,y)) ~ D_u*(Dx(Dx(u(t,x,y))) + Dy(Dy(u(t,x,y))))
                    + u(t,x,y) - u(t,x,y)^3 - v(t,x,y),
    Dt(v(t,x,y)) ~ D_v*(Dx(Dx(v(t,x,y))) + Dy(Dy(v(t,x,y))))
                    + epsilon*(u(t,x,y) - a*v(t,x,y) - b)
]

domains = [t in Interval(0.0, 100.0),
           x in Interval(0.0, 50.0),
           y in Interval(0.0, 50.0)]

discretization = MOLFiniteDifference([x => 0.5, y => 0.5], t)
prob = discretize(PDESystem(eqs, [...], domains, [t,x,y],
                            [u(t,x,y), v(t,x,y)]), discretization)
sol = solve(prob, ROCK2(), saveat=1.0)
```

## Reaction-Diffusion on Graphs

Replace the spatial Laplacian with the **graph Laplacian** L = D_deg - A, where D_deg is the degree matrix and A the adjacency matrix. Turing instability on networks depends on the Laplacian eigenspectrum rather than continuous wavenumbers.

### Julia: RD on Watts-Strogatz Graph

```julia
using Graphs, DifferentialEquations, LinearAlgebra

g = watts_strogatz(100, 4, 0.3)  # 100 nodes, rewiring prob 0.3
A = Float64.(adjacency_matrix(g))
D_deg = Diagonal(vec(sum(A, dims=2)))
L = D_deg - A  # graph Laplacian

function rd_on_graph!(du, u, p, t)
    N = size(L, 1)
    U = @view u[1:N]; V = @view u[N+1:end]
    dU = @view du[1:N]; dV = @view du[N+1:end]
    @. dU = -p.D_u * (L * U) + U - U^3 - V
    @. dV = -p.D_v * (L * V) + p.eps * (U - p.a * V - p.b)
end

u0 = vcat(0.01 * randn(100), zeros(100))
p = (D_u=1.0, D_v=10.0, eps=0.1, a=0.5, b=0.1)
prob = ODEProblem(rd_on_graph!, u0, (0.0, 200.0), p)
sol = solve(prob, ROCK2())
```

## Swift-Hohenberg Equation

The Swift-Hohenberg equation models pattern selection near onset:

```
du/dt = r*u - (1 + nabla^2)^2 u + N(u)
```

where N(u) is typically a cubic nonlinearity (e.g., -u^3 or quadratic-cubic).

### JAX: Pseudo-Spectral Semi-Implicit Step

```python
import jax.numpy as jnp
from jax import jit

@jit
def swift_hohenberg_step(u_hat, dt, r, kx, ky):
    """Semi-implicit pseudo-spectral time step for Swift-Hohenberg."""
    k2 = kx**2 + ky**2
    linear_op = r - (1.0 - k2)**2  # linear part in Fourier space
    # Nonlinearity in physical space
    u = jnp.fft.irfft2(u_hat)
    nl = -u**3
    nl_hat = jnp.fft.rfft2(nl)
    # Semi-implicit: treat linear implicitly, nonlinear explicitly
    u_hat_new = (u_hat + dt * nl_hat) / (1.0 - dt * linear_op)
    return u_hat_new
```

## Spiral Waves

Spiral waves arise in excitable media (e.g., FitzHugh-Nagumo, Barkley model). The spiral **tip** is located at the intersection of nullclines u = u* and v = v* in physical space.

### Tip Tracking via Contour Intersection

```python
from skimage.measure import find_contours

def track_spiral_tip(U, V, u_thresh, v_thresh):
    """Locate spiral tip as intersection of iso-contours."""
    contours_u = find_contours(U, u_thresh)
    contours_v = find_contours(V, v_thresh)
    # Find closest point between the two contour sets
    min_dist = float('inf')
    tip = None
    for cu in contours_u:
        for cv in contours_v:
            dists = jnp.sqrt((cu[:, None, 0] - cv[None, :, 0])**2
                           + (cu[:, None, 1] - cv[None, :, 1])**2)
            idx = jnp.unravel_index(jnp.argmin(dists), dists.shape)
            d = dists[idx]
            if d < min_dist:
                min_dist = d
                tip = cu[idx[0]]
    return tip
```

## Amplitude Equations: Complex Ginzburg-Landau

Near onset of oscillatory instability, the envelope evolves under the Complex Ginzburg-Landau equation (CGLE):

```
dA/dt = A + (1 + i*c1)*nabla^2 A - (1 + i*c2)|A|^2 A
```

### JAX: Pseudo-Spectral CGLE Step

```python
@jit
def cgle_step(A_hat, dt, c1, c2, k2):
    """Semi-implicit pseudo-spectral step for CGLE."""
    linear_coeff = 1.0 - (1.0 + 1j * c1) * k2
    A = jnp.fft.ifft2(A_hat)
    nl = -(1.0 + 1j * c2) * jnp.abs(A)**2 * A
    nl_hat = jnp.fft.fft2(nl)
    A_hat_new = (A_hat + dt * nl_hat) / (1.0 - dt * linear_coeff)
    return A_hat_new
```

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| CFL violation in explicit Euler | Blow-up after few steps | Use dt < dx^2 / (4 * D_max) or semi-implicit scheme |
| Insufficient grid resolution | Patterns merge or disappear | Ensure dx < pi / k_crit (Nyquist for critical mode) |
| Wrong diffusion ratio | No Turing patterns form | Verify D_v / D_u >> 1 and check instability criteria analytically |
| Aliasing in pseudo-spectral | Spurious high-k energy growth | Apply 2/3-rule dealiasing or exponential filter |
| Graph Laplacian sign convention | Diffusion amplifies instead of smooths | Ensure L = D_deg - A (positive semi-definite), flux term is -D * L * u |
