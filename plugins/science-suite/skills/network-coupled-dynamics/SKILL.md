---
name: network-coupled-dynamics
description: Coupled dynamics on networks with NetworkDynamics.jl (Julia) and JAX sparse graph operations. Covers Kuramoto synchronization, master stability function, chimera states, epidemic models (SIR/SIS), graph Laplacian dynamics, and synchronization order parameters. Use Julia for small networks (<1K nodes), JAX for large-scale GPU simulation (>1K nodes).
---

# Network-Coupled Dynamics

Coupled dynamics on networks: synchronization, epidemic spreading, and stability analysis.

## Expert Agents

For network-coupled dynamics tasks, delegate to the appropriate expert agent:

- **`nonlinear-dynamics-expert`**: Synchronization theory, bifurcation analysis, master stability function.
  - *Location*: `plugins/science-suite/agents/nonlinear-dynamics-expert.md`
- **`julia-pro`**: NetworkDynamics.jl, Graphs.jl, DifferentialEquations.jl for small-to-medium networks.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
- **`jax-pro`**: JAX sparse graph operations, GPU-accelerated large-scale network simulation.
  - *Location*: `plugins/science-suite/agents/jax-pro.md`

## Graph Laplacian Construction

### Julia (Graphs.jl)

```julia
using Graphs, LinearAlgebra, SparseArrays

# Standard topologies
g_er = erdos_renyi(N, p)              # random graph
g_ba = barabasi_albert(N, k)          # scale-free
g_ws = watts_strogatz(N, k, beta)     # small-world

# Adjacency and Laplacian matrices
A = adjacency_matrix(g_er)            # sparse adjacency
L = laplacian_matrix(g_er)            # L = D - A

# Weighted Laplacian from edge weights
function weighted_laplacian(g, weights::Vector{Float64})
    n = nv(g)
    W = spzeros(n, n)
    for (i, e) in enumerate(edges(g))
        W[src(e), dst(e)] = weights[i]
        W[dst(e), src(e)] = weights[i]
    end
    D = spdiagm(vec(sum(W, dims=2)))
    return D - W
end
```

### JAX (Sparse BCOO)

```python
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

def build_sparse_laplacian(edge_list, weights, n_nodes):
    """Construct graph Laplacian as BCOO sparse matrix from edge list."""
    src, dst = edge_list[:, 0], edge_list[:, 1]
    # Off-diagonal: -w_{ij}
    indices = jnp.concatenate([
        jnp.stack([src, dst], axis=1),
        jnp.stack([dst, src], axis=1),
    ])
    off_diag = jnp.concatenate([-weights, -weights])
    # Diagonal: sum of incident weights
    deg = jnp.zeros(n_nodes).at[src].add(weights).at[dst].add(weights)
    diag_indices = jnp.stack([jnp.arange(n_nodes)] * 2, axis=1)
    all_indices = jnp.concatenate([indices, diag_indices])
    all_data = jnp.concatenate([off_diag, deg])
    return BCOO((all_data, all_indices), shape=(n_nodes, n_nodes))
```

## Kuramoto Model

### Theory

Coupled phase oscillators on a network:

$$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)$$

**Order parameter** measuring global synchronization:

$$R(t) = \left| \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j(t)} \right|$$

- R = 0: incoherent (phases uniformly distributed)
- R = 1: fully synchronized

**Critical coupling** for mean-field (all-to-all, Lorentzian g(w)):

$$K_c = \frac{2}{\pi \, g(0)}$$

where g(0) is the natural frequency distribution evaluated at the mean.

### Julia (NetworkDynamics.jl)

```julia
using NetworkDynamics, OrdinaryDiffEq, Graphs

function kuramoto_vertex!(dv, v, edges, p, t)
    omega = p[1]
    dv[1] = omega
    for e in edges
        dv[1] += e[1]  # sum coupling from edges
    end
end

function kuramoto_edge!(e, v_s, v_d, p, t)
    K = p[1]
    e[1] = K * sin(v_d[1] - v_s[1])
end

vert = ODEVertex(; f=kuramoto_vertex!, dim=1)
edge = StaticEdge(; f=kuramoto_edge!, dim=1, coupling=:antisymmetric)

g = watts_strogatz(100, 6, 0.3)
nd = network_dynamics(vert, edge, g)

# Natural frequencies from Lorentzian
omega = tan.(pi .* (rand(nv(g)) .- 0.5))
p_v = [(w,) for w in omega]
p_e = [(0.5,) for _ in edges(g)]

theta0 = 2pi .* rand(nv(g))
prob = ODEProblem(nd, theta0, (0.0, 50.0), (p_v, p_e))
sol = solve(prob, Tsit5(); saveat=0.1)

# Order parameter
R(theta) = abs(mean(exp.(im .* theta)))
R_t = [R(sol[:, i]) for i in 1:size(sol, 2)]
```

### JAX (Large-Scale GPU)

```python
import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def kuramoto_rhs(theta, t, L_data, L_indices, K, omega, n):
    """Kuramoto RHS using sparse Laplacian coupling."""
    src, dst = L_indices[:, 0], L_indices[:, 1]
    coupling = jnp.zeros(n).at[src].add(
        K * jnp.sin(theta[dst] - theta[src]) * (-L_data[: len(src)])
    )
    return omega + coupling

def order_parameter(theta):
    """Global synchronization order parameter R(t)."""
    return jnp.abs(jnp.mean(jnp.exp(1j * theta)))

@partial(jax.jit, static_argnums=(4,))
def simulate_kuramoto(theta0, K, omega, L_sparse, n_steps, dt=0.01):
    """Integrate Kuramoto with lax.scan for GPU efficiency."""
    def step(theta, _):
        dtheta = kuramoto_rhs(theta, 0.0, L_sparse.data,
                              L_sparse.indices, K, omega, theta.shape[0])
        theta_new = theta + dt * dtheta
        R = order_parameter(theta_new)
        return theta_new, R
    final_theta, R_trace = jax.lax.scan(step, theta0, jnp.arange(n_steps))
    return final_theta, R_trace

# Sweep over coupling strengths with vmap
K_values = jnp.linspace(0.0, 5.0, 200)
sweep_fn = jax.vmap(lambda K: simulate_kuramoto(
    theta0, K, omega, L_sparse, 5000
)[1][-1])
R_final = sweep_fn(K_values)  # R vs K phase diagram
```

## Master Stability Function (MSF)

Determines synchronization stability from network topology alone.

1. **Laplacian eigenvalues**: Compute spectrum {lambda_k} of L, with lambda_1 = 0.
2. **Variational equation**: Linearize around the synchronized state:
   $$\dot{\xi}_k = [DF + \lambda_k \, \sigma \, DH] \, \xi_k$$
   where DF is the uncoupled Jacobian, DH the coupling function Jacobian, sigma the coupling strength.
3. **MSF**: The maximum Lyapunov exponent Lambda_max(alpha) as a function of alpha = sigma * lambda_k.
4. **Stability criterion**: Synchronization is stable iff Lambda_max(sigma * lambda_k) < 0 for all k >= 2.

```julia
using LinearAlgebra

function check_msf_stability(L, sigma, msf_func)
    eigvals_L = eigvals(Matrix(L))
    sort!(eigvals_L)
    # Skip lambda_1 = 0; check k >= 2
    for k in 2:length(eigvals_L)
        alpha = sigma * real(eigvals_L[k])
        if msf_func(alpha) >= 0.0
            return false, k, alpha
        end
    end
    return true, -1, 0.0
end
```

## Chimera States

Coexistence of synchronized and desynchronized domains in identical oscillators.

### Local Order Parameter

$$R_j = \left| \frac{1}{2\delta+1} \sum_{|k-j| \le \delta} e^{i\theta_k} \right|$$

A chimera is identified when the distribution of R_j is **bimodal**: one cluster near R ~ 1 (coherent) and another near R << 1 (incoherent).

### Detection and Visualization

```julia
using CairoMakie

function local_order_param(theta_matrix, delta)
    N, T = size(theta_matrix)
    R_local = zeros(N, T)
    for t in 1:T, j in 1:N
        neighbors = mod1.(j-delta:j+delta, N)
        R_local[j, t] = abs(mean(exp.(im .* theta_matrix[neighbors, t])))
    end
    return R_local
end

# Space-time heatmap of local order parameter
R_loc = local_order_param(theta_history, 5)
fig = Figure(size=(800, 400))
ax = Axis(fig[1,1]; xlabel="Time", ylabel="Oscillator index")
heatmap!(ax, R_loc; colormap=:viridis, colorrange=(0, 1))
Colorbar(fig[1,2]; colormap=:viridis, label="R_local")
save("chimera_spacetime.png", fig)
```

### Bimodal Distribution Detection

```python
from scipy.signal import find_peaks

def detect_chimera(R_local_snapshot, prominence=0.05):
    """Detect chimera by checking bimodality of local R distribution."""
    hist, bin_edges = jnp.histogram(R_local_snapshot, bins=50)
    peaks, props = find_peaks(hist, prominence=prominence * hist.max())
    return len(peaks) >= 2  # bimodal => chimera
```

## Epidemic Models on Networks

### SIR with NetworkDynamics.jl

```julia
function sir_vertex!(dv, v, edges, p, t)
    beta, gamma = p
    S, I, R = v[1], v[2], v[3]
    infection_pressure = sum(e[1] for e in edges; init=0.0)
    dv[1] = -beta * S * infection_pressure   # dS/dt
    dv[2] =  beta * S * infection_pressure - gamma * I  # dI/dt
    dv[3] =  gamma * I                       # dR/dt
end

function sir_edge!(e, v_s, v_d, p, t)
    e[1] = v_s[2]  # pass infected fraction of source
end

sir_v = ODEVertex(; f=sir_vertex!, dim=3)
sir_e = StaticEdge(; f=sir_edge!, dim=1, coupling=:directed)
```

### Basic Reproduction Number on Networks

$$R_0 = \frac{\beta}{\gamma} \cdot \frac{\langle k^2 \rangle}{\langle k \rangle}$$

where <k> and <k^2> are the first and second moments of the degree distribution. Heterogeneous networks (scale-free) have diverging <k^2>, making epidemic threshold vanishingly small.

## Ecosystem Selection

| Network Size | Recommended Stack | Rationale |
|-------------|-------------------|-----------|
| < 100 nodes | Julia (Graphs.jl + NetworkDynamics.jl) | Full symbolic access, interactive exploration |
| 100 -- 1K nodes | Julia (NetworkDynamics.jl + OrdinaryDiffEq) | Dense linear algebra still feasible, MSF analysis |
| 1K -- 100K nodes | JAX (BCOO sparse + lax.scan) | GPU parallelism, vmap parameter sweeps |
| > 100K nodes | JAX (multi-GPU pmap + sharded BCOO) | Distributed memory, batch coupling sweeps |

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Non-connected graph | Isolated clusters never synchronize globally | Check `is_connected(g)` before simulation; use largest connected component |
| Wrong Laplacian sign convention | Dynamics diverge or damp incorrectly | Ensure L = D - A (positive semi-definite); verify lambda_1 = 0 |
| All-to-all coupling assumption | Mean-field K_c formula fails on sparse networks | Use network-specific MSF or numerical sweep for K_c |
| Finite-size effects | Order parameter R never reaches 0 (scales as 1/sqrt(N)) | Report R with finite-size correction; compare against R_null = 1/sqrt(N) |
| Chimera vs transient | Apparent chimera dissolves after long integration | Integrate for t > 1000/K; verify persistence with time-windowed R_local |
