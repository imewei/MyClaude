---
name: correlation-physical-systems
description: Map correlation functions to physical systems including condensed matter (spin correlations, critical exponents ξ ~ |T-Tc|^(-ν)), soft matter (polymer Rouse/Zimm dynamics, colloidal g(r), glass χ₄), biological systems (protein folding, membrane fluctuations), and non-equilibrium (active matter, transfer entropy). Use for materials characterization, transport predictions, or connecting experiments to theory. Use when mapping correlation functions to condensed matter, soft matter, biological systems, or non-equilibrium processes.
---

# Physical Systems & Correlation Functions

Bridge theoretical predictions with experimental observables across domains.

## Expert Agent

For domain-specific correlation analysis in condensed matter, soft matter, and biophysics, delegate to the expert agent:

- **`statistical-physicist`**: Unified specialist for Physical Systems Analysis.
  - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - *Capabilities*: Critical phenomena analysis, polymer dynamics, and glass transition characterization.

## Domain Selection

| Domain | Key Correlations | Experiments |
|--------|------------------|-------------|
| Condensed Matter | Spin ⟨SᵢSⱼ⟩, density n(r)n(r') | Neutron/X-ray scattering |
| Soft Matter | Polymer ⟨R²(t)⟩, colloidal g(r) | DLS, rheology |
| Biological | Contact maps Cᵢⱼ(t), membrane ⟨h(r)h(0)⟩ | FRET, NMR |
| Non-equilibrium | Active Cvv(r), transfer entropy | Microscopy, tracking |

## Condensed Matter

### Spin Correlations

```python
def ising_correlation_critical(r, xi, eta=0.036):
    """Critical Ising correlation (d=3)."""
    return (r/xi)**(-1+eta) * np.exp(-r/xi)
```

| Model | Correlation | Critical Behavior |
|-------|-------------|-------------------|
| Ising | ⟨SᵢSⱼ⟩ | C(r) ~ r^(-(d-2+η)) |
| Heisenberg | ⟨Sᵢ·Sⱼ⟩ | Vector spins |
| General | ξ ~ \|T-Tc\|^(-ν) | Correlation length diverges |

### Density Correlations

**Van Hove Function**: G(r,t) = ⟨ρ(r,t)ρ(0,0)⟩
- **Self part** Gs: Single-particle propagator
- **Distinct part** Gd: Inter-particle correlations
- **Static limit**: g(r) = lim G(r,t)/ρ

## Soft Matter

### Polymer Dynamics

| Model | Mean-Square Displacement | Regime |
|-------|-------------------------|--------|
| Rouse | ⟨R²(t)⟩ ~ t^(1/2) | No hydrodynamics |
| Zimm | ⟨R²(t)⟩ ~ t^(2/3) | With solvent |
| Reptation | C(t) ~ t^(-1/4) | Entangled, τe < t < τd |

### Colloidal Systems

- **g(r)**: Oscillations at σ, 2σ, 3σ for hard spheres
- **DLS**: f(q,t) intermediate scattering function
- **Short-time**: f(q,t) ≈ exp(-Dq²t)

### Glass Transition

```python
def four_point_susceptibility(positions, dt, w_cutoff=0.3):
    """χ₄(t) for dynamic heterogeneity."""
    displacements = positions[dt] - positions[0]
    w = np.exp(-displacements**2 / (2*w_cutoff**2))
    return len(w) * np.var(w)
```

- **χ₄(t)**: Growing correlation length ξ₄ near Tg
- **KWW**: φ(t) = exp[-(t/τ)^β], β < 1

## Biological Systems

### Protein Folding

| Observable | Correlation | Method |
|------------|-------------|--------|
| Native contacts | Cᵢⱼ(t) = ⟨qᵢⱼ(t)qᵢⱼ(0)⟩ | MD simulations |
| End-to-end distance | FRET efficiency | smFRET |
| Backbone dynamics | S² order parameters | NMR |

### Membrane Fluctuations

- **Height-height**: ⟨h(r)h(0)⟩ ~ ln(r) for 2D membranes
- Extracts bending rigidity κ and tension σ
- Lipid diffusion D ~ 1-10 μm²/s

## Non-Equilibrium Systems

### Active Matter

```python
def velocity_correlation_active(velocities, positions, rmax, dr):
    """Velocity-velocity correlation Cvv(r)."""
    C_vv = np.zeros(int(rmax/dr))
    for i, j in pairs_at_distance(positions, r, dr):
        C_vv[bin] += np.dot(velocities[i], velocities[j])
    return C_vv / counts
```

- **Enhanced diffusion**: D_eff = D_t + v₀²τr/d
- **Giant fluctuations**: ⟨(δρ)²⟩ ~ L^(4/5) in 2D (Toner-Tu)
- **MIPS**: Motility-induced phase separation

### Information Transfer

```python
def transfer_entropy(X, Y, delay=1):
    """TE_{X→Y}: Directional information flow."""
    Y_future, Y_past, X_past = Y[delay:], Y[:-delay], X[:-delay]
    # H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    return conditional_entropy_diff(Y_future, Y_past, X_past)
```

- **Mutual information**: Beyond linear correlations
- **Granger causality**: Linear approximation to TE

## Experimental Mapping

| Experiment | Observable | Correlation |
|------------|------------|-------------|
| DLS | g₂(τ) | Diffusion D |
| SAXS/SANS | I(q) | Structure factor S(q) |
| XPCS | C(t₁,t₂) | Aging, non-stationarity |
| FCS | G(τ) | Concentration, binding |

## Transport Coefficients

| Coefficient | Green-Kubo Formula |
|-------------|-------------------|
| Diffusion D | ∫⟨v(t)·v(0)⟩dt |
| Viscosity η | (V/kT)∫⟨σxy(t)σxy(0)⟩dt |
| Conductivity σ | ∫⟨J(t)·J(0)⟩dt |

## Best Practices

| Practice | Implementation |
|----------|----------------|
| System identification | Match correlation type to observables |
| Timescale separation | Fast (microscopic) vs slow (collective) |
| Experimental connection | Map theory to measurable quantities |
| Model validation | Compare predictions with experimental data |

---

## Python / JAX ecosystem

For physical-systems correlations computed from MD trajectories, scattering data, or particle ensembles, the Python side is dominated by the Glotzer-group `freud` toolkit plus the MDAnalysis / mdtraj trajectory readers:

| Role | Package | Key API |
|---|---|---|
| RDF / pair correlation g(r) | **`freud.density.RDF`**, **`MDAnalysis.analysis.rdf`** | radial distribution, neighbor lists, bin centers |
| Static structure factor S(q) | **`freud.diffraction.StaticStructureFactorDebye`**, **`freud.diffraction.StaticStructureFactorDirect`** | Debye formula (fast) vs direct (q-vector enumerated) |
| Dynamic structure factor F(q,t) / S(q,ω) | **`freud.density.IntermediateScattering`**, **`MDAnalysis.analysis.waterdynamics`** | intermediate scattering, van Hove G_s(r,t) and G_d(r,t) |
| Bond-orientational order Q_l / W_l | **`freud.order.Steinhardt`** | `Q_4`, `Q_6`, `W_6`, neighborhood-averaged Q̄_l, solid/liquid classifier |
| Nematic / hexatic order | **`freud.order.Nematic`**, **`freud.order.Hexatic`** | scalar order parameter + director field |
| Cluster analysis & percolation | **`freud.cluster.Cluster`** | connected components on a neighbor list, cluster-size distributions |
| Time-correlation from trajectories | **`MDAnalysis.analysis.encore`**, **`mdtraj`**, hand-rolled on `numpy.fft` | VACF, stress autocorrelation for Green-Kubo |
| GPU ensemble correlations | **`jax.vmap`** + **`jax.lax.scan`** over replica trajectories | pair-correlation across replicas with zero Python overhead |

### Minimal pattern — g(r) and Q_6 via freud

```python
import freud

box = freud.box.Box.cube(L)

# Pair correlation
rdf = freud.density.RDF(bins=200, r_max=L / 2)
rdf.compute(system=(box, positions))
g_r, r = rdf.rdf, rdf.bin_centers

# Steinhardt bond-order parameter for solid/liquid detection
q6 = freud.order.Steinhardt(l=6, average=True)
q6.compute(system=(box, positions), neighbors={"num_neighbors": 12})
solid_like = q6.particle_order > 0.35      # threshold per system
```

> **Stay in Julia / SciML** when the workflow drives both the trajectory generation and the correlator in one session (MTK → DiffEq → correlator), or when symbolic reaction-diffusion correlations need derivative-level composability. **Drop to `freud` + MDAnalysis / mdtraj** whenever the input is an existing MD trajectory file (LAMMPS dump, DCD, XTC, HOOMD GSD) or when GPU-/TBB-accelerated bond-order / cluster analysis is the bottleneck — `freud`'s C++ / TBB backend is hard to beat on CPU, and its neighbor-list reuse across observables is a significant win on long trajectories.

## Python `freud` ecosystem — the soft-matter reference toolkit

The Glotzer group's `freud` (v3.5.0) is the production tool for physical-system correlation analysis — RDF, structure factors, bond-orientational order, and higher-order correlations. No native Julia equivalent exists; Julia users go through `PythonCall.jl` (see `chaos-attractors`). Strengths: triclinic-box periodic boundaries, O(N log N) cell-list neighbor queries reusable across analyses, optional CuPy GPU backend, clean handoff from `MDAnalysis`/`MDTraj` frame iterators. Install with `pip install freud-analysis`.

### Radial distribution function g(r)

```python
import freud, numpy as np
box    = freud.box.Box.cube(L=10.0)
points = np.random.uniform(-5, 5, size=(1000, 3))
rdf = freud.density.RDF(bins=100, r_max=4.5)    # r_min=0 default
rdf.compute(system=(box, points))               # rdf.bin_centers, rdf.rdf
```

Chain `compute()` over frames with `reset=False` to accumulate trajectory statistics. Cross-check dilute-limit results against the Ornstein-Zernike closure in `correlation-math-foundations`.

### Static structure factor S(q)

- **`freud.diffraction.StaticStructureFactorDebye(num_k_values, k_max, k_min=0)`** — Debye formula, O(N²), works for non-periodic systems. Note the constructor takes `num_k_values`, **not** `bins`.
- **`freud.diffraction.StaticStructureFactorDirect(bins, k_max, k_min=0, num_sampled_k_points=0)`** — direct reciprocal-space sum, requires periodic box, O(N·N_q), faster for dense systems.

Cross-check S(q) against `g(r)` via the Fourier transform of `h(r) = g(r)−1` (the Wiener-Khinchin pair in `correlation-math-foundations`).

### Bond-orientational order parameters

```python
# Steinhardt Q_l (l is a single unsigned int, not a list)
q6 = freud.order.Steinhardt(l=6)
q6.compute(system=(box, points), neighbors={"num_neighbors": 12})
# q6.particle_order (N,), q6.order (scalar)

hex_order = freud.order.Hexatic(k=6)            # 2D hexatic
hex_order.compute(system=(box_2d, points_2d))

nematic = freud.order.Nematic()                 # liquid-crystal S_2
nematic.compute(orientations=director_vectors)
```

For crystalline/liquid phase classification, combine `Steinhardt` with `freud.order.SolidLiquid` (Lechner-Dellago dot-product filter).

### Intermediate scattering F(q, t)

The dynamical counterpart to S(q) — key observable for glass relaxation and dynamic heterogeneity. freud v3.5.0 does **not** ship a dedicated `IntermediateScattering` analyzer in the `density` module [re-verified absent 2026-04-11]; roll F(q,t) by hand from trajectory `positions(t)` via density modes ρ(q,t) = Σ_j exp(iq·r_j(t)) with `numpy.fft`, or use `MDAnalysis.analysis.waterdynamics` for water-like systems. Use `F(q*, t)` at the peak of `S(q)` as the alpha relaxation probe and fit `exp(−(t/τ)^β)` for τ.

### Calling freud from Julia via PythonCall.jl

```julia
using PythonCall
freud = pyimport("freud"); np = pyimport("numpy")
box    = freud.box.Box.cube(L=10.0)
points = np.random.uniform(-5, 5, size=pytuple((1000, 3)))
rdf = freud.density.RDF(bins=100, r_max=4.5)
rdf.compute(system=pytuple((box, points)))
g_of_r = pyconvert(Vector{Float64}, rdf.rdf)
```

`pytuple` wrapping is required because `freud` expects a Python tuple for `system=` and PythonCall.jl does not implicitly convert Julia tuples. See `chaos-attractors` for the general handoff pattern.

### Caveats

- **No native Julia equivalent (use `PythonCall.jl` → `freud`); GPU is CuPy-only** — no ROCm/JAX/PyTorch backend, plan a host-side handoff if the surrounding pipeline is JAX.
- **Trajectory readers and box conventions** — freud does not read trajectory files (use `MDAnalysis`/`MDTraj` as frame iterator); `freud.box.Box` triclinic tilt-factor signs (`xy`, `xz`, `yz`) differ from LAMMPS/HOOMD defaults — double-check on import.

See `correlation-computational-methods` for freud's algorithmic side.

## Composition with neighboring skills

- **Analytical correlation-function foundations** (Wiener-Khinchin theorem, Ornstein-Zernike closures, Green's-function derivations, Fourier-Laplace transforms) → `correlation-math-foundations`
- **O(N log N) algorithms and neighbor-list infrastructure** (cell lists, Ewald summation, FFT-based structure factor) → `correlation-computational-methods`
- **Experimental-data reduction** for DLS / SAXS / XPCS / microscopy → `correlation-experimental-data`
- **Underlying stochastic dynamics** that generate the correlations you're measuring → `stochastic-dynamics` (Langevin, Fokker-Planck, SDE solvers)
- **Julia → Python handoff for `freud`** (no native Julia equivalent) → `chaos-attractors` contains the canonical PythonCall.jl pattern added in v3.1.6 Commit B

## Checklist

- [ ] Physical observable identified
- [ ] Appropriate correlation function selected
- [ ] Timescales understood
- [ ] Experimental comparison planned
- [ ] Critical behavior (if applicable) characterized
