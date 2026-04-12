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

## Python `freud` ecosystem

The Glotzer group's `freud` (v3.5.0) is the production toolkit for physical-system correlations — RDF, structure factors, bond-orientational order. No native Julia equivalent; use `PythonCall.jl` (see `chaos-attractors`). Install: `pip install freud-analysis`.

| Task | API |
|---|---|
| RDF g(r) | `freud.density.RDF(bins, r_max)` |
| S(q) Debye | `freud.diffraction.StaticStructureFactorDebye(num_k_values, k_max)` |
| S(q) Direct | `freud.diffraction.StaticStructureFactorDirect(bins, k_max)` |
| Steinhardt Q_l | `freud.order.Steinhardt(l=6)` |
| Hexatic / Nematic | `freud.order.Hexatic(k=6)` / `freud.order.Nematic()` |
| F(q,t) | Roll by hand via density modes + `numpy.fft` (no freud builtin) |

```python
import freud, numpy as np
box = freud.box.Box.cube(L=10.0)
rdf = freud.density.RDF(bins=100, r_max=4.5)
rdf.compute(system=(box, points))  # rdf.bin_centers, rdf.rdf

q6 = freud.order.Steinhardt(l=6)
q6.compute(system=(box, points), neighbors={"num_neighbors": 12})
```

**Julia interop**: `pytuple` wrapping required for `system=` arg. `freud.box.Box` tilt-factor signs differ from LAMMPS/HOOMD defaults. See `correlation-computational-methods` for algorithmic details.

## Composition with neighboring skills

- Analytical foundations (Wiener-Khinchin, Ornstein-Zernike, Green's functions) → `correlation-math-foundations`
- O(N log N) algorithms, neighbor lists, Ewald → `correlation-computational-methods`
- Experimental data reduction (DLS/SAXS/XPCS) → `correlation-experimental-data`
- Stochastic dynamics generating correlations → `stochastic-dynamics`
- Julia→Python `freud` handoff pattern → `chaos-attractors`

## Checklist

- [ ] Physical observable identified
- [ ] Appropriate correlation function selected
- [ ] Timescales understood
- [ ] Experimental comparison planned
- [ ] Critical behavior (if applicable) characterized
