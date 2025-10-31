---
name: correlation-physical-systems
description: Apply correlation function analysis to condensed matter (spin correlations in Ising/Heisenberg models with critical exponents η and correlation length ξ ~ |T-T_c|^(-ν), electronic density correlations with Friedel oscillations and dynamical structure factor S(q,ω) from EELS/RIXS, density correlations via van Hove function G(r,t) = ⟨ρ(r,t)ρ(0,0)⟩ separating self and distinct parts), soft matter (polymer Rouse/Zimm dynamics with ⟨R²(t)⟩ ~ t^(1/2) or t^(2/3), reptation τ_d for entangled systems, colloidal pair distribution g(r) with DLS f(q,t) intermediate scattering, glass transition dynamic heterogeneity χ₄(t) with growing length ξ₄, KWW stretched exponential φ(t) ~ exp[-(t/τ)^β]), biological systems (protein folding contact map correlations C_ij(t), FRET end-to-end distance dynamics, NMR relaxation with order parameters S², membrane height-height correlations ⟨h(r)h(0)⟩ ~ ln(r) for bending rigidity κ, molecular motor stepping correlations and force-velocity under load), and non-equilibrium systems (active matter velocity correlations C_vv(r) with enhanced diffusion D_eff = D_t + v₀²τ_r/d, dynamic heterogeneity four-point G₄(r,t) for cooperative motion, transfer entropy TE_{X→Y} for causal information flow). Use when mapping physical systems to correlation functions for materials characterization, predicting transport properties (diffusion, viscosity, conductivity) from microscopic dynamics, or connecting experimental observables (scattering, microscopy, spectroscopy) to theoretical predictions.
---

# Physical Systems & Applications

## When to use this skill

- Analyzing spin correlations in magnetic systems: Ising ⟨S_i S_j⟩ for ferromagnets/antiferromagnets, Heisenberg ⟨S_i·S_j⟩ for vector spins, critical scaling C(r) ~ r^(-(d-2+η)) at phase transitions (*.py Monte Carlo simulations, *.jl Julia implementations)
- Computing electronic density correlations for Friedel oscillations with 2k_F modulations, charge density waves at nesting vectors, or dynamical structure factor S(q,ω) from electron energy loss spectroscopy (EELS) or resonant inelastic X-ray scattering (RIXS)
- Analyzing Hubbard model on-site correlations U⟨n_i↑n_i↓⟩ for Mott transitions, antiferromagnetic spin correlations in cuprates, or pairing correlations in high-T_c superconductors
- Computing van Hove correlation G(r,t) = ⟨ρ(r,t)ρ(0,0)⟩ separating self part G_s (single-particle propagator) from distinct part G_d (inter-particle correlations) for liquid structure from neutron/X-ray scattering
- Modeling polymer dynamics: Rouse ⟨R²(t)⟩ ~ t^(1/2) without hydrodynamics, Zimm ⟨R²(t)⟩ ~ t^(2/3) with solvent interactions, reptation τ_d ~ N^3 for entangled melts (*.py polymer simulation codes)
- Analyzing colloidal systems: pair distribution g(r) for hard spheres with oscillations at σ, 2σ, 3σ, DLVO theory for electrostatic+van der Waals interactions, DLS intermediate scattering f(q,t) for short-time vs long-time dynamics
- Studying glass transitions: dynamic heterogeneity χ₄(t) = N[⟨Q(t)²⟩-⟨Q(t)⟩²] with growing correlation length ξ₄ ~ (T-T_g)^(-ν), KWW stretched exponential φ(t) ~ exp[-(t/τ)^β] with β < 1 distribution of relaxation times
- Implementing mode-coupling theory predictions for two-step relaxation: β-relaxation (fast cage rattling) and α-relaxation (slow structural rearrangement) in supercooled liquids approaching T_g
- Analyzing protein folding: contact map correlations C_ij(t) = ⟨q_ij(t)q_ij(0)⟩ for native contact formation/breaking, FRET end-to-end distance dynamics for folding pathways, NMR backbone order parameters S² from local correlations
- Studying membrane fluctuations: height-height correlation ⟨h(r)h(0)⟩ ~ ln(r) for 2D membranes extracting bending rigidity κ and tension σ, lipid lateral diffusion D ~ 1-10 μm²/s with anomalous subdiffusion α < 1 from crowding
- Modeling molecular motors: stepping time correlations detecting memory effects, ATP hydrolysis correlation with mechanical steps, kinesin/myosin head coordination, force-velocity relations under load for efficiency and stall force
- Analyzing active matter: velocity-velocity correlation C_vv(r) = ⟨v(r)·v(0)⟩ for collective motion, enhanced diffusion D_eff = D_t + v₀²τ_r/d from self-propulsion, motility-induced phase separation (MIPS) in bacterial colonies
- Computing four-point susceptibility χ₄(t) for dynamic heterogeneity in active systems, supercooled liquids, or colloidal glasses measuring spatial extent ξ₄ of cooperative motion
- Implementing Toner-Tu hydrodynamics for active fluids: anomalous density fluctuations ⟨(δρ)²⟩ ~ L^(4/5) in 2D, giant number fluctuations, long-range polar order
- Analyzing information transfer: mutual information I(A;B) = Σ P(a,b)log[P(a,b)/(P(a)P(b))] beyond linear correlations, transfer entropy TE_{X→Y} = I(Y_future; X_past | Y_past) for causal inference
- Computing Granger causality for network reconstruction: X Granger-causes Y if past X improves Y prediction, linear approximation to transfer entropy for neural/gene regulatory networks
- Extracting transport coefficients: diffusion D = ∫⟨v(t)·v(0)⟩dt from velocity autocorrelation (Green-Kubo), viscosity η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩dt from stress correlation
- Predicting viscoelastic moduli from polymer stress correlation G(t) = ⟨σ(t)σ(0)⟩ connecting to storage G'(ω) and loss G''(ω) moduli via Fourier transform
- Analyzing critical phenomena: correlation length divergence ξ ~ |T-T_c|^(-ν), critical exponent η from C(r) ~ r^(-(d-2+η)), universal amplitude ratios at phase transitions
- Mapping experimental observables: DLS g₂(τ) to diffusion D, SAXS/SANS I(q) to structure factor S(q), XPCS two-time C(t₁,t₂) to aging/non-stationarity, FCS G(τ) to concentration and binding
- Validating theoretical predictions: compare MD simulation correlations to experimental scattering data, check scaling laws C(r,T) at phase transitions, verify sum rules and FDT

Map physical systems to appropriate correlation functions, bridging theoretical predictions with experimental observables across condensed matter, soft matter, biological, and non-equilibrium systems.

## Condensed Matter Systems

### Spin Correlations

**Ising Model:**
S_i S_j correlation measures magnetic ordering
- C(r) = ⟨S_i S_{i+r}⟩ - ⟨S⟩²
- Critical regime: C(r) ~ r^(-(d-2+η)) for T ≈ T_c
- Correlation length: ξ ~ |T-T_c|^(-ν)

**Heisenberg Model:**
⟨S_i · S_j⟩ for vector spins
- Ferromagnetic: C(r) > 0, long-range order at T < T_c
- Antiferromagnetic: C(r) oscillates, sign changes
- Dynamic structure factor S(q,ω) from neutron scattering

**Applications:**
- Magnetic phase transitions
- Quantum spin liquids: Absence of long-range order
- Spin glass: Frustration, slow dynamics

```python
def ising_correlation_critical(r, xi, eta=0.036):
    """
    Critical Ising correlation (d=3)
    """
    return (r/xi)**(-1+eta) * np.exp(-r/xi)

def analyze_spin_correlation(spins, lattice):
    """
    Compute spin correlation from lattice configuration
    """
    N = len(spins)
    distances = compute_pairwise_distances(lattice)
    
    C_r = []
    for r in np.arange(0, lattice.size/2, 1):
        mask = (distances > r-0.5) & (distances < r+0.5)
        C = np.mean(spins[mask[:,0]] * spins[mask[:,1]])
        C_r.append(C)
    
    return np.array(C_r)
```

### Electronic Correlations

**Density Correlations:**
n(r)n(r') - ⟨n⟩²
- Friedel oscillations: Long-range 2k_F modulations
- Charge density waves: Diverging correlation at nesting vector
- Structure factor S(q) from X-ray/electron scattering

**Dynamical Structure Factor:**
S(q,ω) = ∫ ⟨n(q,t)n(-q,0)⟩ e^(iωt) dt
- Plasmons: Collective charge oscillations
- Single-particle excitations: Electron-hole pairs
- Measured by EELS, RIXS

**Hubbard Model:**
- On-site correlation: U⟨n_i↑ n_i↓⟩
- Mott transition: Correlations suppress metallic behavior
- Antiferromagnetic spin correlations

**Applications:**
- High-T_c superconductors: Pairing correlations
- Topological insulators: Edge state correlations
- Quantum dots: Few-electron correlations

### Density Correlations

**Van Hove Function:**
G(r,t) = ⟨ρ(r,t)ρ(0,0)⟩
- Self part: G_s(r,t) = ⟨δ(r - r_i(t) + r_i(0))⟩, single-particle propagator
- Distinct part: G_d(r,t), correlation between different particles

**Static Limit:**
g(r) = lim_{t→∞} G(r,t)/ρ
- Pair distribution function
- Radial symmetry for liquids

**Applications:**
- Liquid structure from neutron/X-ray scattering
- Supercooled liquids: Cage effect, dynamic arrest
- Glasses: Non-ergodicity, aging

```python
def van_hove_function(positions_t, positions_0, rmax, dr):
    """
    Compute van Hove correlation G(r,t)
    """
    N = len(positions_t)
    displacements = positions_t - positions_0
    
    # Self part
    r_bins = np.arange(0, rmax, dr)
    G_s, _ = np.histogram(np.linalg.norm(displacements, axis=1), 
                           bins=r_bins)
    G_s = G_s / (N * 4*np.pi*r_bins[:-1]**2 * dr)
    
    # Distinct part (all pairs)
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            dr_ij = positions_t[i] - positions_0[j]
            pairs.append(np.linalg.norm(dr_ij))
    
    G_d, _ = np.histogram(pairs, bins=r_bins)
    G_d = G_d / (N * 4*np.pi*r_bins[:-1]**2 * dr)
    
    return r_bins[:-1], G_s, G_d
```

## Soft Matter Systems

### Polymer Dynamics

**Rouse Model:**
⟨R²(t)⟩ ~ t^(1/2) for t < τ_R (Rouse time)
- No hydrodynamic interactions
- Normal mode relaxation: τ_p ~ p^(-2)

**Zimm Model:**
⟨R²(t)⟩ ~ t^(2/3) with hydrodynamics
- Faster relaxation than Rouse
- τ_Z ~ η_s N^(3ν) (solvent viscosity, chain length)

**End-to-End Correlation:**
C(t) = ⟨R(t)·R(0)⟩
- Decay characterizes chain relaxation
- Connection to viscoelasticity G(t)

**Reptation (Entangled Polymers):**
- Tube model: Lateral confinement
- Relaxation times: τ_e (entanglement) < τ_R (Rouse along tube) < τ_d (disentanglement)
- C(t) ~ t^(-1/4) for τ_e < t < τ_d

**Applications:**
- Polymer melts: Rheology from stress correlations
- Polymer solutions: Concentration dependence of dynamics
- Single-chain tracking: Fluorescence microscopy

```python
def rouse_correlation(t, N, b, kT):
    """
    Rouse model end-to-end correlation
    """
    tau_R = b**2 * N**2 / (3*np.pi**2 * kT)
    C = np.exp(-t / tau_R) * (1 - np.exp(-N**2 * t / tau_R))
    return C
```

### Colloidal Interactions

**Pair Distribution Function g(r):**
- Hard spheres: Oscillations at σ, 2σ, 3σ...
- Attractive colloids: Enhanced g(r) at contact
- Depletion forces: g(r) from small depletant

**Dynamic Correlations:**
- DLS: f(q,t) = ⟨ρ(q,t)ρ(-q,0)⟩ intermediate scattering
- Short-time: f(q,t) ≈ exp(-Dq²t), free diffusion
- Long-time: Caging, cooperative relaxation

**DLVO Theory:**
- Electrostatic + van der Waals interactions
- Predict g(r), structure factor S(q)
- Stability diagrams: Aggregation vs stable suspensions

**Applications:**
- Colloidal glasses: Arrested dynamics
- Gels: Percolated networks, fractal structure
- Crystals: Bragg peaks in S(q)

### Glass Transitions

**Dynamic Heterogeneity:**
χ₄(t) = N[⟨Q(t)²⟩ - ⟨Q(t)⟩²]
- Q(t) = ∑_i w(|r_i(t) - r_i(0)|), overlap function
- Growing length scale ξ₄ near T_g
- Cooperative motion: "Fast" and "slow" regions

**Non-Exponential Relaxation:**
φ(t) = exp[-(t/τ)^β], KWW stretched exponential
- β < 1 indicates distribution of relaxation times
- β → 0 as T → T_g

**Mode-Coupling Theory:**
φ_q(t) obeys closed equation with memory kernel
- Predicts two-step relaxation: β-relaxation (fast), α-relaxation (slow)
- Critical temperature T_c > T_g

**Applications:**
- Supercooled liquids: Approach to glass transition
- Polymer glasses: Fragility, T_g prediction
- Granular materials: Jamming transition

```python
def four_point_susceptibility(positions, dt, w_cutoff=0.3):
    """
    Compute χ₄(t) for dynamic heterogeneity
    """
    N = len(positions[0])
    times = len(positions)
    
    chi4 = []
    for t in range(1, times):
        displacements = positions[t] - positions[0]
        w = np.exp(-displacements**2 / (2*w_cutoff**2))
        Q = np.sum(w)
        chi4.append(N * (np.var(w)))
    
    return np.array(chi4)
```

## Biological Systems

### Protein Folding Dynamics

**Contact Map Correlations:**
C_ij(t) = ⟨q_ij(t)q_ij(0)⟩
- q_ij = 1 if residues i,j in contact
- Formation/breaking of native contacts
- Folding pathways from correlation analysis

**FRET Correlations:**
- Distance-dependent energy transfer
- End-to-end distance dynamics
- Folding/unfolding rates

**NMR Relaxation:**
- Backbone dynamics from correlation times
- Order parameters S² from local correlations
- Timescales: ps (fast motions) to ms (conformational changes)

**Applications:**
- Protein folding mechanisms: Nucleation vs diffusion-collision
- Intrinsically disordered proteins: Ensemble dynamics
- Allostery: Long-range coupling via correlations

### Membrane Fluctuations

**Height-Height Correlation:**
⟨h(r)h(0)⟩ ~ ln(r) for 2D membranes
- Bending rigidity κ from decay
- Tension σ modifies correlations

**Lipid Diffusion:**
- Lateral diffusion: D ~ 1-10 μm²/s
- Confined diffusion in domains
- Anomalous diffusion: α < 1 from crowding

**Membrane Proteins:**
- Protein-lipid correlations
- Clustering: g(r) for protein pairs
- Functional coupling via membrane-mediated interactions

**Applications:**
- Membrane mechanics: κ, σ from fluctuation analysis
- Lipid rafts: Domain formation, phase separation
- Ion channels: Gating correlations, cooperative opening

```python
def height_correlation_membrane(positions, box_size):
    """
    Compute height-height correlation for membrane
    """
    # Extract z-coordinates as height field
    h = positions[:, 2] - positions[:, 2].mean()
    
    # Spatial correlation
    r_bins = np.logspace(-1, np.log10(box_size/2), 50)
    C_h = []
    
    for r in r_bins:
        pairs = find_pairs_at_distance(positions[:, :2], r, dr=r*0.1)
        C = np.mean([h[i]*h[j] for i, j in pairs])
        C_h.append(C)
    
    return r_bins, np.array(C_h)
```

### Molecular Motor Correlations

**Stepping Dynamics:**
- Correlation in step times: Memory effects
- ATP hydrolysis correlation with steps
- Coordination between motor heads (kinesin, myosin)

**Force-Velocity Relations:**
- Correlations under load
- Stall force from velocity fluctuations
- Efficiency from work-heat correlations

**Collective Motors:**
- Cargo transport by multiple motors
- Tug-of-war: Opposing motor correlations
- Traffic jams: Density correlations on tracks

**Applications:**
- Intracellular transport efficiency
- Muscle contraction: Cross-bridge kinetics
- DNA replication: Polymerase coordination

## Non-Equilibrium Systems

### Active Matter

**Enhanced Velocity Correlations:**
C_vv(r) = ⟨v(r)·v(0)⟩
- Positive correlations: Collective motion
- Long-range order in 2D flocking
- Giant number fluctuations

**Active Brownian Particles:**
- Enhanced diffusion: D_eff = D_t + v₀²τ_r/d
- Effective temperature: T_eff > T_bath
- Motility-induced phase separation (MIPS)

**Toner-Tu Hydrodynamics:**
- Coarse-grained equations for active fluids
- Anomalous density fluctuations
- Correlation length from activity

**Applications:**
- Bacterial suspensions: Collective swimming
- Active colloids: Self-propelled particles
- Flocking: Birds, fish, robots

```python
def velocity_correlation_active(velocities, positions, rmax, dr):
    """
    Compute velocity-velocity correlation for active matter
    """
    N = len(velocities)
    r_bins = np.arange(0, rmax, dr)
    C_vv = np.zeros(len(r_bins)-1)
    counts = np.zeros(len(r_bins)-1)
    
    for i in range(N):
        for j in range(i+1, N):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            bin_idx = int(r_ij / dr)
            if bin_idx < len(C_vv):
                C_vv[bin_idx] += np.dot(velocities[i], velocities[j])
                counts[bin_idx] += 1
    
    C_vv /= (counts + 1e-10)
    return r_bins[:-1], C_vv
```

### Dynamic Heterogeneity

**Four-Point Correlations:**
G₄(r,t) = ⟨δφ(r,t)δφ(0,t)δφ(0,0)δφ(r,0)⟩
- δφ: Local density or mobility
- Measures spatial extent of cooperativity
- Growing length ξ₄ ~ (T-T_g)^(-ν)

**Fast/Slow Regions:**
- Bimodal distribution of local dynamics
- Spatially correlated: "Islands" of mobility
- Timescale separation

**Dynamic Facilitation:**
- Relaxation facilitated by neighboring mobile regions
- Kinetically constrained models
- Avalanches, intermittency

**Applications:**
- Supercooled liquids: Glass transition mechanism
- Polymer glasses: Spatially heterogeneous dynamics
- Granular materials: Shear bands, jamming

### Information Transfer & Causality

**Mutual Information:**
I(A;B) = ∑ P(a,b) log[P(a,b)/(P(a)P(b))]
- Quantifies correlation beyond linear
- Distinguishes direct vs indirect coupling

**Transfer Entropy:**
TE_{X→Y} = I(Y_future; X_past | Y_past)
- Directional information flow
- Causal inference from time series
- Distinguishes driver vs driven

**Granger Causality:**
- X Granger-causes Y if past X improves Y prediction
- Linear approximation to TE
- Network reconstruction

**Applications:**
- Neural networks: Effective connectivity
- Gene regulatory networks: Causal interactions
- Financial markets: Lead-lag relationships

```python
def transfer_entropy(X, Y, k=1, delay=1):
    """
    Compute transfer entropy TE_{X→Y}
    """
    from scipy.stats import entropy
    
    # Create lagged variables
    Y_future = Y[delay:]
    Y_past = Y[:-delay]
    X_past = X[:-delay]
    
    # Joint and conditional entropies
    H_Y_future = entropy(np.histogram(Y_future, bins=10)[0])
    H_Y_given_Y_past = entropy(np.histogram2d(Y_future, Y_past, bins=10)[0].flatten())
    H_Y_given_XY_past = entropy(np.histogram2d(Y_future, 
                                  np.column_stack([Y_past, X_past]), 
                                  bins=10)[0].flatten())
    
    TE = H_Y_given_Y_past - H_Y_given_XY_past
    return TE
```

## System-Specific Analysis Strategies

**Condensed Matter:**
- Use symmetry to reduce computation (crystal symmetry, isotropy)
- Finite-size scaling for critical phenomena
- Quantum correlations: Density matrix methods

**Soft Matter:**
- Multi-scale analysis: Monomer → chain → macroscopic
- Coarse-graining: Effective interactions from correlations
- Viscoelasticity: Stress correlation → moduli

**Biological Systems:**
- Single-molecule vs ensemble averages
- Time-resolved: Pump-probe, stopped-flow
- Spatial resolution: Super-resolution microscopy

**Non-Equilibrium:**
- Time-dependent correlations: C(t,t') not just C(t-t')
- Aging: Correlation depends on observation time
- Effective temperature from FDT violations

## Best Practices

- **System identification**: Match correlation type to physical observables
- **Timescale separation**: Identify fast (microscopic) vs slow (collective) dynamics
- **Experimental connection**: Map theoretical correlations to measurable quantities
- **Model validation**: Compare correlation predictions with experimental data

References for advanced applications: quantum correlations, topological order, many-body localization.
