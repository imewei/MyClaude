# Physics Validation Checklists

## Molecular Dynamics Validation

### Energy Conservation (NVE Ensemble)
- [ ] Total energy drift < 0.01% over simulation
- [ ] Energy fluctuations Gaussian-distributed
- [ ] No systematic energy trend (up or down)
- [ ] Kinetic + potential energy balance maintained

### Temperature Control (NVT Ensemble)
- [ ] Average temperature within 1% of target
- [ ] Temperature fluctuations < 5% RMS
- [ ] No oscillations or instabilities
- [ ] Equipartition theorem satisfied (3NkT/2 kinetic energy)

### Pressure Control (NPT Ensemble)
- [ ] Average pressure within 5% of target
- [ ] Volume fluctuations reasonable
- [ ] Density converged to equilibrium value
- [ ] No box size oscillations

### Structural Properties
- [ ] Radial distribution function g(r) matches expected structure
- [ ] First peak in g(r) at expected nearest-neighbor distance
- [ ] Coordination number correct
- [ ] No unphysical overlaps (r < σ/2)

### Physical Consistency
- [ ] Momentum conservation (total momentum ≈ 0)
- [ ] Angular momentum conservation
- [ ] Translational invariance (E unchanged by global shift)
- [ ] Rotational invariance (E unchanged by rotation)

## CFD Validation

### Mass Conservation
- [ ] Divergence of velocity field < 10⁻¹⁰ (machine precision)
- [ ] Mass flux through boundaries balanced
- [ ] No artificial sources/sinks
- [ ] Global mass constant over time

### Momentum Conservation
- [ ] Total momentum conserved (periodic BC)
- [ ] Momentum flux balanced at boundaries
- [ ] Forces integrate correctly

### Energy Conservation/Decay
- [ ] Kinetic energy decay rate matches viscous dissipation
- [ ] Energy balance: dE/dt = -ν∫(∇u)² dV
- [ ] Enstrophy cascade for turbulence (2D)
- [ ] Energy cascade for turbulence (3D)

### Boundary Conditions
- [ ] No-slip BC: u = 0 at walls (error < 10⁻⁶)
- [ ] Free-slip BC: ∂u_tangential/∂n = 0
- [ ] Inflow/outflow BC: consistent with mass conservation
- [ ] Pressure BC: correctly specified

### Numerical Stability
- [ ] CFL condition satisfied (Courant number < 1)
- [ ] No oscillations or checkerboard patterns
- [ ] Solution converges with grid refinement
- [ ] Time step convergence verified

### Benchmark Problems
- [ ] Taylor-Green vortex: matches analytical decay
- [ ] Poiseuille flow: parabolic velocity profile
- [ ] Lid-driven cavity: benchmark vortex structure
- [ ] Channel flow: law of the wall (u+ vs y+)

## PINN Validation

### PDE Residual
- [ ] Mean absolute residual < 10⁻³
- [ ] Maximum residual < 10⁻²
- [ ] Residual spatially uniform (no hot spots)
- [ ] Residual decreasing during training

### Boundary Conditions
- [ ] Dirichlet BC error < 10⁻⁵
- [ ] Neumann BC error < 10⁻⁴
- [ ] Robin BC satisfied
- [ ] Periodic BC continuous

### Initial Conditions
- [ ] Initial state error < 10⁻⁴
- [ ] Smooth transition from IC
- [ ] No discontinuities at t=0

### Physical Consistency
- [ ] Solution respects physics (e.g., temperature > 0)
- [ ] Causality preserved (no backward influence)
- [ ] Symmetries respected (if applicable)
- [ ] Conservation laws satisfied

### Numerical Accuracy
- [ ] Converges to analytical solution (if known)
- [ ] Grid-independent (solution doesn't change with more collocation points)
- [ ] Stable under perturbations
- [ ] Generalizes to unseen domain regions

### Training Diagnostics
- [ ] Loss components balanced (no single term dominates)
- [ ] Gradients not vanishing or exploding
- [ ] No mode collapse
- [ ] Validation loss decreasing

## Quantum Computing Validation

### Energy Calculation (VQE)
- [ ] Energy below exact diagonalization result
- [ ] Converges to chemical accuracy (< 1 mHa)
- [ ] Ground state energy variational (upper bound)
- [ ] Energy variance small (<< energy itself)

### Circuit Properties
- [ ] Parameter gradients non-zero (no barren plateaus)
- [ ] Circuit expressiveness sufficient
- [ ] Entanglement entropy appropriate
- [ ] Gate depth manageable for hardware

### Quantum States
- [ ] State normalization |⟨ψ|ψ⟩| = 1
- [ ] Purity Tr(ρ²) ∈ [0,1]
- [ ] Entanglement measures physical
- [ ] Measurement probabilities ∈ [0,1]

## Multi-Physics Validation

### Interface Conditions
- [ ] Continuity of solution across interfaces
- [ ] Flux balance at interfaces
- [ ] No artificial discontinuities
- [ ] Coupling terms physically motivated

### Conservation Across Domains
- [ ] Global energy conserved
- [ ] Mass conserved across coupling
- [ ] Momentum transfer balanced

## Performance Validation

### Computational Efficiency
- [ ] Speedup scales with hardware (weak/strong scaling)
- [ ] Memory usage within bounds
- [ ] No memory leaks
- [ ] Compilation time reasonable

### Numerical Precision
- [ ] Results stable with float32 vs float64
- [ ] No catastrophic cancellation
- [ ] Conditioning acceptable (condition number < 10⁶)
