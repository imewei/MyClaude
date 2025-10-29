---
name: active-matter
description: Model active matter and complex systems including self-propelled particles, flocking, pattern formation, and collective behavior. Use when studying motility-induced phase separation, bacterial colonies, cytoskeletal dynamics, or reaction-diffusion systems with Turing patterns for bio-inspired materials design.
---

# Active Matter & Complex Systems

Model active matter, pattern formation, and emergent collective behavior in self-propelled and reaction-diffusion systems.

## Active Matter Fundamentals

### Active Brownian Particles (ABPs)
**Dynamics:**
dr/dt = v₀n̂ + √(2D_t)ξ_t
dn̂/dt = √(2D_r)ξ_r × n̂
- v₀: self-propulsion speed
- D_r: rotational diffusion
- D_t: translational diffusion

**Simulation:**
```python
def abp_step(r, theta, v0, Dr, Dt, dt):
    r += v0 * np.array([np.cos(theta), np.sin(theta)]) * dt + np.sqrt(2*Dt*dt) * np.random.randn(2)
    theta += np.sqrt(2*Dr*dt) * np.random.randn()
    return r, theta
```

### Motility-Induced Phase Separation (MIPS)
- High activity → dense/dilute phase coexistence
- No equilibrium analog (requires activity)
- Applications: Bacterial colonies, cell sorting

### Vicsek Model (Flocking)
**Alignment rule:**
θ_i(t+1) = ⟨θ_j⟩_{j∈neighbors} + noise
- Polar order at high density/low noise
- Phase transition to collective motion

## Pattern Formation

### Reaction-Diffusion Systems
**General form:**
∂u/∂t = D_u∇²u + f(u,v)
∂v/∂t = D_v∇²v + g(u,v)

**Turing Instability:**
- Diffusion-driven instability
- Requires: D_u ≠ D_v, specific reaction kinetics
- Produces: Spots, stripes, labyrinths

**Examples:**
- Chemical patterns (Belousov-Zhabotinsky)
- Animal coat patterns
- Morphogen gradients in development

### FitzHugh-Nagumo Model
∂u/∂t = D∇²u + u(u-a)(1-u) - v
∂v/∂t = ε(u - γv)
- Excitable media
- Traveling waves, spirals

## Collective Behavior

### Swarm Dynamics
**Boids rules:**
1. Cohesion: Move toward center of mass
2. Alignment: Match neighbors' velocity
3. Separation: Avoid collisions

**Applications**: Fish schools, bird flocks, robot swarms

### Chemotaxis
**Keller-Segel Model:**
∂ρ/∂t = D∇²ρ - χ∇·(ρ∇c)
∂c/∂t = D_c∇²c + αρ - βc
- ρ: cell density, c: chemoattractant
- Aggregation instability

### Cytoskeletal Dynamics
- Actin polymerization/depolymerization
- Motor-driven transport
- Force generation in cells

## Theoretical Frameworks

### Toner-Tu Hydrodynamics
**Hydrodynamic equations for active fluids:**
∂ρ/∂t + ∇·(ρv) = 0
∂v/∂t + (v·∇)v = -α∇ρ + ... + noise
- Long-range order possible in 2D
- Giant number fluctuations

### Active Field Theory
- Coarse-grained descriptions
- Symmetries and conservation laws
- Effective equations of motion

## Computational Methods

### Agent-Based Simulations
```python
class ActiveParticle:
    def __init__(self, r, theta, v0, Dr):
        self.r = r
        self.theta = theta
        self.v0 = v0
        self.Dr = Dr

    def update(self, dt):
        self.r += self.v0 * np.array([np.cos(self.theta), np.sin(self.theta)]) * dt
        self.theta += np.sqrt(2*self.Dr*dt) * np.random.randn()
        self.apply_interactions()  # With other particles
```

### Continuum Simulations
- Solve PDEs with finite differences/elements
- Spectral methods for periodic domains
- Adaptive mesh refinement for sharp interfaces

## Materials Design

**Active Metamaterials:**
- Self-propelling mechanical structures
- Autonomous assembly/reconfiguration
- Collective sensing and response

**Bio-Inspired Materials:**
- Artificial cilia for fluid transport
- Self-healing through active reorganization
- Adaptive camouflage patterns

**Microfluidic Control:**
- Bacteria-driven mixing
- Chemotat design
- Drug delivery systems

References for advanced topics: topological active matter, active turbulence, information processing in swarms.
