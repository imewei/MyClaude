---
name: active-matter
description: Model active matter and complex systems including self-propelled particles, flocking, pattern formation, and collective behavior with Active Brownian Particles (ABPs), Vicsek model for alignment dynamics, motility-induced phase separation (MIPS), reaction-diffusion systems (Turing patterns, FitzHugh-Nagumo), collective behavior (swarm dynamics, chemotaxis, cytoskeletal dynamics), and Toner-Tu hydrodynamics for active fluids. Use when studying bacterial colonies with phase separation, self-organizing materials (active metamaterials, bio-inspired designs), microfluidic control, pattern formation in developmental biology, swarm robotics, or designing autonomous self-healing materials with emergent collective behavior.
---

# Active Matter & Complex Systems

## When to use this skill

- Modeling self-propelled particles with Active Brownian Particle (ABP) dynamics including translational diffusion, rotational diffusion, and self-propulsion speed (*.py, *.jl simulation codes)
- Simulating motility-induced phase separation (MIPS) in bacterial colonies, active colloids, or cell sorting systems to predict dense/dilute phase coexistence
- Implementing Vicsek model for flocking dynamics, collective motion, and polar order phase transitions in bird flocks, fish schools, or robot swarms
- Analyzing reaction-diffusion systems with Turing instability for pattern formation in chemical systems (Belousov-Zhabotinsky reactions), animal coat patterns (spots, stripes, labyrinths), or morphogen gradients in developmental biology
- Designing FitzHugh-Nagumo excitable media for traveling waves, spiral patterns, or cardiac tissue modeling (*.m MATLAB, *.py Python implementations)
- Modeling chemotaxis with Keller-Segel equations for cell aggregation, bacterial chemotactic response, or collective migration in wound healing
- Simulating cytoskeletal dynamics including actin polymerization/depolymerization, motor-driven transport, force generation in cells, or cell division mechanics
- Analyzing swarm dynamics using Boids rules (cohesion, alignment, separation) for autonomous robot coordination, UAV formations, or crowd dynamics
- Implementing Toner-Tu hydrodynamics for active fluids with long-range order in 2D flocking systems, giant number fluctuations, or anomalous density correlations
- Designing active metamaterials with self-propelling mechanical structures, autonomous assembly, reconfiguration capabilities, or collective sensing/response
- Creating bio-inspired materials including artificial cilia for fluid transport, self-healing through active reorganization, or adaptive camouflage patterns
- Optimizing microfluidic control using bacteria-driven mixing, chemostat design for bacterial growth, or drug delivery systems with active transport
- Studying pattern formation in developmental biology including gastrulation, somitogenesis, digit formation, or left-right asymmetry determination
- Analyzing collective behavior in social systems including pedestrian dynamics, traffic flow, opinion dynamics, or epidemic spreading
- Implementing agent-based simulations for active particle systems with interactions, alignment rules, or activity-dependent dynamics (*.py, *.cpp codes)
- Solving continuum simulations for pattern-forming PDEs using finite differences, spectral methods, or adaptive mesh refinement (*.py SciPy, *.jl Julia implementations)
- Predicting phase transitions in active matter systems including flocking transition density, critical noise threshold, or MIPS critical activity
- Designing self-organizing materials for soft robotics, reconfigurable materials, autonomous repair mechanisms, or responsive architectures
- Analyzing topological active matter including topological defects in flocking systems, active turbulence, or active nematics with +1/2 and -1/2 defects
- Modeling information processing in swarms including distributed decision-making, quorum sensing in bacterial colonies, or collective computation in insect colonies
- Studying active matter in confined geometries including channel flows, circular boundaries inducing vortices, or microfluidic chambers with pattern formation

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
