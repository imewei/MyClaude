---
name: active-matter
version: "1.0.7"
maturity: "5-Expert"
specialization: Active Matter Physics
description: Model active matter including self-propelled particles, flocking, pattern formation, and collective behavior. Use when simulating ABPs, Vicsek model, MIPS, reaction-diffusion systems, or designing bio-inspired materials.
---

# Active Matter & Complex Systems

Model self-propelled particles, pattern formation, and collective behavior.

---

## Active Matter Models

| Model | Dynamics | Application |
|-------|----------|-------------|
| Active Brownian Particles | dr/dt = v₀n̂ + √(2D)ξ | Colloids, bacteria |
| Vicsek | θ_i(t+1) = ⟨θ⟩ + noise | Flocking, swarms |
| MIPS | Activity → phase separation | Bacterial colonies |
| Toner-Tu | Hydrodynamic active fluids | 2D flocking order |

---

## Active Brownian Particles

```python
def abp_step(r, theta, v0, Dr, Dt, dt):
    r += v0 * np.array([np.cos(theta), np.sin(theta)]) * dt
    r += np.sqrt(2*Dt*dt) * np.random.randn(2)
    theta += np.sqrt(2*Dr*dt) * np.random.randn()
    return r, theta
```

**Parameters**: v₀ (propulsion), D_r (rotational diffusion), D_t (translational diffusion)

---

## Pattern Formation

### Reaction-Diffusion (Turing)

```
∂u/∂t = D_u∇²u + f(u,v)
∂v/∂t = D_v∇²v + g(u,v)
```

| Condition | Pattern |
|-----------|---------|
| D_u ≠ D_v | Instability |
| Specific kinetics | Spots, stripes, labyrinths |

### FitzHugh-Nagumo

```
∂u/∂t = D∇²u + u(u-a)(1-u) - v
∂v/∂t = ε(u - γv)
```

Produces: Traveling waves, spirals, excitable media

---

## Collective Behavior

### Boids Rules (Swarm Dynamics)

1. **Cohesion**: Move toward center of mass
2. **Alignment**: Match neighbors' velocity
3. **Separation**: Avoid collisions

### Chemotaxis (Keller-Segel)

```
∂ρ/∂t = D∇²ρ - χ∇·(ρ∇c)
```

Applications: Cell aggregation, wound healing

---

## Computational Methods

| Method | Use Case |
|--------|----------|
| Agent-based | Discrete particles, interactions |
| Continuum PDE | Large-scale patterns |
| Spectral methods | Periodic domains |
| Adaptive mesh | Sharp interfaces |

---

## Materials Applications

| Application | Principle |
|-------------|-----------|
| Active metamaterials | Self-propelling structures |
| Bio-inspired materials | Artificial cilia, self-healing |
| Microfluidic control | Bacteria-driven mixing |
| Swarm robotics | Collective decision-making |

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Choose appropriate model | ABP for particles, PDE for continuum |
| Validate phase behavior | Check for MIPS, flocking transitions |
| Scale correctly | Match length/time scales to physics |
| Test collective emergence | Verify order parameters |

---

## Checklist

- [ ] Model selection matches physics
- [ ] Parameters in realistic regime
- [ ] Phase behavior characterized
- [ ] Collective order measured
- [ ] Boundary conditions appropriate
- [ ] Time scales resolved

---

**Version**: 1.0.5
