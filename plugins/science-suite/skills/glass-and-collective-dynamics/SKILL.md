---
name: glass-and-collective-dynamics
description: Glass physics, jamming, and collective phenomena in disordered/soft-matter systems — random landscapes, aging, cooperative dynamics, and percolation theory. Use when analyzing glassy relaxation, jamming transitions, aging dynamics, cooperative/collective particle motion, or percolation thresholds in filler networks or disordered media.
---

# Glass & Collective Dynamics

Cooperative and collective phenomena in disordered and glassy soft-matter systems.

## Expert Agent

- **`statistical-physicist`** — Glass physics, jamming, percolation, and collective-dynamics theory.

## Glass & Jamming

- **Glass transition**: dynamical arrest without long-range structural order — relaxation times diverge (often described phenomenologically by Vogel-Fulcher-Tammann, `τ = τ₀ exp(DT₀/(T-T₀))`) as temperature approaches Tg from above.
- **Jamming**: analogous arrest driven by density/packing rather than temperature — a jammed packing is mechanically rigid despite being disordered, characterized by the jamming point `φⱼ` (random close packing for frictionless spheres, ≈0.64 in 3D).
- **Aging**: glassy/jammed systems never fully equilibrate on experimental timescales — their properties (relaxation time, mechanical response) continue to evolve ("age") with waiting time since preparation/quench, breaking time-translation invariance.

## Cooperative & Collective Dynamics

Dynamics in dense/disordered systems are often dominated by cooperative rearrangements (groups of particles moving together) rather than independent single-particle motion — this is the microscopic origin of the dramatic slowdown near the glass transition despite only modest structural changes. Dynamical heterogeneity (spatially varying local relaxation rates) is the key diagnostic, measured via multi-point correlation functions (e.g. the four-point susceptibility χ₄).

## Percolation Theory

- **Percolation threshold `pc`**: the critical occupation/connection probability at which a system-spanning cluster first appears.
- **Universality**: near `pc`, cluster-size distributions and connectivity properties follow power laws with universal critical exponents depending only on dimensionality and percolation type (site vs. bond), not microscopic details.
- **Application to filler networks**: a nanocomposite's filler particles form a percolating network above their own geometric percolation threshold — this is the statistical-mechanics basis for the property jump referenced (but not derived) in `science-suite:nanocomposites-and-adaptive-materials`.

## Routing Within This Skill

| Question | Answer here covers |
|----------|----------------------|
| "Why does relaxation slow down approaching Tg?" | Glass transition phenomenology, VFT law |
| "What's the jamming point for this packing?" | Jamming transition, random close packing |
| "Why hasn't this system's properties stabilized?" | Aging, breaking of time-translation invariance |
| "Are these particles moving cooperatively?" | Dynamical heterogeneity, four-point susceptibility |
| "What filler fraction gives a percolating network?" | Percolation threshold, critical exponents |

## Delegation

For applying a percolation threshold to nanocomposite property prediction (not deriving the threshold itself), see `science-suite:nanocomposites-and-adaptive-materials`. For active-matter collective phenomena specifically (self-propelled particles, flocking), see `science-suite:active-matter` — that skill covers a different (non-equilibrium, driven) mechanism for collective behavior than the equilibrium/quenched-disorder mechanisms here.
