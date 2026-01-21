---
name: stochastic-dynamics
version: "1.1.0"
description: Model stochastic dynamics using master equations, Fokker-Planck, Langevin dynamics, and Green-Kubo transport theory. Use when simulating noise-driven systems, calculating transport coefficients, or modeling rare events.
---

# Stochastic Dynamics & Transport

Stochastic processes and transport properties from microscopic dynamics.

## Expert Agent

For stochastic modeling, Langevin simulations, and rare event sampling, delegate to the expert agent:

- **`statistical-physicist`** or **`simulation-expert`**:
  - **`statistical-physicist`**: For theoretical framework, Fokker-Planck equations, and transport theory.
    - *Location*: `plugins/science-suite/agents/statistical-physicist.md`
  - **`simulation-expert`**: For numerical implementation of Langevin dynamics and rare event sampling.
    - *Location*: `plugins/science-suite/agents/simulation-expert.md`

## Framework Selection

| Framework | Use Case | Equation |
|-----------|----------|----------|
| Master Equation | Discrete states | dP_n/dt = Σ[W_{mn}P_m - W_{nm}P_n] |
| Fokker-Planck | Probability evolution | ∂P/∂t = -∂(μF·P)/∂x + D·∂²P/∂x² |
| Langevin | Stochastic trajectories | dx/dt = μF(x) + √(2D)ξ(t) |

## Gillespie Algorithm (Master Equation)

```python
def gillespie_step(state, propensities):
    total_rate = sum(propensities)
    tau = np.random.exponential(1/total_rate)
    reaction = np.random.choice(len(propensities), p=propensities/total_rate)
    return tau, reaction
```

## Langevin Dynamics

```python
def langevin_step(x, F, mu, D, dt):
    # Euler-Maruyama integration
    return x + mu*F(x)*dt + np.sqrt(2*D*dt)*np.random.randn()
```

White noise: ⟨ξ(t)ξ(t')⟩ = δ(t-t')

## Green-Kubo Relations

| Property | Formula |
|----------|---------|
| Diffusion | D = ∫⟨v(t)·v(0)⟩ dt |
| Viscosity | η = (V/kT)∫⟨σ_xy(t)σ_xy(0)⟩ dt |
| Thermal conductivity | κ = (V/kT²)∫⟨J_q(t)·J_q(0)⟩ dt |

**Einstein relations**: D = kT·μ (diffusion-mobility)

## Rare Event Sampling

| Method | Approach |
|--------|----------|
| Forward Flux Sampling | Cross interfaces sequentially |
| Transition Path Sampling | Sample reactive trajectories |
| Metadynamics | History-dependent bias |

## Applications

| System | Method |
|--------|--------|
| Brownian motion | Langevin, overdamped |
| Chemical kinetics | Master equation, Gillespie |
| Polymer dynamics | Langevin (Rouse/Zimm) |
| Transport properties | Green-Kubo, NEMD |
| Glass relaxation | Stress autocorrelation |

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Match timescales | Timestep << relaxation time |
| Overdamped limit | Use when inertia negligible |
| Convergence check | Integrate correlations sufficiently |
| Validate Einstein | Check D = kT·μ relation |
| Sample sufficiently | Long trajectories for statistics |

## Checklist

- [ ] Appropriate framework selected
- [ ] Noise strength correctly parameterized
- [ ] Timestep small enough for stability
- [ ] Correlation functions converged
- [ ] Transport coefficients validated
- [ ] Einstein relations checked
