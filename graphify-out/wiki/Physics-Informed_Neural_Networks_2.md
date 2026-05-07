# Physics-Informed Neural Networks

> 12 nodes · cohesion 0.18

## Key Concepts

- **Pinn Heat Equation** (9 connections) — `plugins/science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **compute_pde_residual** (4 connections) — `plugins/science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **analytical_solution** (3 connections) — `plugins/science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **pinn_loss** (3 connections) — `plugins/science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **sample_collocation_points** (3 connections) — `plugins/science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **sample_initial_conditions** (3 connections) — `plugins/science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **Sample boundary points: x=0 and x=1** (3 connections) — `science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **Physics-informed loss = PDE residual + BC + IC** (2 connections) — `science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **Compute PDE residual: ∂u/∂t - α∂²u/∂x²** (2 connections) — `science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **Sample random points in domain for PDE residual** (2 connections) — `science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **Sample initial condition: t=0, u(x,0)=sin(πx)** (2 connections) — `science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- **Analytical solution: u(x,t) = exp(-απ²t)sin(πx)** (2 connections) — `science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`

## Relationships

- [[HMC-ECS Advanced Sampling]] (6 shared connections)
- [[JAX Computing]] (6 shared connections)
- [[Plugin Hooks]] (1 shared connections)
- [[Physics-Informed Neural Networks]] (1 shared connections)

## Source Files

- `plugins/science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`
- `science-suite/skills/jax-physics-applications/scripts/pinn_heat_equation.py`

## Audit Trail

- EXTRACTED: 38 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*