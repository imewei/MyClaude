---
name: neural-pde
description: Master physics-informed neural networks (PINNs) with NeuralPDE.jl for solving PDEs using deep learning while enforcing physical constraints and boundary conditions. Use when solving partial differential equations with neural networks (.jl files with PDESystem, PhysicsInformedNN), enforcing boundary conditions and initial conditions in PINNs, working with time-dependent and spatial PDE problems, incorporating physical laws as loss constraints, using automatic differentiation for PDE residuals, training with Flux.jl chains (Dense, Chain), selecting training strategies (QuadratureTraining, GridTraining), or combining machine learning with physics. Essential for solving complex PDEs where traditional methods struggle, inverse problems, and scientific machine learning applications.
---

# Neural PDE (Physics-Informed Neural Networks)

Solve PDEs using physics-informed neural networks with NeuralPDE.jl.

## When to use this skill

- Solving partial differential equations with neural networks
- Enforcing boundary conditions and initial conditions in PINNs
- Working with time-dependent and spatial PDE problems
- Incorporating physical laws as neural network loss constraints
- Using automatic differentiation for computing PDE residuals
- Training physics-informed models with Flux.jl
- Selecting training strategies (QuadratureTraining, GridTraining, StochasticTraining)
- Solving inverse problems with PINNs
- Combining data-driven and physics-based modeling
- Handling complex geometries with neural networks
- Solving high-dimensional PDEs where traditional methods fail

## PINN Pattern
```julia
using NeuralPDE, Flux

@parameters t x
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

eq = Dt(u(t,x)) ~ Dxx(u(t,x))
bcs = [u(0,x) ~ cos(π*x), u(t,0) ~ 0, u(t,1) ~ 0]
domains = [t ∈ IntervalDomain(0.0, 1.0), x ∈ IntervalDomain(0.0, 1.0)]

chain = Chain(Dense(2, 16, σ), Dense(16, 1))
discretization = PhysicsInformedNN(chain, QuadratureTraining())

@named pde_system = PDESystem(eq, bcs, domains, [t,x], [u])
prob = discretize(pde_system, discretization)
result = solve(prob, Adam(0.01), maxiters=5000)
```

## Resources
- **NeuralPDE.jl**: https://docs.sciml.ai/NeuralPDE/stable/
