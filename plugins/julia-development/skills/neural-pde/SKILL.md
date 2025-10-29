---
name: neural-pde
description: Physics-informed neural networks (PINNs) with NeuralPDE.jl for solving PDEs with neural networks and incorporating physical constraints.
---

# Neural PDE (Physics-Informed Neural Networks)

Solve PDEs using physics-informed neural networks with NeuralPDE.jl.

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
