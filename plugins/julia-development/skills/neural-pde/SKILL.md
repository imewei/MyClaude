---
name: neural-pde
version: "1.0.7"
maturity: "5-Expert"
specialization: Physics-Informed Neural Networks
description: Solve PDEs with physics-informed neural networks using NeuralPDE.jl. Use when solving PDEs with neural networks, enforcing boundary conditions, or combining ML with physics.
---

# NeuralPDE.jl (PINNs)

Physics-informed neural networks for solving PDEs.

---

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

---

## Use Cases

| Use Case | When to Apply |
|----------|---------------|
| Complex PDEs | Traditional methods struggle |
| Inverse problems | Unknown parameters |
| High-dimensional | Curse of dimensionality |
| Irregular domains | Meshing difficult |

---

## Checklist

- [ ] PDE system defined
- [ ] Boundary conditions specified
- [ ] Neural network architecture chosen
- [ ] Training strategy selected
- [ ] Solution validated

---

**Version**: 1.0.5
