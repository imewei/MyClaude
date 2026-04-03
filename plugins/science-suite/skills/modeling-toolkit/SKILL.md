---
name: modeling-toolkit
maturity: "5-Expert"
specialization: Symbolic Modeling
description: Define symbolic differential equations with ModelingToolkit.jl for automatic simplification and code generation. Use when building complex mathematical models declaratively.
---

# ModelingToolkit.jl

## Expert Agent

For symbolic differential equation modeling with ModelingToolkit.jl, delegate to:

- **`julia-pro`**: Julia SciML ecosystem and symbolic modeling workflows.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

Symbolic modeling with automatic equation simplification.

---

## Basic Pattern

```julia
using ModelingToolkit, DifferentialEquations

@variables t x(t) y(t)
@parameters α β
D = Differential(t)

eqs = [D(x) ~ α * x, D(y) ~ -β * y]
@named sys = ODESystem(eqs, t)
sys_simple = structural_simplify(sys)

prob = ODEProblem(sys_simple, [x => 1.0, y => 1.0],
                  (0.0, 10.0), [α => 0.5, β => 0.3])
sol = solve(prob, Tsit5())
```

---

## Key Features

| Feature | Benefit |
|---------|---------|
| Symbolic variables | @variables, @parameters |
| Automatic simplification | structural_simplify() |
| Code generation | Efficient numerical code |
| Component-based | Reusable model parts |

---

## Variable and Parameter Declaration

```julia
using ModelingToolkit

@variables t
@variables x(t) y(t) z(t)           # State variables (time-dependent)
@parameters α β γ δ                  # Constants (estimated or fixed)
D = Differential(t)                   # First derivative operator
D2 = Differential(t)^2               # Second derivative operator
```

## DAE Systems (Differential-Algebraic)

```julia
@variables t x(t) y(t) T(t)
@parameters k cp
D = Differential(t)

eqs = [
    D(x) ~ -k * x * y,              # ODE: reaction kinetics
    D(y) ~ k * x * y - γ * y,       # ODE: product formation
    0 ~ x + y + T - 1.0,            # Algebraic constraint: conservation
]

@named sys = ODESystem(eqs, t)
sys_simple = structural_simplify(sys)  # Reduces DAE index automatically
```

## Component-Based Modeling

```julia
using ModelingToolkit

function Resistor(; name, R = 1.0)
    @variables t v(t) i(t)
    @parameters R = R
    eqs = [v ~ R * i]
    ODESystem(eqs, t, [v, i], [R]; name)
end

function Capacitor(; name, C = 1.0)
    @variables t v(t) i(t)
    @parameters C = C
    D = Differential(t)
    eqs = [D(v) ~ i / C]
    ODESystem(eqs, t, [v, i], [C]; name)
end

# Compose into a circuit
@named resistor = Resistor(R = 100.0)
@named capacitor = Capacitor(C = 1e-6)
connections = [resistor.v ~ capacitor.v, resistor.i ~ -capacitor.i]
@named circuit = ODESystem(connections, t, systems = [resistor, capacitor])
circuit_simple = structural_simplify(circuit)
```

## Structural Simplification

| Transformation | Effect |
|----------------|--------|
| `structural_simplify` | Eliminate algebraic variables, reduce DAE index |
| `alias_elimination` | Remove trivially equal variables |
| `tearing` | Solve for algebraic variables analytically |
| `dae_index_lowering` | Convert high-index DAE to index-1 form |

## Code Generation

```julia
# Generate optimized Julia function from symbolic system
prob = ODEProblem(sys_simple, [x => 1.0, y => 0.5],
                  (0.0, 100.0), [α => 0.1, β => 0.05])

# ModelingToolkit auto-generates sparse Jacobian
prob_sparse = ODEProblem(sys_simple, u0, tspan, p, jac = true, sparse = true)
sol = solve(prob_sparse, TRBDF2())  # Exploit sparsity for stiff systems
```

## Checklist

- [ ] Variables declared as functions of independent variable (`x(t)`)
- [ ] Parameters separated from state variables
- [ ] `structural_simplify` applied before problem construction
- [ ] Sparse Jacobian enabled for systems with >10 equations
- [ ] Component models tested independently before composition
- [ ] Conservation laws verified in algebraic constraints
- [ ] Units consistent across all equations (use Unitful.jl if needed)

---

**Version**: 1.0.5
