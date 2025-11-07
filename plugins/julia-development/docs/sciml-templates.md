# SciML Project Templates

Comprehensive code templates for Scientific Machine Learning projects with DifferentialEquations.jl, ModelingToolkit.jl, and Optimization.jl.

## Table of Contents

- [ODE Templates](#ode-templates)
- [PDE Templates](#pde-templates)
- [SDE Templates](#sde-templates)
- [Optimization Templates](#optimization-templates)

---

## ODE Templates

### Direct API Template

```julia
# Auto-generated ODE template
using DifferentialEquations
using Plots

# System: <description from user>
function system_dynamics!(du, u, p, t)
    # State variables: u[1], u[2], ...
    # Parameters: p[1], p[2], ...

    # TODO: Define derivatives
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end

# Initial conditions
u0 = [1.0, 1.0]  # TODO: Set appropriate initial values

# Time span
tspan = (0.0, 10.0)  # TODO: Adjust time range

# Parameters
p = [1.5, 1.0, 3.0, 1.0]  # TODO: Set parameter values

# Define problem
prob = ODEProblem(system_dynamics!, u0, tspan, p)

# Solve (Tsit5 is a good general-purpose solver)
sol = solve(prob, Tsit5())

# Plot solution
plot(sol, xlabel="Time", ylabel="State", label=["u1" "u2"])

# Access solution at specific times
println("Solution at t=5: ", sol(5.0))
```

### With Callbacks

```julia
# Callback: Terminate when condition met
condition(u, t, integrator) = u[1] < 0.1  # TODO: Define condition
affect!(integrator) = terminate!(integrator)
cb_terminate = ContinuousCallback(condition, affect!)

# Callback: Periodic events
function periodic_event!(integrator)
    integrator.u[1] *= 0.9  # TODO: Define periodic action
end
cb_periodic = PeriodicCallback(periodic_event!, 1.0)  # Every 1 time unit

# Combine callbacks
callbacks = CallbackSet(cb_terminate, cb_periodic)

# Solve with callbacks
sol = solve(prob, Tsit5(), callback=callbacks)
```

### With Ensemble Simulation

```julia
# Ensemble simulation with varying initial conditions
function prob_func(prob, i, repeat)
    # TODO: Customize how each trajectory differs
    u0_varied = u0 .* (1.0 .+ 0.1 * randn(length(u0)))
    remake(prob, u0=u0_varied)
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

# Solve ensemble (using multi-threading)
ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=100)

# Plot ensemble
using DifferentialEquations.EnsembleAnalysis
summ = EnsembleSummary(ensemble_sol)
plot(summ, xlabel="Time", ylabel="State")
```

### With Sensitivity Analysis

```julia
using SciMLSensitivity

# Forward sensitivity to parameters
prob_sens = ODEForwardSensitivityProblem(system_dynamics!, u0, tspan, p)
sol_sens = solve(prob_sens, Tsit5())

# Extract sensitivities
sens = extract_local_sensitivities(sol_sens)

# Plot sensitivity to first parameter
plot(sol_sens.t, [s[1,1] for s in sens], xlabel="Time",
     ylabel="∂u1/∂p1", title="Sensitivity to Parameter 1")
```

### Symbolic ModelingToolkit Template

```julia
# Auto-generated symbolic ODE template
using ModelingToolkit
using DifferentialEquations
using Plots

# Define symbolic variables
@variables t x(t) y(t)
@parameters α β γ δ
D = Differential(t)

# Define system equations symbolically
# TODO: Define your differential equations
eqs = [
    D(x) ~ α * x - β * x * y,
    D(y) ~ -γ * y + δ * x * y
]

# Create ODE system
@named sys = ODESystem(eqs, t)

# Simplify system (symbolic optimization)
sys_simplified = structural_simplify(sys)

# Initial conditions and parameters
u0 = [x => 1.0, y => 1.0]  # TODO: Set initial values
p = [α => 1.5, β => 1.0, γ => 3.0, δ => 1.0]  # TODO: Set parameters
tspan = (0.0, 10.0)

# Create numerical problem
prob = ODEProblem(sys_simplified, u0, tspan, p)

# Solve
sol = solve(prob, Tsit5())

# Plot
plot(sol, xlabel="Time", ylabel="State", label=["x" "y"])
```

---

## PDE Templates

### Method of Lines Template

```julia
# Auto-generated PDE template
using ModelingToolkit
using DifferentialEquations

# Define symbolic variables
@parameters t x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Dx^2  # Second derivative

# PDE equation: ∂u/∂t = ∂²u/∂x²
# TODO: Modify equation for your problem
eq = Dt(u(t, x)) ~ Dxx(u(t, x))

# Boundary and initial conditions
# TODO: Define appropriate conditions
bcs = [
    u(0, x) ~ cos(π * x),  # Initial condition
    u(t, 0) ~ 0,            # Boundary at x=0
    u(t, 1) ~ 0             # Boundary at x=1
]

# Domain
domains = [
    t ∈ IntervalDomain(0.0, 1.0),
    x ∈ IntervalDomain(0.0, 1.0)
]

# Method of Lines discretization
@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u])

# NOTE: For NeuralPDE (PINN) approach, see NeuralPDE.jl documentation
# For Method of Lines, convert to ODE system and solve
```

---

## SDE Templates

### Stochastic Differential Equation Template

```julia
# Auto-generated SDE template
using DifferentialEquations
using Plots

# Drift function (deterministic part)
function drift!(du, u, p, t)
    # TODO: Define drift dynamics
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end

# Diffusion function (stochastic part)
function diffusion!(du, u, p, t)
    # TODO: Define noise strength
    du[1] = p[5] * u[1]  # Multiplicative noise
    du[2] = p[6] * u[2]
end

# Initial conditions
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0, 0.1, 0.1]  # Include noise parameters

# Define SDE problem
prob = SDEProblem(drift!, diffusion!, u0, tspan, p)

# Solve
sol = solve(prob, SOSRI())  # Stochastic solver

# Plot multiple trajectories
ensemble_prob = EnsembleProblem(prob)
ensemble_sol = solve(ensemble_prob, SOSRI(), EnsembleThreads(), trajectories=100)
plot(ensemble_sol, xlabel="Time", ylabel="State", alpha=0.2)
```

---

## Optimization Templates

### Parameter Estimation Template

```julia
# Auto-generated optimization template
using Optimization
using OptimizationOptimJL
using DifferentialEquations

# Define the model (ODE system)
function model!(du, u, p, t)
    # TODO: Define model dynamics
    du[1] = p[1] * u[1]
    du[2] = p[2] * u[2]
end

# Experimental data (TODO: Replace with real data)
t_data = 0.0:0.1:10.0
u_data = rand(2, length(t_data))  # Placeholder data

# Loss function: Compare model to data
function loss_function(p, _)
    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(model!, u0, tspan, p)

    sol = solve(prob, Tsit5(), saveat=t_data, sensealg=ForwardDiffSensitivity())

    # Check if solve was successful
    if sol.retcode != :Success
        return Inf
    end

    # Mean squared error
    loss = sum(abs2, Array(sol) .- u_data)
    return loss
end

# Initial parameter guess
p_init = [0.5, 0.5]

# Setup optimization problem
opt_prob = OptimizationProblem(loss_function, p_init)

# Solve optimization
result = solve(opt_prob, BFGS())

println("Optimized parameters: ", result.u)
println("Final loss: ", result.minimum)

# Visualize fit
prob_opt = ODEProblem(model!, [1.0, 1.0], (0.0, 10.0), result.u)
sol_opt = solve(prob_opt, Tsit5())

using Plots
plot(sol_opt, xlabel="Time", ylabel="State", label=["Model u1" "Model u2"])
scatter!(t_data, u_data[1,:], label="Data u1")
scatter!(t_data, u_data[2,:], label="Data u2")
```

---

## Problem Type Detection Keywords

### ODE Detection
- "ordinary differential", "ODE"
- "dynamics", "time evolution"
- "population", "predator-prey"
- "chemical kinetics", "oscillator"
- "coupled system"

### PDE Detection
- "partial differential", "PDE"
- "spatial", "diffusion"
- "heat equation", "wave equation"
- "boundary conditions", "Laplacian"

### SDE Detection
- "stochastic", "SDE"
- "noise", "random", "Brownian"
- "uncertainty", "fluctuations"

### Optimization Detection
- "minimize", "maximize", "optimal"
- "parameter estimation", "fitting"
- "calibration", "inverse problem"

---

## Solver Selection Guide

### ODE Solvers

**General Purpose**:
- `Tsit5()`: Good default for non-stiff problems
- `Vern7()`: Higher accuracy
- `BS3()`: Lower accuracy, faster

**Stiff Problems**:
- `Rodas5()`: Best for moderately stiff
- `TRBDF2()`: Good for very stiff
- `KenCarp4()`: Implicit-explicit (IMEX)

**Event Detection**:
- `Tsit5()` with callbacks
- Continuous callbacks for zero-crossing

### PDE Solvers

**Method of Lines**:
- Discretize in space → ODE system
- Use appropriate ODE solver

**NeuralPDE (PINN)**:
- `PhysicsInformedNN` for complex geometries
- Good for high-dimensional PDEs

### SDE Solvers

**Additive Noise**:
- `EM()`: Euler-Maruyama (simple)
- `ImplicitEM()`: For stiff SDEs

**General Noise**:
- `SOSRI()`: Recommended default
- `SOSRA()`: Alternative method

### Optimization Algorithms

**Gradient-Based**:
- `BFGS()`: Quasi-Newton (recommended)
- `LBFGS()`: Limited memory BFGS
- `Adam()`: Adaptive learning rate

**Gradient-Free**:
- `NelderMead()`: Simplex method
- `ParticleSwarm()`: Global optimization

---

## Best Practices

### Performance

1. **Use in-place operations**: Functions ending in `!` modify arrays directly
2. **Avoid allocations**: Pre-allocate arrays, use `@views`
3. **Type stability**: Ensure consistent return types
4. **Appropriate solvers**: Match solver to problem characteristics

### Sensitivity Analysis

1. **Forward mode**: Good for few parameters (< 100)
2. **Adjoint mode**: Better for many parameters
3. **Checkpointing**: For memory efficiency with adjoint

### Ensemble Simulations

1. **Threading**: `EnsembleThreads()` for shared memory
2. **Distributed**: `EnsembleDistributed()` for clusters
3. **GPU**: `EnsembleGPUArray()` for GPU parallelization

### Callbacks

1. **Continuous callbacks**: Zero-crossing detection
2. **Discrete callbacks**: Fixed-time events
3. **Callback priority**: Order matters when combining

---

## Common Pitfalls

### ODE Pitfalls

❌ **Don't**: Allocate arrays inside dynamics function
```julia
function bad_dynamics!(du, u, p, t)
    temp = zeros(length(u))  # Allocation!
    # ...
end
```

✅ **Do**: Pre-allocate or use in-place operations
```julia
function good_dynamics!(du, u, p, t)
    # Direct operations on du
    du[1] = p[1] * u[1]
end
```

### Optimization Pitfalls

❌ **Don't**: Ignore solver failures
```julia
sol = solve(prob, Tsit5())
loss = sum(abs2, Array(sol) .- data)  # May fail if solve failed!
```

✅ **Do**: Check return code
```julia
sol = solve(prob, Tsit5())
if sol.retcode != :Success
    return Inf
end
```

### Ensemble Pitfalls

❌ **Don't**: Create new problem from scratch each time
```julia
function prob_func(prob, i, repeat)
    ODEProblem(f, u0_new, tspan, p)  # Inefficient!
end
```

✅ **Do**: Use `remake` to reuse structure
```julia
function prob_func(prob, i, repeat)
    remake(prob, u0=u0_new)  # Efficient!
end
```

---

**Version**: 1.0.3
**Last Updated**: 2025-11-07
**Plugin**: julia-development
