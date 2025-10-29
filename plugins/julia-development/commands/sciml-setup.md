# /sciml-setup - Interactive SciML Project Scaffolding

**Priority**: 1 (Highest)
**Agent**: sciml-pro
**Description**: Auto-detect problem type from user description (ODE, PDE, SDE, optimization) and generate scaffolded solver code with appropriate templates, callbacks, ensemble simulations, and sensitivity analysis setup.

## Overview

The /sciml-setup command provides interactive scaffolding for Scientific Machine Learning projects. It automatically detects the type of problem from a natural language description and generates working template code with DifferentialEquations.jl, ModelingToolkit.jl, or Optimization.jl.

## Usage

```
/sciml-setup "<problem description>"
```

**Examples:**
```
/sciml-setup "coupled oscillator system"
/sciml-setup "heat equation with boundary conditions"
/sciml-setup "stochastic population dynamics with noise"
/sciml-setup "parameter estimation for pharmacokinetics"
```

## Problem Type Auto-Detection

The command analyzes the description for keywords to determine problem type:

**ODE Detection Keywords**:
- "ordinary differential", "ODE"
- "dynamics", "time evolution"
- "population", "predator-prey"
- "chemical kinetics", "oscillator"
- "coupled system"

**PDE Detection Keywords**:
- "partial differential", "PDE"
- "spatial", "diffusion"
- "heat equation", "wave equation"
- "boundary conditions", "Laplacian"

**SDE Detection Keywords**:
- "stochastic", "SDE"
- "noise", "random", "Brownian"
- "uncertainty", "fluctuations"

**Optimization Detection Keywords**:
- "minimize", "maximize", "optimal"
- "parameter estimation", "fitting"
- "calibration", "inverse problem"

## Interactive Prompts

After auto-detection, the command presents interactive prompts:

1. **Confirm Problem Type**: "Detected ODE problem. Is this correct? (yes/no)"
2. **Modeling Approach**: "Use symbolic ModelingToolkit.jl or direct API? (symbolic/direct)"
3. **Add Callbacks?**: "Include callback examples? (yes/no)"
4. **Ensemble Simulation?**: "Include ensemble simulation template? (yes/no)"
5. **Sensitivity Analysis?**: "Include sensitivity analysis setup? (yes/no)"

## Output Templates

### ODE Template (Direct API)

```julia
# Auto-generated ODE template by /sciml-setup
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

**With Callbacks:**
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

**With Ensemble Simulation:**
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

**With Sensitivity Analysis:**
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

### ODE Template (Symbolic ModelingToolkit)

```julia
# Auto-generated symbolic ODE template by /sciml-setup
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

### PDE Template

```julia
# Auto-generated PDE template by /sciml-setup
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

### SDE Template

```julia
# Auto-generated SDE template by /sciml-setup
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

### Optimization Template

```julia
# Auto-generated optimization template by /sciml-setup
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

## Success Criteria

The command successfully:
1. Auto-detects problem type from natural language description
2. Generates syntactically correct Julia code
3. Includes appropriate solver selection (Tsit5 for ODE, SOSRI for SDE, etc.)
4. Provides TODO comments for user customization
5. Generates runnable code (after TODOs are filled)
6. Includes explanatory comments
7. Optionally includes callbacks, ensemble, and sensitivity analysis based on user choice

## Notes

- Generated code uses best practices for each problem type
- Solver selection matches problem characteristics
- Code includes performance hints (e.g., ForwardDiffSensitivity for optimization)
- Templates are starting points; users customize based on specific needs
- All generated files include proper imports and dependencies

## Related Commands

- **/julia-optimize**: Analyze and optimize generated code for performance
- **/julia-scaffold**: Create package structure for organizing SciML projects

## Implementation Notes

The command:
1. Parses user description for keywords
2. Scores each problem type based on keyword matches
3. Selects type with highest score (or prompts if ambiguous)
4. Presents interactive prompts for configuration
5. Generates appropriate template file
6. Saves to `<description>_sciml.jl` or user-specified filename
7. Prints next steps and usage instructions
