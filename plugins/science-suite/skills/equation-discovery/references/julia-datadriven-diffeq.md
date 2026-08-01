# DataDrivenDiffEq.jl — Full Julia API Reference

`--mode deep` content for **equation-discovery**. Covers basis construction, thresholding algorithms, implicit SINDy, and the discover-simulate-validate loop.

## Basic Usage

```julia
using DataDrivenDiffEq, ModelingToolkit

# Define symbolic variables
@variables x y

# Build candidate library (polynomial basis up to degree 3)
basis = Basis(polynomial_basis([x, y], 3), [x, y])

# Create problem from data
prob = DataDrivenProblem(X, DX=DX)  # X: state data, DX: derivative data

# Solve with STLSQ sparse regression
result = solve(prob, basis, STLSQ(threshold=0.1))

# Extract discovered equations
system = result.basis        # Symbolic equations
coefficients = result.coeff  # Sparse coefficient matrix
```

## Custom Library Functions

Extend the basis beyond polynomials for domain-specific dynamics:

```julia
@variables x y

# Trigonometric terms
trig_terms = [sin(x), cos(x), sin(y), cos(y)]

# Exponential terms
exp_terms = [exp(-x), exp(-y)]

# Cross terms
cross_terms = [x * sin(y), y * cos(x), x * exp(-y)]

# Combined custom basis
custom_basis = Basis(
    vcat(polynomial_basis([x, y], 2), trig_terms, exp_terms, cross_terms),
    [x, y]
)
```

> **Rule:** Start with a polynomial basis. Add domain-specific terms only when polynomial SINDy fails or physics suggests oscillatory/exponential behavior.

## Thresholding Algorithms

```julia
# STLSQ — default for clean data
result_stlsq = solve(prob, basis, STLSQ(threshold=0.1))

# SR3 — noisy data
result_sr3 = solve(prob, basis, SR3(threshold=0.05, relaxation=1.0))

# ADMM — constrained
result_admm = solve(prob, basis, ADMM(threshold=0.1, rho=1.0))
```

See the SKILL.md thresholding table for when to pick each algorithm.

## Implicit SINDy

For dynamics that cannot be written as explicit `dX/dt = f(X)` (e.g., implicit ODEs, DAEs):

```julia
# Implicit formulation: F(X, dX/dt) = 0
# Augment library with derivative terms
@variables x dx

implicit_basis = Basis(
    vcat(polynomial_basis([x, dx], 3), [sin(x) * dx, x^2 * dx]),
    [x, dx]
)

# Solve with implicit flag
prob_implicit = DataDrivenProblem(X, DX=DX)
result = solve(prob_implicit, implicit_basis, ImplicitOptimizer(STLSQ(threshold=0.1)))
```

## DifferentialEquations.jl Integration

Discover-simulate-validate loop:

```julia
using DifferentialEquations, DataDrivenDiffEq, ModelingToolkit

# 1. Discover equations from data
result = solve(prob, basis, STLSQ(threshold=0.1))

# 2. Convert to ODESystem for simulation
@named discovered_sys = ODESystem(result.basis)

# 3. Simulate discovered model
ode_prob = ODEProblem(discovered_sys, u0, tspan)
sol = solve(ode_prob, Tsit5())

# 4. Validate: compare simulation vs held-out data
error = norm(sol(t_test) .- X_test) / norm(X_test)
```

> **Rule:** Always validate on held-out data not used in the SINDy fit. In-sample error is misleading for sparse models.

## Threshold Sweep for Validation

```julia
# Pareto front sweep over thresholds
thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
for thresh in thresholds
    res = solve(prob_train, basis, STLSQ(threshold=thresh))
    err = validate_on_test(res, X_test, t_test)
    n_terms = count(!iszero, res.coeff)
    println("Threshold=$thresh, Terms=$n_terms, Error=$err")
end
```
