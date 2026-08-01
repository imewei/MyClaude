# Turing UDE Stage Code — Full Reference

`--mode deep` content for **bayesian-ude-workflow**. Full Stage 1 (deterministic warm-start) and Stage 2 (Turing model) code.

## Stage 1 — Deterministic warm-start

Find a maximum a posteriori (MAP) or maximum likelihood point with `Optimization.jl` driving `SciMLSensitivity` adjoints. This gives the sampler a sensible initial position and surfaces obvious problems (stiff solver failures, unbounded gradients) before you commit to MCMC.

```julia
using OrdinaryDiffEq, Lux, ComponentArrays, Optimization, OptimizationOptimJL, SciMLSensitivity, Zygote, Random

# Define the UDE: known physics + Lux neural correction
nn = Chain(Dense(2, 16, tanh), Dense(16, 2))
ps_init, st = Lux.setup(Random.default_rng(), nn)
ps_ca = ComponentArray(ps_init)             # contiguous, AD-friendly packing

function ude!(du, u, p, t)
    correction, _ = nn(u, p.nn, st)
    du[1] = p.α * u[1] - correction[1]
    du[2] = -p.β * u[2] + correction[2]
end

p0 = ComponentArray(α = 1.0, β = 1.0, nn = ps_ca)
prob = ODEProblem(ude!, u0, tspan, p0)

function loss(p)
    sol = solve(prob, Tsit5(); p = p, saveat = t_obs,
                sensealg = GaussAdjoint(autojacvec = ZygoteVJP()))
    sum(abs2, Array(sol) .- y_obs)
end

opt_prob = OptimizationProblem(OptimizationFunction((p, _) -> loss(p),
                                                    Optimization.AutoZygote()), p0)
map_estimate = solve(opt_prob, BFGS(), maxiters = 500)
```

`ComponentArray` is critical: it gives the optimizer (and later, Turing) a flat parameter vector while preserving the structured `(α, β, nn)` view that the ODE function consumes. See `optimization-patterns` for the full Optimization.jl interface and `sciml-modern-stack` for sensealg selection.

## Stage 2 — Turing model with embedded ODE

Wrap the same `prob` in a `Turing.@model`. Use `remake` to inject sampled parameters without rebuilding the problem each step.

```julia
using Turing

@model function bayesian_ude(y_obs, t_obs, prob, p_template)
    # Priors over physical parameters
    α ~ truncated(Normal(1.0, 0.5), 0, Inf)
    β ~ truncated(Normal(1.0, 0.5), 0, Inf)

    # Prior over neural network weights (flat vector)
    nn_dim = length(p_template.nn)
    nn_flat ~ MvNormal(zeros(nn_dim), 0.5 * I)

    # Repack into ComponentArray with the same structure
    p = ComponentArray(α = α, β = β,
                       nn = reshape(nn_flat, axes(p_template.nn)))

    # Solve and condition on observations
    sol = solve(remake(prob; p = p), Tsit5();
                saveat = t_obs,
                sensealg = ForwardDiffSensitivity())
    σ ~ truncated(Normal(0, 0.1), 0, Inf)
    y_obs .~ Normal.(Array(sol), σ)
end

model = bayesian_ude(y_obs, t_obs, prob, p0)
```

**Sensealg selection in one breath**:

- **`ForwardDiff` AD**: bypasses `sensealg` entirely (uses Dual numbers for `u0` and `p`). The safe default for ≤100 parameters and stiff systems. Just pass `Optimization.AutoForwardDiff()` to the optimizer or let Turing pick ForwardDiff — no `sensealg` keyword needed.
- **Reverse-mode AD** (`Zygote`, `Enzyme`, `ReverseDiff`) for many parameters: pair with **`GaussAdjoint()`** — the SciMLSensitivity team's current general recommendation. It is `O(n³ + p)` for stiff/implicit problems vs `O((n+p)³)` for `BacksolveAdjoint`/`InterpolatingAdjoint`, supports checkpointing, and avoids the backwards-solve instability that breaks `BacksolveAdjoint` on stiff systems and DAEs.
- **`InterpolatingAdjoint`** / **`QuadratureAdjoint`**: now niche, useful only in benchmarking scenarios where you can prove an advantage.
- **`BacksolveAdjoint`**: lowest memory but **avoid for stiff systems and DAEs** — backwards-solution accuracy degrades fast.
