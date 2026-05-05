using DifferentialEquations
using Lux
using ComponentArrays
using SciMLSensitivity
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using Turing
using Random

# Universal Differential Equation: known physics + neural correction
function lotka_volterra_ude!(du, u, p, t)
    correction, _ = nn_model(u, p.nn, st)
    du[1] = p.α * u[1] - correction[1]
    du[2] = -p.β * u[2] + correction[2]
end

# Lux neural network for the unknown correction term
nn_model = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 2))
rng = Random.default_rng()
ps_init, st = Lux.setup(rng, nn_model)
p_ca = ComponentArray(α = 1.5, β = 1.0, nn = ComponentArray(ps_init))

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka_volterra_ude!, u0, tspan, p_ca)

function loss(p)
    sol = solve(prob, Tsit5(); p = p, saveat = 0.1,
                sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))
    sum(abs2, Array(sol) .- y_observed)
end

# Stage 1: deterministic warm-start
opt_func = OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, p_ca)
map_estimate = solve(opt_prob, BFGS(); maxiters = 500)

# Stage 2: Bayesian inference with Turing
@model function bayesian_ude(y_obs, prob)
    α ~ truncated(Normal(1.5, 0.5), 0, Inf)
    β ~ truncated(Normal(1.0, 0.5), 0, Inf)
    σ ~ truncated(Normal(0.0, 0.1), 0, Inf)
    nn_dim = length(p_ca.nn)
    nn_flat ~ MvNormal(zeros(nn_dim), 0.5 * I)

    p = ComponentArray(α = α, β = β, nn = reshape(nn_flat, axes(p_ca.nn)))
    sol = solve(remake(prob; p = p), Tsit5();
                saveat = 0.1, sensealg = ForwardDiffSensitivity())
    y_obs .~ Normal.(Array(sol), σ)
end
