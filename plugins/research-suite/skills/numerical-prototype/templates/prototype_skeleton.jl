"""
prototype_skeleton.jl

Starting scaffold for Stage 6 numerical prototypes implemented in Julia. Copy
into src/<PackageName>.jl, rename `module PrototypeSkeleton` to match the
package name, and fill in the physics for your specific formalism. The
`_self_test` block below the module is for running this file standalone
(`julia prototype_skeleton.jl`) during development; once copied into a real
package, move it into test/runtests.jl per julia_first_rules.md's layout.

Conventions (see _research-commons/code_architecture/julia_first_rules.md):
- Params and State are concrete, type-stable structs
- `step!` mutates state in place; no allocations in the hot loop
- Every stochastic function takes an `AbstractRNG` explicitly
- For stiff ODEs/DAEs/PDEs, replace the manual loop with OrdinaryDiffEq.jl
"""
module PrototypeSkeleton

using Random

export Params, State, step!, integrate!, extract_observable

# --- Dtype policy -------------------------------------------------------
# Pick one and document the reason. Mixed precision requires an explicit
# policy note, same discipline as the JAX skeleton's DTYPE constant.
const T = Float64
# Reason: the observable is a small number (O(1e-3)) accumulated over many
# steps, so Float32 accumulated error exceeds the claimed precision.

# --- Params ---------------------------------------------------------------
"""Physical parameters. Do not vary during integration."""
struct Params
    diffusion::T
    interaction_strength::T   # the "new physics" term; zero recovers the known limit
    dt::T
end

Params(; diffusion=1.0, interaction_strength=0.0, dt=1e-3) =
    Params(diffusion, interaction_strength, dt)

# --- State ------------------------------------------------------------------
"""Simulation state. Type-stable, dtype-consistent per module policy."""
mutable struct State
    positions::Matrix{T}   # (D, N): one particle per column
    velocities::Matrix{T}
    t::T

    function State(positions::Matrix{T}, velocities::Matrix{T}, t::T)
        size(positions) == size(velocities) || throw(DimensionMismatch(
            "positions $(size(positions)) and velocities $(size(velocities)) must match"))
        new(positions, velocities, t)
    end
end

function State(n_particles::Int, dim::Int, rng::AbstractRNG)
    positions = randn(rng, T, dim, n_particles)
    velocities = randn(rng, T, dim, n_particles)
    State(positions, velocities, zero(T))
end

# --- Physics core -------------------------------------------------------
"""
    step!(state, params, rng)

One integration step, mutating `state` in place. Type-stable, allocation-free
in the hot loop. Fill in the physics from Eq. (X) of the formalism here.
"""
function step!(state::State, params::Params, rng::AbstractRNG)
    # Placeholder: overdamped Brownian dynamics with an interaction stub.
    # Replace with the physics from 05_formalism.tex.
    noise_scale = sqrt(2 * params.diffusion * params.dt)
    @inbounds for i in axes(state.positions, 2)
        for d in axes(state.positions, 1)
            drift = -params.interaction_strength * state.positions[d, i]
            state.positions[d, i] += drift * params.dt + noise_scale * randn(rng, T)
        end
    end
    state.t += params.dt
    return state
end

# --- Time integration -----------------------------------------------------
"""
    integrate!(state, params, n_steps, rng) -> trajectory

Roll out `n_steps`, mutating `state` to its final value (the `!` suffix marks
this) and returning the stacked trajectory (a State per step). For stiff
governing equations, replace this loop with `OrdinaryDiffEq.jl`'s
`solve(ODEProblem(...))` instead.
"""
function integrate!(state::State, params::Params, n_steps::Int, rng::AbstractRNG)
    trajectory = Vector{State}(undef, n_steps)
    for k in 1:n_steps
        step!(state, params, rng)
        trajectory[k] = State(copy(state.positions), copy(state.velocities), state.t)
    end
    return trajectory
end

# --- Observable extractor --------------------------------------------------
"""
    extract_observable(trajectory, params) -> NamedTuple

Convert the trajectory into the predicted observable that Stage 7 will design
a measurement for. Replace with the problem-specific observable from
04_theory.md. Returns the identity/values subset of the schema in
templates/predicted_observable.md; combine with uncertainty and units via
scripts/observable_extractor.py's ObservableBuilder before the Stage 6 write-up.
"""
function extract_observable(trajectory::Vector{State}, params::Params)
    t = [s.t for s in trajectory]
    msd = [sum(abs2, s.positions) / size(s.positions, 2) for s in trajectory]
    return (name="mean_square_displacement", t=t, values=msd,
            units=(t="time", values="length^2"))
end

end # module

# --- Minimal self-test -------------------------------------------------
using .PrototypeSkeleton
using Random

"""Runnable self-check: `julia prototype_skeleton.jl`."""
function _self_test()
    rng = Random.MersenneTwister(0)
    params = Params()
    state = State(100, 3, rng)
    trajectory = integrate!(state, params, 1000, rng)
    obs = extract_observable(trajectory, params)
    println("trajectory length: ", length(trajectory))
    println("observable '", obs.name, "' final value: ", round(obs.values[end], digits=4))
    @assert length(obs.t) == length(trajectory)
    @assert all(isfinite, obs.values)
end

if abspath(PROGRAM_FILE) == @__FILE__
    _self_test()
end
