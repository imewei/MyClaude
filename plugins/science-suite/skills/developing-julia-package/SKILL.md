---
name: developing-julia-package
disable-model-invocation: true
description: "Use when you implement or edit code inside an existing Julia package: module/`include` layout, multiple dispatch, type annotations and stability, formatting, and docstrings"
---

# Developing a Julia Package

Assume the package is `MyPkg`; substitute the real name.

Use `src/MyPkg.jl` as the entry point: module declaration, exports, and `include` list only. Put substantial code in focused files under `src/`.

```julia
module MyPkg

export fit_model, FitResult, GaussianModel

include("models.jl")
include("fit.jl")

end
```

Split files for readability, not to recreate class or submodule hierarchies. Prefer one public module unless there is a real user-facing namespace boundary.

## Keep exports small

Do not `export` helpers just so tests can reach them. Test the public API first; import internals explicitly only when they have behavior that is hard to exercise through the public path:

```julia
using Test
using MyPkg
using MyPkg: initial_guess   # internal, imported explicitly

@testset "fit_model" begin
    @test fit_model(x, y; model = GaussianModel()) isa FitResult
end
```

To audit and trim an existing export list, see [fixing-julia-exports](../fixing-julia-exports/SKILL.md).

## Multiple dispatch

Split behavior across methods rather than `if x isa ...` chains:

```julia
f(x) = x               # generic
f(x::Integer) = 2x     # specialized
```

For package APIs, make the dispatch object explicit. Keep configuration and fitted state in separate types:

```julia
abstract type AbstractModel end

struct GaussianModel <: AbstractModel
    baseline::Bool
end
GaussianModel(; baseline = true) = GaussianModel(baseline)

fit_model(x, y; model::AbstractModel = GaussianModel()) = fit_model(model, x, y)

function fit_model(model::GaussianModel, x, y)
    guess = initial_guess(x, y, model)
    # gaussian-specific implementation
end

struct FitResult{P,T}
    params::P
    residuals::Vector{T}
    converged::Bool
end
```

`GaussianModel()` selects the method; `FitResult` is returned state. Do not merge them into one mutable "algorithm plus state" object unless that is the public contract.

Avoid **type piracy**: only add methods to a function you own or for a type you own. Aqua.jl flags violations.

## Types

- **Public signatures:** narrow only when it prevents misuse or documents the contract.
- **Internals:** stay generic (`AbstractVector{<:Real}`) so callers' types, including AD dual numbers, pass through. Over-constrained `::Float64` breaks ForwardDiff and unit-carrying types.
- **Struct fields:** concrete or parametric (`struct S{T<:Real}; x::T; end`), never abstract, on hot paths.
- **Keyword defaults:** use `Base.@kwdef` for config structs with many options.
- **Globals:** `const` or pass as arguments.
- **Stability:** diagnose with [profiling-julia-performance](../profiling-julia-performance/SKILL.md) (JET, `@code_warntype`); guard with `Test.@inferred` on hot-path entry points.

## Optional dependencies

Heavy integrations (plotting, GPU, AD backends) belong in package extensions via `[weakdeps]` and `[extensions]`, not in `[deps]`. This keeps load time down; see [precompiling-julia-package](../precompiling-julia-package/SKILL.md).

## Formatting

Format with [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl); commit a `.JuliaFormatter.toml` so contributors and CI agree:

```julia
using JuliaFormatter
format(".")
```

Run it before merging; check it in CI with `format(".") || error("unformatted")`-style gating if the team wants enforcement. Install it in a tooling environment, not the package's runtime deps.

## Performance

- Profile before optimizing ([profiling-julia-performance](../profiling-julia-performance/SKILL.md)).
- Benchmark with `@btime`/`@benchmark` from BenchmarkTools.jl, interpolating globals (`@btime f($x)`), not a single `@time`.
- Expose in-place variants explicitly (`f!`) so callers opt in.

## Errors and docs

- Throw `ArgumentError`, `DomainError`, `DimensionMismatch`, etc., with messages that tell the caller what to fix.
- Give every exported or `public` name a docstring; see [documenting-julia-package](../documenting-julia-package/SKILL.md).

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
