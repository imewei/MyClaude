---
name: core-julia-patterns
version: "1.0.8"
maturity: "5-Expert"
specialization: Julia Language Fundamentals
description: Master multiple dispatch, type system, parametric types, type stability, and metaprogramming for high-performance Julia. Use when designing type hierarchies, debugging @code_warntype issues, optimizing with @inbounds/@simd/StaticArrays, or writing macros and generated functions.
---

# Core Julia Patterns

Expert-level patterns for leveraging Julia's type system and multiple dispatch for performance and composability.

## Expert Agent

For complex type design, metaprogramming, and performance optimization, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia optimization, including Core Julia, SciML, Turing.jl, and Package Development.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: Performance tuning, type stability analysis, and advanced metaprogramming.

---

## 1. Type System Design

### Abstract Type Hierarchy

Design generic interfaces using abstract types. Avoid concrete types in function signatures unless dispatching specifically.

```julia
abstract type AbstractSolver end
abstract type IterativeSolver <: AbstractSolver end
abstract type DirectSolver <: AbstractSolver end

struct NewtonSolver{T} <: IterativeSolver
    tol::T
    max_iter::Int
end

# Generic fallback
solve(solver::AbstractSolver, problem) = error("Not implemented")

# Specialized implementation
function solve(solver::NewtonSolver, problem)
    # ...
end
```

### Parametric Types

Use parametric types to avoid `Any` fields and ensure type stability.

```julia
# BAD: Fields are abstract, causing allocations
struct BadPoint
    x::Real
    y::Real
end

# GOOD: Fields are concrete and typed
struct Point{T<:Real}
    x::T
    y::T
end
```

---

## 2. Multiple Dispatch Patterns

### Trait-Based Dispatch

Use "Holy Traits" to dispatch on behavior rather than type hierarchy.

```julia
abstract type LinearityStyle end
struct IsLinear <: LinearityStyle end
struct IsNonLinear <: LinearityStyle end

# Trait definition
LinearityStyle(::Type{<:NewtonSolver}) = IsNonLinear()

# Dispatch implementation
solve(s::S, p) where S = solve(LinearityStyle(S), s, p)
solve(::IsLinear, s, p) = # Linear solve
solve(::IsNonLinear, s, p) = # Non-linear solve
```

### Function Barriers

Isolate type instabilities behind a "function barrier" where dynamic types become concrete.

```julia
function kernel(x::Vector{Float64})
    # Type stable inner loop
    s = 0.0
    @inbounds for val in x
        s += val
    end
    return s
end

function driver()
    data = read_data() # might return Vector{Any}
    # Barrier: dynamic dispatch to specialized kernel
    return kernel(convert(Vector{Float64}, data))
end
```

---

## 3. Performance Tuning

### Type Stability

Use `@code_warntype` to find type instabilities (red/pink output).

```julia
# Unstable: returns Int or Float64
function unstable(x)
    x > 0 ? 1 : 1.0
end

# Stable: promotes to common type
function stable(x)
    x > 0 ? 1.0 : 1.0
end
```

### Memory Management

- **Pre-allocation**: Allocate arrays once, reuse them.
- **Views**: Use `@view` to slice arrays without copying.
- **StaticArrays**: Use `SVector` for small, fixed-size vectors (< 100 elements).

```julia
using StaticArrays

function fast_math(a::SVector{3, Float64}, b::SVector{3, Float64})
    return cross(a, b) # No allocations
end
```

### Loop Optimization

- **Row-major vs Column-major**: Julia is column-major. Iterate over columns first.
- **@inbounds**: Disable bounds checking (safe only if verified).
- **@simd**: Enable vectorization.

```julia
function sum_cols(A)
    s = zero(eltype(A))
    @inbounds @simd for i in eachindex(A)
        s += A[i]
    end
    return s
end
```

---

## 4. Metaprogramming

### Macros

Use macros for syntax transformation, not just code generation. Always escape inputs with `esc()`.

```julia
macro time_it(ex)
    quote
        local t0 = time()
        local val = $(esc(ex))
        println("Time: ", time() - t0)
        val
    end
end
```

### Generated Functions

Use `@generated` functions when the implementation depends on type parameters.

```julia
@generated function ntuple_zero(::Val{N}) where N
    t = Expr(:tuple, [0 for _ in 1:N]...)
    return :($t)
end
```

---

## Checklist

- [ ] Structs use parametric types for fields
- [ ] Functions dispatch on abstract types
- [ ] Hot loops are type-stable (verified with `@code_warntype`)
- [ ] Allocations minimized (verified with `@btime`)
- [ ] Views used for slicing
- [ ] Column-major iteration order respected

---

**Version**: 1.0.6
