---
name: core-julia-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Language Fundamentals
description: Master multiple dispatch, type system, parametric types, type stability, and metaprogramming for high-performance Julia. Use when designing type hierarchies, debugging @code_warntype issues, optimizing with @inbounds/@simd/StaticArrays, or writing macros and generated functions.
---

# Core Julia Patterns

Multiple dispatch, type system, and metaprogramming for idiomatic Julia.

---

## Multiple Dispatch

```julia
# Different implementations based on ALL argument types
process(x::Int, y::Int) = x + y
process(x::Float64, y::Float64) = x * y
process(x::String, y::String) = x * " and " * y

# Dispatch on combinations
combine(x::Number, y::String) = string(x) * y
combine(x::String, y::Number) = x * string(y)
```

**Design Principle**: Method selection based on all argument types, not just first (unlike OOP).

---

## Type Hierarchy

```julia
# Abstract type hierarchy
abstract type Animal end
abstract type Pet <: Animal end

# Concrete types
struct Dog <: Pet
    name::String
    breed::String
end

# Methods on abstract types
greet(pet::Pet) = "Hello, $(pet.name)!"
```

**Best Practice**: Abstract types in function signatures (flexibility), concrete types in struct fields (performance).

---

## Parametric Types

```julia
struct Point{T<:Real}
    x::T
    y::T
end

# Different instantiations
Point(1, 2)         # Point{Int64}
Point(1.5, 2.5)     # Point{Float64}

# Parametric methods
function distance(p1::Point{T}, p2::Point{T}) where T
    sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
end
```

---

## Type Stability

```julia
# ✅ TYPE STABLE
function stable_sum(x::Vector{Float64})
    result = 0.0  # Concrete type
    for val in x
        result += val
    end
    return result  # Always Float64
end

# ❌ TYPE UNSTABLE
function unstable(x::Float64)
    if x > 0
        return x^2        # Float64
    else
        return "negative" # String - INSTABILITY!
    end
end

# Check with @code_warntype
@code_warntype my_function(args...)
# Red/pink = unstable (Any, Union)
# Blue = stable
```

---

## Performance Patterns

```julia
using StaticArrays

# 1. StaticArrays for small fixed-size
v = @SVector [1.0, 2.0, 3.0]  # Stack allocation

# 2. @inbounds (after verifying safety)
@inbounds for i in eachindex(x)
    s += x[i]
end

# 3. @simd for vectorization
@simd for i in eachindex(x)
    s += x[i]
end

# 4. const for globals
const PARAM = 0.5  # Fast, inlined

# 5. Preallocate and mutate
function compute!(results::Vector{Float64}, n)
    for i in 1:n
        results[i] = sqrt(i)
    end
end
```

---

## Macros

```julia
# Simple macro
macro time_it(expr)
    quote
        t0 = time_ns()
        result = $(esc(expr))  # esc() prevents variable capture
        println("Time: ", (time_ns() - t0) / 1e9, "s")
        result
    end
end

@time_it sleep(0.1)
```

---

## Generated Functions

```julia
# Compile-time specialization
@generated function tuple_sum(t::Tuple)
    n = length(t.parameters)
    exprs = [:(t[$i]) for i in 1:n]
    return :(+($(exprs...)))
end

tuple_sum((1, 2, 3))  # Generates: +(t[1], t[2], t[3])
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Multiple dispatch | Dispatch on most specific types needed |
| Type system | Abstract for APIs, concrete for struct fields |
| Type stability | Check critical functions with @code_warntype |
| Performance | Profile first, const globals, preallocate |
| Metaprogramming | Prefer functions over macros when possible |

---

## Common Pitfalls

```julia
# ❌ Too specific
function process_data(x::Vector{Float64})
    sum(x)
end

# ✅ Flexible
function process_data(x::AbstractVector{<:Number})
    sum(x)
end

# ❌ Dispatch ambiguity
foo(x::Int, y::Any) = 1
foo(x::Any, y::Int) = 2
foo(3, 4)  # ERROR!

# ✅ Add specific case
foo(x::Int, y::Int) = 3

# ❌ Type changes in loop
result = 0  # Int
result += 1.5  # Now Float64!

# ✅ Consistent type
result = 0.0  # Float64 from start
```

---

## Performance Killers

| Killer | Solution |
|--------|----------|
| Global variables | Use `const` or pass as argument |
| Type instability | Consistent return types |
| Unnecessary allocations | Views, preallocate, mutate |
| Missing @inbounds | Add after verifying bounds safety |

---

## Checklist

- [ ] Multiple dispatch used appropriately
- [ ] Type hierarchy designed (abstract → concrete)
- [ ] Critical functions checked with @code_warntype
- [ ] const for global constants
- [ ] Preallocated arrays where possible
- [ ] @inbounds/@simd for hot loops
- [ ] Macros use esc() properly

---

**Version**: 1.0.5
