---
name: core-julia-patterns
description: Master multiple dispatch, type system, parametric types, and metaprogramming in Julia for high-performance computing. Use when writing or editing Julia source files (.jl), implementing type hierarchies with abstract and concrete types, designing dispatch-based APIs with multiple methods, debugging type stability issues with @code_warntype, ensuring performance with @inbounds and @simd, creating parametric types with where clauses, building macros for code generation, using @generated functions for compile-time specialization, optimizing with StaticArrays for small fixed-size arrays, or working with Julia's core language features (multiple dispatch, type inference, metaprogramming). Essential for all performance-critical Julia code, generic library design, and understanding Julia's unique programming paradigms.
---

# Core Julia Patterns

Master Julia's unique programming paradigms including multiple dispatch, sophisticated type system, parametric types, and metaprogramming capabilities. These patterns are fundamental to writing idiomatic, high-performance Julia code.

## What This Skill Provides

This skill equips you to:

1. **Multiple Dispatch Design** - Leverage Julia's dispatch system for extensible, composable APIs
2. **Type System Mastery** - Design type hierarchies with abstract and concrete types
3. **Parametric Programming** - Write generic, reusable code with type parameters
4. **Type Stability** - Ensure predictable types for optimal performance
5. **Metaprogramming** - Generate code at compile-time with macros and @generated functions
6. **Performance Optimization** - Apply Julia-specific optimization patterns

## When to Use This Skill

Invoke this skill when encountering:

- Multiple dispatch design questions (method specialization, dispatch ambiguities)
- Type system design (abstract types, concrete types, type hierarchies)
- Parametric types and where clauses
- Type stability issues and performance problems
- Metaprogramming needs (macros, generated functions, expression manipulation)
- Performance optimization (const globals, @inbounds, @simd, StaticArrays)
- Generic programming patterns
- API design for extensibility

## Core Concepts

### Multiple Dispatch Fundamentals

Multiple dispatch selects method based on **all** argument types, not just the first (unlike OOP single dispatch).

```julia
# Generic function with multiple methods
function process(x::Int, y::Int)
    x + y
end

function process(x::Float64, y::Float64)
    x * y
end

function process(x::String, y::String)
    x * " and " * y  # String concatenation
end

# Dispatch selects method based on runtime types
process(2, 3)          # → 5 (calls Int method)
process(2.0, 3.0)      # → 6.0 (calls Float64 method)
process("A", "B")      # → "A and B" (calls String method)

# Method with multiple dispatch on different types
function combine(x::Number, y::String)
    string(x) * y
end

function combine(x::String, y::Number)
    x * string(y)
end

combine(42, "!")       # → "42!"
combine("ID:", 7)      # → "ID:7"
```

### Type Hierarchy and Abstract Types

```julia
# Abstract type hierarchy
abstract type Animal end
abstract type Pet <: Animal end  # Subtype of Animal

# Concrete types
struct Dog <: Pet
    name::String
    breed::String
end

struct Cat <: Pet
    name::String
    indoor::Bool
end

struct WildAnimal <: Animal
    species::String
end

# Methods dispatch on abstract types
function greet(animal::Pet)
    "Hello, $(animal.name)!"
end

function greet(animal::WildAnimal)
    "Observing $(animal.species) from a distance"
end

# Works for any Pet subtype
dog = Dog("Rex", "Golden Retriever")
cat = Cat("Whiskers", true)
greet(dog)  # → "Hello, Rex!"
greet(cat)  # → "Hello, Whiskers!"
```

**Design Principle**: Use abstract types in function signatures for flexibility, concrete types in struct fields for performance.

### Parametric Types

```julia
# Parametric struct
struct Point{T<:Real}
    x::T
    y::T
end

# Different instantiations
p1 = Point(1, 2)         # Point{Int64}
p2 = Point(1.5, 2.5)     # Point{Float64}
p3 = Point{Float32}(1.0f0, 2.0f0)  # Explicit type

# Parametric methods
function distance(p1::Point{T}, p2::Point{T}) where T
    sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
end

# Multiple type parameters
struct Pair{T, S}
    first::T
    second::S
end

# Type constraints
struct SortedVector{T<:Real}
    data::Vector{T}

    # Inner constructor with validation
    function SortedVector{T}(data::Vector{T}) where T
        sorted_data = sort(data)
        new{T}(sorted_data)
    end
end

# Convenient outer constructor
SortedVector(data::Vector{T}) where T<:Real = SortedVector{T}(data)
```

**Best Practice**: Use type parameters to write generic code that works with multiple types while maintaining type safety and performance.

### Type Stability

Type stability means the compiler can infer the return type from input types alone.

```julia
# TYPE STABLE (GOOD)
function stable_sum(x::Vector{Float64})
    result = 0.0  # Concrete type
    for val in x
        result += val
    end
    return result  # Always Float64
end

# TYPE UNSTABLE (BAD)
function unstable_conditional(x::Float64)
    if x > 0
        return x^2        # Returns Float64
    else
        return "negative" # Returns String - INSTABILITY!
    end
end

# Check type stability
@code_warntype stable_sum([1.0, 2.0, 3.0])     # Clean (no red)
@code_warntype unstable_conditional(1.0)        # Shows red (Any type)

# Fix: Use Union types if multiple returns needed
function stable_conditional(x::Float64)::Union{Float64, Nothing}
    if x > 0
        return x^2
    else
        return nothing
    end
end

# Better: Redesign to avoid multiple return types
function checked_square(x::Float64)
    x >= 0 || throw(ArgumentError("x must be non-negative"))
    return x^2
end
```

**Debugging Tool**:
```julia
# @code_warntype shows inferred types
@code_warntype my_function(args...)

# Red/pink color = type instability (Any, Union{...} with many types)
# Blue color = stable, inferred types
```

### Performance Optimization Patterns

```julia
using StaticArrays, BenchmarkTools

# 1. Use StaticArrays for small fixed-size arrays
function matrix_multiply_dynamic(v::Vector{Float64})
    M = [1.0 0.0 0.0;
         0.0 2.0 0.0;
         0.0 0.0 3.0]
    return M * v  # Heap allocation
end

function matrix_multiply_static(v::SVector{3, Float64})
    M = @SMatrix [1.0 0.0 0.0;
                  0.0 2.0 0.0;
                  0.0 0.0 3.0]
    return M * v  # Stack allocation, no allocations
end

# 2. Use @inbounds for bounds-check elimination (after ensuring safety)
function sum_with_bounds_check(x::Vector{Float64})
    s = 0.0
    for i in eachindex(x)
        s += x[i]  # Bounds check on every access
    end
    return s
end

function sum_without_bounds_check(x::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(x)
        s += x[i]  # No bounds check (faster)
    end
    return s
end

# 3. Use @simd for SIMD vectorization hints
function sum_simd(x::Vector{Float64})
    s = 0.0
    @simd for i in eachindex(x)
        s += x[i]
    end
    return s
end

# 4. Avoid globals, use const for global constants
# BAD
global_param = 0.5
function uses_global(x)
    return x * global_param  # Slow, global lookup each time
end

# GOOD
const PARAM = 0.5
function uses_const(x)
    return x * PARAM  # Fast, inlined constant
end

# 5. Preallocate and mutate instead of allocating
# BAD
function compute_values(n)
    results = Float64[]
    for i in 1:n
        push!(results, sqrt(i))  # Reallocations
    end
    return results
end

# GOOD
function compute_values!(results::Vector{Float64}, n)
    for i in 1:n
        results[i] = sqrt(i)  # Mutate preallocated array
    end
    return results
end

results = Vector{Float64}(undef, 1000)
compute_values!(results, 1000)
```

### Metaprogramming: Macros

```julia
# Simple macro
macro show_expr(expr)
    println("Expression: ", expr)
    return expr
end

@show_expr 2 + 2  # Prints: Expression: 2 + 2, returns 4

# Macro with quote and esc()
macro time_it(expr)
    quote
        local t0 = time_ns()
        local result = $(esc(expr))  # esc() prevents variable capture
        local t1 = time_ns()
        println("Time: ", (t1 - t0) / 1e9, " seconds")
        result
    end
end

@time_it sleep(0.1)  # Times the expression

# Macro for code generation
macro create_getter_setter(typename, field)
    getter = Symbol("get_", field)
    setter = Symbol("set_", field)
    quote
        function $(esc(getter))(obj::$(esc(typename)))
            return getfield(obj, $(QuoteNode(field)))
        end

        function $(esc(setter))(obj::$(esc(typename)), value)
            return setfield!(obj, $(QuoteNode(field)), value)
        end
    end
end

struct MyStruct
    value::Int
end

@create_getter_setter MyStruct value
# Generates: get_value(obj) and set_value(obj, val)
```

### Metaprogramming: Generated Functions

```julia
# @generated functions execute at compile-time
@generated function tuple_sum(t::Tuple)
    # This code runs once per type at compile time
    n = length(t.parameters)
    exprs = [:(t[$i]) for i in 1:n]
    return :(+($(exprs...)))
end

# Generates specialized code for each tuple size
tuple_sum((1, 2, 3))        # Generates: +(t[1], t[2], t[3])
tuple_sum((1.0, 2.0, 3.0, 4.0))  # Generates: +(t[1], t[2], t[3], t[4])

# Practical use: Type-specific optimizations
@generated function my_dot(a::AbstractArray{T}, b::AbstractArray{T}) where T
    if T <: Integer
        # Integer-specific implementation
        return quote
            s = zero($T)
            @inbounds @simd for i in eachindex(a)
                s += a[i] * b[i]
            end
            s
        end
    else
        # Floating-point implementation with fused multiply-add
        return quote
            s = zero($T)
            @inbounds @simd for i in eachindex(a)
                s = muladd(a[i], b[i], s)
            end
            s
        end
    end
end
```

## Best Practices

### Multiple Dispatch
- Design methods to dispatch on the most specific types needed
- Avoid dispatch ambiguities (methods with overlapping type signatures)
- Use abstract types in signatures for flexibility
- Group related methods under one generic function name

### Type System
- Use abstract types for APIs and function signatures
- Use concrete types for struct fields (performance)
- Avoid Union types with many members (type instability)
- Prefer type parameters over Any
- Document type constraints clearly

### Type Stability
- Always check critical functions with @code_warntype
- Return consistent types from all code paths
- Annotate return types if compiler struggles: `function f(x)::Int`
- Use type assertions in loops: `@assert x isa Float64`
- Avoid changing variable types within a function

### Performance
- Profile before optimizing (@profile, @profview, BenchmarkTools.jl)
- Use const for global constants
- Preallocate arrays and mutate instead of allocating
- Use @inbounds only after verifying bounds safety
- Consider StaticArrays.jl for small fixed-size arrays
- Use @simd for vectorizable loops
- Avoid type instabilities (biggest performance killer)

### Metaprogramming
- Prefer functions over macros when possible
- Use esc() to prevent variable capture in macros
- Document macro hygiene issues
- Test macros with different input expressions
- Use @generated for compile-time specialization only when needed
- Keep generated code readable

## Common Pitfalls

### Multiple Dispatch Anti-Patterns
```julia
# BAD: Too specific, inflexible
function process_data(x::Vector{Float64})
    sum(x)
end

# GOOD: Accept any AbstractVector of Numbers
function process_data(x::AbstractVector{<:Number})
    sum(x)
end

# BAD: Dispatch ambiguity
foo(x::Int, y::Any) = 1
foo(x::Any, y::Int) = 2
foo(3, 4)  # ERROR: Ambiguous!

# GOOD: Define specific case
foo(x::Int, y::Any) = 1
foo(x::Any, y::Int) = 2
foo(x::Int, y::Int) = 3  # Resolves ambiguity
```

### Type Instability Traps
```julia
# BAD: Changes type in loop
function accumulate_mixed(n)
    result = 0  # Int
    for i in 1:n
        result += i / 2  # Now Float64!
    end
    return result
end

# GOOD: Consistent type
function accumulate_consistent(n)
    result = 0.0  # Float64 from start
    for i in 1:n
        result += i / 2
    end
    return result
end
```

### Performance Killers
```julia
# BAD: Global variable lookup in hot loop
global_scale = 2.5
function scale_array_bad(x)
    return [val * global_scale for val in x]  # Slow
end

# GOOD: Use const or pass as argument
const SCALE = 2.5
function scale_array_good(x)
    return [val * SCALE for val in x]  # Fast
end

# BAD: Unnecessary allocations
function moving_average_bad(x::Vector{Float64}, window::Int)
    result = Float64[]
    for i in window:length(x)
        push!(result, mean(x[i-window+1:i]))  # Allocates slice!
    end
    return result
end

# GOOD: Preallocate and avoid slicing
function moving_average_good(x::Vector{Float64}, window::Int)
    n = length(x) - window + 1
    result = Vector{Float64}(undef, n)
    for i in 1:n
        result[i] = mean(@view x[i:i+window-1])  # View, no allocation
    end
    return result
end
```

### Metaprogramming Pitfalls
```julia
# BAD: Variable capture
macro bad_macro(x)
    quote
        result = $x + 1  # 'result' may collide with outer scope
        result
    end
end

# GOOD: Use esc() or gensym()
macro good_macro(x)
    result = gensym("result")  # Generate unique symbol
    quote
        $result = $(esc(x)) + 1
        $result
    end
end
```

## Resources

- **Official Julia Documentation**: https://docs.julialang.org/
- **Performance Tips**: https://docs.julialang.org/en/v1/manual/performance-tips/
- **Type System**: https://docs.julialang.org/en/v1/manual/types/
- **Metaprogramming**: https://docs.julialang.org/en/v1/manual/metaprogramming/
- **Julia Style Guide**: https://docs.julialang.org/en/v1/manual/style-guide/

## Related Skills

- **jump-optimization**: JuMP.jl mathematical optimization patterns
- **package-management**: Project.toml and Pkg.jl workflows
- **performance-tuning**: Profiling and optimization (sciml-pro)
- **visualization-patterns**: Plots.jl and Makie.jl
- **interop-patterns**: Cross-language integration
