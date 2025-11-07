# Julia Optimization Patterns

Comprehensive guide to profiling and optimizing Julia code for maximum performance.

## Table of Contents

- [Type Stability Patterns](#type-stability-patterns)
- [Allocation Reduction](#allocation-reduction)
- [Parallelization Strategies](#parallelization-strategies)
- [Memory Optimization](#memory-optimization)
- [Algorithm Improvements](#algorithm-improvements)

---

## Type Stability Patterns

### What is Type Stability?

A function is type-stable if the type of its output can be inferred from the types of its inputs. This allows the compiler to generate optimized code.

### Detecting Type Instability

```julia
@code_warntype my_function(args...)
```

**Red flags**:
- `Union{...}` types (especially with `Nothing`)
- `Any` type
- Abstract types in returns

### Common Type Instability Patterns

#### Pattern 1: Conditional Return Types

❌ **Type-Unstable**:
```julia
function process_data(x)
    if x > 0
        return x * 2.0        # Returns Float64
    else
        return "negative"     # Returns String
    end
end
```

✅ **Type-Stable**:
```julia
function process_data(x)
    if x > 0
        return x * 2.0
    else
        return 0.0  # Same type
    end
end
```

#### Pattern 2: Abstract Container Types

❌ **Type-Unstable**:
```julia
function sum_array(arr::AbstractArray)
    result = 0  # Int, but will contain Float64
    for x in arr
        result += x
    end
    return result
end
```

✅ **Type-Stable**:
```julia
function sum_array(arr::AbstractArray{T}) where T
    result = zero(T)  # Type matches array elements
    for x in arr
        result += x
    end
    return result
end
```

#### Pattern 3: Empty Container Initialization

❌ **Type-Unstable**:
```julia
function create_array(n)
    arr = []  # Type Vector{Any}
    for i in 1:n
        push!(arr, i^2)
    end
    return arr
end
```

✅ **Type-Stable**:
```julia
function create_array(n)
    arr = Int[]  # Type Vector{Int}
    for i in 1:n
        push!(arr, i^2)
    end
    return arr
end

# Or better: pre-allocate
function create_array_fast(n)
    arr = Vector{Int}(undef, n)
    for i in 1:n
        arr[i] = i^2
    end
    return arr
end
```

#### Pattern 4: Global Variables

❌ **Type-Unstable**:
```julia
x = 1  # Global, type can change

function use_global()
    return x + 1  # Compiler can't infer type
end
```

✅ **Type-Stable**:
```julia
const x = 1  # Const global

function use_global()
    return x + 1  # Type stable
end

# Or pass as argument
function use_param(x)
    return x + 1
end
```

---

## Allocation Reduction

### Measuring Allocations

```julia
using BenchmarkTools

@benchmark my_function(args...)
# Look at "memory" and "allocs" fields
```

### Pre-allocation Patterns

#### Pattern 1: Array Pre-allocation

❌ **Many Allocations**:
```julia
function compute_many_times(n)
    results = Float64[]
    for i in 1:n
        result = expensive_computation(i)
        push!(results, result)
    end
    return results
end
```

✅ **Pre-allocated**:
```julia
function compute_many_times(n)
    results = Vector{Float64}(undef, n)
    for i in 1:n
        results[i] = expensive_computation(i)
    end
    return results
end
```

#### Pattern 2: In-Place Operations

❌ **Allocating**:
```julia
function update_state(state, delta)
    return state + delta  # Allocates new array
end
```

✅ **In-Place**:
```julia
function update_state!(state, delta)
    state .+= delta  # Modifies in-place
    return state
end

# Or using @. macro
function update_state!(state, delta)
    @. state += delta
    return state
end
```

#### Pattern 3: Views Instead of Copies

❌ **Copying**:
```julia
function process_subarray(arr, start, stop)
    sub = arr[start:stop]  # Copies data
    return sum(sub)
end
```

✅ **View**:
```julia
function process_subarray(arr, start, stop)
    sub = @view arr[start:stop]  # No copy
    return sum(sub)
end
```

#### Pattern 4: Broadcasting vs. Loops

```julia
# Both are good, broadcasting is more concise
function scale_array!(arr, factor)
    arr .*= factor  # Broadcasting, in-place
end

function scale_array!(arr, factor)
    @inbounds for i in eachindex(arr)
        arr[i] *= factor
    end
end
```

### SIMD Optimization

```julia
function sum_simd(arr)
    s = zero(eltype(arr))
    @simd for i in eachindex(arr)
        @inbounds s += arr[i]
    end
    return s
end
```

**Requirements**:
- Loop iterations must be independent
- No early returns or breaks
- Use `@inbounds` to skip bounds checking

---

## Parallelization Strategies

### Multi-Threading

#### Pattern 1: Parallel Loop

```julia
using Base.Threads

function parallel_sum(arr)
    n = length(arr)
    sums = zeros(nthreads())

    @threads for i in 1:n
        tid = threadid()
        @inbounds sums[tid] += arr[i]
    end

    return sum(sums)
end
```

**When to use**:
- Independent iterations
- Workload > threading overhead
- Shared memory sufficient

#### Pattern 2: Parallel Map

```julia
function process_parallel(data)
    results = Vector{Float64}(undef, length(data))
    @threads for i in eachindex(data)
        results[i] = expensive_function(data[i])
    end
    return results
end
```

### Distributed Computing

```julia
using Distributed

# Add workers
addprocs(4)

@everywhere function worker_task(x)
    # Heavy computation
    return result
end

# Parallel map
results = pmap(worker_task, data)

# Or using @distributed
@distributed (+) for i in 1:1000000
    compute(i)
end
```

**When to use**:
- Data too large for single machine
- Network overhead < computation time
- Tasks are independent

### GPU Computing

```julia
using CUDA

function gpu_compute(arr)
    d_arr = CuArray(arr)  # Transfer to GPU
    d_result = d_arr .^ 2 .+ 1  # GPU computation
    return Array(d_result)  # Transfer back
end
```

**When to use**:
- Large data arrays (> 10^6 elements)
- Embarrassingly parallel operations
- Transfer overhead < computation speedup

---

## Memory Optimization

### Static Arrays

```julia
using StaticArrays

# For small fixed-size arrays (< 100 elements)
function compute_fast()
    v = @SVector [1.0, 2.0, 3.0]  # Stack-allocated
    return v .* 2
end

# Mutation with MVector
function compute_mutable()
    v = @MVector [1.0, 2.0, 3.0]
    v[1] = 5.0
    return SVector(v)
end
```

**Benefits**:
- Stack allocation (no GC)
- SIMD optimization
- Inlining

**Use when**: Array size ≤ 100 elements and known at compile time

### Memory Layout

#### Column-Major vs Row-Major

Julia uses **column-major** ordering (like Fortran, MATLAB).

❌ **Slow (row-major)**:
```julia
function sum_rows_slow(mat)
    m, n = size(mat)
    sums = zeros(m)
    for i in 1:m
        for j in 1:n
            sums[i] += mat[i, j]  # Poor cache locality
        end
    end
    return sums
end
```

✅ **Fast (column-major)**:
```julia
function sum_rows_fast(mat)
    m, n = size(mat)
    sums = zeros(m)
    for j in 1:n
        for i in 1:m
            sums[i] += mat[i, j]  # Good cache locality
        end
    end
    return sums
end
```

**Rule**: Inner loop should iterate over first index.

### Struct of Arrays vs Array of Structs

#### Array of Structs (AoS)

```julia
struct Particle
    x::Float64
    y::Float64
    vx::Float64
    vy::Float64
end

particles = [Particle(rand(), rand(), rand(), rand()) for _ in 1:10000]

function update_aos!(particles, dt)
    for p in particles
        # Bad: scattered memory access
        p.x += p.vx * dt
    end
end
```

#### Struct of Arrays (SoA)

```julia
struct ParticleSystem
    x::Vector{Float64}
    y::Vector{Float64}
    vx::Vector{Float64}
    vy::Vector{Float64}
end

function update_soa!(sys, dt)
    @. sys.x += sys.vx * dt  # Vectorized, good cache
    @. sys.y += sys.vy * dt
end
```

**Use SoA when**: Accessing same field across many objects.

---

## Algorithm Improvements

### Algorithmic Complexity

Before micro-optimizing, check if you're using the right algorithm.

#### Example: Finding Duplicates

❌ **O(n²)**:
```julia
function has_duplicates_slow(arr)
    n = length(arr)
    for i in 1:n
        for j in (i+1):n
            if arr[i] == arr[j]
                return true
            end
        end
    end
    return false
end
```

✅ **O(n)**:
```julia
function has_duplicates_fast(arr)
    seen = Set{eltype(arr)}()
    for x in arr
        if x ∈ seen
            return true
        end
        push!(seen, x)
    end
    return false
end
```

### Memoization

```julia
# Naive Fibonacci (exponential time)
function fib_slow(n)
    n ≤ 2 && return 1
    return fib_slow(n-1) + fib_slow(n-2)
end

# Memoized (linear time)
const fib_cache = Dict{Int,Int}()

function fib_fast(n)
    n ≤ 2 && return 1
    if haskey(fib_cache, n)
        return fib_cache[n]
    end
    result = fib_fast(n-1) + fib_fast(n-2)
    fib_cache[n] = result
    return result
end
```

### Loop Fusion

❌ **Multiple Passes**:
```julia
function multi_pass(arr)
    temp1 = arr .^ 2
    temp2 = temp1 .+ 1
    result = sqrt.(temp2)
    return result
end
```

✅ **Single Pass**:
```julia
function single_pass(arr)
    return @. sqrt(arr^2 + 1)
end
```

### Function Specialization

```julia
# Generic function
function compute(x, op)
    return op(x)
end

# Specialized for common case
function compute_square(x)
    return x * x
end

# Compiler generates optimized code for compute_square
```

---

## Profiling Workflow

### Step 1: Measure Baseline

```julia
using BenchmarkTools

@benchmark my_function($args)
```

### Step 2: Profile

```julia
using Profile
using ProfileView

@profile my_function(args)
ProfileView.view()
```

Look for:
- Hot spots (red/yellow in flame graph)
- Type instabilities
- Allocations

### Step 3: Check Type Stability

```julia
@code_warntype my_function(args)
```

Look for `Any`, `Union`, abstract types.

### Step 4: Optimize Hot Spots

Focus on functions that take most time.

### Step 5: Verify Improvement

```julia
@benchmark my_function_optimized($args)
```

Compare with baseline.

---

## Optimization Checklist

- [ ] Type stability verified with `@code_warntype`
- [ ] Allocations minimized (check with `@benchmark`)
- [ ] Pre-allocated arrays where possible
- [ ] In-place operations used (`!` functions)
- [ ] Views used instead of copies (`@view`)
- [ ] Correct loop order (column-major)
- [ ] SIMD used for independent loops (`@simd`, `@inbounds`)
- [ ] Parallelization considered (threads, distributed, GPU)
- [ ] Algorithm complexity optimal (O(n) vs O(n²))
- [ ] Static arrays for small fixed-size data
- [ ] Function barriers for type stability
- [ ] Profiling done to find bottlenecks

---

## Common Performance Pitfalls

### Pitfall 1: Premature Optimization

Optimize only after profiling identifies bottlenecks.

### Pitfall 2: Global Variables

Use `const` or pass as arguments.

### Pitfall 3: Type Instability

Check with `@code_warntype` early.

### Pitfall 4: Small Functions Not Inlining

Julia usually inlines automatically, but check with `@inline`.

### Pitfall 5: Over-Parallelization

Threading overhead can exceed benefits for small workloads.

---

**Version**: 1.0.3
**Last Updated**: 2025-11-07
**Plugin**: julia-development
