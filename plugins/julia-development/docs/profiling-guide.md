# Julia Profiling Guide

Comprehensive guide to profiling Julia code using modern tools and techniques.

## Table of Contents

- [Profiling Tools Overview](#profiling-tools-overview)
- [BenchmarkTools.jl](#benchmarktoolsjl)
- [Profile.jl & ProfileView.jl](#profilejl--profileviewjl)
- [Type Stability Analysis](#type-stability-analysis)
- [Memory Profiling](#memory-profiling)
- [Interpretation Guide](#interpretation-guide)

---

## Profiling Tools Overview

### Tool Selection Matrix

| Tool | Use Case | Output | Overhead |
|------|----------|--------|----------|
| **@time** | Quick check | Text (time, allocations) | Minimal |
| **@benchmark** | Accurate timing | Statistics | Low |
| **@profile** | Find bottlenecks | Flame graph | Medium |
| **@code_warntype** | Type stability | Colored IR | None |
| **@code_llvm** | LLVM inspection | LLVM IR | None |
| **@code_native** | Assembly code | Native assembly | None |

---

## BenchmarkTools.jl

### Basic Usage

```julia
using BenchmarkTools

# Single run with statistics
@benchmark my_function($args)

# Quick timing (less accurate)
@btime my_function($args)
```

### Interpreting Results

```julia
julia> @benchmark sort(rand(1000))
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  42.125 μs … 125.458 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     43.500 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   44.234 μs ±   2.894 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

 Memory estimate: 15.88 KiB, allocs estimate: 2.
```

**Key metrics**:
- **Median time**: Most representative (use this)
- **Mean ± σ**: Average with standard deviation
- **Min/max**: Range of timings
- **GC**: Garbage collection percentage
- **Memory estimate**: Total allocations
- **Allocs estimate**: Number of allocations

### $ Interpolation (Important!)

```julia
x = rand(1000)

# ❌ Wrong: x captured as global, type-unstable
@benchmark sort(x)

# ✅ Correct: x interpolated, type-stable
@benchmark sort($x)
```

**Rule**: Always use `$` for variables in `@benchmark`.

### Setup and Teardown

```julia
@benchmark expensive_computation($data) setup=(data=prepare_data())
```

### Comparing Implementations

```julia
# Baseline
baseline = @benchmark implementation_v1($data)

# Optimized
optimized = @benchmark implementation_v2($data)

# Compare
ratio = median(baseline).time / median(optimized).time
println("Speedup: $(round(ratio, digits=2))x")
```

---

## Profile.jl & ProfileView.jl

### Sampling Profiler

Julia's profiler is statistical: samples the call stack at regular intervals.

### Basic Workflow

```julia
using Profile
using ProfileView

# Profile your code
@profile my_function(args)

# View flame graph
ProfileView.view()
```

### Understanding Flame Graphs

**Anatomy**:
- **Width**: Time spent in function (wider = slower)
- **Height**: Call stack depth (taller = deeper)
- **Color**: Red/yellow = hot (optimize these)

**Reading Tips**:
1. Focus on wide red/yellow boxes
2. Ignore thin slivers (not worth optimizing)
3. Look for surprising functions (unexpected calls)

### Profile Options

```julia
# Clear previous profile data
Profile.clear()

# Profile with more samples (default: 1M per second)
@profile my_function(args)

# Set sample rate (samples per second)
Profile.init(n=10^7, delay=0.0001)
```

### Console Output

```julia
Profile.print()  # Text output

# Format options
Profile.print(format=:flat)  # Flat listing
Profile.print(format=:tree)  # Call tree
Profile.print(mincount=100)  # Filter rare functions
```

---

## Type Stability Analysis

### @code_warntype

Most important tool for performance debugging.

```julia
@code_warntype my_function(args)
```

### Color Guide

- **Green/Blue**: Type-stable (good!)
- **Yellow**: Abstract type (caution)
- **Red**: `Any` or `Union` (bad!)

### Example Analysis

```julia
function unstable(x)
    if x > 0
        return x
    else
        return "negative"
    end
end

@code_warntype unstable(5.0)
```

**Output** (simplified):
```
Body::Union{Float64, String}  # Red flag: Union return type
```

**Fix**:
```julia
function stable(x)
    if x > 0
        return x
    else
        return 0.0  # Same type
    end
end

@code_warntype stable(5.0)
# Body::Float64  # Green: stable!
```

### Common Type Issues

#### Issue 1: Abstract Container Types

```julia
# Bad
function sum_bad(arr::AbstractArray)
    s = 0  # Type Int, but arr might be Float64
    for x in arr
        s += x
    end
    return s
end

# Good
function sum_good(arr::AbstractArray{T}) where T
    s = zero(T)
    for x in arr
        s += x
    end
    return s
end
```

#### Issue 2: Empty Container Initialization

```julia
# Bad
function collect_bad(n)
    result = []  # Vector{Any}
    for i in 1:n
        push!(result, i^2)
    end
    return result
end

# Good
function collect_good(n)
    result = Int[]  # Vector{Int}
    for i in 1:n
        push!(result, i^2)
    end
    return result
end
```

#### Issue 3: Conditional Return Types

```julia
# Bad
function process_bad(x)
    if x > 0
        return x
    else
        return nothing  # Union{Float64, Nothing}
    end
end

# Good: Use sentinel value
function process_good(x)
    return x > 0 ? x : 0.0
end

# Or: Throw exception
function process_exception(x)
    x > 0 || throw(ArgumentError("x must be positive"))
    return x
end
```

### Function Barriers

When type instability is unavoidable, use function barriers:

```julia
# Type-unstable caller
function caller(flag)
    data = flag ? [1, 2, 3] : [1.0, 2.0, 3.0]  # Union type
    return process_stable(data)  # Barrier
end

# Type-stable worker
function process_stable(data)
    # Julia specializes this for actual type
    sum = zero(eltype(data))
    for x in data
        sum += x
    end
    return sum
end
```

---

## Memory Profiling

### Allocation Tracking

```julia
using BenchmarkTools

@benchmark my_function($args)
# Look at "memory" and "allocs" fields
```

### Identifying Allocation Sources

```julia
# Run with allocation tracking
@time my_function(args)

# Sample output:
# 0.450000 seconds (1.50 M allocations: 68.665 MiB, 5.23% gc time)
```

**High allocations** = potential optimization target.

### @allocations Macro

```julia
@allocations my_function(args)
# Returns number of allocations
```

### Finding Allocation Hot Spots

```julia
using Profile

# Allocations profiler
Profile.Allocs.clear()
Profile.Allocs.@profile my_function(args)

# View results
Profile.Allocs.print()
```

---

## Interpretation Guide

### Benchmark Results

#### Good Performance Indicators

```julia
julia> @benchmark my_function($data)
BenchmarkTools.Trial: 10000 samples
 Time  (median):     50.000 μs
 Memory estimate: 0 bytes, allocs estimate: 0  # ✅ Zero allocations!
 GC (median):    0.00%  # ✅ No GC pauses
```

#### Performance Issues

```julia
julia> @benchmark my_function($data)
BenchmarkTools.Trial: 100 samples
 Time  (median):     5.500 ms
 Memory estimate: 150.00 MiB, allocs estimate: 1500000  # ❌ Too many!
 GC (median):    45.23%  # ❌ Half the time in GC!
```

**Diagnosis**: Memory allocation problem. Check for:
- Missing pre-allocation
- Unnecessary copies
- Type instabilities causing boxing

### Profile Flame Graph

#### Healthy Profile

```
┌───────────────────────────────────────────┐
│ my_function                                │ 100%
├─────────────────┬─────────────────────────┤
│ compute_kernel  │ other_work              │ 60% | 40%
└─────────────────┴─────────────────────────┘
```

**Good**: Time distributed across multiple functions, no single bottleneck.

#### Problematic Profile

```
┌───────────────────────────────────────────┐
│ my_function                                │ 100%
├───────────────────────────────────────────┤
│ unnecessary_allocation                     │ 95% ❌
└───────────────────────────────────────────┘
```

**Bad**: 95% of time in one function. Optimize this!

### Type Stability

#### Type-Stable (Good)

```julia
Body::Float64
  # ... green/blue text ...
```

#### Type-Unstable (Bad)

```julia
Body::Union{Float64, Int64, Nothing}  # ❌ Red
  # ... yellow/red text ...
```

**Fix**: Refactor to return consistent types.

---

## Profiling Workflow

### Complete Workflow Example

```julia
using BenchmarkTools, Profile, ProfileView

# Step 1: Baseline measurement
println("=== Baseline ===")
@benchmark my_function($test_data)

# Step 2: Statistical profiling
println("\n=== Profiling ===")
Profile.clear()
@profile for _ in 1:1000
    my_function(test_data)
end
ProfileView.view()  # Examine flame graph

# Step 3: Type stability check
println("\n=== Type Stability ===")
@code_warntype my_function(test_data)

# Step 4: Make optimizations
# ... (your changes) ...

# Step 5: Verify improvement
println("\n=== After Optimization ===")
@benchmark my_function_optimized($test_data)

# Step 6: Compare
baseline = @benchmark my_function($test_data)
optimized = @benchmark my_function_optimized($test_data)

speedup = median(baseline).time / median(optimized).time
println("\nSpeedup: $(round(speedup, digits=2))x")

alloc_reduction = (median(baseline).memory - median(optimized).memory) / median(baseline).memory * 100
println("Allocation reduction: $(round(alloc_reduction, digits=1))%")
```

---

## Advanced Techniques

### CPU Cycle Counting

```julia
using BenchmarkTools

# Get CPU cycles instead of time
b = @benchmark my_function($data)
cycles = median(b).time * 2.5e9  # Assuming 2.5 GHz CPU
println("CPU cycles: $cycles")
```

### Cache Analysis

```julia
# Estimate cache misses (requires Linux perf)
using Profile
Profile.init(n=10^7)
@profile my_function(data)

# Look for wide cache-miss patterns in flame graph
```

### Compilation Time

```julia
# Measure compilation overhead
@time my_function(args)  # First run: includes compilation
@time my_function(args)  # Second run: no compilation

# Precompile
precompile(my_function, (typeof(args),))
```

---

## Best Practices

### DO ✅

1. **Use $ interpolation** in `@benchmark`
2. **Profile before optimizing** (find bottlenecks)
3. **Check type stability** with `@code_warntype`
4. **Measure allocations** with `@benchmark`
5. **Compare before/after** with median times
6. **Focus on hot spots** (top 20% of functions)
7. **Run multiple samples** for statistical significance

### DON'T ❌

1. **Don't optimize without profiling** (premature optimization)
2. **Don't use @time for benchmarking** (use `@benchmark`)
3. **Don't ignore type instabilities** (major performance killer)
4. **Don't micro-optimize cold code** (rarely executed)
5. **Don't trust first run** (includes compilation time)
6. **Don't forget $ in @benchmark** (creates type instability)

---

## Profiling Checklist

- [ ] Run `@benchmark` for accurate timing
- [ ] Use `$` interpolation for variables
- [ ] Profile with `@profile` and `ProfileView`
- [ ] Check type stability with `@code_warntype`
- [ ] Measure allocations (memory estimate)
- [ ] Compare before/after optimizations
- [ ] Document speedup and allocation reduction
- [ ] Verify correctness after optimization

---

## Common Profiling Scenarios

### Scenario 1: Function is Slow

1. **Profile**: Find hot spot with `@profile`
2. **Type check**: Run `@code_warntype` on hot function
3. **Fix types**: Eliminate `Union` and `Any` types
4. **Re-profile**: Verify improvement

### Scenario 2: High Memory Usage

1. **Measure**: Run `@benchmark`, check memory estimate
2. **Find allocations**: Use `Profile.Allocs.@profile`
3. **Pre-allocate**: Create arrays before loops
4. **Use views**: Replace slicing with `@view`
5. **Verify**: Confirm allocation reduction

### Scenario 3: Slow Startup

1. **Measure compilation**: First vs second `@time`
2. **Precompile**: Add `precompile` statements
3. **Reduce generic code**: Use concrete types
4. **Consider PackageCompiler.jl**: Create system image

---

**Version**: 1.0.3
**Last Updated**: 2025-11-07
**Plugin**: julia-development
