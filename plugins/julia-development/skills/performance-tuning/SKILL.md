---
name: performance-tuning
description: Master profiling and optimizing Julia code with @code_warntype, @profview, BenchmarkTools.jl, and allocation reduction techniques for maximum performance. Use when debugging slow Julia code, checking type stability with @code_warntype (looking for red/Any types), profiling with @profview and ProfileView.jl, benchmarking with @benchmark and @btime, reducing memory allocations, identifying performance bottlenecks in hot loops, optimizing @inbounds and @simd usage, analyzing with @time and @allocated, comparing algorithm performance, or improving execution speed. Foundation for /julia-optimize command analysis and essential for high-performance computing applications.
---

# Performance Tuning

Master profiling and optimizing Julia code for maximum performance.

## When to use this skill

- Debugging slow Julia code and identifying bottlenecks
- Checking type stability with @code_warntype (red = type instability)
- Profiling execution with @profview and ProfileView.jl flame graphs
- Benchmarking code with @benchmark, @btime from BenchmarkTools.jl
- Reducing memory allocations in hot loops
- Analyzing execution time with @time and @allocated
- Optimizing performance-critical sections with @inbounds, @simd
- Comparing algorithm performance
- Improving startup time with precompilation
- Optimizing array operations and memory access patterns
- Reducing garbage collection pressure
- Tuning parallel and GPU code performance

## Profiling Tools

```julia
using BenchmarkTools, ProfileView

# Benchmark
@benchmark my_function(args)

# Profile
@profview my_function(large_input)

# Type stability
@code_warntype my_function(args)  # Look for red/Any types

# Allocations
@time my_function(args)  # Shows allocations
```

## Optimization Checklist

1. Check type stability (@code_warntype)
2. Profile to find bottlenecks (@profview)
3. Reduce allocations
4. Use @inbounds in safe loops
5. Consider StaticArrays for small arrays
6. Use @simd for vectorization
7. Consider parallelization

## Resources
- **Performance Tips**: https://docs.julialang.org/en/v1/manual/performance-tips/
