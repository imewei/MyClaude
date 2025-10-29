---
name: performance-tuning
description: Profile Julia code and optimize performance with @code_warntype, @profview, BenchmarkTools.jl, and allocation reduction. Foundation for /julia-optimize command analysis.
---

# Performance Tuning

Master profiling and optimizing Julia code for maximum performance.

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
