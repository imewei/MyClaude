---
name: performance-tuning
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Performance
description: Profile and optimize Julia code with @code_warntype, @profview, and BenchmarkTools.jl. Use when debugging slow code, reducing allocations, or improving execution speed.
---

# Julia Performance Tuning

Profiling and optimizing Julia code for maximum performance.

---

## Profiling Tools

```julia
using BenchmarkTools, ProfileView

# Benchmark
@benchmark my_function(args)

# Profile (flame graph)
@profview my_function(large_input)

# Type stability (red = bad)
@code_warntype my_function(args)

# Allocations
@time my_function(args)
```

---

## Optimization Workflow

1. **Identify**: Profile to find bottlenecks
2. **Type stability**: Check @code_warntype
3. **Reduce allocations**: Preallocate, avoid temporaries
4. **Optimize loops**: @inbounds, @simd
5. **Parallelize**: If still slow

---

## Common Fixes

| Problem | Solution |
|---------|----------|
| Red types in @code_warntype | Add type annotations |
| High allocations | Preallocate arrays |
| Slow loops | @inbounds, @simd |
| Small arrays | Use StaticArrays |

---

## Checklist

- [ ] Profiled with @profview
- [ ] Type stability checked
- [ ] Allocations minimized
- [ ] @inbounds used in safe loops
- [ ] Benchmarked improvements

---

**Version**: 1.0.5
