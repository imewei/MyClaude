---
description: Profile Julia code and provide optimization recommendations through type
  stability, allocation, and bottleneck analysis
triggers:
- /julia-optimize
- profile julia code and
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<file_path>`
The agent should parse these arguments from the user's request.

# Performance Profiling and Optimization

Target: $ARGUMENTS

## Phase 1: Baseline

```julia
using BenchmarkTools
@benchmark target_function($args)  # Capture: median time, memory, allocs, GC%
```

## Phase 2: Type Stability

```julia
@code_warntype target_function(args)
```

| Color | Fix |
|-------|-----|
| Red (`Any`, `Union`) | Critical - fix immediately |
| Yellow (Abstract) | Consider concretizing |

**Common issues**: Conditional returns, abstract containers, empty `[]`, globals

## Phase 3: Profiling

```julia
using Profile, ProfileView
@profile for _ in 1:1000; target_function(args); end
ProfileView.view()  # Width = time, red = hot spots
```

## Phase 4: Allocations

```julia
@allocations target_function(args)
Profile.Allocs.@profile target_function(args)
```

**Fixes**: Pre-allocate, use `@view`, fix type instabilities, reuse buffers

## Recommendations Priority

| Priority | Category | Speedup |
|----------|----------|---------|
| ⭐⭐⭐⭐⭐ | Type stability | 3-10x |
| ⭐⭐⭐⭐⭐ | Algorithm | 10-100x |
| ⭐⭐⭐⭐ | Allocations | 2-5x |
| ⭐⭐⭐ | Pre-allocation | 1.5-3x |

## Verification

```julia
baseline = @benchmark original($data)
optimized = @benchmark improved($data)
speedup = median(baseline).time / median(optimized).time
```

**Docs**: `optimization-patterns.md`, `profiling-guide.md`
