---
version: "1.0.6"
category: "julia-development"
command: "/julia-optimize"
description: Profile Julia code and provide optimization recommendations through type stability, allocation, and bottleneck analysis
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<file_path>"
color: green
execution_modes:
  quick: "5-10 minutes"
  standard: "15-25 minutes"
  comprehensive: "30-45 minutes"
agents:
  primary:
    - julia-pro
  conditional:
    - agent: sciml-pro
      trigger: pattern "differential.*equation|ode|pde|sciml"
    - agent: hpc-numerical-coordinator
      trigger: pattern "parallel|distributed|gpu|mpi"
  orchestrated: false
---

# Performance Profiling and Optimization

Profile Julia code and provide actionable optimization recommendations.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 5-10 min | Baseline + type stability + top 3 opportunities |
| Standard (default) | 15-25 min | Full analysis with code examples |
| Comprehensive | 30-45 min | Deep analysis + parallelization + implementation guide |

---

## Phase 1: Baseline Measurement

```julia
using BenchmarkTools
@benchmark target_function($args)
```

### Metrics to Capture
| Metric | Purpose |
|--------|---------|
| Median time | Most reliable timing |
| Memory estimate | Total allocations |
| Allocs estimate | Allocation count |
| GC percentage | Garbage collection overhead |

---

## Phase 2: Type Stability Analysis

```julia
@code_warntype target_function(args)
```

### Interpretation

| Color | Type | Action |
|-------|------|--------|
| Red | `Any`, `Union` | ❌ Critical - fix immediately |
| Yellow | Abstract | ⚠️ Caution - consider concretizing |
| Green/Blue | Concrete | ✅ Good |

### Common Issues
- Conditional return types
- Abstract container types
- Empty container initialization (`[]`)
- Global variable usage

---

## Phase 3: Profiling

```julia
using Profile, ProfileView
@profile for _ in 1:1000; target_function(args); end
ProfileView.view()
```

### Flame Graph Analysis
| Indicator | Meaning |
|-----------|---------|
| Width | Time spent (wider = slower) |
| Red/Yellow | Hot spots (optimize these) |
| Unexpected functions | Hidden allocations |

---

## Phase 4: Allocation Analysis

```julia
@allocations target_function(args)
Profile.Allocs.@profile target_function(args)
```

### Common Culprits
| Issue | Fix |
|-------|-----|
| Missing pre-allocation | Allocate before loops |
| Array copies | Use `@view` for slicing |
| Type instabilities | Fix causes boxing |
| Temporary arrays | Reuse buffers |

---

## Phase 5: Recommendations

### Priority Ranking

| Priority | Category | Typical Speedup |
|----------|----------|-----------------|
| ⭐⭐⭐⭐⭐ | Type stability fixes | 3-10x |
| ⭐⭐⭐⭐⭐ | Algorithm improvements | 10-100x |
| ⭐⭐⭐⭐ | Allocation elimination | 2-5x |
| ⭐⭐⭐ | Pre-allocation | 1.5-3x |
| ⭐⭐⭐ | Views vs copies | 1.5-2x |
| ⭐⭐ | SIMD/loop order | 1.2-2x |

---

## Phase 6: Verification

```julia
baseline = @benchmark original_function($data)
optimized = @benchmark optimized_function($data)
speedup = median(baseline).time / median(optimized).time
println("Speedup: $(round(speedup, digits=2))x")
```

---

## Optimization Categories

### Type Stability (Critical)
**Symptoms:** `Union`/`Any` in @code_warntype, slow despite simple code
**Fix patterns:** See `optimization-patterns.md#type-stability-patterns`

### Allocations (High Impact)
**Symptoms:** High memory, large GC%, allocations in hot loops
**Fix patterns:** See `optimization-patterns.md#allocation-reduction`

### Parallelization (Medium-High)
**Consider when:** Independent iterations, large data, embarrassingly parallel
**Strategies:** See `optimization-patterns.md#parallelization-strategies`

---

## Output Format

### Quick Mode
```
## Top 3 Optimization Opportunities

1. **Fix Type Instability** (⭐⭐⭐⭐⭐)
   - Location: `compute_result` line 42
   - Issue: Returns `Union{Float64, Nothing}`
   - Estimated: 3-5x speedup

2. **Reduce Allocations** (⭐⭐⭐⭐)
   - Location: Main loop line 67
   - Issue: Array allocated each iteration
   - Estimated: 50% memory reduction
```

### Standard Mode
Adds: Detailed profiling results, code examples, benchmark comparisons

### Comprehensive Mode
Adds: Parallelization analysis, memory layout, step-by-step implementation guide

---

## Success Criteria

- ✅ Baseline benchmark captured
- ✅ Type stability analyzed
- ✅ Bottlenecks identified
- ✅ Allocation sources found
- ✅ Recommendations prioritized
- ✅ Code examples provided
- ✅ Expected improvements estimated

---

## External Documentation

- `optimization-patterns.md` - Complete patterns (~400 lines)
- `profiling-guide.md` - Detailed techniques (~350 lines)

## Related Commands
- `/julia-scaffold` - Create performant package structure
