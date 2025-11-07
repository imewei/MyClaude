---
version: "1.0.3"
category: "julia-development"
command: "/julia-optimize"
description: Profile Julia code and provide optimization recommendations through systematic analysis of type stability, allocations, and performance bottlenecks
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<file_path>"
color: green
execution_modes:
  quick: "5-10 minutes - Basic profiling with top optimization opportunities"
  standard: "15-25 minutes - Comprehensive analysis with prioritized recommendations"
  comprehensive: "30-45 minutes - Deep analysis with implementation examples and benchmarks"
agents:
  primary:
    - julia-pro
  conditional:
    - agent: sciml-pro
      trigger: pattern "differential.*equation|ode|pde|sde|sciml"
    - agent: hpc-numerical-coordinator
      trigger: pattern "parallel|distributed|gpu|mpi|simd"
  orchestrated: false
---

# Performance Profiling and Optimization

Systematically profile Julia code and provide actionable optimization recommendations through type stability analysis, allocation profiling, and bottleneck identification.

## Quick Reference

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| **Optimization Patterns** | [optimization-patterns.md](../docs/optimization-patterns.md) | ~400 |
| **Profiling Guide** | [profiling-guide.md](../docs/profiling-guide.md) | ~350 |
| **Type Stability** | [optimization-patterns.md#type-stability-patterns](../docs/optimization-patterns.md#type-stability-patterns) | ~120 |
| **Allocation Reduction** | [optimization-patterns.md#allocation-reduction](../docs/optimization-patterns.md#allocation-reduction) | ~100 |
| **Parallelization** | [optimization-patterns.md#parallelization-strategies](../docs/optimization-patterns.md#parallelization-strategies) | ~80 |

**Total External Documentation**: ~750 lines of detailed patterns and examples

## Requirements

$ARGUMENTS

## Core Workflow

### Phase 1: Baseline Measurement

**Establish performance baseline** using BenchmarkTools.jl:

```julia
using BenchmarkTools

# Accurate timing with statistics
@benchmark target_function($args)
```

**Capture metrics**:
- **Median time**: Most reliable timing
- **Memory estimate**: Total allocations
- **Allocs estimate**: Number of allocations
- **GC percentage**: Garbage collection overhead

**Documentation**: [profiling-guide.md#benchmarktoolsjl](../docs/profiling-guide.md#benchmarktoolsjl)

### Phase 2: Type Stability Analysis

**Check for type instabilities** using `@code_warntype`:

```julia
@code_warntype target_function(args)
```

**Look for**:
- ❌ **Red**: `Any`, `Union` types (critical issues)
- ⚠️ **Yellow**: Abstract types (caution)
- ✅ **Green/Blue**: Concrete types (good!)

**Common patterns**:
- Conditional return types
- Abstract container types
- Empty container initialization
- Global variable usage

**Fix patterns**: [optimization-patterns.md#type-stability-patterns](../docs/optimization-patterns.md#type-stability-patterns)

### Phase 3: Profiling & Bottleneck Identification

**Statistical profiling** using Profile.jl:

```julia
using Profile, ProfileView

# Profile execution
@profile for _ in 1:1000
    target_function(args)
end

# View flame graph
ProfileView.view()
```

**Analyze flame graph**:
- **Width**: Time spent (wider = slower)
- **Red/Yellow**: Hot spots (optimize these)
- **Unexpected functions**: Hidden allocations, type conversions

**Documentation**: [profiling-guide.md#profilejl--profileviewjl](../docs/profiling-guide.md#profilejl--profileviewjl)

### Phase 4: Allocation Analysis

**Identify allocation sources**:

```julia
# Check allocation count
@allocations target_function(args)

# Profile allocations
using Profile
Profile.Allocs.@profile target_function(args)
Profile.Allocs.print()
```

**Common culprits**:
- Missing pre-allocation
- Array copies vs views
- Type instabilities causing boxing
- Temporary arrays in loops

**Fix patterns**: [optimization-patterns.md#allocation-reduction](../docs/optimization-patterns.md#allocation-reduction)

### Phase 5: Optimization Recommendations

**Generate prioritized recommendations** based on analysis:

**High Priority** (largest impact):
1. **Fix type instabilities** → Enables compiler optimizations
2. **Eliminate allocations in hot loops** → Reduces GC pressure
3. **Algorithm improvements** → Change O(n²) to O(n)

**Medium Priority**:
4. **Pre-allocation** → Avoid repeated allocations
5. **In-place operations** → Use `!` functions
6. **Views vs copies** → Use `@view` for slicing

**Low Priority** (micro-optimizations):
7. **SIMD annotations** → `@simd`, `@inbounds`
8. **Loop reordering** → Column-major order
9. **Static arrays** → For small fixed-size data

### Phase 6: Implementation & Verification

**For each recommendation**:

1. **Implement fix** based on pattern from docs
2. **Verify correctness** with tests
3. **Re-benchmark**: Compare before/after
4. **Document improvement**: Speedup and allocation reduction

**Benchmark comparison**:
```julia
baseline = @benchmark original_function($data)
optimized = @benchmark optimized_function($data)

speedup = median(baseline).time / median(optimized).time
println("Speedup: $(round(speedup, digits=2))x")
```

## Mode-Specific Execution

### Quick Mode (5-10 minutes)

**Phases**: 1 (baseline), 2 (type stability), 4 (allocation check)

**Output**:
- Top 3 optimization opportunities
- Quick wins (low effort, high impact)
- Estimated improvement potential

**Skip**: Detailed profiling, implementation examples

### Standard Mode (15-25 minutes) - DEFAULT

**Phases**: All 6 phases

**Output**:
- Comprehensive analysis (type stability, allocations, bottlenecks)
- Prioritized recommendations with rationale
- Code examples for top issues
- Performance estimates

**Include**: Flame graph analysis, before/after benchmarks

### Comprehensive Mode (30-45 minutes)

**Phases**: All 6 phases with extended analysis

**Output**:
- Deep analysis across all optimization dimensions
- Complete implementation examples
- Parallelization opportunities assessment
- Algorithm complexity review
- Step-by-step optimization guide
- Expected performance improvements with confidence levels

**Include**:
- Multi-threading/distributed/GPU analysis
- Cache optimization opportunities
- Memory layout analysis
- Integration with external docs

## Optimization Categories

### Type Stability Issues

**Impact**: ⭐⭐⭐⭐⭐ (Critical)

**Symptoms**:
- `Union` or `Any` types in `@code_warntype`
- Slow performance despite simple code
- High allocation counts

**Common patterns**: [optimization-patterns.md#type-stability-patterns](../docs/optimization-patterns.md#type-stability-patterns)

### Allocation Hot Spots

**Impact**: ⭐⭐⭐⭐ (High)

**Symptoms**:
- High memory estimate in `@benchmark`
- Large GC percentage (> 5%)
- Allocations in hot loops

**Fix patterns**: [optimization-patterns.md#allocation-reduction](../docs/optimization-patterns.md#allocation-reduction)

### Algorithm Complexity

**Impact**: ⭐⭐⭐⭐⭐ (Critical for large inputs)

**Check**:
- Does performance scale linearly with input size?
- Is there a more efficient algorithm?
- Can preprocessing reduce repeated work?

**Examples**: [optimization-patterns.md#algorithm-improvements](../docs/optimization-patterns.md#algorithm-improvements)

### Parallelization Opportunities

**Impact**: ⭐⭐⭐ (Medium to High)

**Consider when**:
- Independent iterations in loops
- Large data arrays
- Embarrassingly parallel operations

**Strategies**: [optimization-patterns.md#parallelization-strategies](../docs/optimization-patterns.md#parallelization-strategies)

### Memory Layout

**Impact**: ⭐⭐ (Medium)

**Check**:
- Loop order (column-major for Julia)
- Struct of Arrays vs Array of Structs
- Static arrays for small fixed-size data

**Details**: [optimization-patterns.md#memory-optimization](../docs/optimization-patterns.md#memory-optimization)

## Output Format

### Quick Mode Output

```markdown
## Profiling Results
- **Baseline**: 2.5ms (median), 150 KiB allocations
- **Performance**: ⚠️ 45% time in GC

## Top Optimization Opportunities

1. **Fix Type Instability** (⭐⭐⭐⭐⭐ Critical)
   - Location: `compute_result` function, line 42
   - Issue: Returns `Union{Float64, Nothing}`
   - Fix: Return `0.0` instead of `nothing`
   - **Estimated improvement**: 3-5x speedup

2. **Reduce Allocations** (⭐⭐⭐⭐ High)
   - Location: Main loop, line 67
   - Issue: Allocating array on each iteration
   - Fix: Pre-allocate before loop
   - **Estimated improvement**: 50% memory reduction

3. **Use Views** (⭐⭐⭐ Medium)
   - Location: `process_chunk`, line 89
   - Issue: Slicing creates copies
   - Fix: Use `@view arr[start:stop]`
   - **Estimated improvement**: 30% allocation reduction
```

### Standard Mode Output

```markdown
## Performance Analysis

### Baseline Metrics
- **Time**: 2.5ms (median), range: 2.3-2.8ms
- **Memory**: 150 KiB
- **Allocations**: 1,247
- **GC**: 45% (very high!)

### Type Stability Analysis
❌ **Critical Issues** (2 found):
1. `compute_result:42` - Returns `Union{Float64, Nothing}`
2. `aggregate:108` - Empty array initialization (`[]`)

⚠️ **Warnings** (1 found):
1. `process:67` - Uses `AbstractArray` without type parameter

### Profiling Results
**Hot Spots** (top 3):
1. `compute_result` - 60% of total time
2. `allocate_temps` - 25% of time (allocation heavy)
3. `process_chunk` - 10% of time

### Allocation Analysis
- **Total allocations**: 1,247 (150 KiB)
- **Hot loop**: 1,000 allocations (80% of total)
- **Largest allocation**: 32 KiB temporary array

## Prioritized Recommendations

[... detailed recommendations with code examples ...]

## Expected Improvements
After implementing all recommendations:
- **Speed**: 5-8x faster
- **Memory**: 95% reduction
- **GC**: < 1%
```

## Success Criteria

✅ Baseline benchmark captured
✅ Type stability analyzed with `@code_warntype`
✅ Bottlenecks identified via profiling
✅ Allocation sources found
✅ Recommendations prioritized by impact
✅ Code examples provided for fixes
✅ Expected improvements estimated
✅ External documentation referenced

## Agent Integration

- **julia-pro**: Primary agent for general Julia optimization
- **sciml-pro**: Triggered for SciML-specific optimizations (ODE/PDE solvers, sensitivity analysis)
- **hpc-numerical-coordinator**: Triggered for parallel/distributed/GPU optimization opportunities

## Post-Optimization

After implementing recommendations:

1. **Verify correctness**: Run test suite
2. **Re-profile**: Confirm improvements
3. **Compare benchmarks**: Document speedup
4. **Iterate**: Focus on remaining bottlenecks if needed

**See Also**:
- `/julia-scaffold` - Create performant package structure
- [optimization-patterns.md](../docs/optimization-patterns.md) - Complete pattern library
- [profiling-guide.md](../docs/profiling-guide.md) - Detailed profiling techniques

---

Focus on **high-impact optimizations first**, **measure everything**, and **verify correctness** after each change.
