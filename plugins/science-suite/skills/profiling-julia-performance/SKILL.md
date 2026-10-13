---
name: profiling-julia-performance
disable-model-invocation: true
description: Use when Julia code is slower than expected at steady state (not just on its first call), or you suspect type instability or unnecessary allocations, or the user mentions profiling, Profile, JET.jl, AllocCheck.jl, @code_warntype, allocations, flamegraphs, or optimizing Julia performance. Measure and diagnose before guessing at optimizations.
---

# Profiling Julia Performance

Do not start by rewriting. Two causes account for most slow Julia: **type instability** (inference fails, dynamic dispatch follows) and **needless allocation** (GC dominates hot loops). Both are measurable, and fixing them beats micro-tuning arithmetic.

Tools: `Profile` (stdlib, CPU and allocation), `JET.jl` (static instability and error analysis), `AllocCheck.jl` (static allocation proof), `BenchmarkTools.jl` (reliable timings).

First confirm it is steady-state cost. If only the first call is slow, go to [precompiling-julia-package](../precompiling-julia-package/SKILL.md). Quick wins and idioms: [performance-tuning](../performance-tuning/SKILL.md). GPU and distributed work have their own tooling: [gpu-acceleration](../gpu-acceleration/SKILL.md), [julia-hpc-distributed](../julia-hpc-distributed/SKILL.md).

## Step 1: Profile to find hot code

Profile a **function** with a representative workload, after warm-up. Global-scope code is not optimized the same way.

```julia
using Profile

function main()
    # representative workload
end

main()                   # warm up: finish compilation first
Profile.clear()
@profile main()
Profile.print(format=:flat, sortedby=:count)
```

Script form for reproducibility:

```sh
julia --startup-file=no --project -e '
using Profile
include("bench.jl")
main()
Profile.clear()
@profile main()
Profile.print(format=:flat, sortedby=:count, maxdepth=12, noisefloor=2.0)
'
```

Large shares in `jl_apply_generic`, `jl_gc_*`, `jl_gc_pool_alloc`, or `typeinf` point to dynamic dispatch or allocation pressure. `ProfileView.@profview main()` and `PProf` give graphical flamegraphs from the same samples. Multithreaded code: `Profile.print(groupby=:thread)`.

### Allocation profiling

```julia
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=1.0 main()
Profile.Allocs.print()
```

`@allocated main()` reports how much; the allocation profiler reports where. Use both. For stable micro-benchmarks use BenchmarkTools, interpolating non-constant globals:

```julia
using BenchmarkTools
@benchmark f($x)
```

## Step 2: Type instability with JET.jl

JET reuses inference to find calls that cannot be resolved statically. Pass representative **values**:

```julia
using JET
@report_opt target_function(1, 2.0)
```

Look for `runtime dispatch detected` in frames from your own code; ignore noise from Base and dependencies. `report_package(MyPkg; target_modules=(MyPkg,))` covers a package; `@report_call` finds potential runtime errors instead.

Caveats:

- JET executes top-level statements and `__init__` while loading. Analyze only trusted code.
- JET tracks compiler internals and supports a limited range of Julia versions. Check `JET.JET_AVAILABLE` after loading; if `false` it is a no-op.
- On dependency conflicts (notably with Revise via JuliaInterpreter), install JET in a temporary environment (`Pkg.activate(; temp=true)`). Never make it a runtime dependency.

`@code_warntype f(1, 2.0)` is the zero-dependency check: red `Any`/`Union` marks instability. `@code_typed` shows the full inferred IR.

## Step 3: Prove allocation-freedom with AllocCheck.jl

AllocCheck inspects LLVM IR to prove a function *cannot* allocate, including on paths your inputs never hit:

```julia
using AllocCheck

@check_allocs hot_loop(x) = sum(x)
hot_loop(rand(100))            # passes
```

A violation raises `AllocCheckFailure`; its `errors` field locates the allocating site.

- Throw paths are ignored by default; add `ignore_throw=false` to include them.
- Calls into a `@check_allocs` function behave like dynamic dispatch with small entry allocations. Wrap a top-level entry point or main loop, not a per-iteration helper.
- Annotate only genuine hot paths.

## Step 4: Fix the root cause

In rough order of impact:

- **Typed containers.** `Float64[]` or `Vector{Float64}(undef, n)`, never `Any[]`.
- **`const` globals**, or pass them as arguments.
- **Function barriers.** Do unstable setup, then call a type-stable inner function.
- **Consistent return types** across branches (`zero(T)`, not `0`).
- **Concrete struct fields** (parametric if generic).
- **Preallocate and mutate:** `mul!`, `@views`, `.=` broadcasting, `StaticArrays` for small fixed sizes.
- **Annotate `::T` only at boundaries**, not everywhere.

Once stable and allocation-light, consider `@inbounds`/`@simd` and threading, after correctness tests exist.

## Step 5: Re-measure and guard

Redo the measurement that exposed the problem, then re-run `@report_opt` and `@check_allocs` on changed functions. Report before/after numbers, not impressions. If the hot spot moved, follow it.

Lock in the gain with regression tests ([testing-julia-package](../testing-julia-package/SKILL.md)):

```julia
@test (@inferred f(x)) isa Float64
@test (@allocated f!(out, x)) == 0   # after a warm-up call
```

## Pitfalls

- **Profiling global scope or a cold call** measures compilation.
- **`@time` without warm-up** includes compilation; prefer `@benchmark`.
- **Chasing JET noise** in Base/dependencies.
- **Over-annotating** with `@check_allocs`/`::T`.
- **Optimizing before measuring.** Profile first.
- **Benchmarking on a loaded machine** or with a changing thread count; pin both.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
