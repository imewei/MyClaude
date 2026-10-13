---
name: precompiling-julia-package
disable-model-invocation: true
description: Use when measured Julia package load or first-call latency is a problem, or when the user mentions precompilation, time-to-first-execution, time-to-first-load, slow package load, slow first call, startup latency, invalidations, or PrecompileTools.jl. Measure before adding a PrecompileTools.jl workload.
---

# Reducing Julia TTFL and TTFX with PrecompileTools.jl

Julia compiles on demand, so the first `using MyPkg` (**TTFL**, time to first load) and the first call of a function (**TTFX**, time to first execution) can dominate interactive latency. `PrecompileTools.jl` runs a small representative workload during precompilation and caches the result, so the cost is paid at install/update, not in every fresh process.

Treat thresholds as prompts to measure, not pass/fail rules. Starting signals: TTFL ≥ 1 s, TTFX ≥ 3 s. Newer Julia releases keep improving precompile and load time, so measure on the Julia version and hardware users actually have. A one-shot batch job that runs for hours gains little.

If the second call is also slow, it is runtime cost, not latency: see [profiling-julia-performance](../profiling-julia-performance/SKILL.md).

## Step 1: Measure first

Start from a fresh, current precompile cache and skip the startup file:

```sh
julia --project -e 'using Pkg; Pkg.precompile()'
```

**TTFL**, minimum of several fresh processes (the first may be cold disk cache):

```sh
for i in 1 2 3; do
  julia --startup-file=no --project -e 't=@elapsed using MyPkg; println(round(t, digits=3), " s")'
done
```

Break TTFL down per dependency to see whether your package or a dependency is the cost:

```sh
julia --startup-file=no --project -e '@time_imports using MyPkg'
```

**TTFX**, first call only, one function per process (a prior call shares compilation):

```sh
julia --startup-file=no --project -e '
using MyPkg
t = @elapsed MyPkg.foo(1, 2)
println("TTFX foo: ", round(t, digits=3), " s")
'
```

To see what compiles at runtime (candidates for the workload):

```sh
julia --startup-file=no --project --trace-compile=stderr -e 'using MyPkg; MyPkg.foo(1, 2)'
```

## Step 2: Add a PrecompileTools workload

`PrecompileTools` must be in the package's `[deps]`. Put the workload at the end of the module, after the code it exercises:

```julia
module MyPkg

using PrecompileTools: @setup_workload, @compile_workload

# ... package code ...

@setup_workload begin
    # Runs only during precompilation, never at load time.
    x = [1.0, 2.0, 3.0]
    y = 2.0

    @compile_workload begin
        result = compute(x, y)
        summarize(result)
    end
end

end # module
```

The workload executes during `Pkg.precompile()`, install, and update, so precompiling runs this code. Only precompile code you trust.

- `@setup_workload` is skipped at runtime and swallows errors, so a broken workload does not make the package unloadable. It also means a failing workload silently does nothing. Confirm it ran (Step 4).
- `@compile_workload` caches native code for the methods it hits (pkgimages).
- The cache is keyed to Julia version, dependency versions, and package source, and rebuilds automatically.

### Choosing a workload

- **Real input types.** `foo(1, 2)` does nothing for `foo(1.0, 2.0)`. Cover the concrete types callers use (`Float64`, `Vector{Float64}`, `Matrix{Float64}`), and AD dual types if users differentiate through it.
- **Entry points.** Exercising a top-level API compiles most callees.
- **Small.** Each line slows `Pkg.precompile()` and grows the image. Target the measured TTFX offenders.
- **No side effects.** No files, servers, network, or global mutation; it may run in CI or read-only caches.
- **Tiny inputs** for expensive functions; you are paying for compilation, not numerics.
- **No GPU/MPI/threads** in the workload; guard or leave those to extensions.

## Step 3: Fix invalidations when TTFL does not improve

If loading your package invalidates methods compiled elsewhere, they recompile on every load. First detect it:

```julia
using SnoopCompileCore
invs = @snoop_invalidations using MyPkg
using SnoopCompile
trees = invalidation_trees(invs)
```

Typical causes are type piracy and methods added for abstract signatures to Base functions. The best fix is to remove the cause (narrow the signature, own the type). Otherwise wrap the offending code:

```julia
using PrecompileTools
PrecompileTools.@recompile_invalidations begin
    include("api.jl")
end
```

This re-precompiles invalidated methods. It mainly helps packages that heavily extend others' functions.

## Step 4: Verify

Rebuild the cache and repeat Step 1:

```sh
julia --project -e 'using Pkg; Pkg.precompile()'
```

Report before/after numbers against the package's real latency goal. If nothing improved, check that the workload ran (temporarily remove the `@setup_workload` wrapper to surface errors) and that its types match what callers use.

## Optional: SnoopCompile.jl

When measurement and `--trace-compile` cannot locate the cost, `SnoopCompile.@snoop_inference` records inference timing per method; use it to find hot methods, then write a hand-tuned workload. A curated workload is usually smaller than raw generated output. Reach for it only when needed.

## When precompilation is not enough

For scripts and CLIs that start in a fresh process every time, a custom system image or relocatable app can remove most remaining latency; see [compiler-patterns](../compiler-patterns/SKILL.md) and [creating-julia-app](../creating-julia-app/SKILL.md).

## Pitfalls

- **Over-precompiling.** A big dependency graph slows installation and can hit CI precompile timeouts.
- **Stale numbers.** Any edit, dependency change, or Julia upgrade invalidates the cache; re-measure.
- **One type only.** A `Float64` workload does not help an `Int` or `Matrix` first call.
- **Mistaking runtime for latency.** Slow second calls are algorithmic.
- **Hard-depending on heavy packages** just for the workload. Put integrations in extensions.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
