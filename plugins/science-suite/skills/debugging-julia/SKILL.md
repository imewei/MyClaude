---
name: debugging-julia
disable-model-invocation: true
description: "Use when Julia code errors, throws an exception, produces wrong results, or crashes and you need to find the cause from the command line: capturing and reading stack traces, locating the failing method with @which/@code_warntype/@code_typed, finding potential runtime errors with JET.jl, and narrowing to a minimal reproduction"
---

# Debugging Julia from the CLI

Use reproducible, non-interactive steps. An interactive debugger can complement them when a human is present, but never assume one is available. For NaN/Inf propagation, JAX-vs-Julia numerics, or AD failures, the `/dev-suite:smart-debug` workflow and [julia-ad-backends](../julia-ad-backends/SKILL.md) cover those cases.

## Step 1: Reproduce deterministically

```sh
julia --startup-file=no --project script.jl args...
```

Drop `--startup-file=no` only if the failure depends on the user's startup file. Shrink arguments and inputs to the smallest case that still fails. Seed RNGs (`Random.seed!(1)`) so stochastic failures repeat; record `julia -v` and `Pkg.status()`.

## Step 2: Capture the full stack trace

If code swallows the exception, log it with its backtrace:

```julia
try
    main()
catch e
    @error "uncaught" exception = (e, catch_backtrace())
    rethrow()
end
```

or print directly: `showerror(stderr, e, catch_backtrace())`.

Missing frames were probably inlined: rerun with `julia --debug-info=2` and mark the suspect `Base.@noinline`. The exception is at the top; the bug is usually at the first frame in **your** code, not the deepest Base frame. When explaining it, summarize the error and that frame instead of pasting the trace. For Python users, note Python puts the error at the bottom.

## Step 3: Locate the failing method

`InteractiveUtils` ships with Julia:

```julia
@which f(args...)         # method the call resolves to
methods(f)                # all methods
@code_warntype f(args)    # inferred types, unstable ones flagged
@code_typed f(args)       # inferred IR (non-interactive stand-in for Cthulhu)
@code_lowered f(args)     # lowered, no inference
```

For a `MethodError`, compare the printed argument types against the "closest candidates"; often a concrete type was narrowed upstream (an `Int` where `Float64` was expected, a `Vector{Any}`, a `Missing`).

## Step 4: Potential runtime errors with JET.jl

```julia
using JET
@report_call target_function(1, 2.0)
```

Pass representative values; JET infers from types. `report_package(MyPkg)` covers a package. JET runs top-level code while loading, so analyze only trusted code. Version support and the temporary-environment install workaround are in [profiling-julia-performance](../profiling-julia-performance/SKILL.md), which also covers performance (not correctness) analysis.

## Step 5: Narrow with instrumentation

- `@debug "msg" x y` with `JULIA_DEBUG=MyPkg` keeps noise off by default.
- `@assert cond "message"` fails fast where a wrong value first appears.
- `julia --check-bounds=yes` turns out-of-bounds access (including under `@inbounds`) into errors.
- Silent numeric corruption: check `isfinite`/`isnan` at loop boundaries; `Test.@inferred` for type surprises; compare against a reference or higher-precision (`BigFloat`) run.
- Wrong results with threads: rerun with `JULIA_NUM_THREADS=1`; a change in output indicates a data race.
- Hangs: run with `timeout` and send `SIGQUIT` (or `kill -QUIT`) for a Julia stack dump on Unix, `Ctrl-\` on a terminal.

Remove instrumentation afterwards; keep only genuine invariants.

## Step 6: Minimize and fix

- Cut to one argument, one input, one code path.
- If recent, `git bisect run` between a known-good and known-bad commit finds the introducing change. Each step executes historical code, so bisect only commits you trust.
- Confirm the fix with the exact minimal reproduction and add a regression test ([testing-julia-package](../testing-julia-package/SKILL.md)).
- Show a small before/after snippet when presenting the fix.

## Pitfalls

- **Discarding the backtrace:** `catch; end` or `catch e; @warn e` loses it. Attach `catch_backtrace()`.
- **Reading only the top frame.** Find the first user-code frame.
- **Missing inlined frames:** `--debug-info=2` and `Base.@noinline`.
- **Stale compile cache:** if behavior contradicts the source, run `Pkg.precompile()` or restart; check `Revise` is not hiding a stale definition.
- **Redefined structs in a session:** changing a `struct` needs a restart, not `include`.
- **Interactive debuggers in automation.** `Debugger`, `Infiltrator`, and `Cthulhu` wait for a human and hang non-interactive runs.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
