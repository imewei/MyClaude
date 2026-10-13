---
name: testing-julia-package
disable-model-invocation: true
description: Use when you set up or run tests for a Julia package, including workspace-based test dependencies, Pkg.test(), focused local runs, and optional parallel test files
---

# Testing a Julia Package

Keep test-only dependencies in `test/Project.toml`. This works on every Julia version (`Pkg.test()` adds the package under test to that environment automatically). Opt into extra features only when `[compat] julia` allows them:

| Feature | Needs | Effect |
|---|---|---|
| `test/Project.toml` | any (1.2+) | separate test environment |
| `[sources]` in `test/Project.toml` | 1.11+ | pins the package to `path = ".."` |
| `[workspace]` in the root | 1.12+ | test project shares the root manifest |

Check the declared bound first and use the lowest-common layout:

```sh
grep -A2 '^\[compat\]' Project.toml
```

## Test project (any Julia)

`test/Project.toml` declares `Test` and every package the tests import. Workspace and test projects do not inherit root dependencies.

```toml
[deps]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
```

Add the package itself only if you run `julia --project=test` directly (see Run tests): `Pkg.develop(path=".")` with `--project=test`. `Pkg.test()` does not need it.

`test/runtests.jl`, exercising the public API:

```julia
using Test
using MyPkg

@testset "MyPkg" begin
    @test my_function(2) == 4
end
```

## Optional: workspace (compat `julia >= "1.12"` only)

Do not add this if the package supports older Julia. Pre-1.12 Pkg cannot instantiate a workspace, and CI legs on older versions will diverge.

Root `Project.toml`:

```toml
[workspace]
projects = ["test"]
```

`test/Project.toml` adds the package and a source pin:

```toml
[deps]
MyPkg = "<UUID from the root Project.toml>"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[sources]
MyPkg = {path = ".."}
```

Resolve with `julia --project=test -e 'using Pkg; Pkg.resolve()'`.

Legacy `[extras]`/`[targets]` in the root `Project.toml` still work. When `test/Project.toml` exists, delete them to avoid two sources of truth.

## Run tests

Fast iteration (needs the package in the test environment: workspace/`[sources]`, or `Pkg.develop(path=".")` once):

```sh
julia --project=test test/runtests.jl
```

Canonical full check, before merging and in CI:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

`Pkg.test()` runs `test/runtests.jl` in a fresh process with test dependencies. Add `coverage=true` for coverage, or `allow_reresolve=false` to forbid re-resolution.

Commit `Manifest.toml` and `test/Manifest.toml` for reproducible CI, or gitignore them to resolve fresh each run. Libraries usually ignore them; applications and research code usually commit them.

For a focused run, make test files independently runnable and invoke one with `--project=test`. [TestRunner.jl](https://github.com/aviatesk/TestRunner.jl) can select testsets or line ranges; confirm the summary shows the expected count, since filters can match more than intended.

## Scientific test content

For floating-point results use `isapprox` with explicit tolerances (`@test x ≈ y rtol=1e-8`), fix RNG seeds (`StableRNGs.jl` for cross-version stability), and add `Aqua.test_all`, `@inferred`, and JET checks. See [julia-testing-patterns](../julia-testing-patterns/SKILL.md) for ReTestItems, Aqua, and JET detail, and [ci-cd-patterns](../ci-cd-patterns/SKILL.md) for the CI matrix.

## Optional: parallel test files

For a large suite of independent files, add ParallelTestRunner to the test project:

```sh
julia --project=test -e 'using Pkg; Pkg.add("ParallelTestRunner")'
```

```julia
using MyPkg
using ParallelTestRunner

runtests(MyPkg, ARGS)
```

Each discovered file runs in isolation and must load its own dependencies and define its own testsets. Remove manual `include` calls. Pass options through `Pkg.test`:

```sh
julia --project -e 'using Pkg; Pkg.test(; test_args=["--jobs=4"])'
```

Use it only when files are truly independent; keep `Pkg.test()` as the canonical full-suite check.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
