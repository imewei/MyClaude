---
name: generating-julia-package
disable-model-invocation: true
description: Use when you create a new Julia package or project environment — choosing the layout, writing `Project.toml`/`src`, generating a UUID, adding dependencies — or prepare a package for registration
---

# Generating a Julia Package

Pick the lightest structure that fits: a **project environment** for scripts, a **package directory** for reusable code.

## Project environment (scripts)

A `Project.toml` with `[deps]` is enough. Let Pkg write it:

```sh
julia --project -e 'using Pkg; Pkg.add("TargetPackage")'
```

Commit `Project.toml` and `Manifest.toml` for reproducible research code. Never use the global environment.

## Package directory

### Choosing the layout

- **Default: in-place.** `./Project.toml` and `./src/MyPkg.jl` in the current directory, no wrapping subdirectory. Use this for "create a package `MyPkg`" without further qualification.
- **Opt-in: subdirectory.** Only when the user explicitly asks for one, or when bootstrapping several packages side by side.

`Pkg.generate("MyPkg")` always creates a subdirectory, so it cannot produce the in-place layout; write the files by hand.

### In-place layout

1. Generate a UUID:

   ```sh
   julia -E 'using UUIDs; string(uuid4())'
   ```

2. Write `./Project.toml`:

   ```toml
   name = "MyPkg"
   uuid = "<uuid from step 1>"
   authors = ["Your Name <you@example.com>"]
   version = "0.1.0"

   [compat]
   julia = "1.10"
   ```

   Set the `julia` bound to the oldest version you test in CI (LTS is `1.10`), and keep the CI matrix's lowest leg equal to it. The bound decides which setup the other skills may use: `[sources]` and `public` need 1.11, `[workspace]` and `[apps]` need 1.12. With `julia = "1.10"`, use plain `test/Project.toml` and `Pkg.develop` for docs; raise the bound to `1.12` before adopting workspaces.

3. Write `./src/MyPkg.jl`:

   ```julia
   module MyPkg

   greet() = "Hello"

   end # module MyPkg
   ```

4. Verify:

   ```sh
   julia --project=. -e 'using MyPkg; println(MyPkg.greet())'
   ```

Next: tests ([testing-julia-package](../testing-julia-package/SKILL.md)), implementation style ([developing-julia-package](../developing-julia-package/SKILL.md)).

### Subdirectory layout (templated)

[PkgTemplates.jl](https://juliaci.github.io/PkgTemplates.jl/stable/) gives repeatable scaffolding with docs and CI files. See [package-development-workflow](../package-development-workflow/SKILL.md) for plugin configuration.

```julia
using Pkg
Pkg.activate(; temp = true)
Pkg.add("PkgTemplates")
using PkgTemplates

Template(dir = pwd())("MyPkg")
```

Minimal alternative, only when a subdirectory is requested: `julia -e 'using Pkg; Pkg.generate("MyPkg")'`.

## Before registering

Registration in General is not automatic. Get these in order first:

- **`[compat]` for every dependency and `julia`.** The registry's auto-merge requires it.

  ```toml
  [compat]
  julia = "1.10"
  SomeDep = "1.2, 2"
  ```

- **Aqua.jl quality checks** (ambiguities, undefined exports, stale deps, missing compat, piracy). Declare Aqua in the test project, not in runtime `[deps]`:

  ```julia
  using Aqua
  Aqua.test_all(MyPkg)
  ```

- **Full test suite:** `julia --project -e 'using Pkg; Pkg.test()'`.
- **Register** with the `@JuliaRegistrator register` comment on a commit or PR (Registrator.jl). Add TagBot and CompatHelper workflows (see [ci-cd-patterns](../ci-cd-patterns/SKILL.md)).

Register only when the version, compat bounds, and tests are green.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
