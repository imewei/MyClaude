---
name: fixing-julia-exports
disable-model-invocation: true
description: Use when a Julia package exports too much — internal helpers exported only so `Pkg.test` can reach them, unused or stale exports, or namespace pollution and name collisions — or when you need to audit and trim a package's public API. Covers finding `export` statements, classifying public vs internal names, replacing `export` with explicit test imports or the `public` keyword, and verifying with tests and Aqua.jl.
---

# Fixing Excessive Exports in a Julia Package

`using MyPkg` brings every exported name into the caller's scope. An internal helper exported only for the package's tests becomes de-facto public API: it appears in `names(MyPkg)`, the docs namespace, and collision warnings for every user.

The fix is almost always on the **test** side: stop exporting the helper and import it explicitly in tests. General style rule: [developing-julia-package](../developing-julia-package/SKILL.md).

## Step 1: Inventory

On Julia 1.11+, `names(MyPkg)` lists names marked `public` **or** `export`. Use `Base.isexported` to isolate exports:

```sh
julia --project -e 'using MyPkg; println(sort(filter(n -> Base.isexported(MyPkg, n), names(MyPkg; all=true))))'
rg -n '^\s*export\b' src/
```

A single `export` line can list many names; read the statements instead of grepping one name.

## Step 2: Classify each name

- **Documented or intended API** (in `docs/` `@docs` blocks, README, examples): public. Keep.
- **Referenced by another package or a user-facing entry point**: public. Keep.
- **Used only in `src/` or only in `test/`**: internal. Remove the export.

```sh
rg -n --glob '!Manifest.toml' '\bhelper_name\b'
```

If the only hits outside `src/` are in `test/`, the export exists solely for tests. Docs are the contract: a documented name is public even if nothing in `src/` calls it.

## Step 3: Remove the export, import in tests

Tell the user which files change before editing an existing package. Delete the name from `export`; leave the definition alone. Then:

```julia
# src/MyPkg.jl
export fit_model            # initial_guess no longer exported
```

```julia
# test/runtests.jl
using Test
using MyPkg
using MyPkg: initial_guess  # explicit internal import
```

- `using MyPkg: name` or `import MyPkg` plus `MyPkg.name` are both fine.
- Macros need the same: `using MyPkg: @internal_macro`.
- Many tests reaching into many internals suggests testing implementation, not behavior. Rewrite through the public API or move fixtures into `test/`.

## Step 4: `public` for supported-but-unexported names

On Julia 1.11+, `public` marks a name as supported API without adding it to `using MyPkg` scope:

```julia
public advanced_hook
advanced_hook(x) = ...
```

Callers write `MyPkg.advanced_hook`. `public` is a syntax error before 1.11. If `[compat] julia` allows older versions, use `Compat.jl`'s `@compat public advanced_hook`. Do not use `public` to avoid fixing test imports; it does not bring the name into scope.

## Step 5: Verify

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Re-run the Step 1 inventory and confirm the list shrank. Add Aqua for undefined exports and related mistakes (see [generating-julia-package](../generating-julia-package/SKILL.md)):

```julia
using Aqua
Aqua.test_all(MyPkg)
```

Test setup: [testing-julia-package](../testing-julia-package/SKILL.md).

## What counts as breaking

Removing an export breaks `using MyPkg; helper` for anyone relying on it.

- Truly internal (undocumented, test-only): low risk; patch or minor release.
- Documented in docs, README, or examples: public. Removal is breaking under SemVer (major bump). To stop injecting it into caller scope without breaking, keep it as `public` and deprecate the bare name:

  ```julia
  Base.@deprecate_binding old_name new_name   # or @deprecate f(x) g(x)
  ```

## Pitfalls

- **Name listed on several `export` lines** after a refactor. Grep the name.
- **Implicit test import left behind.** A bare call after removing the export fails with `UndefVarError`.
- **Treating a documented symbol as internal.** Check `docs/` and the README first.
- **Using `public` to paper over test access.**
- **Forgetting macros.** `export @m` and `public @m` behave the same.
- **Forgetting downstream packages.** When in doubt, deprecate rather than delete.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
