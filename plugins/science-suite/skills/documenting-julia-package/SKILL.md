---
name: documenting-julia-package
disable-model-invocation: true
description: "Use when you add, build, or publish documentation for a Julia package with Documenter.jl: the `docs/` layout, `docs/Project.toml`, `makedocs`/`deploydocs`, `@docs` and docstring blocks, doctests, and GitHub Pages deployment"
---

# Documenting a Julia Package with Documenter.jl

[Documenter.jl](https://documenter.juliadocs.org/stable/) builds a cross-linked HTML site from Markdown plus docstrings and publishes it to GitHub Pages. Doctests keep examples from going stale. Assume package `MyPkg` at the repo root ([generating-julia-package](../generating-julia-package/SKILL.md)); write docstrings per [developing-julia-package](../developing-julia-package/SKILL.md).

## Step 1: `docs/` layout

```
MyPkg/
├── docs/
│   ├── Project.toml
│   ├── make.jl
│   └── src/index.md     # required name for HTML output
├── src/MyPkg.jl
└── Project.toml
```

## Step 2: Doc environment

`docs/Project.toml` declares Documenter (a doc-build dependency, never a runtime one). Which wiring to use depends on `[compat] julia` in the root `Project.toml`:

**Any Julia (default).** Develop the local package into the docs environment:

```toml
[deps]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"

[compat]
Documenter = "1"
```

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

**Julia 1.11+ only:** add `MyPkg = "<UUID>"` to `[deps]` and `[sources] MyPkg = {path = ".."}` in `docs/Project.toml`; then plain `Pkg.instantiate()` is enough.

**Julia 1.12+ only:** additionally add `docs` to the root `[workspace] projects` list (with `test`, `benchmark` if present) and run `Pkg.instantiate(; workspace=true)` from the root. The `workspace` keyword raises an error on older Julia. Do not use it if the package supports pre-1.12 versions.

The docs build itself can still run on the newest Julia in CI even when the package supports older ones; only the instantiate step must match the layout.

## Step 3: `docs/make.jl`

```julia
using Documenter, MyPkg

makedocs(;
    sitename = "MyPkg.jl",
    modules = [MyPkg],
    authors = "Your Name",
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
)

deploydocs(; repo = "github.com/USER_NAME/MyPkg.jl.git")
```

- `modules = [MyPkg]` names the docstring owners and stops extended `Base` methods dragging in unrelated docs.
- `checkdocs = :exports` fails the build when an exported name lacks a docstring in the manual.
- If the package is a local path dependency, pass `remotes = nothing` or source links break.
- `deploydocs` takes `repo` without `https://` or `git@`; it is a no-op outside CI deploy conditions.
- `linkcheck = true` verifies external URLs (slow, flaky; enable in CI only if wanted).

## Step 4: Pages and docstrings

````markdown
# MyPkg.jl

```@meta
CurrentModule = MyPkg
```

```@docs
fit_model
FitResult
```
````

`@autodocs` includes every docstring in a module, `@index` builds an index, `@contents` a TOC. Cross-reference with `` [`fit_model`](@ref) ``. Scientific packages should state units, shapes, and conventions in docstrings, and use `$$ ... $$` or `` ``x`` `` for math.

## Step 5: Build locally

```sh
julia --project=docs docs/make.jl
```

Output goes to `docs/build/` (gitignore it). Pretty URLs do not resolve over `file://`; preview over HTTP:

```sh
python3 -m http.server --bind localhost --directory docs/build
```

## Step 6: Doctests

`jldoctest` blocks run during `makedocs`; mismatches fail the build.

````markdown
```jldoctest
julia> 1 + 1
2
```
````

- Definitions do not carry across blocks: share a label (`jldoctest mylabel`) or use `setup = :(...)`.
- Filter non-deterministic output (timings, addresses) with `filter = r"..." => s"..."` or `DocTestFilters` in an `@meta` block.
- Docstring doctests run only for modules in `modules`.
- Regenerate outdated output with `doctest(MyPkg; fix = true)` and review the diff.

Run doctests with the normal tests ([testing-julia-package](../testing-julia-package/SKILL.md)); add Documenter to the test project:

```julia
@testset "doctests" begin
    doctest(MyPkg; manual = false)
end
```

## Step 7: Deploy to GitHub Pages

`.github/workflows/documentation.yml`. Action majors were current when written; verify before pinning with `git ls-remote --tags --refs https://github.com/<owner>/<action>`. More CI patterns: [ci-cd-patterns](../ci-cd-patterns/SKILL.md).

```yaml
name: Documentation
on:
  push:
    branches: [main]
    tags: '*'
  pull_request:
  workflow_dispatch:
jobs:
  build:
    permissions:
      contents: write
      pull-requests: read
      statuses: write
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: julia-actions/setup-julia@v3
        with:
          version: '1'
      - uses: julia-actions/cache@v3
      - name: Install dependencies
        shell: julia --color=yes --project=docs {0}
        run: |
          using Pkg
          Pkg.develop(PackageSpec(path=pwd()))
          Pkg.instantiate()
          # Julia 1.12+ workspace layout instead: Pkg.instantiate(; workspace=true)
      - name: Build and deploy
        run: julia --color=yes --project=docs docs/make.jl
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- Settings → Pages: source is the `gh-pages` branch root (created on first deploy).
- Same-repo deploys work with `GITHUB_TOKEN` and `contents: write`. For tag-triggered builds, TagBot must be able to trigger the workflow, which needs an SSH deploy key.
- Cross-repo or tag-triggered: `DocumenterTools.genkeys(user="USER", repo="MyPkg.jl")`, add the public key as a write-enabled **Deploy key**, store the private key as the `DOCUMENTER_KEY` **repository secret**.
- Docs badge: `[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://USER.github.io/MyPkg.jl/stable)`.

## Security

- `DOCUMENTER_KEY` is an unencrypted private key with write access. Never log, paste, or commit it; use an encrypted repository secret (not a variable, not an environment secret).
- `makedocs` and `doctest` execute package code and every `jldoctest` block, and CI repeats that on each push. Build only trusted code, and be careful with `pull_request` builds from forks.

## Pitfalls

- **Missing `docs/src/index.md`.**
- **Docstrings not found:** module absent from `modules`, or no `CurrentModule`.
- **`remotes = nothing` forgotten** for a path dependency.
- **Committed `docs/build/`.**
- **Hand-edited doctest output.** Use `fix = true` and inspect the diff.
- **Unpinned Documenter.** Keep a `[compat]` bound.

---
*Adapted from [AtelierArith/atelier-arith-julia-development-skills](https://github.com/AtelierArith/atelier-arith-julia-development-skills) (MIT). See `THIRD_PARTY_NOTICES.md`.*
