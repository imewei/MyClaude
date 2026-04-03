---
name: package-development-workflow
maturity: "5-Expert"
specialization: Julia Packages
description: Create Julia packages following community standards with proper structure, exports, and PkgTemplates.jl. Use when creating new packages or organizing source code.
---

# Julia Package Development

Create packages following ecosystem standards.

## Expert Agent

For package structure, CI/CD configuration, and documentation workflows, delegate to the expert agent:

- **`julia-pro`**: Unified specialist for Julia package development and testing.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`
  - *Capabilities*: PkgTemplates.jl configuration, GitHub Actions setup, and comprehensive test suite generation.

---

## Package Structure

```
MyPackage/
├── Project.toml
├── src/MyPackage.jl
├── test/runtests.jl
├── docs/make.jl
├── README.md
└── LICENSE
```

---

## Module Pattern

```julia
module MyPackage

export public_function, PublicType

function public_function(x)
    internal_helper(x)
end

function internal_helper(x)
    # Not exported
end

end # module
```

---

## PkgTemplates.jl

```julia
using PkgTemplates

tpl = Template(;
    user="username",
    plugins=[
        Git(),
        License(name="MIT"),
        GitHubActions(),
        Documenter{GitHubActions}()
    ]
)

tpl("MyPackage")
```

---

## Project.toml Configuration

```toml
name = "MyPackage"
uuid = "12345678-1234-1234-1234-123456789abc"
authors = ["Author Name <email@example.com>"]
version = "0.1.0"

[deps]
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7caa7"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
DifferentialEquations = "7"
julia = "1.10"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
Aqua = "4c88cf16-eb10-579e-8560-4a9242c79595"

[targets]
test = ["Test", "Aqua"]
```

## Documenter.jl Setup

```julia
# docs/make.jl
using Documenter, MyPackage

makedocs(
    sitename = "MyPackage.jl",
    modules = [MyPackage],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Tutorials" => "tutorials.md"
    ],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true")
)

deploydocs(
    repo = "github.com/username/MyPackage.jl.git",
    devbranch = "main"
)
```

## CI with TagBot and CompatHelper

```yaml
# .github/workflows/CI.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        julia-version: ['1.10', '1.11']
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v4
```

```yaml
# .github/workflows/TagBot.yml
name: TagBot
on:
  issue_comment:
    types: [created]
jobs:
  TagBot:
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
```

## Registration Workflow

| Step | Action |
|------|--------|
| 1. Prepare | Ensure all tests pass, docs build, compat entries set |
| 2. Register | Comment `@JuliaRegistrator register` on a commit |
| 3. Wait | Registrator opens PR to General registry (3-day wait for new packages) |
| 4. Tag | TagBot auto-creates GitHub release after merge |
| 5. Maintain | CompatHelper PRs keep dependency bounds current |

## Checklist

- [ ] Project.toml has UUID, version, and `[compat]` for all deps
- [ ] Module exports cover the public API
- [ ] Test suite in `test/` includes Aqua.jl quality checks
- [ ] Documenter.jl configured with `deploydocs` for GitHub Pages
- [ ] CI matrix tests against Julia 1.10+ (LTS and stable)
- [ ] TagBot workflow auto-tags releases on registration
- [ ] CompatHelper workflow keeps dependency bounds updated
- [ ] LICENSE file present (MIT recommended for Julia ecosystem)

---

**Version**: 1.0.6
