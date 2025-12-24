---
version: "1.0.5"
category: "julia-development"
command: "/julia-scaffold"
description: Bootstrap new Julia package with proper structure, testing, documentation, and CI/CD
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: "<PackageName>"
color: cyan
execution_modes:
  quick: "5-10 minutes"
  standard: "15-20 minutes"
  comprehensive: "25-35 minutes"
agents:
  primary:
    - julia-developer
  conditional:
    - agent: julia-pro
      trigger: pattern "performance|optimization"
  orchestrated: false
---

# Bootstrap New Julia Package

Create a well-structured Julia package following modern best practices with PkgTemplates.jl.

## Package Name

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Plugins |
|------|----------|---------|
| Quick | 5-10 min | Git, License only |
| Standard (default) | 15-20 min | + GitHubActions, Codecov |
| Comprehensive | 25-35 min | + Documenter, CompatHelper, TagBot, cross-platform CI |

---

## Phase 1: Configuration

### Validation
- Package name is PascalCase ending with `.jl`
- Directory doesn't exist
- Author info available

### Template Selection

| Mode | Template Configuration |
|------|----------------------|
| Quick | `Git(manifest=false), License("MIT")` |
| Standard | `+ GitHubActions(coverage=true), Codecov()` |
| Comprehensive | `+ Documenter, CompatHelper(), TagBot(), Develop()` |

---

## Phase 2: Generation

```julia
using PkgTemplates
t = Template(; user="username", julia=v"1.6", plugins=[...])
t("PackageName")
```

### Generated Structure

```
PackageName/
├── .github/workflows/     # CI/CD (if included)
├── docs/                  # Documentation (if included)
│   ├── make.jl
│   └── src/index.md
├── src/PackageName.jl     # Main module
├── test/runtests.jl       # Test suite
├── Project.toml           # Package metadata
├── LICENSE
└── README.md
```

---

## Phase 3: Module Setup

### Single File Package
```julia
module PackageName
export useful_function

"""
    useful_function(x)

Brief description.
"""
useful_function(x) = x + 1

end
```

### Multi-File Package
```julia
module PackageName
export Type1, function1

include("types.jl")
include("functions.jl")

end
```

---

## Phase 4: Test Setup

```julia
# test/runtests.jl
using PackageName, Test

@testset "PackageName.jl" begin
    @testset "Feature 1" begin
        @test useful_function(1) == 2
    end
end
```

---

## Phase 5: Post-Creation Checklist

1. **Navigate**: `cd PackageName`
2. **Git remote**: `git remote add origin git@github.com:user/PackageName.jl.git`
3. **Doc keys** (if Documenter): `DocumenterTools.genkeys(user="...", repo="...")`
4. **Add deps**: `Pkg.add("DependencyName")`
5. **Update compat**: Add `[compat]` entries in Project.toml
6. **Push**: `git push -u origin main`

---

## Phase 6: Development Workflow

```julia
using Pkg; Pkg.activate(".")
using Revise, PackageName  # Auto-reload on changes
Pkg.test()                 # Run tests
```

---

## Package Type Recommendations

| Type | Mode | Why |
|------|------|-----|
| Research/Academic | Standard | Docs for citations, moderate automation |
| Open Source Library | Comprehensive | Professional quality, full automation |
| Internal Tool | Quick | Fast setup, minimal overhead |
| Performance-Critical | Comprehensive | Need benchmarking, optimization tracking |

---

## Module Organization

| Size | Strategy |
|------|----------|
| < 500 lines | Single file |
| 500-5000 lines | Multi-file with `include()` |
| > 5000 lines | Submodules |

---

## Naming Conventions

| ✅ Good | ❌ Bad |
|---------|--------|
| `DataStructures.jl` | `my_package` (no .jl) |
| `HTTP.jl` | `MP.jl` (unclear) |
| `MyPackage.jl` | `JuliaMyPackage.jl` (redundant) |

---

## Success Criteria

- ✅ Package generated with proper structure
- ✅ Project.toml has correct metadata
- ✅ Initial module and tests created
- ✅ CI/CD configured (if included)
- ✅ Documentation set up (if included)
- ✅ Post-creation checklist provided

---

## External Documentation

- `package-scaffolding.md` - Complete guide (~450 lines)

## Related Commands
- `/julia-package-ci` - Add/update CI workflows
- `/sciml-setup` - Add SciML functionality
