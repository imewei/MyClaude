---
description: Bootstrap new Julia package with proper structure, testing, documentation,
  and CI/CD
triggers:
- /julia-scaffold
- bootstrap new julia package
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



## User Input
Input arguments pattern: `<PackageName>`
The agent should parse these arguments from the user's request.

# Bootstrap New Julia Package

Package: $ARGUMENTS

## Validation

- PascalCase ending with `.jl`
- Directory doesn't exist
- Author info available

## Generation

```julia
using PkgTemplates
t = Template(; user="username", julia=v"1.6", plugins=[...])
t("PackageName")
```

### Structure
```
PackageName/
├── .github/workflows/
├── docs/
├── src/PackageName.jl
├── test/runtests.jl
├── Project.toml
└── LICENSE
```

## Module Setup

**Single file** (<500 lines):
```julia
module PackageName
export useful_function
useful_function(x) = x + 1
end
```

**Multi-file** (500-5000 lines):
```julia
module PackageName
export Type1, function1
include("types.jl")
include("functions.jl")
end
```

## Tests

```julia
using PackageName, Test
@testset "PackageName.jl" begin
    @test useful_function(1) == 2
end
```

## Post-Creation

1. `cd PackageName`
2. `git remote add origin git@github.com:user/PackageName.jl.git`
3. `DocumenterTools.genkeys()` (if Documenter)
4. `Pkg.add("Dependencies")`
5. Add `[compat]` entries
6. `git push -u origin main`

## Development

```julia
using Pkg; Pkg.activate(".")
using Revise, PackageName
Pkg.test()
```

**Docs**: `package-scaffolding.md`
