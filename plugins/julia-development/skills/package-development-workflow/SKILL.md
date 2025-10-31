---
name: package-development-workflow
description: Master Julia package structure, module organization, exports, and PkgTemplates.jl conventions for creating robust packages. Use when creating new Julia packages (.jl files in src/ directory), organizing package source code with modules, setting up Project.toml and Manifest.toml files, defining module exports and public APIs, structuring test/ directories with runtests.jl, creating package documentation in docs/, using PkgTemplates.jl for automated package scaffolding, organizing multi-file packages with include() patterns, or bootstrapping development environments. Foundation for /julia-scaffold command and essential for all Julia package development workflows.
---

# Package Development Workflow

Master creating and organizing Julia packages following community standards.

## When to use this skill

- Creating new Julia packages from scratch
- Structuring source files in src/ directory with proper module organization
- Setting up Project.toml with dependencies and compatibility bounds
- Defining package exports and public vs internal APIs
- Organizing test suites in test/ directory
- Using PkgTemplates.jl for automated package scaffolding
- Creating multi-file packages with proper include() patterns
- Setting up package documentation structure
- Establishing development environments for package work
- Following Julia package ecosystem conventions

## Standard Package Structure
```
MyPackage/
├── Project.toml
├── src/MyPackage.jl
├── test/runtests.jl
├── docs/make.jl
├── README.md
└── LICENSE
```

## Module Pattern
```julia
module MyPackage

export public_function, PublicType

# Implementation
function public_function(x)
    internal_helper(x)
end

function internal_helper(x)
    # Not exported
end

end # module
```

## Resources
- **PkgTemplates.jl**: https://github.com/JuliaCI/PkgTemplates.jl
