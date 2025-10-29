---
name: package-development-workflow
description: Package structure, module organization, exports, and PkgTemplates.jl conventions. Use when creating Julia packages, organizing code, or setting up development environments. Foundation for /julia-scaffold command.
---

# Package Development Workflow

Master creating and organizing Julia packages following community standards.

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
