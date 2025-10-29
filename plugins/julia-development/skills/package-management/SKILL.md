---
name: package-management
description: Master Project.toml structure, Pkg.jl workflows, and dependency management in Julia. Use when managing package dependencies, specifying compatibility bounds, creating reproducible environments, or working with Julia's package system.
---

# Package Management

Master Julia's package management system with Pkg.jl, Project.toml, and Manifest.toml for reproducible, well-managed Julia projects.

## Core Concepts

### Project.toml Structure

```toml
name = "MyPackage"
uuid = "12345678-1234-1234-1234-123456789012"
authors = ["Your Name <you@example.com>"]
version = "0.1.0"

[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"

[compat]
julia = "1.6"
DataFrames = "1.3"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

### Pkg.jl Workflows

```julia
using Pkg

# Environment management
Pkg.activate(".")              # Activate current project
Pkg.activate("path/to/project") # Activate specific project
Pkg.instantiate()              # Install exact versions from Manifest.toml

# Adding packages
Pkg.add("DataFrames")          # Add package
Pkg.add(["Plots", "CSV"])      # Add multiple
Pkg.add(name="Example", version="0.5") # Specific version
Pkg.add(url="https://github.com/user/Pkg.git") # From URL

# Development
Pkg.develop("MyPackage")       # Link local package for development
Pkg.develop(path="path/to/pkg") # Link from path
Pkg.free("MyPackage")          # Stop developing, use registry version

# Updating
Pkg.update()                   # Update all compatible packages
Pkg.update("DataFrames")       # Update specific package

# Removing
Pkg.rm("DataFrames")           # Remove package

# Status and info
Pkg.status()                   # List installed packages
Pkg.status("DataFrames")       # Show specific package info

# Testing
Pkg.test()                     # Run package tests
Pkg.test("DataFrames")         # Test specific package

# Other operations
Pkg.precompile()               # Precompile all packages
Pkg.gc()                       # Garbage collect old package versions
```

### Semantic Versioning and [compat]

```toml
[compat]
# Caret (^): Allow patch and minor updates
julia = "^1.6"          # >=1.6.0, <2.0.0
DataFrames = "^1.3.2"   # >=1.3.2, <2.0.0

# Tilde (~): Allow only patch updates
CSV = "~0.10.4"         # >=0.10.4, <0.11.0

# Range
Plots = "1.25-1.30"     # >=1.25.0, <1.31.0

# Single version (prefer ranges)
JSON = "0.21"           # >=0.21.0, <0.22.0
```

## Best Practices

- Always specify [compat] bounds in Project.toml
- Use semantic versioning: MAJOR.MINOR.PATCH
- Commit Project.toml for all projects
- Commit Manifest.toml for applications, NOT for packages
- Use Pkg.instantiate() for reproducibility
- Regularly update dependencies with Pkg.update()
- Use Revise.jl for interactive development

## Resources

- **Pkg.jl Documentation**: https://pkgdocs.julialang.org/
- **Semantic Versioning**: https://semver.org/
