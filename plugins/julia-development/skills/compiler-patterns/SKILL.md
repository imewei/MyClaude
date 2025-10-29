---
name: compiler-patterns
description: PackageCompiler.jl for static compilation, creating executables, system images, and deployment optimization. Use for reducing startup time and creating standalone applications.
---

# Compiler Patterns (PackageCompiler.jl)

Create system images and standalone executables with PackageCompiler.jl.

## System Image (Faster Startup)
```julia
using PackageCompiler

create_sysimage(
    [:MyPackage, :Plots],
    sysimage_path="custom_sysimage.so",
    precompile_execution_file="precompile.jl"
)

# Use: julia --sysimage custom_sysimage.so
```

## Standalone Executable
```julia
create_app(
    "path/to/MyPackage",
    "MyApp",
    precompile_execution_file="precompile.jl"
)
```

## Resources
- **PackageCompiler.jl**: https://github.com/JuliaLang/PackageCompiler.jl
