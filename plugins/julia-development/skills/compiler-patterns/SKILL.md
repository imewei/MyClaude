---
name: compiler-patterns
description: Master PackageCompiler.jl for static compilation, creating standalone executables, custom system images, and deployment optimization to reduce Julia startup time. Use when creating standalone executables from Julia packages (.jl files compiled to binaries), building custom system images (sysimage.so) for faster startup, reducing latency with precompilation files, deploying Julia applications without requiring Julia installation, creating distributable applications with create_app(), optimizing startup time with create_sysimage(), working with precompile_execution_file for warmup, or packaging Julia code for production deployment. Essential for production deployments, distribution, and minimizing Julia's startup latency.
---

# Compiler Patterns (PackageCompiler.jl)

Create system images and standalone executables with PackageCompiler.jl.

## When to use this skill

- Creating standalone executables from Julia packages (no Julia installation needed)
- Building custom system images (sysimage.so) for faster startup
- Reducing Julia startup latency with precompilation
- Deploying Julia applications to production environments
- Creating distributable applications with create_app()
- Optimizing startup time with create_sysimage()
- Working with precompile_execution_file for package warmup
- Packaging Julia code for users without Julia
- Reducing time-to-first-plot or time-to-first-execution
- Creating Docker images with optimized Julia startup
- Deploying performance-critical Julia services

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
