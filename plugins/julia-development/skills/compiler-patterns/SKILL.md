---
name: compiler-patterns
version: "2.1.0"
maturity: "5-Expert"
specialization: Julia Compilation
description: Create system images and standalone executables with PackageCompiler.jl. Use when reducing startup time or deploying Julia applications without requiring Julia installation.
---

# PackageCompiler.jl Patterns

System images and standalone executables for Julia deployment.

---

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

---

## Standalone Executable

```julia
create_app(
    "path/to/MyPackage",
    "MyApp",
    precompile_execution_file="precompile.jl"
)
```

---

## Use Cases

| Method | Purpose |
|--------|---------|
| System image | Faster startup for development |
| Executable | Deploy without Julia installation |
| Docker | Containerized Julia services |

---

## Parallel Compilation Strategies

| Strategy | Implementation | Benefit |
|----------|----------------|---------|
| **Parallel Precompile** | `JULIA_NUM_THREADS=auto` during build | Faster system image generation |
| **Incremental Builds** | Split large apps into packages | Recompile only changed components |
| **Artifact Caching** | Cache `~/.julia/artifacts` | Reuse binary dependencies |
| **Cloud Build** | Remote massive core machines | Scaling compilation vertically |

---

## Checklist

- [ ] Precompile script covers common paths
- [ ] System image reduces startup time
- [ ] Executable works standalone
- [ ] Deployment tested

---

**Version**: 1.0.5
