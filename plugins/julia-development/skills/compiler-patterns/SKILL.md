---
name: compiler-patterns
version: "1.0.7"
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

## Checklist

- [ ] Precompile script covers common paths
- [ ] System image reduces startup time
- [ ] Executable works standalone
- [ ] Deployment tested

---

**Version**: 1.0.5
