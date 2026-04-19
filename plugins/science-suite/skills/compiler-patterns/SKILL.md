---
name: compiler-patterns
description: Create system images and standalone executables with PackageCompiler.jl. Use when reducing startup time or deploying Julia applications without requiring Julia installation. Also use when building Docker images for Julia services, writing precompile scripts, bundling apps with artifacts, troubleshooting method invalidations with SnoopCompile, or shipping Julia code to end users. Use proactively when the user complains about Julia startup latency or asks how to distribute a Julia application.
---

# PackageCompiler.jl Patterns

## Expert Agent

For Julia compilation, system images, and standalone executables, delegate to:

- **`julia-pro`**: Julia performance optimization and deployment.
  - *Location*: `plugins/science-suite/agents/julia-pro.md`

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

## Precompile Script

```julia
# precompile.jl — exercise all hot paths to record native code
using MyPackage

# Cover common entry points
data = load_sample_data()
result = MyPackage.process(data)
MyPackage.export_results(result, "/tmp/test_output.csv")

# Cover plotting paths if included in sysimage
using Plots
plot(rand(100), title="warmup")
savefig("/tmp/warmup.png")
```

## Advanced System Image Options

```julia
using PackageCompiler

create_sysimage(
    [:MyPackage, :DifferentialEquations, :Plots],
    sysimage_path = "custom_sysimage.so",
    precompile_execution_file = "precompile.jl",
    precompile_statements_file = "precompile_statements.jl",
    cpu_target = "generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)",
    incremental = false
)
```

## App Bundling with Artifacts

```julia
create_app(
    "path/to/MyPackage",
    "build/MyApp",
    precompile_execution_file = "precompile.jl",
    include_lazy_artifacts = true,
    force = true
)
# Result: build/MyApp/bin/MyPackage executable
# Ships with bundled Julia runtime and all dependencies
```

## Docker Deployment

```dockerfile
FROM julia:1.10 AS builder
WORKDIR /app
COPY . .
RUN julia --project -e 'using Pkg; Pkg.instantiate()'
RUN julia --project build_sysimage.jl

FROM debian:bookworm-slim
COPY --from=builder /app/build /app
ENTRYPOINT ["/app/bin/MyPackage"]
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Missing methods at runtime | Precompile script incomplete | Add the failing call path to precompile.jl |
| Large sysimage (>1 GB) | Too many packages | Minimize included packages |
| Segfault on load | CPU target mismatch | Use `cpu_target="generic"` |
| Slow first call despite sysimage | Method invalidations | Check with `SnoopCompile.@snoopr` |

## Checklist

- [ ] Precompile script exercises all user-facing entry points
- [ ] System image tested on target hardware
- [ ] `cpu_target` matches deployment architecture
- [ ] Executable runs without Julia installation
- [ ] Docker image uses multi-stage build for minimal size
- [ ] Method invalidations checked with SnoopCompile
- [ ] CI builds sysimage and runs smoke tests

---

**Version**: 1.0.5
