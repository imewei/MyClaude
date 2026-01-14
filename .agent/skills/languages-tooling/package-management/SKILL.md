---
name: package-management
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia Package Management
description: Master Julia package management with Pkg.jl, Project.toml, and Manifest.toml for reproducible environments. Use when managing dependencies, specifying compatibility bounds, or setting up project environments.
---

# Julia Package Management

Pkg.jl workflows for reproducible Julia environments.

---

## Project.toml Structure

```toml
name = "MyPackage"
uuid = "12345678-1234-1234-1234-123456789012"
version = "0.1.0"

[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"

[compat]
julia = "1.6"
DataFrames = "1.3"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
```

---

## Core Pkg.jl Commands

| Command | Purpose |
|---------|---------|
| `Pkg.activate(".")` | Activate current project |
| `Pkg.instantiate()` | Install from Manifest.toml |
| `Pkg.add("Package")` | Add package |
| `Pkg.update()` | Update all packages |
| `Pkg.develop("Pkg")` | Link local package |
| `Pkg.status()` | List installed packages |
| `Pkg.test()` | Run package tests |

---

## Compatibility Bounds

```toml
[compat]
# Caret: Allow minor + patch updates
DataFrames = "^1.3.2"   # >=1.3.2, <2.0.0

# Tilde: Allow only patch updates
CSV = "~0.10.4"         # >=0.10.4, <0.11.0

# Range
Plots = "1.25 - 1.30"   # >=1.25.0, <1.31.0
```

---

## Environment Workflow

```julia
using Pkg

# Activate project environment
Pkg.activate(".")

# Install exact versions from Manifest
Pkg.instantiate()

# Add packages
Pkg.add("DataFrames")
Pkg.add(url="https://github.com/user/Pkg.git")

# Development mode
Pkg.develop(path="path/to/local/pkg")
Pkg.free("Package")  # Back to registry version
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Always specify [compat] | Prevents breaking updates |
| Commit Project.toml | Required for all projects |
| Commit Manifest.toml | For apps, NOT packages |
| Use instantiate() | For reproducibility |
| Regular updates | `Pkg.update()` frequently |
| Use Revise.jl | For interactive development |

---

## Checklist

- [ ] Project.toml created with deps
- [ ] [compat] bounds specified
- [ ] Environment activated
- [ ] Dependencies installed
- [ ] Test dependencies in [extras]
- [ ] Manifest.toml committed (for apps)

---

**Version**: 1.0.5
