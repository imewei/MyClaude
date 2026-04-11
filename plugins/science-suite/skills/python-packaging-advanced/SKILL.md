---
name: python-packaging-advanced
description: Master modern Python packaging using uv, focusing on workspaces, monorepos, and reproducible builds. Use when configuring pyproject.toml for uv, setting up monorepo workspaces, managing toolchains with uv, defining dependency groups, or publishing high-performance Python libraries.
---

# Python Packaging with uv

Modern Python packaging centers on `uv` for speed, reliability, and workspace management.

## Expert Agent

For advanced packaging, workspace configuration, or CI/CD integration, delegate to:

- **`python-pro`**: Expert in modern Python systems engineering and `uv` toolchains.
  - *Location*: `plugins/science-suite/agents/python-pro.md`

## 1. uv Workspaces (Monorepos)

Workspaces allow managing multiple packages in a single repository with shared dependencies and a single lockfile.

### Root `pyproject.toml`
```toml
[project]
name = "my-monorepo"
version = "0.1.0"
dependencies = []

[tool.uv.workspace]
members = ["packages/*"]
```

### Member `packages/core/pyproject.toml`
```toml
[project]
name = "my-core"
version = "0.1.0"
dependencies = [
    "numpy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Workspace Commands
```bash
# Add dependency to a specific package
uv add --package my-core pandas

# Run command in a package context
uv run --package my-core pytest

# Sync the entire workspace
uv sync
```

## 2. Project Configuration

Use `hatchling` as the build backend for modern performance.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "science-tool"
dynamic = ["version"]
requires-python = ">=3.12"
dependencies = [
    "pydantic",
    "scipy",
]

[project.optional-dependencies]
dev = [
    "ruff",
    "mypy",
    "pytest",
]

[tool.hatch.version]
path = "src/science_tool/__init__.py"
```

## 3. Dependency Management

| Scenario | Command |
|----------|---------|
| Add dependency | `uv add requests` |
| Add dev dependency | `uv add --dev pytest` |
| Add optional group | `uv add --optional science numpy` |
| Sync environment | `uv sync` |
| Lock versions | `uv lock` |
| Update packages | `uv lock --upgrade` |

## 4. Best Practices

- **Lockfiles**: Always commit `uv.lock` for applications; optional for libraries.
- **Python Versions**: Use `uv python pin 3.12` to ensure consistent execution.
- **Source Layout**: Always use the `src/` layout to prevent accidental imports of the local package.
- **Tools**: Configure `ruff` and `mypy` in `pyproject.toml` to centralize settings.

## Checklist

- [ ] `pyproject.toml` uses `hatchling` or similar modern backend.
- [ ] `uv.lock` is present and committed.
- [ ] Workspace members are correctly defined if using a monorepo.
- [ ] `src/` layout is implemented.
- [ ] Development dependencies are in a named group (`dev`).
