---
name: uv-package-manager
version: "1.0.7"
description: Master uv for blazing-fast Python dependency management, virtual environments, and lockfiles. Use when running uv commands (init, add, sync, venv, run), managing dependencies 10-100x faster than pip/poetry, installing Python versions, working with uv.lock for reproducible builds, optimizing Docker/CI builds, or migrating from pip/poetry to uv.
---

# UV Package Manager

Ultra-fast Python package management with Rust-powered uv.

## Speed Comparison

| Tool | Install Time | Speedup |
|------|-------------|---------|
| pip | ~30s | 1x |
| poetry | ~20s | 1.5x |
| uv | ~2-3s | 10-15x |

## Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip
pip install uv
```

## Quick Start

```bash
# Create new project
uv init my-project && cd my-project

# Add dependencies
uv add requests pandas

# Add dev dependencies
uv add --dev pytest black ruff

# Sync (install from lock)
uv sync

# Run command in venv
uv run python app.py
uv run pytest
```

## Essential Commands

| Command | Purpose |
|---------|---------|
| `uv init` | Initialize project |
| `uv add PKG` | Add dependency |
| `uv add --dev PKG` | Add dev dependency |
| `uv remove PKG` | Remove dependency |
| `uv sync` | Install from lockfile |
| `uv lock` | Generate uv.lock |
| `uv run CMD` | Run in venv |
| `uv venv` | Create venv |
| `uv python install 3.12` | Install Python |
| `uv python pin 3.12` | Set Python version |

## Virtual Environments

```bash
# Create venv
uv venv
uv venv --python 3.12

# Use uv run (no activation needed)
uv run python script.py
uv run pytest

# Or activate manually
source .venv/bin/activate
```

## Lockfile Workflows

```bash
# Create/update lockfile
uv lock

# Install exact versions from lock
uv sync --frozen

# Upgrade all packages
uv lock --upgrade

# Upgrade specific package
uv lock --upgrade-package requests

# Export to requirements.txt
uv export --format requirements-txt > requirements.txt
```

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
        with: { enable-cache: true }
      - run: uv python install 3.12
      - run: uv sync --all-extras --dev
      - run: uv run pytest
```

## Docker Integration

```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen = exact versions)
RUN uv sync --frozen --no-dev

COPY . .
CMD ["uv", "run", "python", "app.py"]
```

## Migration

### From pip

```bash
# Before
pip install -r requirements.txt

# After
uv init
uv add -r requirements.txt
# Or: uv pip install -r requirements.txt
```

### From poetry

```bash
# Before
poetry install

# After (reads existing pyproject.toml)
uv sync
```

## pyproject.toml

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.31.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.1", "mypy>=1.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Best Practices

| Practice | Command |
|----------|---------|
| Pin Python | `uv python pin 3.12` |
| Use lockfile | Commit `uv.lock` |
| CI frozen | `uv sync --frozen` |
| No venv activation | Use `uv run` |
| Update regularly | `uv lock --upgrade` |

## Checklist

- [ ] Project initialized with `uv init`
- [ ] Python version pinned (.python-version)
- [ ] Dependencies in pyproject.toml
- [ ] uv.lock committed to git
- [ ] CI uses `uv sync --frozen`
- [ ] Docker uses uv for builds
- [ ] Using `uv run` instead of activating venv
