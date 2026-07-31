---
name: python-packaging-advanced
description: Master modern Python packaging using uv, focusing on workspaces, monorepos, reproducible builds, and PyPI publishing. Use when configuring pyproject.toml for uv, setting up monorepo workspaces, choosing a build backend (hatchling, setuptools, flit), defining CLI entry points, managing uv.lock and virtual environments, wiring uv into CI or Docker, publishing to PyPI/TestPyPI, or migrating a project from pip or poetry to uv.
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

## 2. Project Layout and Build Backend

Always use the `src/` layout so tests import the *installed* package, never the
working directory.

```
my-package/
|-- pyproject.toml
|-- README.md
|-- LICENSE
|-- src/
|   `-- my_package/
|       |-- __init__.py
|       |-- core.py
|       `-- py.typed        # ship this so downstream mypy/pyright see your types
`-- tests/
    `-- test_core.py
```

| Backend | Use case |
|---------|----------|
| `hatchling` | Default. Modern, fast, opinionated. |
| `setuptools` | Needed for C extensions and `setuptools-scm` versioning. |
| `flit` | Pure-Python packages with minimal metadata. |
| `maturin` | Rust extensions — see [Rust Extensions](../rust-extensions/SKILL.md). |

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

[project.scripts]
science-cli = "science_tool.cli:main"      # console entry point

[tool.hatch.build.targets.wheel]
packages = ["src/science_tool"]
```

Read bundled data files through `importlib.resources`, never a relative path:

```python
from importlib.resources import files

config = files("science_tool").joinpath("data/config.json").read_text()
```

### Versioning

Either declare it statically in `__init__.py` (the `[tool.hatch.version]` block
above), or derive it from git tags:

```toml
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
dynamic = ["version"]

[tool.setuptools_scm]
write_to = "src/science_tool/_version.py"
```

### Namespace packages

For sibling distributions sharing a top-level name (`lab.core`, `lab.io`),
omit `__init__.py` from the shared `lab/` directory entirely — implicit
namespace packages (PEP 420). An `__init__.py` there breaks the other
distributions.

## 3. Dependency and Environment Management

| Scenario | Command |
|----------|---------|
| Initialize project | `uv init` |
| Add dependency | `uv add requests` |
| Add dev dependency | `uv add --dev pytest` |
| Add optional group | `uv add --optional science numpy` |
| Remove dependency | `uv remove requests` |
| Sync environment | `uv sync` |
| Sync exactly from lock (CI) | `uv sync --frozen` |
| Lock versions | `uv lock` |
| Update all packages | `uv lock --upgrade` |
| Update one package | `uv lock --upgrade-package numpy` |
| Create venv | `uv venv --python 3.12` |
| Run without activating | `uv run pytest` |
| Install/pin interpreter | `uv python install 3.12` / `uv python pin 3.12` |
| Editable dev install | `uv pip install -e ".[dev]"` |
| Export for legacy tools | `uv export --format requirements-txt > requirements.txt` |

`uv run` resolves and syncs before executing, so venv activation is never
required — prefer it in scripts, Makefiles, and CI.

### Dependency constraints

| Syntax | Meaning |
|--------|---------|
| `>=2.28` | Minimum version |
| `>=2.28,<3.0` | Compatible range — preferred for libraries |
| `~=2.28.0` | `>=2.28.0,<2.29` |
| `==2.28.3` | Exact — pin in `uv.lock`, not in `pyproject.toml` |

### Migrating to uv

```bash
# From pip
uv init && uv add -r requirements.txt

# From poetry — uv reads the existing pyproject.toml
uv sync
```

## 4. Building and Publishing

```bash
uv build                                   # -> dist/*.tar.gz, dist/*.whl
uvx twine check dist/*                     # metadata sanity check
uv publish --publish-url https://test.pypi.org/legacy/   # TestPyPI first
uv publish                                 # then PyPI
```

Verify the TestPyPI artifact in a clean environment before the real upload:

```bash
uv run --with science-tool --no-project --index https://test.pypi.org/simple/ \
  python -c "import science_tool; print(science_tool.__version__)"
```

### CI and container integration

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v2
        with: { enable-cache: true }
      - run: uv python install 3.12
      - run: uv sync --frozen --all-extras
      - run: uv run pytest
```

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev          # cached layer: deps only
COPY . .
CMD ["uv", "run", "python", "-m", "science_tool"]
```

Publish from CI with a PyPI Trusted Publisher (OIDC) rather than a long-lived
`PYPI_API_TOKEN` secret where the registry supports it.

## 5. Best Practices

- **Lockfiles**: Always commit `uv.lock` for applications; optional for libraries.
- **Python Versions**: Use `uv python pin 3.12` to ensure consistent execution.
- **Source Layout**: Always use the `src/` layout to prevent accidental imports of the local package.
- **Tools**: Configure `ruff` and `mypy` in `pyproject.toml` to centralize settings.

## Checklist

- [ ] `pyproject.toml` uses `hatchling` or similar modern backend.
- [ ] `uv.lock` is present and committed.
- [ ] Workspace members are correctly defined if using a monorepo.
- [ ] `src/` layout is implemented, with a `py.typed` marker if the package is typed.
- [ ] Development dependencies are in a named group (`dev`).
- [ ] `uv build` succeeds and `twine check dist/*` passes.
- [ ] Release verified from TestPyPI in a clean environment before PyPI upload.
- [ ] CI uses `uv sync --frozen`; Docker copies `pyproject.toml` + `uv.lock` before source.
