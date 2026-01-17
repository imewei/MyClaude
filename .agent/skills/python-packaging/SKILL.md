---
name: python-packaging
version: "1.0.7"
description: Create distributable Python packages with pyproject.toml, proper project structure, and publishing to PyPI. Use when writing pyproject.toml or setup.py, creating package structures with __init__.py, building wheel/source distributions, setting up CLI entry points, configuring build backends (setuptools, hatchling), or publishing to PyPI/TestPyPI.
---

# Python Packaging

Modern Python packaging with pyproject.toml and PyPI distribution.

## Build Backend Selection

| Backend | Use Case | Speed |
|---------|----------|-------|
| setuptools | Traditional, C extensions | Standard |
| hatchling | Modern, opinionated | Fast |
| flit | Pure Python only | Fast |
| poetry | Deps + packaging | Moderate |

## Project Structure (src layout)

```
my-package/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core.py
│       └── py.typed
└── tests/
    └── test_core.py
```

## Complete pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
version = "1.0.0"
description = "Package description"
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
authors = [{name = "Your Name", email = "you@example.com"}]
keywords = ["example", "package"]
classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "requests>=2.28.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.1", "mypy>=1.0"]

[project.urls]
Homepage = "https://github.com/user/my-package"
Repository = "https://github.com/user/my-package"

[project.scripts]
my-cli = "my_package.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
my_package = ["py.typed", "data/*.json"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[tool.mypy]
python_version = "3.12"
strict = true
```

## CLI Entry Points

```python
# src/my_package/cli.py
import click

@click.group()
@click.version_option()
def cli():
    """My CLI tool."""
    pass

@cli.command()
@click.argument("name")
@click.option("--greeting", default="Hello")
def greet(name: str, greeting: str):
    """Greet someone."""
    click.echo(f"{greeting}, {name}!")

def main():
    cli()
```

```toml
# pyproject.toml
[project.scripts]
my-cli = "my_package.cli:main"
```

## Version Management

### Static Version

```python
# src/my_package/__init__.py
__version__ = "1.0.0"
```

### Dynamic Version (Git-based)

```toml
[build-system]
requires = ["setuptools>=61.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
dynamic = ["version"]

[tool.setuptools_scm]
write_to = "src/my_package/_version.py"
```

## Building and Publishing

```bash
# Install build tools
pip install build twine

# Build distributions
python -m build
# Creates: dist/my_package-1.0.0.tar.gz, my_package-1.0.0-py3-none-any.whl

# Check package
twine check dist/*

# Test on TestPyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ my-package

# Publish to PyPI
twine upload dist/*
```

## GitHub Actions Publishing

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: "3.12" }
      - run: pip install build twine
      - run: python -m build
      - run: twine check dist/*
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

## Including Data Files

```toml
[tool.setuptools.package-data]
my_package = ["data/*.json", "templates/*.html", "py.typed"]
```

```python
# Access data files
from importlib.resources import files

data = files("my_package").joinpath("data/config.json").read_text()
```

## Namespace Packages

For multi-repo packages under same namespace:

```
# company-core/
company/core/__init__.py

# company-api/
company/api/__init__.py

# NO __init__.py in company/ directory!
```

```toml
[tool.setuptools.packages.find]
include = ["company.core*"]
```

## Development Installation

```bash
# Editable install
pip install -e .

# With optional dependencies
pip install -e ".[dev]"
```

## Best Practices

| Practice | Guideline |
|----------|-----------|
| Layout | Use src/ layout for cleaner imports |
| Config | Use pyproject.toml, not setup.py |
| Versioning | Semantic versioning (MAJOR.MINOR.PATCH) |
| Type hints | Include py.typed marker |
| Testing | Test in clean environment before release |
| TestPyPI | Always test on TestPyPI first |
| CI/CD | Automate publishing with GitHub Actions |

## Dependency Constraints

| Syntax | Meaning |
|--------|---------|
| `>=2.28.0` | Minimum version |
| `>=2.28.0,<3.0` | Compatible range |
| `~=2.28.0` | >=2.28.0,<2.29 |
| `==2.28.3` | Exact (avoid) |

## Checklist

- [ ] src/ layout with proper __init__.py
- [ ] Complete pyproject.toml metadata
- [ ] Version number updated
- [ ] README.md and LICENSE included
- [ ] Tests passing
- [ ] Builds without errors (`python -m build`)
- [ ] Tested in clean venv
- [ ] Published to TestPyPI first
- [ ] Git tag created for release
