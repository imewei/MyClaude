# Python Library Packaging

Create distributable Python packages with modern build backends, type hints, and PyPI publishing workflows.

## Directory Structure

```
library-name/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── library_name/
│       ├── __init__.py
│       ├── py.typed
│       ├── core.py
│       ├── utils.py
│       └── exceptions.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/
│   ├── conf.py
│   ├── index.md
│   └── api.md
└── examples/
    └── basic_usage.py
```

## pyproject.toml (Modern Build Backend)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "library-name"
version = "0.1.0"
description = "A modern Python library"
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
maintainers = [
    {name = "Your Name", email = "your.email@example.com"},
]
keywords = ["python", "library", "package"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Typing :: Typed",
]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.24.0",
]

[project.urls]
Homepage = "https://github.com/username/library-name"
Documentation = "https://library-name.readthedocs.io"
Repository = "https://github.com/username/library-name"
Changelog = "https://github.com/username/library-name/blob/main/CHANGELOG.md"

[tool.hatch.version]
path = "src/library_name/__init__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/library_name"]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
    "/LICENSE",
]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "W",   # pycodestyle warnings
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src --cov-report=term-missing --cov-report=html"

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## src/library_name/__init__.py

```python
"""
Library Name - A modern Python library

This library provides...
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import main_function
from .exceptions import LibraryException

__all__ = [
    "main_function",
    "LibraryException",
    "__version__",
]
```

## src/library_name/py.typed

Create an empty `py.typed` file to indicate type hint support:

```bash
touch src/library_name/py.typed
```

This signals to type checkers that your package exports type information.

## src/library_name/core.py

```python
"""Core functionality of the library"""

from typing import Any


def main_function(data: str, *, option: bool = False) -> dict[str, Any]:
    """
    Main function that does something useful.

    Args:
        data: Input data to process
        option: Optional flag to enable feature

    Returns:
        Dictionary with processed results

    Raises:
        ValueError: If data is invalid

    Examples:
        >>> main_function("hello")
        {'result': 'hello', 'processed': True}
    """
    if not data:
        raise ValueError("Data cannot be empty")

    result = {
        "result": data,
        "processed": True,
        "option_enabled": option,
    }

    return result
```

## src/library_name/exceptions.py

```python
"""Custom exceptions for the library"""


class LibraryException(Exception):
    """Base exception for library-specific errors"""

    pass


class ValidationError(LibraryException):
    """Raised when validation fails"""

    pass


class ProcessingError(LibraryException):
    """Raised when processing fails"""

    pass
```

## tests/test_core.py

```python
import pytest

from library_name import main_function
from library_name.exceptions import ValidationError


def test_main_function_basic():
    """Test basic functionality"""
    result = main_function("test")
    assert result["result"] == "test"
    assert result["processed"] is True


def test_main_function_with_option():
    """Test with optional flag"""
    result = main_function("test", option=True)
    assert result["option_enabled"] is True


def test_main_function_empty_data():
    """Test error handling for empty data"""
    with pytest.raises(ValueError, match="Data cannot be empty"):
        main_function("")


@pytest.mark.parametrize(
    "data,expected",
    [
        ("hello", "hello"),
        ("world", "world"),
        ("test123", "test123"),
    ],
)
def test_main_function_parametrized(data, expected):
    """Test with multiple inputs"""
    result = main_function(data)
    assert result["result"] == expected
```

## README.md Template

```markdown
# Library Name

[![PyPI](https://img.shields.io/pypi/v/library-name.svg)](https://pypi.org/project/library-name/)
[![Python Version](https://img.shields.io/pypi/pyversions/library-name.svg)](https://pypi.org/project/library-name/)
[![License](https://img.shields.io/pypi/l/library-name.svg)](https://github.com/username/library-name/blob/main/LICENSE)
[![Tests](https://github.com/username/library-name/workflows/tests/badge.svg)](https://github.com/username/library-name/actions)

A modern Python library for...

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
uv uv pip install library-name
```

## Quick Start

```python
from library_name import main_function

result = main_function("hello")
print(result)
```

## Documentation

Full documentation is available at [https://library-name.readthedocs.io](https://library-name.readthedocs.io)

## Development

```bash
# Clone repository
git clone https://github.com/username/library-name.git
cd library-name

# Install with dev dependencies
uv sync

# Run tests
pytest

# Run type checking
mypy src

# Run linting
ruff check .
```

## License

MIT License - see LICENSE file for details.
```

## Building and Publishing

### Build Package

```bash
# Install build tool
uv add --dev build

# Build distribution files
python -m build

# This creates:
# dist/library_name-0.1.0-py3-none-any.whl
# dist/library_name-0.1.0.tar.gz
```

### Publish to PyPI

```bash
# Install twine
uv add --dev twine

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

### Using API Tokens (Recommended)

Create `.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
username = __token__
password = pypi-your-test-api-token-here
```

## Version Management

### Semantic Versioning

Follow [SemVer](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Add functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

### Automated Versioning

```bash
# Install bump2version
uv add --dev bump2version

# Bump version
bump2version patch  # 0.1.0 -> 0.1.1
bump2version minor  # 0.1.1 -> 0.2.0
bump2version major  # 0.2.0 -> 1.0.0
```

## Best Practices

### 1. Type Hints
- Add type hints to all public APIs
- Include `py.typed` marker file
- Run `mypy` in strict mode

### 2. Documentation
- Write docstrings (Google/NumPy style)
- Include examples in docstrings
- Generate API docs with `mkdocstrings`

### 3. Testing
- Aim for >90% code coverage
- Test edge cases and error conditions
- Use parametrized tests for similar cases

### 4. CI/CD
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - run: uv uv pip install uv
    - run: uv sync
    - run: pytest
    - run: mypy src
    - run: ruff check .
```

### 5. Changelog
Maintain `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/):

```markdown
# Changelog

## [0.1.0] - 2025-01-15

### Added
- Initial release
- Feature X
- Feature Y

### Fixed
- Bug fix Z
```
