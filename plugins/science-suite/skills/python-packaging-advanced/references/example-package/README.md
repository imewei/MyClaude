# Example Package

> A modern Python package template with best practices for 2024/2025

## Features

- ✅ Modern `pyproject.toml` configuration
- ✅ src/ layout for cleaner package structure
- ✅ Type hints with mypy
- ✅ Code formatting with ruff
- ✅ Testing with pytest and coverage
- ✅ Python 3.12+ support

## Installation

```bash
# From PyPI (once published)
pip install example-package

# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
from example_package import greet

# Basic usage
message = greet("World")
print(message)  # Hello, World!
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/example-package.git
cd example-package

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=example_package --cov-report=html

# Run specific test
pytest tests/test_core.py::test_greet
```

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

### Building and Publishing

```bash
# Build distribution
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Project Structure

```
example-package/
├── src/
│   └── example_package/
│       ├── __init__.py
│       └── core.py
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## API Documentation

### `greet(name: str) -> str`

Returns a greeting message.

**Parameters:**
- `name` (str): The name to greet

**Returns:**
- str: A personalized greeting message

**Example:**
```python
>>> from example_package import greet
>>> greet("Alice")
'Hello, Alice!'
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Support

- Documentation: https://example-package.readthedocs.io
- Issues: https://github.com/yourusername/example-package/issues
- Discussions: https://github.com/yourusername/example-package/discussions
