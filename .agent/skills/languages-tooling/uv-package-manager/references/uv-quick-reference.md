# UV Quick Reference Guide

> **Fast, reliable Python package management** with uv - 10-100x faster than pip!

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Migration from pip](#migration-from-pip)
- [Migration from poetry](#migration-from-poetry)
- [Common Workflows](#common-workflows)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip (if already installed)
pip install uv

# Verify installation
uv --version
```

---

## Quick Start

```bash
# Create a new project
uv init my-project
cd my-project

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install requests pandas

# Install from requirements.txt
uv pip install -r requirements.txt

# Run a script with uv
uv run python script.py
```

---

## Command Reference

### Project Management

```bash
# Initialize new project
uv init <project-name>

# Create virtual environment
uv venv                    # Create .venv in current directory
uv venv my-env             # Create named environment
uv venv --python 3.12      # Specify Python version

# Activate environment (manual)
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Package Installation

```bash
# Install packages
uv pip install <package>              # Single package
uv pip install <package>==1.2.3       # Specific version
uv pip install <package>>=1.0.0       # Version constraint
uv pip install package1 package2      # Multiple packages

# Install from requirements file
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# Install in editable mode
uv pip install -e .                   # Current directory
uv pip install -e path/to/package     # Specific path

# Install with extras
uv pip install package[dev,test]
```

### Package Management

```bash
# List installed packages
uv pip list
uv pip list --format=json

# Show package information
uv pip show <package>

# Freeze installed packages
uv pip freeze
uv pip freeze > requirements.txt

# Uninstall packages
uv pip uninstall <package>
uv pip uninstall -r requirements.txt

# Upgrade packages
uv pip install --upgrade <package>
uv pip install --upgrade -r requirements.txt
```

### Lock Files

```bash
# Generate lock file
uv pip compile requirements.in -o requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# Update lock file
uv pip compile requirements.in -o requirements.txt --upgrade

# Install from lock file
uv pip sync requirements.txt
```

### Running Commands

```bash
# Run Python script with uv
uv run python script.py

# Run with automatic dependency management
uv run --with requests python script.py

# Run in specific environment
uv run --python 3.12 python script.py
```

### Advanced Operations

```bash
# Check for dependency conflicts
uv pip check

# Generate requirements from installed packages
uv pip freeze > requirements.txt

# Install without dependencies
uv pip install --no-deps <package>

# Reinstall package
uv pip install --force-reinstall <package>

# Cache management
uv cache clean               # Clear cache
uv cache dir                 # Show cache directory
```

---

## Migration from pip

### Direct Command Mapping

| pip Command | uv Equivalent | Notes |
|-------------|---------------|-------|
| `pip install package` | `uv pip install package` | 10-100x faster |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` | Parallel downloads |
| `pip uninstall package` | `uv pip uninstall package` | Same syntax |
| `pip list` | `uv pip list` | Same output format |
| `pip freeze` | `uv pip freeze` | Identical output |
| `pip show package` | `uv pip show package` | Package metadata |
| `python -m venv .venv` | `uv venv` | Faster creation |

### Migration Steps

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create new virtual environment with uv
uv venv

# 3. Activate environment
source .venv/bin/activate

# 4. Install existing dependencies
uv pip install -r requirements.txt

# 5. Verify installation
uv pip list

# 6. Optional: Generate lock file
uv pip compile requirements.txt
```

### Replacing pip in existing projects

```bash
# Before (pip)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# After (uv) - much faster!
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

---

## Migration from poetry

### Command Mapping

| poetry Command | uv Equivalent | Notes |
|----------------|---------------|-------|
| `poetry init` | `uv init` | Project initialization |
| `poetry install` | `uv pip install -e .` | Install project |
| `poetry add package` | `uv pip install package` | Add dependency |
| `poetry remove package` | `uv pip uninstall package` | Remove dependency |
| `poetry show` | `uv pip list` | List packages |
| `poetry run python` | `uv run python` | Run command |
| `poetry lock` | `uv pip compile pyproject.toml` | Generate lock |
| `poetry update` | `uv pip install --upgrade -r requirements.txt` | Update deps |

### Converting pyproject.toml

```toml
# poetry pyproject.toml
[tool.poetry]
name = "my-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.12"
requests = "^2.31.0"
pandas = "^1.0.2"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.0.0"
```

Converts to:

```toml
# uv-compatible pyproject.toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.31.0",
    "pandas>=1.0.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
]
```

### Migration Steps

```bash
# 1. Backup poetry files
cp pyproject.toml pyproject.toml.bak
cp poetry.lock poetry.lock.bak

# 2. Export dependencies to requirements.txt
poetry export -f requirements.txt -o requirements.txt --without-hashes

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Create virtual environment with uv
rm -rf .venv  # Remove poetry venv
uv venv

# 5. Install dependencies
source .venv/bin/activate
uv pip install -r requirements.txt

# 6. Update pyproject.toml format (see example above)
# Manually convert [tool.poetry] to [project]

# 7. Generate lock file
uv pip compile pyproject.toml -o requirements.txt

# 8. Test your application
uv run python -m pytest
```

---

## Common Workflows

### New Project Setup

```bash
# Create project structure
uv init my-project
cd my-project

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install requests pandas numpy
uv pip install -e .

# Save dependencies
uv pip freeze > requirements.txt
```

### Adding Dependencies

```bash
# Install new package
uv pip install new-package

# Update requirements.txt
uv pip freeze > requirements.txt

# Or use requirements.in + compile
echo "new-package>=1.0.0" >> requirements.in
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt
```

### Development Workflow

```bash
# Install project in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Run formatter
uv run black .

# Run type checker
uv run mypy src/
```

### CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Create virtual environment
        run: uv venv

      - name: Install dependencies
        run: |
          source .venv/bin/activate
          uv pip install -r requirements.txt
          uv pip install -e ".[dev]"

      - name: Run tests
        run: |
          source .venv/bin/activate
          uv run pytest
```

### Lock File Workflow

```bash
# Create requirements.in with loose constraints
cat > requirements.in << EOF
requests>=2.31.0
pandas>=1.0.2
numpy>=1.24.0
EOF

# Generate lock file (exact versions)
uv pip compile requirements.in -o requirements.txt

# Install exact versions from lock file
uv pip sync requirements.txt

# Update dependencies (get latest compatible versions)
uv pip compile requirements.in -o requirements.txt --upgrade

# Update specific package
uv pip compile requirements.in -o requirements.txt --upgrade-package requests
```

---

## Configuration

### pyproject.toml Configuration

```toml
[tool.uv]
# Python version constraint
python = ">=3.12"

# Index URL (custom PyPI mirror)
index-url = "https://pypi.org/simple"

# Extra index URLs
extra-index-url = [
    "https://download.pytorch.org/whl/cpu"
]

# Trusted hosts
trusted-hosts = ["download.pytorch.org"]

# Cache directory
cache-dir = ".uv-cache"
```

### Environment Variables

```bash
# Custom index URL
export UV_INDEX_URL="https://pypi.org/simple"

# Extra index URL
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# Cache directory
export UV_CACHE_DIR="$HOME/.cache/uv"

# System Python location
export UV_PYTHON="/usr/bin/python3.12"

# Disable cache
export UV_NO_CACHE=1

# Verbose output
export UV_VERBOSE=1
```

---

## Troubleshooting

### Common Issues

#### Issue: "No such file or directory: 'uv'"

```bash
# Solution: Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Issue: "Could not find a version that matches..."

```bash
# Solution 1: Update lock file
uv pip compile requirements.in -o requirements.txt --upgrade

# Solution 2: Clear cache
uv cache clean
uv pip install -r requirements.txt

# Solution 3: Check for conflicts
uv pip check
```

#### Issue: "Permission denied" errors

```bash
# Solution: Use virtual environment
uv venv
source .venv/bin/activate
uv pip install package

# Never use sudo with uv!
```

#### Issue: Slow network / timeouts

```bash
# Solution: Increase timeout
uv pip install --timeout 300 package

# Or use retry
uv pip install --retries 5 package
```

#### Issue: Package not found in index

```bash
# Solution: Add extra index URL
uv pip install --extra-index-url https://test.pypi.org/simple package

# Or configure in pyproject.toml (see Configuration section)
```

### Debugging

```bash
# Verbose output
uv -v pip install package
uv -vv pip install package  # Extra verbose

# Show what would be installed (dry run)
uv pip install --dry-run package

# Check dependency tree
uv pip show package

# Verify environment
uv pip list
python -m site  # Show Python paths
```

### Performance Tips

```bash
# Use lock files for reproducibility
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt  # Faster than install

# Pre-compile wheels (for large packages)
uv pip install --no-deps package  # Install without dependencies first

# Use cache effectively
uv cache dir  # Show cache location
# Don't clear cache unless necessary

# Parallel downloads (default)
# uv automatically uses parallel downloads

# Use system Python (faster venv creation)
uv venv --python /usr/bin/python3.12
```

---

## Best Practices

### 1. Use Lock Files

```bash
# Generate lock file for reproducibility
uv pip compile requirements.in -o requirements.txt

# Install from lock file
uv pip sync requirements.txt

# Commit lock file to version control
git add requirements.txt
```

### 2. Separate Dev Dependencies

```bash
# requirements.in (production)
requests>=2.31.0
pandas>=1.0.2

# requirements-dev.in (development)
-c requirements.txt  # Constrain to production versions
pytest>=7.4.0
black>=23.0.0
mypy>=1.7.0
```

### 3. Use Virtual Environments

```bash
# Always use virtual environments
uv venv
source .venv/bin/activate

# Never install globally (no sudo!)
```

### 4. Pin Python Version

```toml
# pyproject.toml
[project]
requires-python = ">=3.12,<3.13"
```

### 5. Cache Management

```bash
# Cache is good - don't clear unless necessary
# Cache location
uv cache dir

# Only clear if corrupted
uv cache clean
```

---

## Performance Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|-----|---------|
| Install Flask | 5.2s | 0.3s | **17x faster** |
| Install Django | 8.1s | 0.5s | **16x faster** |
| Install pandas | 12.3s | 0.8s | **15x faster** |
| Create venv | 2.1s | 0.1s | **21x faster** |
| Resolve deps (large project) | 45s | 2s | **22x faster** |

---

## Additional Resources

- **Official Documentation**: https://docs.astral.sh/uv/
- **GitHub Repository**: https://github.com/astral-sh/uv
- **Migration Guide**: https://docs.astral.sh/uv/guides/migration/
- **Configuration Reference**: https://docs.astral.sh/uv/reference/settings/
- **Community**: https://discord.gg/astral-sh
