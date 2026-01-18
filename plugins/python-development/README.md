# Python Development

Master Python 3.12+ with modern tools, async patterns, FastAPI/Django frameworks, performance optimization, and production-ready practices. Includes expert agents, comprehensive skills, and scaffolding commands for the 2024/2025 Python ecosystem.

**Version:** 2.1.0 | **Category:** development | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/python-development.html) | [CHANGELOG](CHANGELOG.md)

---

## 🚀 What's New in v2.1.0

### Command Optimization with Execution Modes

The `/python-scaffold` command now features:
- **3 Execution Modes** (quick/standard/enterprise) for flexible scaffolding
- **Comprehensive External Documentation** (~2,050 lines across 5 guides)
- **41% Command File Reduction** while preserving all content
- **Version Consistency** across all components (1.0.3)

#### `/python-scaffold` - 3 Execution Modes

| Mode | Duration | Agents | Scope |
|------|----------|--------|-------|
| **Quick** | 1-2 hours | 1 agent | Simple script, prototype, basic CLI (~15 files) |
| **Standard** | 3-6 hours | 2 agents | Production FastAPI/Django, library (~50 files) |
| **Enterprise** | 1-2 days | 3 agents | Microservices, K8s, observability (~100 files) |

**External Documentation** (5 files - ~2,050 lines):
- FastAPI Structure (~470 lines) - SQLAlchemy 2.0, Pydantic V2, async patterns, testing
- Django Structure (~410 lines) - Django 5.x, DRF, Celery, async views
- Library Packaging (~530 lines) - Modern build backends, PyPI publishing
- CLI Tools (~390 lines) - Typer, Rich console, subcommands, testing
- Development Tooling (~450 lines) - Makefile, Docker, CI/CD, pre-commit hooks

---


## What's New in v2.1.0

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## Agents (3)

### python-pro

**Status:** active | **Model:** sonnet | **Performance:** comprehensive

Master Python 3.12+ with modern features, async programming, and production-ready practices. Now includes systematic development process with 8 self-verification steps, quality assurance principles, and comprehensive examples showing 10x performance improvements with async optimization.

**Key Capabilities:**
- Modern Python 3.12+ features and best practices
- Async/await patterns and asyncio event loop optimization
- Modern tooling (uv, ruff, mypy, pytest) selection and integration
- Performance profiling and optimization strategies
- Production-ready error handling and logging

### fastapi-pro

**Status:** active | **Model:** sonnet | **Performance:** framework-specific

Build high-performance async APIs with FastAPI, SQLAlchemy 2.0, and Pydantic V2. Enhanced with production-ready microservice examples including JWT authentication, comprehensive tests, and N+1 query optimization achieving 400x performance improvements.

**Key Capabilities:**
- FastAPI async endpoints with dependency injection
- SQLAlchemy 2.0 async ORM with proper relationship loading
- Pydantic V2 models with validation and serialization
- JWT authentication and authorization patterns
- API testing with TestClient and pytest-asyncio

### django-pro

**Status:** active | **Model:** sonnet | **Performance:** framework-specific

Master Django 5.x with async views, DRF, Celery, and scalable architecture. Includes Django ORM optimization examples showing 50x query reduction with prefetch_related, DRF ViewSet patterns with >95% test coverage, and production-ready deployment configurations.

**Key Capabilities:**
- Django 5.x async views and middleware
- ORM optimization (select_related, prefetch_related, annotate)
- DRF ViewSets, serializers, and authentication
- Celery task queues and beat scheduling
- Django testing with fixtures and factory patterns

## Commands (1)

### `/python-scaffold`

**Status:** active | **Version:** 2.1.0

Scaffold production-ready Python projects with modern tooling, 3 execution modes (quick: 1-2h, standard: 3-6h, enterprise: 1-2d), and comprehensive external documentation

## Skills (5)

### async-python-patterns

Master async/await patterns, asyncio event loop internals, concurrent programming, async context managers, async generators, aiohttp/httpx clients, async database operations, WebSocket servers, background task coordination, and async testing with pytest-asyncio for building high-performance I/O-bound Python applications.

**When to use:** Writing async/await syntax, building async web APIs (FastAPI, aiohttp, Sanic), implementing WebSocket servers, creating async database queries (SQLAlchemy, asyncpg, motor), coordinating background tasks, and 14+ more scenarios.

### python-testing-patterns

Comprehensive testing strategies with pytest fixtures (conftest.py, autouse, parametrization), unittest.mock for isolation, monkeypatch and pytest-mock utilities, test coverage analysis, TDD workflows, async testing with pytest-asyncio, integration testing patterns, performance testing, and CI/CD test automation for ensuring code quality and reliability.

**When to use:** Creating pytest fixtures, using unittest.mock for isolation, implementing TDD red-green-refactor workflows, testing async code, setting up CI/CD test automation, and 17+ more scenarios.

### python-packaging

Create distributable Python packages with pyproject.toml configuration, setuptools/hatchling build backends, semantic versioning, README and LICENSE files, entry points for CLI tools, wheel/sdist building, PyPI/private repository publishing, dependency specification with version constraints, namespace packages, and automated release workflows for sharing Python code effectively.

**When to use:** Writing pyproject.toml configuration files, building wheels and source distributions, publishing to PyPI or private repositories, creating CLI tools with entry points, managing dependency version constraints, and 15+ more scenarios.

### python-performance-optimization

Profile and optimize Python code using cProfile, line_profiler, memory_profiler, py-spy, and Scalene for hotspot identification; apply NumPy/Pandas vectorization, Numba JIT compilation, dataclass optimizations, generator expressions, itertools patterns, caching strategies, algorithmic improvements, and memory optimization techniques to achieve 10-100x performance gains in production applications.

**When to use:** Running profiling tools (cProfile, line_profiler, memory_profiler, py-spy), optimizing NumPy/Pandas operations with vectorization, applying Numba JIT compilation for numerical code, implementing caching strategies (functools.lru_cache, cachetools), optimizing memory usage with generators and __slots__, and 16+ more scenarios.

### uv-package-manager

Master the uv package manager (Rust-based, 10-100x faster than pip) for Python dependency management with uv init for project scaffolding, uv add/remove for dependency management, uv sync for lockfile-based installations, uv venv for virtual environments, uv run for command execution, uv lock for deterministic builds, and integration with pyproject.toml for modern, fast, reliable Python workflows.

**When to use:** Running uv commands (init, add, remove, sync, venv, run, lock), setting up new Python projects with uv init, managing dependencies with uv.lock for deterministic builds, migrating from pip/poetry/pipenv to uv, integrating with pyproject.toml, and 17+ more scenarios.

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `python-development` plugin
3. Activate an agent (e.g., `@python-pro`)
4. Try a command (e.g., `/python-scaffold`)

## Integration

See the full documentation for integration patterns and compatible plugins.

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/python-development.html)

To build documentation locally:

```bash
cd docs/
make html
```
