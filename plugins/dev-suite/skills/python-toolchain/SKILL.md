---
name: python-toolchain
description: Meta-orchestrator for Python ecosystem tooling. Routes to packaging, performance optimization, uv, error handling, and migration skills. Use when packaging Python projects, profiling performance, configuring uv, implementing error handling, or migrating legacy Python code.
---

# Python Toolchain

Orchestrator for Python ecosystem tooling and project lifecycle. Routes to the appropriate specialized skill based on the packaging need, performance concern, or migration target.

## Expert Agent

- **`systems-engineer`**: Specialist for Python build systems, native extensions, and performance-critical toolchain decisions.
  - *Location*: `plugins/dev-suite/agents/systems-engineer.md`
  - *Capabilities*: PEP compliance, package publishing, C extension integration, profiling, and codebase modernization.

## Core Skills

### [Python Packaging](../python-packaging/SKILL.md)
`pyproject.toml`, build backends (hatch, flit, setuptools), and PyPI publishing workflows.

### [Python Performance Optimization](../python-performance-optimization/SKILL.md)
Profiling, Cython, native extensions, and algorithmic optimization for CPython.

### [uv Package Manager](../uv-package-manager/SKILL.md)
`uv` workspace setup, lockfile management, virtual environments, and script running.

### [Error Handling Patterns](../error-handling-patterns/SKILL.md)
Python exception hierarchies, structured error propagation, and retry/circuit-breaker patterns. Scoped to Python — for language-agnostic error handling (Node.js, Go, etc.), see `backend-patterns`.

### [Modernization & Migration](../modernization-migration/SKILL.md)
Python 2→3 migration, dependency upgrades, and legacy Python codebase modernization. For general architecture modernization (Strangler Fig, microservice extraction), see `architecture-and-infra`.

## Routing Decision Tree

```
What is the Python toolchain concern?
|
+-- Package metadata / build backend / PyPI publish?
|   --> python-packaging
|
+-- Profiling / Cython / speed bottleneck?
|   --> python-performance-optimization
|
+-- uv workspace / lockfile / venv management?
|   --> uv-package-manager
|
+-- Python exception design / retry / error propagation?
|   --> error-handling-patterns
|
+-- Legacy upgrade / Python version migration?
    --> modernization-migration
```

## Routing Table

| Trigger                                    | Sub-skill                         |
|--------------------------------------------|-----------------------------------|
| pyproject.toml, hatch, flit, PyPI          | python-packaging                  |
| cProfile, Cython, ctypes, perf bottleneck  | python-performance-optimization   |
| uv add, uv sync, uv.lock, workspace        | uv-package-manager                |
| Python try/except, retry, circuit breaker   | error-handling-patterns          |
| 2to3, deprecated APIs, version upgrade     | modernization-migration           |

## Checklist

- [ ] Confirm `uv` is the package manager before applying packaging patterns
- [ ] Verify `pyproject.toml` is the single source of project metadata
- [ ] Profile before optimizing — confirm the bottleneck is in Python, not I/O
- [ ] Ensure error hierarchies distinguish recoverable from fatal exceptions
- [ ] Check migration scripts are tested on a branch before touching main
- [ ] Validate lockfile (`uv.lock`) is committed and reproducible across environments
