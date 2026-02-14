---
name: python-development
version: "2.2.0"
description: Master modern Python systems engineering for scientific computing. Covers type-driven design, Rust extensions (PyO3), structured concurrency (TaskGroups), robust testing (Hypothesis), and uv-based packaging.
---

# Python Systems Engineering

Advanced patterns and toolchains for high-performance, robust Python development.

## Expert Agent

For all advanced Python development, architecture, and optimization tasks, delegate to:

- **`python-pro`**: Expert Python Systems Engineer focusing on rigor, performance, and modern standards.
  - *Location*: `plugins/science-suite/agents/python-pro.md`

## Core Skills

### 1. [Type-Driven Design](./type-driven-design/SKILL.md)
Structural typing with Protocols, Generics for reusability, and strict static analysis with Pyright/Mypy.

### 2. [Rust Extensions](./rust-extensions/SKILL.md)
Writing performance-critical bottlenecks in Rust using PyO3 and Maturin for 100x speedups.

### 3. [Modern Concurrency](./modern-concurrency/SKILL.md)
Structured concurrency using `asyncio.TaskGroup` (Python 3.11+) for reliable task management and error propagation.

### 4. [Robust Testing](./robust-testing/SKILL.md)
Property-based testing with Hypothesis and advanced Pytest patterns to ensure algorithmic correctness.

### 5. [Modern Packaging](./python-packaging/SKILL.md)
Using `uv` for blazing-fast dependency management, workspaces (monorepos), and reproducible environments.

## The Python Pro Mindset

1.  **Strict Typing**: `mypy --strict` or `pyright` strict mode must always pass.
2.  **Zero Global State**: Use dependency injection and avoid the `global` keyword.
3.  **Performance First**: Profile with `py-spy`, vectorize with NumPy/JAX, and offload to Rust when necessary.
4.  **Structured Concurrency**: Avoid `asyncio.gather` in favor of `TaskGroup`.
5.  **Modern Tooling**: Standardize on `uv`, `ruff`, and `hatchling`.

## Checklist

- [ ] Strict type checking enabled and passing.
- [ ] `uv.lock` managed and consistent across the workspace.
- [ ] Performance bottlenecks identified via profiling and addressed.
- [ ] Algorithmic properties verified with property-based tests.
- [ ] Async code uses structured concurrency primitives.
