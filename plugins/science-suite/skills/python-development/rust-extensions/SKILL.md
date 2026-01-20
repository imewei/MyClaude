---
name: rust-extensions
version: "1.0.0"
description: Build high-performance Python extensions using Rust, PyO3, and Maturin. Use when optimizing performance-critical bottlenecks, implementing native modules, or bridging Rust libraries to Python.
---

# Rust Extensions for Python

Leverage the safety and performance of Rust within your Python applications.

## Expert Agent

For complex FFI, memory management between Rust/Python, or performance architecture, delegate to:

- **`python-pro`**: Expert in systems engineering and Rust/Python integration via PyO3.
  - *Location*: `plugins/science-suite/agents/python-pro.md`

## 1. Tooling (Maturin)

`maturin` is the recommended build system for PyO3 projects.

```bash
# Initialize a new project
maturin init --mixed

# Build and install in development
maturin develop

# Build for release
maturin build --release
```

## 2. PyO3 Basics

Define Python modules and functions in Rust.

```rust
use pyo3::prelude::*;

#[pyfunction]
fn compute_heavy_task(data: Vec<f64>) -> PyResult<f64> {
    // Rust-speed computation
    Ok(data.iter().sum())
}

#[pymodule]
fn my_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_heavy_task, m)?)?;
    Ok(())
}
```

## 3. pyproject.toml Integration

Use `maturin` as the build backend.

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "my-extension"
requires-python = ">=3.12"
dependencies = []
```

## 4. Best Practices

- **Zero-Copy**: Use `numpy` with `rust-numpy` for zero-copy data sharing.
- **Error Handling**: Map Rust `Result` to `PyResult` to raise proper Python exceptions.
- **GIL Management**: Release the GIL with `py.allow_threads(|| ...)` for long-running CPU tasks.
- **Type Safety**: Use `PyAny`, `PyDict`, etc., carefully; prefer specialized types.

## Checklist

- [ ] `maturin` used as the build backend.
- [ ] Functions decorated with `#[pyfunction]`.
- [ ] GIL released for heavy CPU work.
- [ ] Comprehensive tests written in Python to verify extension behavior.
- [ ] Type stubs (`.pyi` files) generated for better IDE support.
