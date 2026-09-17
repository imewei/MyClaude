---
name: systems-cli-engineering
description: Design high-performance systems and production-grade CLI tools. Covers memory management, concurrency, and CLI UX design. Use when building CLI applications, optimizing system-level code, or implementing concurrent data pipelines in C, C++, Rust, or Go.
---

# Systems & CLI Engineering

## Expert Agent

No dev-suite agent specializes in systems/CLI programming directly. For language-specific
patterns and build fixes, defer to ecc's toolchain skills (e.g. `ecc:rust-patterns`,
`ecc:golang-patterns`, `ecc:cpp-coding-standards`, `ecc:build-fix`).

Expert guide for building low-level systems and developer-facing command-line tools.

## 1. Systems Programming Patterns

### Memory Management
- **Allocators**: Use Arena or Pool allocators for high-frequency, temporary allocations.
- **Ownership**: Master manual memory (C) vs. smart pointers (C++) vs. ownership (Rust).

### Concurrency
- **Primitives**: Use Mutexes, RWLocks, and Atomics appropriately.
- **Lock-Free**: Design non-blocking data structures for high-throughput requirements.

## 2. CLI Tool Design

- **UX Design**: Follow the POSIX standard. Provide clear help text, progress indicators, and structured output (JSON/YAML).
- **Performance**: Optimize startup time and minimize binary size.
- **Distribution**: Package for multiple platforms using tools like `GoReleaser` or `Cargo Dist`.

## 3. Engineering Checklist

- [ ] **Resource Efficiency**: Are memory leaks eliminated? Is CPU usage optimized?
- [ ] **Concurrency**: Are data races prevented? Is locking granularity appropriate?
- [ ] **CLI UX**: Does the tool provide a consistent and intuitive interface?
- [ ] **Robustness**: Are signals (SIGINT, SIGTERM) handled gracefully?
