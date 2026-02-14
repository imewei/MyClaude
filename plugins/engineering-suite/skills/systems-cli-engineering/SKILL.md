---
name: systems-cli-engineering
version: "2.2.0"
description: Design high-performance systems and production-grade CLI tools. Covers memory management, concurrency, and CLI UX design.
---

# Systems & CLI Engineering

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
