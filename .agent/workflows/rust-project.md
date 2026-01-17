---
description: Scaffold production-ready Rust projects with cargo tooling and idiomatic
  patterns
triggers:
- /rust-project
- workflow for rust project
version: 1.0.7
command: /rust-project
argument-hint: '[project-type] [project-name]'
execution_modes:
  quick: 1-2 hours
  standard: 4-6 hours
  enterprise: 1-2 days
allowed-tools: Bash, Write, Read, Edit
---


# Rust Project Scaffolding

Scaffold production-ready Rust projects with cargo tooling, proper module organization, and idiomatic patterns.

## Context

$ARGUMENTS

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 1-2h | cargo new, basic Cargo.toml, minimal dependencies |
| Standard (default) | 4-6h | Full crate with async Tokio, error handling, tests, benchmarks, docs |
| Enterprise | 1-2d | Cargo workspace with multi-crate architecture and CI/CD |

## Phase 1: Project Type Determination

| Type | Use Case | Structure |
|------|----------|-----------|
| Binary | CLI tools, applications, services | Single crate with main.rs |
| Library | Reusable crates, shared utilities | Single crate with lib.rs |
| Workspace | Multi-crate projects, monorepos | Root Cargo.toml + members |
| Web API | Axum services, REST APIs | Binary with Axum + tower |
| WebAssembly | Browser applications | wasm-bindgen + wasm-pack |

**Reference:** [Rust Project Structures](../../plugins/systems-programming/docs/rust-project/rust-project-structures.md)

## Phase 2: Initialize with Cargo

| Mode | Command | Notes |
|------|---------|-------|
| Quick (binary) | `cargo new project-name` | Single binary crate |
| Quick (library) | `cargo new --lib lib-name` | Single library crate |
| Standard | Same as Quick + enhanced Cargo.toml | Add features, tests, benchmarks |
| Enterprise | `cargo new --workspace workspace-name` | Create crates/api, crates/core, crates/cli |

## Phase 3: Cargo.toml Configuration

### Dependencies by Mode

| Mode | Dependencies |
|------|--------------|
| Quick | clap (CLI) |
| Standard | + tokio, anyhow, thiserror, tracing, serde |
| Enterprise | + workspace.dependencies for sharing |

### Essential Sections

| Section | Purpose | Mode |
|---------|---------|------|
| `[package]` | Name, version, edition, rust-version, license | All |
| `[dependencies]` | Runtime dependencies | All |
| `[dev-dependencies]` | Testing/benchmark deps (criterion, tokio-test) | Standard+ |
| `[profile.release]` | Optimization (lto, codegen-units) | Standard+ |
| `[workspace]` | Multi-crate coordination | Enterprise |

**Full reference:** [Rust Cargo Configuration](../../plugins/systems-programming/docs/rust-project/rust-cargo-config.md)

## Phase 4: Source Code Generation

### Entry Point by Type

| Type | File | Pattern |
|------|------|---------|
| Binary (Quick) | main.rs | clap::Parser + simple main |
| Binary (Standard) | main.rs | async main + tracing + error handling |
| Library | lib.rs | Module exports + documentation |
| Workspace | Per-crate | Clear crate boundaries |

### Enterprise Crate Structure

| Crate | Purpose |
|-------|---------|
| api | Axum web service, HTTP handlers |
| core | Business logic, domain types |
| cli | Command-line interface |
| types | Shared types across crates |

**Templates:** [Rust Project Structures](../../plugins/systems-programming/docs/rust-project/rust-project-structures.md)

## Phase 5: Testing and Benchmarks

### Test Organization

| Test Type | Location | Command |
|-----------|----------|---------|
| Unit tests | Same file as code (`#[cfg(test)]`) | `cargo test` |
| Integration tests | tests/ directory | `cargo test` |
| Benchmarks | benches/ directory | `cargo bench` |
| Doc tests | Inline code examples | `cargo test --doc` |

### Benchmark Setup (Standard+)

```toml
[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "benchmark_name"
harness = false
```

**Reference:** [Rust Project Structures - Testing](../../plugins/systems-programming/docs/rust-project/rust-project-structures.md#testing-structure)

## Phase 6: Async Patterns (Standard+)

| Pattern | Use Case |
|---------|----------|
| `#[tokio::main]` | Async entry point |
| `tokio::spawn` | Background tasks |
| `tokio::join!` | Concurrent operations |
| `tokio::select!` | First-to-complete |
| Channels | Message passing |

**Comprehensive patterns:** [Rust Async Patterns](../../plugins/systems-programming/docs/rust-project/rust-async-patterns.md)

## Phase 7: Documentation

| Mode | Documentation |
|------|--------------|
| Quick | README.md with build instructions |
| Standard | + lib.rs doc comments, function docs with examples |
| Enterprise | + Architecture docs, API docs for all crates |

## Output Deliverables

| Mode | Deliverables |
|------|--------------|
| Quick | Working binary/library, basic Cargo.toml, README |
| Standard | + Async setup, error handling, tests, benchmarks, examples |
| Enterprise | + Workspace, multi-crate architecture, CI/CD, cross-compilation |

## External Documentation

| Document | Content | Lines |
|----------|---------|-------|
| [Rust Project Structures](../../plugins/systems-programming/docs/rust-project/rust-project-structures.md) | Binary/library/workspace patterns, Axum | ~600 |
| [Rust Cargo Configuration](../../plugins/systems-programming/docs/rust-project/rust-cargo-config.md) | Cargo.toml, profiles, dependencies | ~567 |
| [Rust Async Patterns](../../plugins/systems-programming/docs/rust-project/rust-async-patterns.md) | Tokio, concurrency, channels | ~718 |

## Quality Checklist

### All Modes
- [ ] `cargo build` succeeds without warnings
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
- [ ] README.md present

### Standard+
- [ ] `cargo test` passes
- [ ] `cargo bench` runs
- [ ] `cargo doc` builds

### Enterprise
- [ ] All workspace members build
- [ ] CI/CD workflow validates
- [ ] Cross-compilation tested
