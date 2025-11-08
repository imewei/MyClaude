---
version: 1.0.3
command: /rust-project
description: Scaffold production-ready Rust projects with 3 execution modes from simple binaries to enterprise workspaces with async support
argument-hint: [project-type] [project-name]
execution_modes:
  quick:
    duration: "1-2 hours"
    description: "Simple binary or library crate"
    scope: "cargo new, basic Cargo.toml, simple main.rs/lib.rs, minimal dependencies"
    deliverables: "Runnable binary or publishable library crate"
  standard:
    duration: "4-6h"
    description: "Production crate with proper error handling and async"
    scope: "Full Cargo.toml with features, async Tokio setup, error handling (thiserror/anyhow), testing, benchmarks, examples, documentation"
    deliverables: "Production-ready crate with comprehensive testing and docs"
  enterprise:
    duration: "1-2 days"
    description: "Workspace with multiple crates, benchmarks, WASM support"
    scope: "Cargo workspace, multiple crates (api/core/cli), shared dependencies, cross-compilation, CI/CD workflows, comprehensive documentation"
    deliverables: "Enterprise workspace with multi-crate architecture and full automation"
workflow_type: "sequential"
interactive_mode: true
color: orange
allowed-tools: Bash, Write, Read, Edit
---

# Rust Project Scaffolding

Scaffold production-ready Rust projects with cargo tooling, proper module organization, testing, and idiomatic patterns.

## Context

$ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "What level of Rust project do you need?"
    header: "Project Scope"
    multiSelect: false
    options:
      - label: "Quick (1-2h)"
        description: "Simple binary or library crate with basic setup and minimal dependencies."

      - label: "Standard (4-6h)"
        description: "Production crate with features, async Tokio, proper error handling (thiserror/anyhow), testing, benchmarks, and comprehensive documentation."

      - label: "Enterprise (1-2d)"
        description: "Cargo workspace with multiple crates (api/core/cli), shared dependencies, cross-compilation, CI/CD workflows, and full documentation suite."
</AskUserQuestion>

## Instructions

### 1. Determine Project Type

**Project Types:**
- **Binary**: CLI tools, applications, services
- **Library**: Reusable crates, shared utilities
- **Workspace**: Multi-crate projects, monorepos
- **Web API**: Axum web services, REST APIs
- **WebAssembly**: Browser-based applications

**Mode Scoping:**
- **Quick**: Single crate, minimal structure
- **Standard**: Full crate with testing/benchmarks/examples
- **Enterprise**: Workspace with multiple crates

**Reference:** [Rust Project Structures](../docs/rust-project/rust-project-structures.md)

---

### 2. Initialize with Cargo

**Quick Mode:**
```bash
# Binary
cargo new my-project
cd my-project

# Or library
cargo new --lib my-lib
```

**Standard Mode:** Same as Quick, but enhance Cargo.toml

**Enterprise Mode:** Workspace
```bash
mkdir my-workspace && cd my-workspace
cargo init
# Create crates/api, crates/core, crates/cli
```

---

### 3. Configure Cargo.toml

**Quick Mode: Minimal**
```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4.5", features = ["derive"] }
```

**Standard Mode: Comprehensive**
```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <email@example.com>"]
license = "MIT OR Apache-2.0"

[dependencies]
tokio = { version = "1.36", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"
clap = { version = "4.5", features = ["derive"] }

[dev-dependencies]
criterion = "0.5"

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

**Enterprise Mode: Workspace**
```toml
[workspace]
members = ["crates/api", "crates/core", "crates/cli"]
resolver = "2"

[workspace.dependencies]
tokio = { version = "1.36", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }

[profile.release]
opt-level = 3
lto = "thin"
```

**Full reference:** [Rust Cargo Configuration](../docs/rust-project/rust-cargo-config.md)

---

### 4. Generate Source Code

**Quick Mode: Simple main.rs**
```rust
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    name: String,
}

fn main() {
    let cli = Cli::parse();
    println!("Hello, {}!", cli.name);
}
```

**Standard Mode: Async with Error Handling**
```rust
use anyhow::Result;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Async operations
    process().await?;

    Ok(())
}

async fn process() -> Result<()> {
    // Implementation
    Ok(())
}
```

**Enterprise Mode:**
- API crate: Axum web service
- Core crate: Business logic
- CLI crate: Command-line interface
- Shared types crate

**Templates:** [Rust Project Structures](../docs/rust-project/rust-project-structures.md)

---

### 5. Setup Testing and Benchmarks

**Standard Mode:**
```toml
[dev-dependencies]
criterion = "0.5"
tokio-test = "0.4"

[[bench]]
name = "my_benchmark"
harness = false
```

**benches/my_benchmark.rs:**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark(c: &mut Criterion) {
    c.bench_function("process", |b| {
        b.iter(|| my_lib::process(black_box(input)));
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
```

**Reference:** [Rust Project Structures - Testing](../docs/rust-project/rust-project-structures.md#testing-structure)

---

### 6. Async Patterns (Standard+)

**Tokio Runtime:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Spawn tasks
    let handle = tokio::spawn(async {
        worker().await
    });

    // Concurrent operations
    let (result1, result2) = tokio::join!(
        fetch_data("url1"),
        fetch_data("url2"),
    );

    handle.await?;
    Ok(())
}
```

**Comprehensive async patterns:** [Rust Async Patterns](../docs/rust-project/rust-async-patterns.md)

---

### 7. Documentation

**Quick Mode:**
- README.md with build instructions

**Standard Mode:**
```rust
//! # My Library
//!
//! `my-lib` provides utilities for...
//!
//! ## Quick Start
//!
//! ```
//! use my_lib::Core;
//!
//! let core = Core::new();
//! let result = core.process()?;
//! # Ok::<(), my_lib::Error>(())
//! ```

/// Performs a specific operation.
///
/// # Examples
///
/// ```
/// use my_lib::function;
///
/// let result = function("input")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn function(input: &str) -> Result<(), Error> {
    Ok(())
}
```

**Enterprise Mode:**
- API documentation for all crates
- Architecture documentation
- Examples for all crates

---

## Output Deliverables

### Quick Mode (1-2h):
✅ Working binary or library
✅ Basic Cargo.toml
✅ README.md

### Standard Mode (4-6h):
✅ Production-ready crate
✅ Comprehensive Cargo.toml with features
✅ Async Tokio setup
✅ Error handling (thiserror/anyhow)
✅ Tests and benchmarks
✅ Examples and documentation

### Enterprise Mode (1-2d):
✅ Cargo workspace
✅ Multiple crates with clear boundaries
✅ Shared workspace dependencies
✅ CI/CD workflows
✅ Cross-compilation setup
✅ Comprehensive documentation

---

## External Documentation

Comprehensive guides:

- **[Rust Project Structures](../docs/rust-project/rust-project-structures.md)** (~600 lines)
  - Binary, library, workspace structures
  - Web API patterns with Axum
  - Testing and benchmarking organization

- **[Rust Cargo Configuration](../docs/rust-project/rust-cargo-config.md)** (~567 lines)
  - Cargo.toml patterns and features
  - Workspace configuration
  - Build profiles and optimization
  - Dependency management

- **[Rust Async Patterns](../docs/rust-project/rust-async-patterns.md)** (~718 lines)
  - Tokio runtime and async/await
  - Concurrency patterns (join!, select!)
  - Channels and message passing
  - Production patterns (graceful shutdown, worker pools)

---

## Quality Checklist

**All Modes:**
- [ ] Builds with `cargo build`
- [ ] No compiler warnings
- [ ] README.md present

**Standard+:**
- [ ] Tests pass with `cargo test`
- [ ] Benchmarks run with `cargo bench`
- [ ] Lints clean with `cargo clippy`
- [ ] Formatted with `cargo fmt`
- [ ] Documentation builds with `cargo doc`

**Enterprise:**
- [ ] All workspace members build
- [ ] CI/CD workflow validates builds
- [ ] Cross-compilation tested
- [ ] Documentation comprehensive
