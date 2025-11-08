# Rust Cargo Configuration Reference

Comprehensive guide to Cargo.toml configuration, dependency management, features, profiles, and build optimization for Rust projects.

---

## Cargo.toml Fundamentals

### Package Metadata

```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"               # Rust edition (2015, 2018, 2021)
rust-version = "1.75"          # Minimum supported Rust version (MSRV)
authors = ["Your Name <email@example.com>"]
description = "A brief description"
documentation = "https://docs.rs/my-project"
readme = "README.md"
homepage = "https://example.com"
repository = "https://github.com/user/my-project"
license = "MIT OR Apache-2.0" # Dual licensing (Rust standard)
license-file = "LICENSE"      # Or custom license file
keywords = ["cli", "tool", "utility"]  # Max 5 keywords
categories = ["command-line-utilities"]  # From crates.io categories
publish = true                # Allow publishing to crates.io
```

---

## Dependencies

### Basic Dependencies

```toml
[dependencies]
# Simple version
serde = "1.0"

# Specific version
tokio = "=1.36.0"

# Version requirements
axum = ">=0.7, <0.8"

# With features
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.36", features = ["full"] }

# Optional dependencies (for features)
serde_json = { version = "1.0", optional = true }

# Git dependencies
my-lib = { git = "https://github.com/user/my-lib" }
my-lib = { git = "https://github.com/user/my-lib", branch = "main" }
my-lib = { git = "https://github.com/user/my-lib", tag = "v0.1.0" }
my-lib = { git = "https://github.com/user/my-lib", rev = "abc123" }

# Path dependencies
utils = { path = "../utils" }

# Platform-specific
[target.'cfg(unix)'.dependencies]
libc = "0.2"

[target.'cfg(windows)'.dependencies]
winapi = { version = "0.3", features = ["winuser"] }
```

### Development Dependencies

```toml
[dev-dependencies]
criterion = "0.5"        # Benchmarking
tokio-test = "0.4"       # Async test utilities
tempfile = "3.10"        # Temporary files for tests
proptest = "1.4"         # Property-based testing
```

### Build Dependencies

```toml
[build-dependencies]
cc = "1.0"               # Compile C/C++ code
bindgen = "0.69"         # Generate Rust FFI bindings
```

---

## Features

### Defining Features

```toml
[features]
# Default features enabled by default
default = ["serde"]

# Feature that enables optional dependency
serde = ["dep:serde", "dep:serde_json"]

# Feature that enables other features
full = ["serde", "async", "cli"]

# Feature with conditional compilation
async = ["tokio"]
cli = ["clap"]

[dependencies]
serde = { version = "1.0", optional = true }
serde_json = { version = "1.0", optional = true }
tokio = { version = "1.36", features = ["full"], optional = true }
clap = { version = "4.5", optional = true }
```

**Usage in code:**
```rust
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Data {
    value: i32,
}
```

**Build with features:**
```bash
cargo build --features serde
cargo build --features "serde,async"
cargo build --all-features
cargo build --no-default-features
cargo build --no-default-features --features serde
```

---

## Build Profiles

### Standard Profiles

```toml
[profile.dev]
opt-level = 0            # No optimization
debug = true             # Include debug info
split-debuginfo = "..."  # Debug info split strategy
debug-assertions = true  # Enable debug assertions
overflow-checks = true   # Integer overflow checks
lto = false              # Link-time optimization
panic = "unwind"         # Panic strategy
incremental = true       # Incremental compilation
codegen-units = 256      # Parallel code generation

[profile.release]
opt-level = 3            # Maximum optimization
debug = false            # No debug info
strip = "symbols"        # Strip symbols
debug-assertions = false
overflow-checks = false
lto = "thin"             # Thin LTO (faster than "fat", smaller than false)
panic = "abort"          # Abort on panic (smaller binary)
codegen-units = 1        # Single codegen unit (better optimization)

[profile.test]
# Inherits from dev

[profile.bench]
# Inherits from release
```

### Custom Profiles

```toml
[profile.release-with-debug]
inherits = "release"
debug = true             # Keep debug info in release

[profile.release-small]
inherits = "release"
opt-level = "z"          # Optimize for size
lto = true
codegen-units = 1
strip = true
```

**Use custom profile:**
```bash
cargo build --profile release-with-debug
```

### Optimization Levels

```toml
opt-level = 0  # No optimizations
opt-level = 1  # Basic optimizations
opt-level = 2  # Some optimizations
opt-level = 3  # All optimizations
opt-level = "s"  # Optimize for size
opt-level = "z"  # Optimize for size even more
```

---

## Workspace Configuration

### Workspace Root

```toml
[workspace]
members = [
    "crates/api",
    "crates/core",
    "crates/cli",
]
exclude = ["old-crate"]
resolver = "2"           # Dependency resolver version

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <email@example.com>"]
license = "MIT OR Apache-2.0"

[workspace.dependencies]
# Shared dependencies with consistent versions
tokio = { version = "1.36", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"

# Internal crates
core = { path = "crates/core" }
api = { path = "crates/api" }

[profile.release]
opt-level = 3
lto = "thin"
```

### Workspace Member

```toml
[package]
name = "api"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[dependencies]
# Use workspace dependency
tokio = { workspace = true }
serde = { workspace = true }
core = { workspace = true }

# Member-specific dependency
axum = "0.7"
```

---

## Binary and Library Targets

### Multiple Binaries

```toml
[[bin]]
name = "main-app"
path = "src/main.rs"

[[bin]]
name = "worker"
path = "src/bin/worker.rs"

[[bin]]
name = "admin"
path = "src/bin/admin.rs"
```

**File structure:**
```
src/
├── main.rs         # main-app binary
├── lib.rs
└── bin/
    ├── worker.rs   # worker binary
    └── admin.rs    # admin binary
```

### Library Configuration

```toml
[lib]
name = "my_lib"
path = "src/lib.rs"
crate-type = ["lib"]       # Rust library
# crate-type = ["dylib"]   # Dynamic library
# crate-type = ["staticlib"]  # Static library (C-compatible)
# crate-type = ["cdylib"]  # C-compatible dynamic library
```

### Benchmarks

```toml
[[bench]]
name = "my_benchmark"
harness = false    # Use custom benchmark framework (e.g., criterion)
```

### Examples

```toml
[[example]]
name = "basic"
path = "examples/basic.rs"
```

---

## Build Scripts

### build.rs

```toml
[build-dependencies]
cc = "1.0"
```

**build.rs:**
```rust
fn main() {
    // Compile C code
    cc::Build::new()
        .file("src/native/helper.c")
        .compile("helper");

    // Re-run if files change
    println!("cargo:rerun-if-changed=src/native/helper.c");

    // Set environment variable
    println!("cargo:rustc-env=BUILD_TIME={}", chrono::Utc::now());

    // Link library
    println!("cargo:rustc-link-lib=ssl");

    // Add link search path
    println!("cargo:rustc-link-search=/usr/local/lib");
}
```

---

## Dependency Versioning

### SemVer Requirements

```toml
[dependencies]
# Caret requirements (default)
serde = "^1.0"       # >=1.0.0, <2.0.0
serde = "1.0"        # Same as ^1.0

# Tilde requirements
serde = "~1.0"       # >=1.0.0, <1.1.0
serde = "~1.0.5"     # >=1.0.5, <1.1.0

# Wildcard
serde = "1.*"        # >=1.0.0, <2.0.0

# Comparison
serde = ">= 1.0, < 2.0"
serde = ">= 1.0.5"

# Exact version
serde = "=1.0.5"

# Multiple requirements
serde = ">= 1.0, < 1.3"
```

---

## Cargo Commands

### Build and Run

```bash
# Build
cargo build                    # Debug build
cargo build --release          # Release build
cargo build --target x86_64-unknown-linux-gnu

# Run
cargo run                      # Run default binary
cargo run --bin worker         # Run specific binary
cargo run --example basic      # Run example
cargo run --release            # Run release build

# Check (faster than build, no codegen)
cargo check                    # Check for errors
cargo check --all-features
```

### Testing

```bash
# Run tests
cargo test                     # All tests
cargo test test_name           # Specific test
cargo test --test integration_test  # Specific integration test
cargo test --lib               # Library tests only
cargo test --bins              # Binary tests only
cargo test --doc               # Doc tests only
cargo test -- --nocapture      # Show println! output
cargo test -- --test-threads=1 # Single-threaded tests

# Benchmarks
cargo bench                    # Run all benchmarks
cargo bench bench_name         # Specific benchmark
```

### Dependencies

```bash
# Update dependencies
cargo update                   # Update to latest compatible versions
cargo update -p serde          # Update specific package

# Show dependency tree
cargo tree                     # Full tree
cargo tree -i serde            # Reverse tree (what depends on serde)
cargo tree --duplicates        # Show duplicates

# Audit dependencies for security vulnerabilities
cargo audit
```

### Documentation

```bash
# Generate and open documentation
cargo doc                      # Generate docs
cargo doc --open               # Generate and open in browser
cargo doc --no-deps            # Only for current crate
cargo doc --document-private-items  # Include private items
```

### Publishing

```bash
# Prepare for publishing
cargo publish --dry-run        # Test publish
cargo package                  # Create .crate file
cargo publish                  # Publish to crates.io

# Yanking versions
cargo yank --vers 0.1.0        # Yank version
cargo yank --vers 0.1.0 --undo # Undo yank
```

---

## .cargo/config.toml

### Project-Level Configuration

```toml
[build]
target = "x86_64-unknown-linux-gnu"
jobs = 4
rustflags = ["-C", "link-arg=-fuse-ld=lld"]  # Use lld linker

[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[alias]
b = "build"
r = "run"
t = "test"
c = "check"
br = "build --release"

[net]
git-fetch-with-cli = true
```

---

## Performance Optimization

### Release Profile Tuning

```toml
[profile.release]
opt-level = 3                  # Maximum optimization
lto = "fat"                    # Full LTO (slower build, smaller/faster binary)
codegen-units = 1              # Single codegen unit
strip = true                   # Strip symbols
panic = "abort"                # Abort on panic
```

### Size Optimization

```toml
[profile.release]
opt-level = "z"                # Optimize for size
lto = true
codegen-units = 1
strip = true
panic = "abort"

[dependencies]
# Use smaller allocator
[target.'cfg(not(target_env = "msvc"))'.dependencies]
jemallocator = "0.5"
```

**In main.rs:**
```rust
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
```

---

## Cross-Compilation

### Install Targets

```bash
rustup target add x86_64-unknown-linux-musl
rustup target add aarch64-unknown-linux-gnu
rustup target add x86_64-pc-windows-gnu
```

### Build for Target

```bash
cargo build --target x86_64-unknown-linux-musl
cargo build --release --target aarch64-unknown-linux-gnu
```

### .cargo/config.toml for Cross-Compilation

```toml
[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"

[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
```

---

## Summary: Cargo Best Practices

1. **Versioning**: Use semantic versioning, specify MSRV
2. **Dependencies**: Minimal dependencies, use workspace dependencies for consistency
3. **Features**: Use features for optional functionality
4. **Profiles**: Optimize release profile, use custom profiles for different needs
5. **Workspace**: Use workspaces for multi-crate projects
6. **Documentation**: Comprehensive Cargo.toml metadata
7. **Testing**: Separate dev-dependencies, use `cargo test`
8. **Publishing**: Test with `--dry-run`, maintain changelog
9. **Cross-compilation**: Use targets, configure linkers
10. **Performance**: LTO, single codegen-unit, optimize for size if needed
