# Rust Project Structures Reference

Comprehensive guide to Rust project organization patterns for binaries, libraries, workspaces, and web applications with production-ready scaffolding.

---

## Binary Project Structure

### Standard CLI Application

```
cli-app/
├── Cargo.toml
├── Cargo.lock
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
├── .gitignore
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── cli.rs              # Clap CLI definitions
│   ├── config.rs           # Configuration management
│   ├── error.rs            # Error types
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── init.rs
│   │   ├── build.rs
│   │   └── deploy.rs
│   └── utils/
│       ├── mod.rs
│       ├── fs.rs
│       └── net.rs
├── tests/
│   ├── integration_test.rs
│   └── common/
│       └── mod.rs
├── benches/
│   └── benchmark.rs
├── examples/
│   ├── basic.rs
│   └── advanced.rs
└── .cargo/
    └── config.toml
```

**Cargo.toml (Binary)**
```toml
[package]
name = "cli-app"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <email@example.com>"]
description = "A production-ready CLI application"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/cli-app"
keywords = ["cli", "tool"]
categories = ["command-line-utilities"]

[[bin]]
name = "cli-app"
path = "src/main.rs"

[dependencies]
clap = { version = "4.5", features = ["derive", "cargo"] }
tokio = { version = "1.36", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
tokio-test = "0.4"
tempfile = "3.10"

[[bench]]
name = "benchmark"
harness = false

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = true

[profile.dev]
opt-level = 0

[profile.bench]
inherits = "release"
```

**src/main.rs**
```rust
use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod cli;
mod commands;
mod config;
mod error;
mod utils;

use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Execute command
    match cli.command {
        Commands::Init(args) => commands::init::execute(args).await?,
        Commands::Build(args) => commands::build::execute(args).await?,
        Commands::Deploy(args) => commands::deploy::execute(args).await?,
    }

    Ok(())
}
```

**src/cli.rs**
```rust
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cli-app")]
#[command(author, version, about = "Production CLI application", long_about = None)]
pub struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize a new project
    Init(InitArgs),
    /// Build the project
    Build(BuildArgs),
    /// Deploy to production
    Deploy(DeployArgs),
}

#[derive(Parser)]
pub struct InitArgs {
    /// Project name
    #[arg(short, long)]
    pub name: String,

    /// Output directory
    #[arg(short, long, value_name = "DIR")]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct BuildArgs {
    /// Enable release mode
    #[arg(short, long)]
    pub release: bool,

    /// Number of parallel jobs
    #[arg(short, long, default_value_t = num_cpus::get())]
    pub jobs: usize,
}

#[derive(Parser)]
pub struct DeployArgs {
    /// Target environment
    #[arg(short, long, value_enum)]
    pub env: Environment,
}

#[derive(clap::ValueEnum, Clone)]
pub enum Environment {
    Dev,
    Staging,
    Production,
}
```

**src/error.rs**
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, AppError>;
```

---

## Library Project Structure

### Standard Library Crate

```
my-lib/
├── Cargo.toml
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
├── src/
│   ├── lib.rs
│   ├── core.rs
│   ├── types.rs
│   ├── error.rs
│   └── utils/
│       ├── mod.rs
│       └── helpers.rs
├── tests/
│   ├── integration_test.rs
│   └── common/
│       └── mod.rs
├── benches/
│   └── benchmark.rs
└── examples/
    ├── basic.rs
    └── advanced.rs
```

**Cargo.toml (Library)**
```toml
[package]
name = "my-lib"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <email@example.com>"]
description = "A production-ready Rust library"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/my-lib"
documentation = "https://docs.rs/my-lib"
keywords = ["library", "utility"]
categories = ["development-tools"]

[lib]
name = "my_lib"
path = "src/lib.rs"
crate-type = ["lib"]

[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"], optional = true }

[dev-dependencies]
criterion = "0.5"
tokio-test = "0.4"

[features]
default = []
serde = ["dep:serde"]
```

**src/lib.rs**
```rust
//! # My Library
//!
//! `my-lib` provides utilities for...
//!
//! ## Quick Start
//!
//! ```rust
//! use my_lib::Core;
//!
//! let core = Core::new();
//! let result = core.process()?;
//! # Ok::<(), my_lib::Error>(())
//! ```

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod core;
pub mod error;
pub mod types;
mod utils;

// Re-export main types
pub use core::Core;
pub use error::{Error, Result};
pub use types::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
```

---

## Workspace Structure

### Multi-Crate Workspace

```
workspace/
├── Cargo.toml              # Workspace root
├── Cargo.lock
├── README.md
├── crates/
│   ├── api/                # Web API crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       └── routes/
│   ├── core/               # Core library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   ├── cli/                # CLI tool
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs
│   └── types/              # Shared types
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs
├── tests/                  # Integration tests
│   └── workspace_test.rs
└── benches/
    └── workspace_bench.rs
```

**Cargo.toml (Workspace Root)**
```toml
[workspace]
members = [
    "crates/api",
    "crates/core",
    "crates/cli",
    "crates/types",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["Your Name <email@example.com>"]
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/workspace"

[workspace.dependencies]
# Shared dependencies with consistent versions
tokio = { version = "1.36", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
anyhow = "1.0"

# Internal crates
api = { path = "crates/api" }
core = { path = "crates/core" }
types = { path = "crates/types" }

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

**crates/core/Cargo.toml**
```toml
[package]
name = "core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
authors.workspace = true
license.workspace = true

[dependencies]
types = { workspace = true }
tokio = { workspace = true }
thiserror = { workspace = true }
```

**crates/api/Cargo.toml**
```toml
[package]
name = "api"
version.workspace = true
edition.workspace = true

[[bin]]
name = "api-server"
path = "src/main.rs"

[dependencies]
core = { workspace = true }
types = { workspace = true }
tokio = { workspace = true }
axum = "0.7"
tower = "0.4"
```

---

## Web API Project Structure

### Axum Web Service

```
web-api/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs
│   ├── app.rs              # App state and initialization
│   ├── config.rs           # Configuration
│   ├── error.rs            # Error handling
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── health.rs
│   │   ├── users.rs
│   │   └── api_v1.rs
│   ├── handlers/
│   │   ├── mod.rs
│   │   └── user_handler.rs
│   ├── models/
│   │   ├── mod.rs
│   │   └── user.rs
│   ├── services/
│   │   ├── mod.rs
│   │   └── user_service.rs
│   ├── middleware/
│   │   ├── mod.rs
│   │   ├── auth.rs
│   │   └── logging.rs
│   └── db/
│       ├── mod.rs
│       └── pool.rs
├── migrations/
│   └── 001_init.sql
└── tests/
    └── api_tests.rs
```

**Cargo.toml (Web API)**
```toml
[package]
name = "web-api"
version = "0.1.0"
edition = "2021"

[dependencies]
# Web framework
axum = { version = "0.7", features = ["macros"] }
tokio = { version = "1.36", features = ["full"] }
tower = { version = "0.4", features = ["limit"] }
tower-http = { version = "0.5", features = ["trace", "cors", "compression-gzip"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Database
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres", "migrate"] }

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# Auth
jsonwebtoken = "9.2"
bcrypt = "0.15"

# Validation
validator = { version = "0.18", features = ["derive"] }
```

**src/main.rs (Axum)**
```rust
use axum::Router;
use std::net::SocketAddr;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    compression::CompressionLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod app;
mod config;
mod db;
mod error;
mod handlers;
mod middleware;
mod models;
mod routes;
mod services;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "web_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = config::Config::from_env()?;

    // Initialize database pool
    let db_pool = db::create_pool(&config.database_url).await?;

    // Run migrations
    sqlx::migrate!("./migrations").run(&db_pool).await?;

    // Build application state
    let app_state = app::AppState::new(db_pool);

    // Build router
    let app = Router::new()
        .nest("/api/v1", routes::api_v1::router())
        .route("/health", axum::routing::get(routes::health::health_check))
        .layer(CorsLayer::permissive())
        .layer(CompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(app_state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received");
}
```

**src/app.rs**
```rust
use sqlx::PgPool;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    // Add other shared state here
}

impl AppState {
    pub fn new(db: PgPool) -> Self {
        Self { db }
    }
}
```

**src/routes/api_v1.rs**
```rust
use axum::{
    routing::{get, post},
    Router,
};
use crate::{app::AppState, handlers};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/users", get(handlers::user_handler::list_users))
        .route("/users", post(handlers::user_handler::create_user))
        .route("/users/:id", get(handlers::user_handler::get_user))
}
```

**src/handlers/user_handler.rs**
```rust
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use crate::{
    app::AppState,
    error::AppError,
    models::user::{User, CreateUser},
    services::user_service,
};

pub async fn list_users(
    State(state): State<AppState>,
) -> Result<Json<Vec<User>>, AppError> {
    let users = user_service::list_users(&state.db).await?;
    Ok(Json(users))
}

pub async fn get_user(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<User>, AppError> {
    let user = user_service::get_user(&state.db, id).await?;
    Ok(Json(user))
}

pub async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUser>,
) -> Result<(StatusCode, Json<User>), AppError> {
    let user = user_service::create_user(&state.db, payload).await?;
    Ok((StatusCode::CREATED, Json(user)))
}
```

---

## Testing Structure

### Integration Tests

**tests/integration_test.rs**
```rust
use my_lib::Core;

#[tokio::test]
async fn test_core_functionality() {
    let core = Core::new();
    let result = core.process().await;
    assert!(result.is_ok());
}

// Common test utilities
mod common;

#[tokio::test]
async fn test_with_common_setup() {
    let context = common::setup().await;
    // Use context for testing
    common::teardown(context).await;
}
```

**tests/common/mod.rs**
```rust
pub struct TestContext {
    // Shared test state
}

pub async fn setup() -> TestContext {
    // Initialize test environment
    TestContext {}
}

pub async fn teardown(_ctx: TestContext) {
    // Cleanup
}
```

### Benchmarks

**benches/benchmark.rs**
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use my_lib::Core;

fn benchmark_process(c: &mut Criterion) {
    let core = Core::new();

    c.bench_function("core_process", |b| {
        b.iter(|| {
            let _ = black_box(core.process());
        });
    });
}

fn benchmark_with_setup(c: &mut Criterion) {
    c.bench_function("with_setup", |b| {
        b.iter_batched(
            || Core::new(),  // Setup (not measured)
            |core| core.process(),  // Measured operation
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, benchmark_process, benchmark_with_setup);
criterion_main!(benches);
```

---

## Configuration Management

### Environment-Based Config

**src/config.rs**
```rust
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub database_url: String,
    pub port: u16,
    pub log_level: String,
    #[serde(default = "default_workers")]
    pub workers: usize,
}

fn default_workers() -> usize {
    num_cpus::get()
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        dotenv::dotenv().ok();

        let config = envy::from_env::<Config>()?;
        Ok(config)
    }
}
```

**.env.example**
```
DATABASE_URL=postgres://user:pass@localhost/dbname
PORT=3000
LOG_LEVEL=info
WORKERS=4
```

---

## Documentation Best Practices

### Module Documentation

```rust
//! # Module Name
//!
//! Brief description of the module's purpose.
//!
//! ## Examples
//!
//! ```
//! use my_lib::module;
//!
//! let result = module::function();
//! ```

/// Performs a specific operation.
///
/// # Arguments
///
/// * `input` - The input parameter
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error.
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
    // Implementation
    Ok(())
}
```

---

## Summary: Best Practices

1. **Project Organization**: Clear separation of concerns (src/, tests/, benches/)
2. **Cargo.toml**: Comprehensive metadata, feature flags, workspace dependencies
3. **Error Handling**: Use thiserror for libraries, anyhow for applications
4. **Testing**: Unit tests in modules, integration tests in tests/, benchmarks in benches/
5. **Documentation**: Doc comments for public API, examples in doc tests
6. **Async**: Tokio for async runtime, careful with blocking operations
7. **Configuration**: Environment-based config with sensible defaults
8. **Logging**: tracing for structured logging
9. **CLI**: clap with derive macros for ergonomic CLI parsing
10. **Web**: Axum for modern async web services
