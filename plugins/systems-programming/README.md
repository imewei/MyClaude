# Systems Programming

Production-grade systems programming with C, C++, Rust, and Go. Features systematic development processes, quality assurance checkpoints, comprehensive examples, and battle-tested patterns for memory safety, concurrency, and performance.

**Version:** 1.0.5 | **Category:** development | **License:** MIT

[Full Documentation →](https://myclaude.readthedocs.io/en/latest/plugins/systems-programming.html) | [CHANGELOG](CHANGELOG.md)

---


## What's New in v1.0.5

This release implements **Opus 4.5 optimization** with enhanced token efficiency and standardized documentation.

### Key Improvements

- **Format Standardization**: All components now include consistent YAML frontmatter with version, maturity, specialization, and description fields
- **Token Efficiency**: 40-50% line reduction through tables over prose, minimal code examples, and structured sections
- **Enhanced Discoverability**: Clear "Use when..." trigger phrases for better Claude Code activation
- **Actionable Checklists**: Task-oriented guidance for common workflows
- **Cross-Reference Tables**: Quick-reference format for delegation and integration patterns


## What's New in v1.0.1

All 4 agents now include:
- **8-step systematic development workflow** with self-verification checkpoints
- **8 quality assurance principles** as Constitutional AI checkpoints
- **16 strategic ambiguity questions** across 4 domains
- **3 comprehensive examples** (Good, Bad, Annotated) per agent
- **3 common patterns** with validation criteria
- **~400-800 lines** of structured guidance per agent

Expected improvements:
- +50-75% faster task completion
- +80% reduction in bugs
- +60% better error handling
- +70% improved code quality

[View Full Changelog →](./CHANGELOG.md)

## Agents (4)

### rust-pro

**Status:** active

Master Rust 1.75+ with systematic development process, ownership patterns, async Tokio, type safety, and zero-cost abstractions.

**Key Capabilities:**
- 8-step systematic development workflow
- Ownership and borrowing patterns
- Async/await with Tokio runtime
- Error handling with Result and thiserror
- Thread-safe rate limiter example
- Production web service with graceful shutdown

**Example Usage:**
```rust
// Thread-safe rate limiter with async support
let limiter = RateLimiter::new(100.0, 10);
limiter.acquire().await?;
```

---

### c-pro

**Status:** active

Master C programming with systematic development process, memory safety validation, POSIX APIs, and error handling patterns.

**Key Capabilities:**
- 8-step systematic development workflow
- Memory management with pools and arenas
- POSIX API expertise
- Error handling with goto cleanup pattern
- Thread-safe memory pool example
- Valgrind and AddressSanitizer validation

**Example Usage:**
```c
// Thread-safe memory pool
Pool* pool = pool_create(sizeof(Data), 1000);
void* obj = pool_alloc(pool);
pool_free(pool, obj);
pool_destroy(pool);
```

---

### cpp-pro

**Status:** active

Master modern C++11/14/17/20/23 with systematic development process, RAII patterns, template metaprogramming, and move semantics.

**Key Capabilities:**
- 8-step systematic development workflow
- RAII and Rule of Zero/Three/Five
- Modern C++20/23 features with concepts
- Template metaprogramming and SFINAE
- Thread-safe LRU cache example
- Compile-time string parser with consteval

**Example Usage:**
```cpp
// Thread-safe LRU cache with std::shared_mutex
LRUCache<std::string, int> cache(100);
auto value = cache.get("key");
cache.put("key", 42);
```

---

### golang-pro

**Status:** active

Master Go 1.21+ with systematic development process, goroutines, channels, context patterns, and observability.

**Key Capabilities:**
- 8-step systematic development workflow
- Goroutines and channel patterns
- Context for cancellation and timeouts
- Structured logging with slog
- Worker pool with graceful shutdown
- HTTP server with Prometheus and tracing

**Example Usage:**
```go
// Worker pool with graceful shutdown
pool := NewPool(10)
pool.Submit(func() error { return doWork() })
pool.Shutdown()  // Graceful shutdown with WaitGroup
```

## Commands (3)

### `/rust-project`

**Status:** active

Scaffold production-ready Rust projects with proper structure, cargo tooling, and testing

### `/c-project`

**Status:** active

Scaffold production-ready C projects with Makefile/CMake, testing, and memory safety tools

### `/profile-performance`

**Status:** active

Comprehensive performance profiling workflow using perf, valgrind, and hardware counters

## Skills (1)

### systems-programming-patterns

Battle-tested patterns for memory management, concurrency, performance optimization, and debugging across C/C++/Rust/Go.

**Key Capabilities:**
- Memory management patterns (pools, arenas, RAII)
- Concurrency patterns (lock-free, work-stealing)
- Performance optimization (SIMD, cache-friendly layouts)
- Debugging with Valgrind, perf, and sanitizers
- Zero-copy algorithms and atomic operations
- 20+ specific use cases with file patterns

**When to Use:**
- Implementing memory allocators or custom pools
- Designing lock-free data structures with atomics
- Optimizing cache performance (SoA layouts, prefetching)
- SIMD vectorization for performance-critical code
- Profiling with perf, valgrind, flamegraph
- Debugging memory issues with sanitizers
- Building thread pools or work-stealing schedulers

**Example Scenarios:**
```
- Writing *.c, *.cpp, *.rs, *.go files with low-level code
- Implementing custom allocators (200× speedup achievable)
- Lock-free queue design with compare-and-swap
- SIMD optimization for array processing
- Cache-friendly struct-of-arrays layouts
```

## Key Features

### Systematic Development Process

Each agent follows an 8-step workflow with self-verification:
1. **Analyze Requirements** - Understand constraints and platform needs
2. **Design Type-Safe APIs** - Clear ownership and error handling
3. **Implement with Idioms** - Language-specific best practices
4. **Concurrent Patterns** - Safe parallelism and synchronization
5. **Error Handling** - Comprehensive coverage of failure modes
6. **Optimize** - Profile-driven performance improvements
7. **Test Rigorously** - Unit, integration, and property-based tests
8. **Document** - Safety invariants and API contracts

### Quality Assurance Checkpoints

8 Constitutional AI principles ensure:
- Memory safety (no leaks, use-after-free, buffer overflows)
- Thread safety (race-free, proper synchronization)
- Error handling (no ignored errors, proper propagation)
- Type safety (prevent invalid states at compile-time)
- Performance (profile before optimizing)
- Async correctness (task lifecycle, cancellation)
- Testing (comprehensive coverage)
- Documentation (safety comments, examples)

### Strategic Ambiguity Handling

16 questions across 4 domains help clarify requirements:
- **Platform & Dependencies** (build system, tooling, versions)
- **Design & Architecture** (ownership, lifetimes, patterns)
- **Performance & Optimization** (targets, hot paths, tradeoffs)
- **Concurrency & Testing** (threading model, test coverage)

### Comprehensive Examples

Each agent includes 3 detailed examples:
- **Good Example**: Production-ready implementation (~100-150 lines)
- **Bad Example**: Common antipatterns to avoid
- **Annotated Example**: Step-by-step walkthrough with reasoning

### Common Patterns Library

3 battle-tested patterns per agent with:
- Clear use cases and decision criteria
- Step-by-step implementation guide
- Validation checklist for correctness

## Quick Start

To use this plugin:

1. Ensure Claude Code is installed
2. Enable the `systems-programming` plugin
3. Activate an agent (e.g., `@rust-pro`)
4. Try a command (e.g., `/rust-project`)

**Example workflow:**
```bash
# Activate Rust agent
@rust-pro

# Ask for implementation with systematic process
"Implement a thread-safe connection pool with graceful shutdown"

# The agent will:
# 1. Ask strategic questions about requirements
# 2. Design API with ownership semantics
# 3. Implement with Tokio patterns
# 4. Add comprehensive error handling
# 5. Include tests and benchmarks
# 6. Document safety invariants
```

## Integration

See the full documentation for integration patterns and compatible plugins.

### Compatible Plugins
- **unit-testing**: Enhanced E2E testing patterns for systems code
- **debugging-toolkit**: AI-assisted debugging with observability
- **cicd-automation**: CI/CD pipelines for C/C++/Rust/Go projects
- **performance-engineering**: Advanced profiling and optimization

## Documentation

For comprehensive documentation, see: [Plugin Documentation](https://myclaude.readthedocs.io/en/latest/plugins/systems-programming.html)

To build documentation locally:

```bash
cd docs/
make html
```
