# Systems Programming Plugin

Comprehensive systems programming toolkit for C, C++, Rust, and Go development with production-ready agents, project scaffolding commands, performance profiling workflows, and battle-tested programming patterns.

## Overview

This plugin provides expert-level systems programming capabilities across four major systems languages, covering the entire development lifecycle from project initialization to performance optimization. It includes enhanced language-specific agents, automated project scaffolding, comprehensive profiling workflows, and a rich library of systems programming patterns.

## Features

### 🚀 **Enhanced Language Agents**

Four deeply capable agents covering modern systems programming:

- **rust-pro**: Rust 1.75+ expert with async patterns, advanced type system, zero-cost abstractions
- **c-pro**: C systems programming specialist with POSIX APIs, embedded development, memory safety
- **cpp-pro**: Modern C++ (C++11-23) expert with templates, RAII, STL, and performance optimization
- **golang-pro**: Go concurrent systems specialist with goroutines, channels, and cloud-native patterns

### ⚡ **Project Scaffolding Commands**

Automated project initialization with industry best practices:

- **`/rust-project`**: Cargo projects with proper module organization, async setup, and testing
- **`/c-project`**: C projects with Makefile/CMake, memory safety tools, and POSIX compliance
- **`/profile-performance`**: Comprehensive profiling workflow with perf, valgrind, and analysis

### 📚 **Systems Programming Patterns Skill**

Battle-tested patterns and techniques:

- Memory management (pools, arenas, RAII)
- Concurrent programming (lock-free structures, thread pools)
- Performance optimization (cache-aware programming, SIMD)
- Debugging workflows (GDB, Valgrind, sanitizers)
- Profiling techniques (perf, flamegraphs, hardware counters)

## Agents

### rust-pro

**Master Rust programmer** specializing in:
- Modern Rust 1.75+ features and compiler improvements
- Advanced async programming with Tokio ecosystem
- Type system mastery (GATs, traits, lifetimes)
- Zero-cost abstractions and performance optimization
- Unsafe code and FFI when necessary
- Comprehensive testing (unit, integration, property-based)

**Use for**:
- High-performance async services
- Memory-safe systems programming
- Lock-free concurrent data structures
- FFI wrappers around C libraries
- WebAssembly compilation targets

**Example**: "Design a high-performance async web service with proper error handling"

### c-pro

**Master C programmer** specializing in:
- C99/C11/C17/C23 standards
- Systems programming and POSIX APIs
- Embedded systems and bare-metal development
- Manual memory management and custom allocators
- Real-time systems and deterministic code
- Performance optimization with profiling tools

**Use for**:
- System utilities and daemons
- Embedded firmware development
- Kernel modules and device drivers
- High-performance libraries
- Memory-constrained applications

**Example**: "Implement a memory pool allocator for fixed-size objects"

### cpp-pro

**Master C++ programmer** specializing in:
- Modern C++ (C++11/14/17/20/23)
- Template metaprogramming and concepts
- RAII and smart pointer patterns
- STL algorithms and containers
- Concurrency (std::thread, atomics, futures)
- Exception safety and error handling

**Use for**:
- High-performance applications
- Template libraries
- Real-time systems
- Game engines
- Scientific computing

**Example**: "Design a thread-safe object pool using modern C++"

### golang-pro

**Master Go programmer** specializing in:
- Concurrent programming with goroutines
- Channel patterns and select statements
- Network services and HTTP servers
- Microservices architecture
- Cloud-native development
- Interface design and composition

**Use for**:
- Microservices and APIs
- Concurrent data processing
- Network servers
- CLI tools
- Cloud infrastructure

**Example**: "Build a concurrent HTTP server with graceful shutdown"

## Commands

### /rust-project

Scaffold production-ready Rust projects with:
- Proper Cargo.toml configuration
- Module organization (binary/library/workspace)
- Async runtime setup (Tokio)
- Testing infrastructure (unit, integration, benchmarks)
- CI/CD configuration
- Documentation setup

**Usage**:
```bash
/rust-project "Create a web API project with actix-web"
/rust-project "Initialize a library for parsing JSON"
/rust-project "Setup a workspace with multiple crates"
```

**Generates**:
- Complete directory structure
- Cargo.toml with dependencies
- Source files with proper organization
- Test files and examples
- README and documentation

### /c-project

Scaffold production-ready C projects with:
- Makefile or CMake build system
- POSIX-compliant code structure
- Memory safety tooling (Valgrind, ASan)
- Unit testing framework
- Header organization
- Documentation

**Usage**:
```bash
/c-project "Create a CLI tool with logging"
/c-project "Initialize a shared library"
/c-project "Setup embedded firmware project"
```

**Generates**:
- Build system (Makefile/CMake)
- Source and header files
- Test framework setup
- Development scripts (valgrind, install)
- Comprehensive error handling

### /profile-performance

Comprehensive performance profiling workflow:
- CPU profiling with perf and flamegraphs
- Memory profiling with massif and heaptrack
- Hardware counter analysis (cache, branches, IPC)
- Micro-benchmarking setup
- Optimization recommendations
- Verification workflows

**Usage**:
```bash
/profile-performance "Optimize this hot loop"
/profile-performance "Reduce memory allocations in parser"
/profile-performance "Improve cache locality in data structure"
```

**Provides**:
- Profiling commands for your code
- Results interpretation
- Specific optimization recommendations
- Benchmark code to verify improvements
- Before/after measurements

## Skills

### systems-programming-patterns

Comprehensive patterns and techniques covering:

#### Memory Management Patterns
- **Memory pools**: Fixed-size allocation with reduced overhead
- **Arena allocators**: Bulk allocation with single deallocation
- **RAII and smart pointers**: Automatic resource management
- **Custom allocators**: Application-specific allocation strategies

#### Concurrent Programming Patterns
- **Lock-free data structures**: CAS-based queues and stacks
- **Thread pools and work stealing**: Efficient task distribution
- **Reader-writer locks and RCU**: Optimized read-heavy workloads
- **Atomic operations**: Memory ordering and synchronization

#### Performance Optimization Patterns
- **Cache-aware programming**: SoA vs AoS, alignment, prefetching
- **SIMD vectorization**: Auto and manual vectorization
- **Zero-copy techniques**: Move semantics, memory mapping
- **Branch optimization**: Elimination and prediction hints

#### Debugging and Profiling
- **Memory debugging**: Valgrind, ASan, leak detection
- **CPU profiling**: perf, flamegraphs, hardware counters
- **Concurrency debugging**: TSan, helgrind, race detection
- **Performance analysis**: Bottleneck identification and optimization

**Reference materials** in `references/`:
- `profiling-guide.md`: Comprehensive profiling workflows and tool usage
- Additional references for specialized topics

## Installation

### From GitHub Marketplace

```bash
/plugin marketplace add <your-username>/scientific-computing-workflows
/plugin install systems-programming
```

### Local Installation

```bash
/plugin add ./plugins/systems-programming
```

## Usage Examples

### Example 1: Create a Rust HTTP Server

```
User: Create a production-ready Rust web API using axum
Claude: /rust-project "Create web API with axum"
```

Result: Complete Rust project with:
- Async Tokio runtime
- axum web framework setup
- Proper error handling
- Test structure
- Docker configuration

### Example 2: Optimize C++ Performance

```
User: This C++ function is slow, how can I optimize it?
Claude: /profile-performance "Analyze and optimize the function"
```

Result:
- Profiling commands and execution
- Performance analysis report
- Specific optimization recommendations
- Benchmark code to verify
- Before/after measurements

### Example 3: Debug Memory Leak

```
User: My C program has a memory leak
Claude (c-pro agent): Let me help debug this with Valgrind...
```

Result:
- Valgrind commands and analysis
- Leak location identification
- Fix implementation
- Verification with clean output

### Example 4: Implement Lock-Free Queue

```
User: Implement a lock-free MPMC queue in Rust
Claude (rust-pro agent): [Uses systems-programming-patterns skill]
```

Result:
- Lock-free queue implementation
- ABA problem mitigation
- Memory ordering explanation
- Comprehensive tests
- Performance benchmarks

## Best Practices

### When to Use Each Agent

**rust-pro**:
- Memory safety is critical
- Need async/concurrent code
- Building high-performance services
- Interfacing with C libraries
- Compile-time guarantees desired

**c-pro**:
- Embedded or resource-constrained systems
- Need maximum performance
- Interfacing with hardware
- Legacy C codebase maintenance
- POSIX systems programming

**cpp-pro**:
- Need template metaprogramming
- Large existing C++ codebase
- Real-time constraints with RAII
- High-performance computing
- Modern C++ features desired

**golang-pro**:
- Building microservices
- Network-heavy applications
- Need fast development iteration
- Concurrent data processing
- Cloud-native applications

### Development Workflow

1. **Project Initialization**: Use scaffolding commands (`/rust-project`, `/c-project`)
2. **Development**: Use language-specific agents for implementation
3. **Optimization**: Use `/profile-performance` to identify bottlenecks
4. **Patterns**: Reference `systems-programming-patterns` skill for best practices
5. **Debugging**: Use agent expertise for tool-assisted debugging
6. **Testing**: Comprehensive testing with sanitizers and coverage

### Code Quality Standards

All code generated follows:
- **Safety**: Memory safety, thread safety, error handling
- **Performance**: Profiled optimizations, zero-cost abstractions
- **Testing**: Unit tests, integration tests, benchmarks
- **Documentation**: Clear comments, API docs, examples
- **Tooling**: Sanitizers, linters, formatters configured

## Advanced Features

### Memory Management Expertise

- Custom allocators (pools, arenas, bump allocators)
- RAII patterns across languages
- Smart pointer strategies
- Zero-copy techniques
- Memory profiling and leak detection

### Concurrency Mastery

- Lock-free data structures
- Thread pool implementations
- Work stealing algorithms
- Atomic operations and memory ordering
- Race condition debugging

### Performance Optimization

- Cache-aware programming
- SIMD vectorization
- Branch prediction optimization
- Algorithm selection
- Hardware counter analysis

### Cross-Language Integration

- FFI between Rust and C/C++
- Safe wrappers around unsafe code
- ABI compatibility
- Build system integration
- Cross-compilation

## Troubleshooting

### Common Issues

**Build failures**:
- Check compiler version requirements
- Verify all dependencies installed
- Review compilation flags

**Performance issues**:
- Profile before optimizing
- Check debug vs release builds
- Verify optimization flags enabled

**Memory problems**:
- Run with Valgrind or ASan
- Check for leaks and use-after-free
- Verify proper resource cleanup

### Getting Help

For issues with this plugin:
1. Check agent descriptions for capabilities
2. Review skill reference materials
3. Use specific language agent for detailed help
4. Consult profiling workflow for performance issues

## Version History

### 2.0.0 (2025-10-27)
- **NEW**: Enhanced C and C++ agents with comprehensive capabilities
- **NEW**: Added `/c-project` scaffolding command
- **NEW**: Added `/profile-performance` profiling workflow command
- **NEW**: Created `systems-programming-patterns` skill with extensive patterns
- **IMPROVED**: Rust agent with expanded capabilities
- **IMPROVED**: Comprehensive documentation and examples

### 1.0.0
- Initial release with basic language agents
- Rust project scaffolding

## Contributing

Contributions welcome! Areas for expansion:
- Additional scaffolding commands (C++, Go)
- More reference materials in skills
- Language-specific optimization guides
- Real-world case studies

## License

MIT

## Authors

Wei Chen

## Related Plugins

- **python-development**: Python with FastAPI, Django, async patterns
- **javascript-typescript**: Modern JS/TS with Node.js and frameworks
- **hpc-computing**: High-performance numerical computing
- **machine-learning**: ML model development and deployment

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [CERT C Coding Standard](https://wiki.sei.cmu.edu/confluence/display/c/)
- [Go Blog](https://go.dev/blog/)
- [Systems Performance Book](http://www.brendangregg.com/systems-performance-2nd-edition-book.html)