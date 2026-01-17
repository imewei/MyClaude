---
description: Scaffold production-ready C projects with Makefile/CMake and memory safety
  tools
triggers:
- /c-project
- workflow for c project
version: 1.0.7
command: /c-project
argument-hint: '[project-type] [project-name]'
execution_modes:
  quick: 1-2 hours
  standard: 4-6 hours
  enterprise: 1-2 days
allowed-tools: Bash, Write, Read, Edit
---


# C Project Scaffolding

Scaffold production-ready C projects with proper structure, build systems, testing, and memory safety tools.

## Context

$ARGUMENTS

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 1-2h | Basic Makefile, single-file, simple main.c with logger |
| Standard (default) | 4-6h | Makefile + CMake, modular structure, tests, Valgrind |
| Enterprise | 1-2d | CMake with feature detection, platform abstraction, CI/CD |

## Phase 1: Project Type Analysis

| Type | Use Case | Structure |
|------|----------|-----------|
| Application | CLI tools, daemons, utilities | Single executable |
| Library | Shared (.so) or static (.a) | Header + implementation |
| Embedded | Bare-metal, RTOS | Cross-compilation setup |
| System Service | Long-running background | Daemonization, signals |

**Mode Scoping:**
- Quick: Single-directory, minimal
- Standard: Modular src/include/tests
- Enterprise: Multi-module with platform abstraction

**Reference:** [C Project Structures](../../plugins/systems-programming/docs/c-project/c-project-structures.md)

## Phase 2: Project Structure

### Quick Mode
```
project/
├── Makefile
├── main.c
├── logger.h / logger.c
└── README.md
```

### Standard/Enterprise Mode
```
project/
├── Makefile
├── CMakeLists.txt
├── src/main.c, config.c, logger.c
├── include/project.h
├── tests/test_main.c
└── scripts/valgrind.sh
```

## Phase 3: Build System Generation

### Makefile Targets

| Target | Purpose | Mode |
|--------|---------|------|
| all | Build main executable | All |
| clean | Remove build artifacts | All |
| test | Run test suite | Standard+ |
| valgrind | Memory leak check | Standard+ |
| asan | AddressSanitizer build | Standard+ |
| coverage | Code coverage report | Enterprise |
| install | Install to system | Enterprise |

### CMake Features (Enterprise)

| Feature | Purpose |
|---------|---------|
| CheckIncludeFile | Feature detection |
| Platform conditionals | OS-specific sources |
| Sanitizer integration | Debug builds |
| CTest | Testing framework |
| CPack | Packaging |

**Full reference:** [C Build Systems Guide](../../plugins/systems-programming/docs/c-project/c-build-systems.md)

## Phase 4: Source Templates

### Core Components

| Component | Purpose | Mode |
|-----------|---------|------|
| main.c | Entry point, argument parsing | All |
| logger.c/h | Thread-safe logging | All |
| config.c/h | Configuration management | Standard+ |
| error.h | Error handling macros | Standard+ |
| utils/ | Memory pools, string utilities | Enterprise |

**Templates:** [C Project Structures Guide](../../plugins/systems-programming/docs/c-project/c-project-structures.md)

## Phase 5: Testing Setup

| Mode | Framework | Features |
|------|-----------|----------|
| Quick | None | Manual testing |
| Standard | Simple test macros | ASSERT_EQ, RUN_TEST |
| Enterprise | CTest + Valgrind | Automated, memory-safe |

### Test Integration

| Tool | Command | Purpose |
|------|---------|---------|
| Unit tests | `make test` | Correctness |
| Valgrind | `make valgrind` | Memory leaks |
| ASan | `make asan` | Memory errors |
| UBSan | Compiler flag | Undefined behavior |
| TSan | Compiler flag | Data races |

**Reference:** [C Build Systems - Testing](../../plugins/systems-programming/docs/c-project/c-build-systems.md#testing-integration)

## Phase 6: Memory Safety Validation

### Tools by Mode

| Tool | Purpose | Mode |
|------|---------|------|
| Valgrind | Leak detection, memory errors | Standard+ |
| AddressSanitizer | Buffer overflows, use-after-free | Standard+ |
| UBSan | Undefined behavior | Standard+ |
| ThreadSanitizer | Data races | Enterprise |
| ASAN_OPTIONS | Fine-tuning | Enterprise |

### Valgrind Flags

| Flag | Purpose |
|------|---------|
| --leak-check=full | Complete leak report |
| --show-leak-kinds=all | All leak types |
| --track-origins=yes | Uninitialized value origins |
| --error-exitcode=1 | CI/CD integration |

**Comprehensive reference:** [C Memory Safety Guide](../../plugins/systems-programming/docs/c-project/c-memory-safety.md)

## Phase 7: Documentation and Scripts

| Mode | Documentation |
|------|--------------|
| Quick | README.md with build instructions |
| Standard | + API.md, install script, valgrind helper |
| Enterprise | + docs/ (API, ARCHITECTURE, BUILDING, CONTRIBUTING), CI/CD, packaging |

## Output Deliverables

| Mode | Deliverables |
|------|--------------|
| Quick | Working executable, basic Makefile, logger, README |
| Standard | + Modular structure, CMakeLists.txt, tests, Valgrind, API docs |
| Enterprise | + Platform abstraction, CI/CD, cross-compilation, packaging |

## External Documentation

| Document | Content | Lines |
|----------|---------|-------|
| [C Project Structures](../../plugins/systems-programming/docs/c-project/c-project-structures.md) | Application, library, embedded patterns | ~600 |
| [C Build Systems](../../plugins/systems-programming/docs/c-project/c-build-systems.md) | Makefile, CMake, sanitizers, CI/CD | ~500 |
| [C Memory Safety](../../plugins/systems-programming/docs/c-project/c-memory-safety.md) | Valgrind, ASan, best practices | ~450 |

## Quality Checklist

### All Modes
- [ ] Compiles without warnings (`-Wall -Wextra`)
- [ ] README.md with build instructions
- [ ] Git repository initialized

### Standard+
- [ ] `make test` passes
- [ ] Valgrind clean (no memory leaks)
- [ ] Both Makefile and CMake work
- [ ] Consistent code style

### Enterprise
- [ ] CI/CD validates builds
- [ ] Cross-compilation tested
- [ ] Packaging scripts work
- [ ] Documentation comprehensive
