---
version: 1.0.3
command: /c-project
description: Scaffold production-ready C projects with 3 execution modes for simple CLI tools to enterprise multi-module systems
argument-hint: [project-type] [project-name]
execution_modes:
  quick:
    duration: "1-2 hours"
    description: "Simple CLI tool or utility"
    scope: "Basic Makefile, single-file structure, simple main.c with logger"
    deliverables: "Runnable executable with basic error handling"
  standard:
    duration: "4-6 hours"
    description: "Production application with comprehensive setup"
    scope: "Makefile + CMakeLists.txt, modular structure, testing framework, Valgrind integration, memory safety validation"
    deliverables: "Production-ready project with tests, sanitizers, and documentation"
  enterprise:
    duration: "1-2 days"
    description: "Multi-module system with advanced build configuration"
    scope: "CMake with feature detection, platform-specific code, comprehensive testing, CI/CD workflows, cross-compilation setup, documentation suite"
    deliverables: "Enterprise-grade project with full automation and deployment pipeline"
workflow_type: "sequential"
interactive_mode: true
color: blue
allowed-tools: Bash, Write, Read, Edit
---

# C Project Scaffolding

Scaffold production-ready C projects with proper structure, build systems, testing, and memory safety tools.

## Context

$ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "What level of project scaffolding do you need?"
    header: "Project Scope"
    multiSelect: false
    options:
      - label: "Quick (1-2h)"
        description: "Simple CLI tool or utility with basic Makefile, single-file structure, and essential error handling."

      - label: "Standard (4-6h)"
        description: "Production application with Makefile/CMake, modular structure, testing framework, Valgrind integration, and memory safety validation."

      - label: "Enterprise (1-2d)"
        description: "Multi-module system with CMake feature detection, platform-specific code, comprehensive testing, CI/CD workflows, and cross-compilation setup."
</AskUserQuestion>

## Instructions

### 1. Analyze Project Requirements

Determine project type from user requirements:
- **Application**: CLI tools, daemons, system utilities
- **Library**: Shared (.so) or static (.a) libraries
- **Embedded**: Bare-metal or RTOS-based firmware
- **System Service**: Long-running background services

**Execution Mode Scoping:**
- **Quick**: Single-directory, minimal structure
- **Standard**: Modular organization, separate src/include/tests
- **Enterprise**: Multi-module with platform abstraction

**Reference:** [C Project Structures](../docs/c-project/c-project-structures.md)

---

### 2. Initialize Project Structure

**Quick Mode Structure:**
```
project/
├── Makefile
├── main.c
├── logger.h
├── logger.c
└── README.md
```

**Standard/Enterprise Mode Structure:**
```
project/
├── Makefile
├── CMakeLists.txt
├── README.md
├── LICENSE
├── src/
│   ├── main.c
│   ├── config.c
│   ├── logger.c
│   └── utils/
├── include/
│   └── project.h
├── tests/
│   └── test_main.c
└── scripts/
    ├── install.sh
    └── valgrind.sh
```

**Full reference:** [C Project Structures Guide](../docs/c-project/c-project-structures.md)

---

### 3. Generate Build System

**Quick Mode: Simple Makefile**
```makefile
CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -O2 -g
TARGET := program

all: $(TARGET)

$(TARGET): main.c logger.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
```

**Standard Mode: Makefile with Tests + Valgrind**
```makefile
# See full template in external docs
CFLAGS := -std=c11 -Wall -Wextra -Werror -O2 -g
LDFLAGS :=
LIBS := -lpthread

# Auto dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(INC_DIR) -MMD -MP -c -o $@ $<

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do $$test || exit 1; done

valgrind: $(TARGET)
	valgrind --leak-check=full --error-exitcode=1 $(TARGET)

asan: CFLAGS += -fsanitize=address
asan: clean all
```

**Enterprise Mode: CMake + Feature Detection**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject VERSION 1.0.0 LANGUAGES C)

# Feature detection
include(CheckIncludeFile)
check_include_file("pthread.h" HAVE_PTHREAD_H)

# Platform-specific sources
if(UNIX AND NOT APPLE)
    set(PLATFORM_SOURCES src/platform/linux.c)
elseif(APPLE)
    set(PLATFORM_SOURCES src/platform/macos.c)
endif()

# Sanitizers for debug
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()
```

**Full build system reference:** [C Build Systems Guide](../docs/c-project/c-build-systems.md)

---

### 4. Generate Source Templates

**Quick Mode: Minimal main.c**
```c
#include <stdio.h>
#include <stdlib.h>
#include "logger.h"

int main(int argc, char *argv[]) {
    logger_init("program.log");

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <argument>\n", argv[0]);
        return EXIT_FAILURE;
    }

    LOG_INFO("Program started");
    // TODO: Implement functionality

    logger_shutdown();
    return EXIT_SUCCESS;
}
```

**Standard/Enterprise Mode:**
- Generate logger.c/logger.h (thread-safe logging)
- Generate config.c/config.h (configuration management)
- Generate utils/ modules (memory pools, string utilities)
- Generate error.h (error handling macros)

**Templates available in:** [C Project Structures Guide](../docs/c-project/c-project-structures.md)

---

### 5. Setup Testing

**Standard Mode: Simple Test Framework**
```c
#include "test_framework.h"

TEST(example_test) {
    ASSERT_EQ(1 + 1, 2);
}

int main(void) {
    RUN_TEST(example_test);
    printf("All tests passed!\n");
    return 0;
}
```

**Enterprise Mode:**
- CMake CTest integration
- Valgrind test targets
- AddressSanitizer builds
- Code coverage reports

**Reference:** [C Build Systems - Testing](../docs/c-project/c-build-systems.md#testing-integration)

---

### 6. Memory Safety Validation

**Standard Mode: Valgrind Script**
```bash
#!/bin/bash
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --error-exitcode=1 \
         ./build/program "$@"
```

**Enterprise Mode:**
- Makefile targets for ASan, UBSan, TSan
- CI/CD integration for automated memory checks
- Suppressions file for known false positives

**Comprehensive reference:** [C Memory Safety Guide](../docs/c-project/c-memory-safety.md)

---

### 7. Documentation and Scripts

**Quick Mode:**
- README.md with build instructions
- Usage examples

**Standard Mode:**
- README.md + API.md
- Installation script
- Valgrind helper script

**Enterprise Mode:**
- Full docs/ directory (API, ARCHITECTURE, BUILDING, CONTRIBUTING)
- CI/CD workflow templates (GitHub Actions)
- Cross-compilation scripts
- Packaging configurations (Debian .deb, RPM)

---

## Output Deliverables

### Quick Mode (1-2h):
✅ Working executable
✅ Basic Makefile
✅ Logger implementation
✅ README with build instructions

### Standard Mode (4-6h):
✅ Modular project structure
✅ Makefile + CMakeLists.txt
✅ Test framework with examples
✅ Valgrind integration
✅ Memory safety validation
✅ API documentation

### Enterprise Mode (1-2d):
✅ Multi-module architecture
✅ CMake with feature detection
✅ Platform abstraction layer
✅ Comprehensive test suite
✅ CI/CD workflows
✅ Cross-compilation support
✅ Packaging configurations
✅ Full documentation suite

---

## External Documentation

Comprehensive guides for deep dives:

- **[C Project Structures](../docs/c-project/c-project-structures.md)** (~600 lines)
  - Application, library, embedded structures
  - Configuration management patterns
  - Testing and documentation organization

- **[C Build Systems](../docs/c-project/c-build-systems.md)** (~500 lines)
  - Makefile patterns (debug/release, static/shared libraries)
  - CMake configuration and advanced features
  - Sanitizers, coverage, CI/CD integration

- **[C Memory Safety](../docs/c-project/c-memory-safety.md)** (~450 lines)
  - Valgrind, AddressSanitizer, UBSan usage
  - Memory safety best practices
  - Common pitfalls and solutions

---

## Quality Checklist

**All Modes:**
- [ ] Project compiles without warnings
- [ ] README.md with clear build instructions
- [ ] Git repository initialized

**Standard+:**
- [ ] Tests pass with `make test`
- [ ] No memory leaks (Valgrind clean)
- [ ] Makefile and CMakeLists.txt both work
- [ ] Code follows consistent style

**Enterprise:**
- [ ] CI/CD workflow validates builds
- [ ] Cross-compilation tested
- [ ] Documentation comprehensive
- [ ] Packaging scripts work
