# C Project Structures Reference

Comprehensive guide to C project organization patterns for applications, libraries, and embedded systems with production-ready scaffolding templates.

---

## Application Project Structure

### Standard CLI Application

```
cli-app/
├── Makefile
├── CMakeLists.txt (alternative)
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── main.c
│   ├── config.c
│   ├── config.h
│   ├── logger.c
│   ├── logger.h
│   ├── argparse.c
│   ├── argparse.h
│   └── utils/
│       ├── memory.c
│       ├── memory.h
│       ├── string_utils.c
│       └── string_utils.h
├── include/
│   └── cli_app.h
├── tests/
│   ├── test_main.c
│   ├── test_config.c
│   ├── test_utils.c
│   └── test_framework.h
├── docs/
│   ├── API.md
│   └── USAGE.md
└── scripts/
    ├── install.sh
    ├── valgrind.sh
    └── run_tests.sh
```

**Key Components:**
- **src/main.c**: Entry point with argument parsing and initialization
- **src/config.c**: Configuration management (file parsing, environment variables)
- **src/logger.c**: Thread-safe logging with levels (DEBUG, INFO, WARN, ERROR)
- **src/utils/**: Utility functions (memory management, string operations)
- **include/**: Public header files for library interface
- **tests/**: Unit and integration tests
- **scripts/**: Development and deployment automation

**File Organization Principles:**
1. Separate public headers (include/) from implementation (src/)
2. Group related functionality in subdirectories (utils/, commands/)
3. One header per module with clear dependencies
4. Test files mirror source structure

### System Daemon Structure

```
daemon/
├── Makefile
├── README.md
├── src/
│   ├── daemon.c          # Main daemon loop
│   ├── daemon.h
│   ├── signal_handler.c  # SIGTERM, SIGHUP handling
│   ├── signal_handler.h
│   ├── worker.c          # Worker threads
│   ├── worker.h
│   ├── ipc.c             # Inter-process communication
│   ├── ipc.h
│   └── pid_file.c        # PID file management
│       └── pid_file.h
├── config/
│   ├── daemon.conf       # Configuration file
│   └── daemon.service    # systemd unit file
├── tests/
│   └── test_daemon.c
└── scripts/
    ├── start.sh
    ├── stop.sh
    └── reload.sh
```

**Daemon-Specific Requirements:**
- Double fork() for proper daemonization
- Signal handling (SIGTERM for shutdown, SIGHUP for reload)
- PID file management in /var/run/
- Logging to syslog or log files
- Privilege dropping after initialization
- Resource cleanup on shutdown

**Example: Daemonization Code**
```c
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>

int daemonize(void) {
    pid_t pid;

    // First fork
    pid = fork();
    if (pid < 0) return -1;
    if (pid > 0) exit(EXIT_SUCCESS);  // Parent exits

    // Create new session
    if (setsid() < 0) return -1;

    // Second fork
    pid = fork();
    if (pid < 0) return -1;
    if (pid > 0) exit(EXIT_SUCCESS);

    // Change working directory
    if (chdir("/") < 0) return -1;

    // Close standard file descriptors
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    // Redirect to /dev/null
    int fd = open("/dev/null", O_RDWR);
    if (fd != -1) {
        dup2(fd, STDIN_FILENO);
        dup2(fd, STDOUT_FILENO);
        dup2(fd, STDERR_FILENO);
        if (fd > 2) close(fd);
    }

    return 0;
}
```

---

## Library Project Structure

### Shared Library (.so) Structure

```
libproject/
├── Makefile
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   ├── project/
│   │   ├── core.h        # Core functionality
│   │   ├── types.h       # Type definitions
│   │   ├── error.h       # Error codes
│   │   ├── version.h     # Version macros
│   │   └── config.h.in   # Build configuration template
│   └── project.h         # Main header (includes all)
├── src/
│   ├── core.c
│   ├── internal.h        # Internal-only declarations
│   ├── error.c
│   ├── version.c
│   └── platform/
│       ├── linux.c       # Linux-specific code
│       ├── macos.c       # macOS-specific code
│       └── windows.c     # Windows-specific code
├── tests/
│   ├── test_core.c
│   ├── test_error.c
│   └── test_framework.h
├── examples/
│   ├── basic_usage.c
│   ├── advanced_usage.c
│   └── multithreaded.c
├── pkg-config/
│   └── libproject.pc.in
└── docs/
    ├── API.md
    └── INTERNALS.md
```

**Library Design Principles:**
1. **Clear API boundary**: Public headers in include/, internals in src/
2. **Symbol visibility**: Use `__attribute__((visibility("default")))` for public API
3. **Namespace prefix**: Prefix all public symbols (e.g., `proj_init`, `PROJ_ERROR`)
4. **Version management**: Semantic versioning with SONAME
5. **Platform abstraction**: Isolate platform-specific code
6. **Thread safety**: Document thread-safety guarantees per function

**Example: Public Header Pattern**
```c
/* include/project/core.h */
#ifndef PROJECT_CORE_H
#define PROJECT_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* Version macros */
#define PROJECT_VERSION_MAJOR 1
#define PROJECT_VERSION_MINOR 0
#define PROJECT_VERSION_PATCH 0

/* Visibility macros */
#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef PROJECT_BUILDING_DLL
    #define PROJECT_API __declspec(dllexport)
  #else
    #define PROJECT_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define PROJECT_API __attribute__((visibility("default")))
#else
  #define PROJECT_API
#endif

/* Opaque type for encapsulation */
typedef struct proj_context proj_context_t;

/* Public API functions */
PROJECT_API proj_context_t* proj_init(void);
PROJECT_API int proj_configure(proj_context_t* ctx, const char* key, const char* value);
PROJECT_API int proj_execute(proj_context_t* ctx);
PROJECT_API void proj_destroy(proj_context_t* ctx);

/* Thread-safe: yes (when using separate contexts) */
/* Reentrant: yes */

#ifdef __cplusplus
}
#endif

#endif /* PROJECT_CORE_H */
```

### Static Library (.a) Structure

```
libutils/
├── Makefile
├── include/
│   └── utils/
│       ├── hash.h
│       ├── list.h
│       ├── pool.h
│       └── string.h
├── src/
│   ├── hash.c
│   ├── list.c
│   ├── pool.c
│   └── string.c
└── tests/
    └── test_all.c
```

**Static Library Considerations:**
- No ABI compatibility concerns (statically linked)
- Can be more aggressive with inlining
- Suitable for utility libraries
- Header-only option for simple utilities

---

## Embedded Systems Structure

### Bare-Metal Firmware

```
firmware/
├── Makefile
├── startup.s             # Assembly startup code
├── linker.ld             # Linker script
├── src/
│   ├── main.c
│   ├── interrupts.c
│   ├── interrupts.h
│   ├── drivers/
│   │   ├── uart.c
│   │   ├── uart.h
│   │   ├── gpio.c
│   │   └── gpio.h
│   ├── hal/              # Hardware abstraction layer
│   │   ├── hal_uart.h
│   │   └── hal_gpio.h
│   └── bsp/              # Board support package
│       ├── stm32f4.h
│       └── memory_map.h
├── include/
│   └── config.h
└── tests/
    └── host/             # Host-based tests
        └── test_logic.c
```

**Embedded-Specific Patterns:**
1. **Startup code**: Assembly initialization before main()
2. **Linker script**: Memory layout (Flash, RAM, peripherals)
3. **Interrupt vectors**: ISR table definition
4. **HAL**: Hardware abstraction for portability
5. **BSP**: Board-specific configurations
6. **Resource constraints**: Minimize stack usage, static allocation

**Example: Linker Script (ARM Cortex-M4)**
```ld
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 128K
}

SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
    } > FLASH

    .data :
    {
        _data_start = .;
        *(.data*)
        _data_end = .;
    } > RAM AT> FLASH

    .bss :
    {
        _bss_start = .;
        *(.bss*)
        *(COMMON)
        _bss_end = .;
    } > RAM

    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
```

### RTOS-Based Application

```
rtos-app/
├── Makefile
├── FreeRTOSConfig.h
├── src/
│   ├── main.c
│   ├── tasks/
│   │   ├── sensor_task.c
│   │   ├── sensor_task.h
│   │   ├── control_task.c
│   │   └── control_task.h
│   ├── queues.c          # Queue definitions
│   ├── queues.h
│   └── drivers/
│       └── ...
└── FreeRTOS/             # RTOS source (submodule)
```

**RTOS Design Patterns:**
- Task-based architecture with clear responsibilities
- Inter-task communication via queues/semaphores
- Priority assignment based on deadlines
- Stack size tuning per task
- Idle task hook for power management

---

## Multi-Module Project Structure

### Large Application with Modules

```
large-app/
├── Makefile
├── CMakeLists.txt
├── src/
│   ├── main.c
│   ├── core/
│   │   ├── engine.c
│   │   ├── engine.h
│   │   └── CMakeLists.txt
│   ├── network/
│   │   ├── http_client.c
│   │   ├── http_client.h
│   │   ├── websocket.c
│   │   └── CMakeLists.txt
│   ├── database/
│   │   ├── sqlite_wrapper.c
│   │   ├── sqlite_wrapper.h
│   │   └── CMakeLists.txt
│   └── ui/
│       ├── terminal_ui.c
│       ├── terminal_ui.h
│       └── CMakeLists.txt
├── include/
│   └── app/
│       ├── core.h
│       ├── network.h
│       ├── database.h
│       └── ui.h
├── tests/
│   ├── core/
│   ├── network/
│   ├── database/
│   └── ui/
└── third_party/
    ├── json/
    └── crypto/
```

**Module Organization:**
1. Each module has its own directory under src/
2. Module-specific CMakeLists.txt for independent build
3. Public API in include/, implementation in src/
4. Test directory mirrors module structure
5. Third-party dependencies isolated

---

## Cross-Platform Project Structure

```
cross-platform/
├── Makefile
├── CMakeLists.txt
├── src/
│   ├── main.c
│   ├── common/           # Platform-independent code
│   │   ├── algorithm.c
│   │   └── algorithm.h
│   ├── platform/
│   │   ├── platform.h    # Platform abstraction interface
│   │   ├── linux/
│   │   │   ├── filesystem.c
│   │   │   └── threads.c
│   │   ├── windows/
│   │   │   ├── filesystem.c
│   │   │   └── threads.c
│   │   └── macos/
│   │       ├── filesystem.c
│   │       └── threads.c
│   └── config/
│       ├── config_linux.h
│       ├── config_windows.h
│       └── config_macos.h
└── tests/
    └── test_common.c
```

**Cross-Platform Strategies:**
1. **Platform abstraction layer**: Single interface, multiple implementations
2. **Conditional compilation**: `#ifdef _WIN32` for platform-specific code
3. **CMake platform detection**: Automatic source selection
4. **POSIX compliance**: Maximize portable code
5. **Testing**: Test on all target platforms

**Example: Platform Abstraction**
```c
/* platform/platform.h */
#ifndef PLATFORM_H
#define PLATFORM_H

typedef struct {
    void* handle;
} plat_thread_t;

/* Platform-independent interface */
int plat_thread_create(plat_thread_t* thread, void* (*func)(void*), void* arg);
int plat_thread_join(plat_thread_t* thread);
void plat_thread_destroy(plat_thread_t* thread);

#endif
```

---

## Configuration Management

### Build-Time Configuration

**config.h.in (CMake template)**
```c
#ifndef CONFIG_H
#define CONFIG_H

#define PROJECT_NAME "@PROJECT_NAME@"
#define PROJECT_VERSION "@PROJECT_VERSION@"
#define PROJECT_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define PROJECT_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define PROJECT_VERSION_PATCH @PROJECT_VERSION_PATCH@

/* Feature flags */
#cmakedefine HAVE_PTHREAD
#cmakedefine HAVE_OPENSSL
#cmakedefine ENABLE_LOGGING
#cmakedefine ENABLE_DEBUG

/* Platform detection */
#cmakedefine PLATFORM_LINUX
#cmakedefine PLATFORM_WINDOWS
#cmakedefine PLATFORM_MACOS

/* Compiler features */
#cmakedefine HAVE_STDATOMIC_H
#cmakedefine HAVE_BUILTIN_EXPECT

#endif
```

**CMakeLists.txt generation**
```cmake
configure_file(
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/include/config.h"
)
```

### Runtime Configuration

**Example: Configuration File Parser**
```c
/* config.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

typedef struct {
    char* server_host;
    int server_port;
    int worker_threads;
    int enable_logging;
} config_t;

int config_load(config_t* cfg, const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;

        // Parse key=value
        char key[64], value[192];
        if (sscanf(line, "%63[^=]=%191s", key, value) == 2) {
            if (strcmp(key, "server_host") == 0) {
                cfg->server_host = strdup(value);
            } else if (strcmp(key, "server_port") == 0) {
                cfg->server_port = atoi(value);
            } else if (strcmp(key, "worker_threads") == 0) {
                cfg->worker_threads = atoi(value);
            } else if (strcmp(key, "enable_logging") == 0) {
                cfg->enable_logging = (strcmp(value, "true") == 0) ? 1 : 0;
            }
        }
    }

    fclose(fp);
    return 0;
}

void config_free(config_t* cfg) {
    free(cfg->server_host);
}
```

---

## Testing Structure

### Unit Testing Framework

**test_framework.h**
```c
#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>

#define TEST(name) void test_##name(void)
#define RUN_TEST(name) do { \
    printf("Running test_%s...", #name); \
    test_##name(); \
    printf(" PASSED\n"); \
} while(0)

#define ASSERT(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "ASSERTION FAILED: %s (line %d)\n", #condition, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NULL(ptr) ASSERT((ptr) == NULL)
#define ASSERT_NOT_NULL(ptr) ASSERT((ptr) != NULL)

#endif
```

**Example Test File**
```c
#include "test_framework.h"
#include "string_utils.h"

TEST(string_trim) {
    char str[] = "  hello  ";
    string_trim(str);
    ASSERT_EQ(strcmp(str, "hello"), 0);
}

TEST(string_split) {
    char* tokens[10];
    int count = string_split("a,b,c", ",", tokens, 10);
    ASSERT_EQ(count, 3);
    ASSERT_EQ(strcmp(tokens[0], "a"), 0);
    ASSERT_EQ(strcmp(tokens[1], "b"), 0);
    ASSERT_EQ(strcmp(tokens[2], "c"), 0);
}

int main(void) {
    printf("Running string_utils tests...\n");
    RUN_TEST(string_trim);
    RUN_TEST(string_split);
    printf("All tests passed!\n");
    return 0;
}
```

---

## Common Utilities

### Memory Pool Allocator

```c
/* utils/pool.h */
#ifndef POOL_H
#define POOL_H

#include <stddef.h>

typedef struct pool pool_t;

pool_t* pool_create(size_t object_size, size_t capacity);
void* pool_alloc(pool_t* pool);
void pool_free(pool_t* pool, void* ptr);
void pool_destroy(pool_t* pool);
size_t pool_available(pool_t* pool);

#endif
```

### String Utilities

```c
/* utils/string_utils.h */
#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <stddef.h>

char* string_dup(const char* str);
char* string_trim(char* str);
int string_split(const char* str, const char* delim, char** tokens, int max_tokens);
int string_starts_with(const char* str, const char* prefix);
int string_ends_with(const char* str, const char* suffix);

#endif
```

### Error Handling

```c
/* error.h */
#ifndef ERROR_H
#define ERROR_H

typedef enum {
    ERR_OK = 0,
    ERR_NOMEM = -1,
    ERR_INVALID_ARG = -2,
    ERR_NOT_FOUND = -3,
    ERR_IO = -4,
    ERR_TIMEOUT = -5,
    ERR_PERMISSION = -6
} error_t;

const char* error_string(error_t err);

/* Error handling with goto cleanup pattern */
#define CHECK(expr) do { \
    int _ret = (expr); \
    if (_ret != ERR_OK) { \
        err = _ret; \
        goto cleanup; \
    } \
} while(0)

#endif
```

---

## Build Artifacts Organization

```
build/
├── debug/
│   ├── bin/           # Debug executables
│   ├── lib/           # Debug libraries
│   └── obj/           # Object files
├── release/
│   ├── bin/           # Release executables
│   ├── lib/           # Release libraries
│   └── obj/           # Object files
└── coverage/          # Coverage reports
```

**Separation Benefits:**
- Clean separation of build types
- Easy to switch between debug and release
- Facilitates parallel builds
- Simplifies .gitignore

---

## Documentation Structure

```
docs/
├── API.md             # Public API reference
├── ARCHITECTURE.md    # System design overview
├── BUILDING.md        # Build instructions
├── CONTRIBUTING.md    # Contribution guidelines
├── INTERNALS.md       # Internal implementation details
└── examples/
    ├── basic.md
    ├── advanced.md
    └── multithreading.md
```

---

## Summary: Best Practices

1. **Separate concerns**: Public API (include/) vs implementation (src/)
2. **Modular design**: One module per directory with clear boundaries
3. **Platform abstraction**: Isolate platform-specific code
4. **Testing**: Mirror source structure in tests/
5. **Configuration**: Build-time (config.h.in) and runtime (config files)
6. **Documentation**: API docs, architecture, build instructions
7. **Build artifacts**: Separate debug/release builds
8. **Utilities**: Common utilities (memory, strings, errors) in utils/
9. **Thread safety**: Document and enforce thread-safety guarantees
10. **Version management**: Semantic versioning with SONAME for libraries
