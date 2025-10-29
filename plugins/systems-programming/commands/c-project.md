# C Project Scaffolding

You are a C project architecture expert specializing in scaffolding production-ready C applications and libraries. Generate complete project structures with proper build systems, testing setup, and configuration following modern C best practices.

## Context

The user needs automated C project scaffolding that creates safe, efficient applications with proper structure, dependency management, testing, and build configuration. Focus on POSIX compliance, memory safety, and systems programming best practices.

## Requirements

$ARGUMENTS

## Instructions

### 1. Analyze Project Type

Determine the project type from user requirements:
- **Application**: CLI tools, daemons, system utilities
- **Library**: Shared (.so) or static (.a) libraries
- **Embedded**: Bare-metal or RTOS-based firmware
- **System Service**: Long-running background services

### 2. Initialize Project Structure

**Application Project Structure**:
```
project-name/
├── Makefile
├── CMakeLists.txt (alternative)
├── README.md
├── LICENSE
├── src/
│   ├── main.c
│   ├── config.c
│   ├── config.h
│   ├── logger.c
│   ├── logger.h
│   └── utils/
│       ├── memory.c
│       └── memory.h
├── include/
│   └── project_name.h
├── tests/
│   ├── test_main.c
│   ├── test_config.c
│   └── test_utils.c
├── docs/
│   └── API.md
└── scripts/
    ├── install.sh
    └── valgrind.sh
```

**Library Project Structure**:
```
libproject/
├── Makefile
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   ├── project/
│   │   ├── core.h
│   │   ├── types.h
│   │   └── error.h
│   └── project.h (main header)
├── src/
│   ├── core.c
│   ├── internal.h
│   └── platform/
│       ├── linux.c
│       └── macos.c
├── tests/
│   └── test_library.c
├── examples/
│   ├── basic_usage.c
│   └── advanced_usage.c
└── pkg-config/
    └── libproject.pc.in
```

### 3. Generate Makefile

**Basic Makefile Template**:
```makefile
# Project configuration
PROJECT := project-name
VERSION := 0.1.0

# Compiler and flags
CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Werror -Wpedantic \
          -Wconversion -Wstrict-prototypes -Wmissing-prototypes \
          -O2 -g
LDFLAGS :=
LIBS := -lpthread

# Directories
SRC_DIR := src
BUILD_DIR := build
INC_DIR := include
TEST_DIR := tests

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/**/*.c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS := $(TEST_SRCS:$(TEST_DIR)/%.c=$(BUILD_DIR)/%)

# Targets
TARGET := $(BUILD_DIR)/$(PROJECT)

.PHONY: all clean test install valgrind

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

test: $(TEST_BINS)
	@for test in $(TEST_BINS); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done

$(BUILD_DIR)/%: $(TEST_DIR)/%.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS))
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INC_DIR) -o $@ $^ $(LIBS)

valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all \
	         --track-origins=yes --error-exitcode=1 \
	         $(TARGET)

clean:
	rm -rf $(BUILD_DIR)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

.SUFFIXES:
```

**Library Makefile Template**:
```makefile
# Library configuration
LIB_NAME := project
LIB_VERSION := 0.1.0
LIB_SONAME := lib$(LIB_NAME).so.0

# Compiler and flags
CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Werror -fPIC \
          -O2 -g -I$(INC_DIR)
LDFLAGS := -shared -Wl,-soname,$(LIB_SONAME)

# Directories
SRC_DIR := src
BUILD_DIR := build
INC_DIR := include

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Targets
STATIC_LIB := $(BUILD_DIR)/lib$(LIB_NAME).a
SHARED_LIB := $(BUILD_DIR)/lib$(LIB_NAME).so.$(LIB_VERSION)

.PHONY: all clean install

all: $(STATIC_LIB) $(SHARED_LIB)

$(STATIC_LIB): $(OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^
	cd $(BUILD_DIR) && ln -sf lib$(LIB_NAME).so.$(LIB_VERSION) lib$(LIB_NAME).so

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

install: all
	install -d $(DESTDIR)/usr/local/lib
	install -d $(DESTDIR)/usr/local/include/$(LIB_NAME)
	install -m 644 $(STATIC_LIB) $(DESTDIR)/usr/local/lib/
	install -m 755 $(SHARED_LIB) $(DESTDIR)/usr/local/lib/
	install -m 644 include/$(LIB_NAME)/*.h $(DESTDIR)/usr/local/include/$(LIB_NAME)/
	ldconfig

clean:
	rm -rf $(BUILD_DIR)
```

### 4. Generate CMakeLists.txt (Alternative)

```cmake
cmake_minimum_required(VERSION 3.15)
project(ProjectName VERSION 0.1.0 LANGUAGES C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(
    -Wall
    -Wextra
    -Werror
    -Wpedantic
    -Wconversion
    -Wstrict-prototypes
    -Wmissing-prototypes
)

# Sanitizers for debug builds
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.c")
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.c$")

# Main executable
add_executable(${PROJECT_NAME} src/main.c ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE pthread)

# Library (if building library)
# add_library(${PROJECT_NAME} SHARED ${SOURCES})
# set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

# Testing
enable_testing()
file(GLOB TEST_SOURCES "tests/*.c")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} ${SOURCES})
    target_link_libraries(${test_name} PRIVATE pthread)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
```

### 5. Generate Source Code Templates

**main.c**:
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "config.h"
#include "logger.h"

int main(int argc, char *argv[]) {
    // Initialize logger
    if (logger_init("program.log") != 0) {
        fprintf(stderr, "Failed to initialize logger\n");
        return EXIT_FAILURE;
    }

    // Parse command line arguments
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <argument>\n", argv[0]);
        logger_shutdown();
        return EXIT_FAILURE;
    }

    // Main program logic
    LOG_INFO("Program started");
    LOG_DEBUG("Processing argument: %s", argv[1]);

    // TODO: Implement main functionality

    LOG_INFO("Program completed successfully");

    // Cleanup
    logger_shutdown();
    return EXIT_SUCCESS;
}
```

**logger.h**:
```c
#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h>

typedef enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR
} log_level_t;

int logger_init(const char *filename);
void logger_shutdown(void);
void logger_set_level(log_level_t level);
void logger_log(log_level_t level, const char *file, int line, const char *fmt, ...);

#define LOG_DEBUG(...) logger_log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)  logger_log(LOG_LEVEL_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  logger_log(LOG_LEVEL_WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) logger_log(LOG_LEVEL_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#endif /* LOGGER_H */
```

**logger.c**:
```c
#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

static FILE *log_file = NULL;
static log_level_t current_level = LOG_LEVEL_INFO;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

int logger_init(const char *filename) {
    log_file = fopen(filename, "a");
    if (!log_file) {
        return -1;
    }
    return 0;
}

void logger_shutdown(void) {
    if (log_file) {
        fclose(log_file);
        log_file = NULL;
    }
}

void logger_set_level(log_level_t level) {
    current_level = level;
}

void logger_log(log_level_t level, const char *file, int line, const char *fmt, ...) {
    if (level < current_level || !log_file) {
        return;
    }

    const char *level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};

    pthread_mutex_lock(&log_mutex);

    // Timestamp
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    fprintf(log_file, "[%04d-%02d-%02d %02d:%02d:%02d] [%s] [%s:%d] ",
            t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
            t->tm_hour, t->tm_min, t->tm_sec,
            level_str[level], file, line);

    // Message
    va_list args;
    va_start(args, fmt);
    vfprintf(log_file, fmt, args);
    va_end(args);

    fprintf(log_file, "\n");
    fflush(log_file);

    pthread_mutex_unlock(&log_mutex);
}
```

### 6. Generate Test Template

**test_main.c**:
```c
#include <stdio.h>
#include <assert.h>
#include <string.h>

// Simple test framework macros
#define TEST(name) void test_##name(void)
#define RUN_TEST(name) do { \
    printf("Running test_%s...", #name); \
    test_##name(); \
    printf(" PASSED\n"); \
} while(0)

// Example test
TEST(example) {
    assert(1 + 1 == 2);
    assert(strcmp("hello", "hello") == 0);
}

int main(void) {
    printf("Running tests...\n");

    RUN_TEST(example);

    printf("All tests passed!\n");
    return 0;
}
```

### 7. Generate Development Scripts

**scripts/valgrind.sh**:
```bash
#!/bin/bash
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --error-exitcode=1 \
         ./build/program "$@"
```

**scripts/install.sh**:
```bash
#!/bin/bash
set -e

PREFIX=${PREFIX:-/usr/local}

echo "Installing to $PREFIX"
install -d "$PREFIX/bin"
install -m 755 build/program "$PREFIX/bin/"
echo "Installation complete"
```

## Output Format

Provide complete project with:
1. **Directory structure** matching project type
2. **Build system** (Makefile or CMake)
3. **Source files** with proper error handling
4. **Test framework** with example tests
5. **README** with build and usage instructions
6. **Scripts** for common development tasks

Focus on creating production-ready C projects with comprehensive error handling, memory safety, and testing infrastructure.
