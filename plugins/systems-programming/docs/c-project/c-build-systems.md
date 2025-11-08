# C Build Systems Reference

Comprehensive guide to Makefile and CMake build systems for C projects with testing, sanitizers, and production configurations.

---

## Makefile Fundamentals

### Basic Application Makefile

```makefile
# Project configuration
PROJECT := myapp
VERSION := 1.0.0

# Compiler and flags
CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Werror -Wpedantic \
          -Wconversion -Wstrict-prototypes -Wmissing-prototypes \
          -O2 -g
LDFLAGS :=
LIBS := -lpthread -lm

# Directories
SRC_DIR := src
BUILD_DIR := build
INC_DIR := include
TEST_DIR := tests

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/**/*.c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

# Test files
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS := $(TEST_SRCS:$(TEST_DIR)/%.c=$(BUILD_DIR)/test_%)

# Target executable
TARGET := $(BUILD_DIR)/$(PROJECT)

# Phony targets
.PHONY: all clean test install valgrind asan ubsan

# Default target
all: $(TARGET)

# Link executable
$(TARGET): $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Build complete: $@"

# Compile source files with dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INC_DIR) -MMD -MP -c -o $@ $<

# Include dependency files
-include $(DEPS)

# Build and run tests
test: $(TEST_BINS)
	@echo "Running tests..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done
	@echo "All tests passed!"

# Build test executables
$(BUILD_DIR)/test_%: $(TEST_DIR)/%.c $(filter-out $(BUILD_DIR)/main.o,$(OBJS))
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INC_DIR) -o $@ $^ $(LIBS)

# Memory leak detection with Valgrind
valgrind: $(TARGET)
	valgrind --leak-check=full \
	         --show-leak-kinds=all \
	         --track-origins=yes \
	         --verbose \
	         --error-exitcode=1 \
	         $(TARGET)

# AddressSanitizer build
asan: CFLAGS += -fsanitize=address -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address
asan: clean $(TARGET)

# UndefinedBehaviorSanitizer build
ubsan: CFLAGS += -fsanitize=undefined
ubsan: LDFLAGS += -fsanitize=undefined
ubsan: clean $(TARGET)

# Install to system
install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Disable built-in suffix rules
.SUFFIXES:
```

**Key Features:**
- Automatic dependency generation (`-MMD -MP`)
- Parallel builds supported (`make -j`)
- Debug symbols included (`-g`)
- Comprehensive warnings (`-Wall -Wextra -Werror`)
- Sanitizer targets (asan, ubsan)
- Valgrind integration
- Directory creation handled automatically

---

## Advanced Makefile Patterns

### Multi-Target Makefile (Debug/Release)

```makefile
# Build configurations
BUILD_TYPE ?= debug

# Configuration-specific flags
ifeq ($(BUILD_TYPE),release)
    CFLAGS := -std=c11 -Wall -Wextra -O3 -DNDEBUG
    BUILD_DIR := build/release
else ifeq ($(BUILD_TYPE),debug)
    CFLAGS := -std=c11 -Wall -Wextra -Werror -O0 -g -DDEBUG
    BUILD_DIR := build/debug
else
    $(error Unknown BUILD_TYPE: $(BUILD_TYPE))
endif

# Usage:
# make BUILD_TYPE=release
# make BUILD_TYPE=debug
```

### Library Makefile (Static and Shared)

```makefile
# Library configuration
LIB_NAME := mylib
LIB_VERSION := 1.0.0
LIB_SONAME := lib$(LIB_NAME).so.1

# Compiler flags
CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Werror -fPIC -O2 -g
LDFLAGS := -shared -Wl,-soname,$(LIB_SONAME)

# Directories
SRC_DIR := src
BUILD_DIR := build
INC_DIR := include

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Library targets
STATIC_LIB := $(BUILD_DIR)/lib$(LIB_NAME).a
SHARED_LIB := $(BUILD_DIR)/lib$(LIB_NAME).so.$(LIB_VERSION)

.PHONY: all clean install

all: $(STATIC_LIB) $(SHARED_LIB)

# Static library
$(STATIC_LIB): $(OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $^
	@echo "Static library created: $@"

# Shared library
$(SHARED_LIB): $(OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $^
	@cd $(BUILD_DIR) && ln -sf lib$(LIB_NAME).so.$(LIB_VERSION) lib$(LIB_NAME).so.1
	@cd $(BUILD_DIR) && ln -sf lib$(LIB_NAME).so.1 lib$(LIB_NAME).so
	@echo "Shared library created: $@"

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

# Install libraries
install: all
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/include/$(LIB_NAME)
	install -m 644 $(STATIC_LIB) $(DESTDIR)$(PREFIX)/lib/
	install -m 755 $(SHARED_LIB) $(DESTDIR)$(PREFIX)/lib/
	install -m 644 $(INC_DIR)/$(LIB_NAME)/*.h $(DESTDIR)$(PREFIX)/include/$(LIB_NAME)/
	ldconfig

clean:
	rm -rf $(BUILD_DIR)
```

### Cross-Compilation Makefile

```makefile
# Cross-compilation toolchain
CROSS_COMPILE ?=
CC := $(CROSS_COMPILE)gcc
AR := $(CROSS_COMPILE)ar
STRIP := $(CROSS_COMPILE)strip

# Architecture-specific flags
ARCH ?= native
ifeq ($(ARCH),arm)
    CROSS_COMPILE := arm-linux-gnueabihf-
    CFLAGS += -march=armv7-a -mfpu=neon
else ifeq ($(ARCH),aarch64)
    CROSS_COMPILE := aarch64-linux-gnu-
    CFLAGS += -march=armv8-a
else ifeq ($(ARCH),x86_64)
    CFLAGS += -march=x86-64
endif

# Usage:
# make ARCH=arm
# make ARCH=aarch64
```

---

## CMake Build System

### Basic CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyApp VERSION 1.0.0 LANGUAGES C)

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

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -DDEBUG")
# Release flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

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
target_link_libraries(${PROJECT_NAME} PRIVATE pthread m)

# Testing
enable_testing()
file(GLOB TEST_SOURCES "tests/*.c")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src} ${SOURCES})
    target_link_libraries(${test_name} PRIVATE pthread m)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# CPack for packaging
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
include(CPack)
```

**Usage:**
```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build

# Install
cmake --install build --prefix /usr/local
```

---

## Library CMake Configuration

### Shared and Static Library

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyLib VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Source files
file(GLOB SOURCES "src/*.c")

# Shared library
add_library(${PROJECT_NAME}_shared SHARED ${SOURCES})
set_target_properties(${PROJECT_NAME}_shared PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "include/${PROJECT_NAME}.h"
)
target_include_directories(${PROJECT_NAME}_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Static library
add_library(${PROJECT_NAME}_static STATIC ${SOURCES})
set_target_properties(${PROJECT_NAME}_static PROPERTIES
    OUTPUT_NAME ${PROJECT_NAME}
    PUBLIC_HEADER "include/${PROJECT_NAME}.h"
)
target_include_directories(${PROJECT_NAME}_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Installation
install(TARGETS ${PROJECT_NAME}_shared ${PROJECT_NAME}_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/${PROJECT_NAME}
)

# pkg-config file generation
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/pkg-config/${PROJECT_NAME}.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
    DESTINATION lib/pkgconfig
)
```

**pkg-config template (mylib.pc.in)**
```
prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=${prefix}
libdir=${prefix}/lib
includedir=${prefix}/include

Name: @PROJECT_NAME@
Description: My library description
Version: @PROJECT_VERSION@
Libs: -L${libdir} -l@PROJECT_NAME@
Cflags: -I${includedir}
```

---

## Advanced CMake Features

### Feature Detection

```cmake
include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckSymbolExists)

# Check for headers
check_include_file("pthread.h" HAVE_PTHREAD_H)
check_include_file("stdatomic.h" HAVE_STDATOMIC_H)

# Check for functions
check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
check_symbol_exists(pthread_setname_np "pthread.h" HAVE_PTHREAD_SETNAME_NP)

# Generate config header
configure_file(
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/include/config.h"
)
include_directories(${PROJECT_BINARY_DIR}/include)
```

### Platform-Specific Code

```cmake
if(UNIX AND NOT APPLE)
    # Linux-specific
    set(PLATFORM_SOURCES src/platform/linux.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_LINUX)
elseif(APPLE)
    # macOS-specific
    set(PLATFORM_SOURCES src/platform/macos.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_MACOS)
elseif(WIN32)
    # Windows-specific
    set(PLATFORM_SOURCES src/platform/windows.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_WINDOWS)
endif()

target_sources(${PROJECT_NAME} PRIVATE ${PLATFORM_SOURCES})
```

### External Dependencies with find_package

```cmake
# Find pthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# Find OpenSSL
find_package(OpenSSL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenSSL::SSL OpenSSL::Crypto)

# Find custom library
find_package(MyCustomLib REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE MyCustomLib::MyCustomLib)
```

### Submodules and ExternalProject

```cmake
# Git submodule
add_subdirectory(third_party/json)
target_link_libraries(${PROJECT_NAME} PRIVATE json)

# ExternalProject for external dependencies
include(ExternalProject)
ExternalProject_Add(
    libcurl
    URL https://curl.se/download/curl-8.0.0.tar.gz
    CONFIGURE_COMMAND ./configure --prefix=${CMAKE_BINARY_DIR}/deps
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install
)
```

---

## Testing Integration

### CTest Configuration

```cmake
enable_testing()

# Add test executable
add_executable(test_core tests/test_core.c ${SOURCES})
target_link_libraries(test_core PRIVATE pthread)

# Register test
add_test(NAME CoreTests COMMAND test_core)

# Test with timeout
add_test(NAME LongRunningTest COMMAND test_long)
set_tests_properties(LongRunningTest PROPERTIES TIMEOUT 60)

# Test with environment variables
add_test(NAME EnvTest COMMAND test_env)
set_tests_properties(EnvTest PROPERTIES
    ENVIRONMENT "TEST_VAR=value;ANOTHER_VAR=123"
)

# Valgrind test
find_program(VALGRIND valgrind)
if(VALGRIND)
    add_test(NAME ValgrindTest
        COMMAND ${VALGRIND} --leak-check=full --error-exitcode=1
        $<TARGET_FILE:test_core>
    )
endif()
```

**Run tests:**
```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R CoreTests  # Run specific test
ctest --test-dir build -j4           # Parallel testing
```

---

## Coverage Analysis

### GCC/Clang Coverage

**CMakeLists.txt:**
```cmake
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

if(ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()
```

**Generate coverage report:**
```bash
# Build with coverage
cmake -B build -DENABLE_COVERAGE=ON
cmake --build build

# Run tests
ctest --test-dir build

# Generate report (gcov)
cd build
gcov -r ../src/*.c

# Generate HTML report (lcov)
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

---

## Sanitizers Configuration

### AddressSanitizer (ASan)

Detects memory errors: use-after-free, buffer overflows, memory leaks.

**Makefile:**
```makefile
asan: CFLAGS += -fsanitize=address -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address
asan: clean all
```

**CMake:**
```cmake
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
```

**Usage:**
```bash
make asan
./build/program
# ASan will report errors with stack traces
```

### UndefinedBehaviorSanitizer (UBSan)

Detects undefined behavior: integer overflow, null pointer dereference, etc.

```makefile
ubsan: CFLAGS += -fsanitize=undefined
ubsan: LDFLAGS += -fsanitize=undefined
ubsan: clean all
```

### MemorySanitizer (MSan)

Detects uninitialized memory reads (Clang only).

```makefile
msan: CC = clang
msan: CFLAGS += -fsanitize=memory -fno-omit-frame-pointer
msan: LDFLAGS += -fsanitize=memory
msan: clean all
```

### ThreadSanitizer (TSan)

Detects data races in multithreaded programs.

```makefile
tsan: CFLAGS += -fsanitize=thread
tsan: LDFLAGS += -fsanitize=thread
tsan: clean all
```

**CMake all sanitizers:**
```cmake
option(SANITIZER "Enable sanitizer (address, memory, thread, undefined)" "")

if(SANITIZER STREQUAL "address")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
elseif(SANITIZER STREQUAL "memory")
    add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
    add_link_options(-fsanitize=memory)
elseif(SANITIZER STREQUAL "thread")
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
elseif(SANITIZER STREQUAL "undefined")
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()
```

**Usage:**
```bash
cmake -B build -DSANITIZER=address
cmake --build build
./build/program
```

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: C/C++ CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        compiler: [gcc, clang]
        build_type: [Debug, Release]

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y valgrind lcov

    - name: Configure
      run: |
        cmake -B build \
          -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} \
          -DCMAKE_C_COMPILER=${{ matrix.compiler }} \
          -DENABLE_COVERAGE=ON

    - name: Build
      run: cmake --build build -j

    - name: Test
      run: ctest --test-dir build --output-on-failure

    - name: Valgrind
      if: matrix.build_type == 'Debug'
      run: |
        valgrind --leak-check=full --error-exitcode=1 \
          ./build/program

    - name: Coverage
      if: matrix.compiler == 'gcc'
      run: |
        lcov --capture --directory build --output-file coverage.info
        lcov --remove coverage.info '/usr/*' --output-file coverage.info
        bash <(curl -s https://codecov.io/bash)
```

---

## Packaging and Distribution

### Debian Package (.deb)

**CMakeLists.txt:**
```cmake
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Your Name")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.27), libpthread-stubs0-dev")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
include(CPack)
```

**Build package:**
```bash
cmake -B build
cmake --build build
cpack --config build/CPackConfig.cmake
# Generates: MyApp-1.0.0-Linux.deb
```

### RPM Package

```cmake
set(CPACK_GENERATOR "RPM")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
include(CPack)
```

### Tarball

```cmake
set(CPACK_GENERATOR "TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/.git/;.gitignore")
include(CPack)
```

---

## Build Optimization

### Parallel Builds

**Makefile:**
```bash
make -j$(nproc)  # Use all CPU cores
```

**CMake:**
```bash
cmake --build build -j  # Automatic parallel build
```

### Compiler Optimization Flags

```makefile
# Debug: No optimization, debug symbols
CFLAGS_DEBUG := -O0 -g -DDEBUG

# Release: Maximum optimization
CFLAGS_RELEASE := -O3 -DNDEBUG -march=native -flto

# Size-optimized: Minimize binary size
CFLAGS_SIZE := -Os -DNDEBUG

# Link-time optimization (LTO)
LDFLAGS_RELEASE := -flto -Wl,--strip-all
```

### Precompiled Headers (CMake)

```cmake
target_precompile_headers(${PROJECT_NAME} PRIVATE
    <stdio.h>
    <stdlib.h>
    <string.h>
    <pthread.h>
)
```

---

## Summary: Build System Best Practices

**Makefile:**
- ✅ Automatic dependency generation (`-MMD -MP`)
- ✅ Parallel build support
- ✅ Separate debug/release configurations
- ✅ Sanitizer targets (asan, ubsan, tsan)
- ✅ Testing integration
- ✅ Clean separation of concerns

**CMake:**
- ✅ Modern CMake (3.15+)
- ✅ Target-based design
- ✅ Feature detection
- ✅ CTest integration
- ✅ CPack for packaging
- ✅ Cross-platform support
- ✅ External dependencies with find_package

**Testing:**
- ✅ Automated test discovery
- ✅ Valgrind integration
- ✅ Sanitizer builds
- ✅ Coverage reporting

**CI/CD:**
- ✅ GitHub Actions workflows
- ✅ Matrix builds (compilers, build types)
- ✅ Automated testing and coverage
