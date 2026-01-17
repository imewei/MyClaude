# Common Systems Programming Bugs and Fixes

## Overview

This guide catalogs the most common bugs in systems programming across C, C++, Rust, and Go, with detailed examples, debugging techniques, and prevention strategies. Each bug includes real-world examples, detection methods, and fixes.

## Category 1: Memory Bugs

### Bug 1.1: Memory Leaks

**Description**: Allocated memory never freed, causing gradual memory exhaustion.

#### Example (C):
```c
void process_data() {
    char* buffer = malloc(1024);
    if (complex_condition()) {
        return;  // BUG: buffer leaked
    }
    process(buffer);
    free(buffer);
}
```

#### Detection:
```bash
# Valgrind memcheck
valgrind --leak-check=full --show-leak-kinds=all ./program

# Expected output:
# ==12345== 1,024 bytes in 1 blocks are definitely lost
# ==12345==    at malloc (vg_replace_malloc.c:309)
# ==12345==    by process_data (program.c:42)
```

#### Fix:
```c
// Fix 1: RAII-style (C with cleanup attribute)
void process_data() {
    __attribute__((cleanup(free_buffer))) char* buffer = malloc(1024);
    if (complex_condition()) {
        return;  // buffer automatically freed
    }
    process(buffer);
}

// Fix 2: C++ RAII
void process_data() {
    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(1024);
    if (complex_condition()) {
        return;  // Automatic cleanup
    }
    process(buffer.get());
}

// Fix 3: Rust (impossible by design)
fn process_data() {
    let buffer = vec![0u8; 1024];
    if complex_condition() {
        return;  // Drop automatically called
    }
    process(&buffer);
}  // Drop called here too
```

### Bug 1.2: Use-After-Free

**Description**: Accessing memory after it has been freed.

#### Example:
```c
char* create_message() {
    char* msg = malloc(100);
    strcpy(msg, "Hello");
    free(msg);
    return msg;  // BUG: returning freed memory
}

void use_message() {
    char* msg = create_message();
    printf("%s\n", msg);  // BUG: use-after-free
}
```

#### Detection:
```bash
# AddressSanitizer (ASan)
gcc -fsanitize=address -g program.c

# Output:
# ==12345==ERROR: AddressSanitizer: heap-use-after-free
# READ of size 1 at 0x60300000eff0 thread T0
#     #0 0x4a83d7 in use_message program.c:15
```

#### Fix:
```cpp
// C++: Use smart pointers
std::unique_ptr<char[]> create_message() {
    auto msg = std::make_unique<char[]>(100);
    strcpy(msg.get(), "Hello");
    return msg;  // Ownership transferred
}

// Rust: Impossible (caught at compile time)
fn create_message() -> String {
    let msg = String::from("Hello");
    msg  // Ownership transferred
}
```

### Bug 1.3: Double Free

**Description**: Freeing the same memory twice.

#### Example:
```c
void double_free_bug() {
    char* ptr = malloc(100);
    free(ptr);
    // ... code ...
    free(ptr);  // BUG: double free
}
```

#### Detection:
```bash
# ASan detects this
# ==12345==ERROR: AddressSanitizer: attempting double-free
```

#### Fix:
```c
// Fix 1: Nullify after free
void safe_version() {
    char* ptr = malloc(100);
    free(ptr);
    ptr = NULL;  // Subsequent free(NULL) is safe
}

// Fix 2: Use ownership semantics
// C++
std::unique_ptr<char[]> ptr = std::make_unique<char[]>(100);
// Impossible to double-free

// Rust: Impossible by design (ownership system)
```

### Bug 1.4: Buffer Overflow

**Description**: Writing past the end of allocated memory.

#### Example:
```c
void overflow_bug() {
    char buffer[10];
    strcpy(buffer, "This is way too long");  // BUG: overflow
}
```

#### Detection:
```bash
# ASan stack overflow detection
gcc -fsanitize=address -g program.c

# Output:
# ==12345==ERROR: AddressSanitizer: stack-buffer-overflow
# WRITE of size 21 at 0x7ffd1234 thread T0
```

#### Fix:
```c
// Fix 1: Use bounded functions
void safe_copy() {
    char buffer[10];
    strncpy(buffer, "Long string", sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';  // Ensure null termination
}

// Fix 2: C++ std::string
void cpp_safe() {
    std::string buffer = "This is way too long";  // Safe
}

// Fix 3: Rust (compile-time bounds checking)
fn rust_safe() {
    let buffer = String::from("This is way too long");  // Safe
}
```

### Bug 1.5: Uninitialized Memory Read

**Description**: Reading memory before initializing it.

#### Example:
```c
void uninitialized_bug() {
    int x;
    if (x > 10) {  // BUG: x uninitialized
        process();
    }
}
```

#### Detection:
```bash
# Valgrind memcheck
valgrind --track-origins=yes ./program

# Output:
# ==12345== Conditional jump or move depends on uninitialised value(s)
# ==12345==    at 0x400536: uninitialized_bug (program.c:5)
```

#### Fix:
```c
// Always initialize variables
void fixed_version() {
    int x = 0;  // Initialize
    if (x > 10) {
        process();
    }
}

// C++: Use {} initialization
void cpp_safe() {
    int x{};  // Zero-initialized
}

// Rust: Impossible (compiler enforces initialization)
fn rust_safe() {
    let x: i32;  // Declared but not initialized
    // if x > 10 { }  // Compile error: use of possibly-uninitialized variable
    x = 0;  // Must initialize before use
    if x > 10 { }  // OK
}
```

## Category 2: Concurrency Bugs

### Bug 2.1: Data Race

**Description**: Concurrent unsynchronized access to shared data.

#### Example:
```c
int counter = 0;

void* increment_thread(void* arg) {
    for (int i = 0; i < 100000; ++i) {
        counter++;  // BUG: data race
    }
    return NULL;
}
```

#### Detection:
```bash
# ThreadSanitizer (TSan)
gcc -fsanitize=thread -g program.c

# Output:
# WARNING: ThreadSanitizer: data race (pid=12345)
#   Write of size 4 at 0x7b04 by thread T1:
#     #0 increment_thread program.c:5
#   Previous write of size 4 at 0x7b04 by thread T2:
#     #0 increment_thread program.c:5
```

#### Fix:
```c
// Fix 1: Mutex
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* safe_increment(void* arg) {
    for (int i = 0; i < 100000; ++i) {
        pthread_mutex_lock(&mutex);
        counter++;
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

// Fix 2: Atomic operations
#include <stdatomic.h>
_Atomic int counter = 0;

void* atomic_increment(void* arg) {
    for (int i = 0; i < 100000; ++i) {
        atomic_fetch_add(&counter, 1);
    }
    return NULL;
}
```

```cpp
// C++: std::atomic
std::atomic<int> counter{0};

void increment_thread() {
    for (int i = 0; i < 100000; ++i) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}
```

```rust
// Rust: Impossible (caught at compile time)
use std::sync::atomic::{AtomicI32, Ordering};

static COUNTER: AtomicI32 = AtomicI32::new(0);

fn increment_thread() {
    for _ in 0..100000 {
        COUNTER.fetch_add(1, Ordering::Relaxed);
    }
}
```

### Bug 2.2: Deadlock

**Description**: Circular wait where threads block each other indefinitely.

#### Example:
```c
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void* thread1(void* arg) {
    pthread_mutex_lock(&mutex1);
    sleep(1);  // Simulate work
    pthread_mutex_lock(&mutex2);  // BUG: deadlock
    // ...
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void* thread2(void* arg) {
    pthread_mutex_lock(&mutex2);
    sleep(1);
    pthread_mutex_lock(&mutex1);  // BUG: deadlock
    // ...
    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);
    return NULL;
}
```

#### Detection:
```bash
# TSan can detect potential deadlocks
gcc -fsanitize=thread -g program.c

# Helgrind (Valgrind)
valgrind --tool=helgrind ./program

# Output:
# ==12345== Thread #1: lock order violation
```

#### Fix:
```c
// Fix: Always acquire locks in same order
void* safe_thread1(void* arg) {
    pthread_mutex_lock(&mutex1);  // Always acquire in order: 1, 2
    pthread_mutex_lock(&mutex2);
    // ...
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void* safe_thread2(void* arg) {
    pthread_mutex_lock(&mutex1);  // Same order: 1, 2
    pthread_mutex_lock(&mutex2);
    // ...
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}
```

```cpp
// C++17: std::scoped_lock (deadlock-free)
void safe_transfer() {
    std::scoped_lock lock(mutex1, mutex2);  // Acquires in deadlock-free manner
    // Critical section
}
```

### Bug 2.3: Race Condition (Time-of-Check to Time-of-Use)

**Description**: State changes between check and use.

#### Example:
```c
void write_file(const char* filename) {
    if (access(filename, F_OK) == -1) {  // Check
        // File doesn't exist
        int fd = open(filename, O_CREAT | O_WRONLY, 0644);  // Use
        // BUG: Another process might create file between check and open
        write(fd, "data", 4);
        close(fd);
    }
}
```

#### Fix:
```c
// Fix: Atomic check-and-create
void safe_write_file(const char* filename) {
    // O_EXCL with O_CREAT fails if file exists (atomic)
    int fd = open(filename, O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd == -1) {
        if (errno == EEXIST) {
            // File already exists
            return;
        }
        perror("open");
        return;
    }

    write(fd, "data", 4);
    close(fd);
}
```

## Category 3: Logic Bugs

### Bug 3.1: Off-by-One Errors

**Description**: Loop or array access off by one element.

#### Example:
```c
void copy_array(int* dest, int* src, int size) {
    for (int i = 0; i <= size; ++i) {  // BUG: should be i < size
        dest[i] = src[i];  // Writes one past end
    }
}
```

#### Detection:
```bash
# ASan detects buffer overflow
gcc -fsanitize=address -g program.c
```

#### Fix:
```c
// Fix: Correct loop condition
void safe_copy(int* dest, int* src, int size) {
    for (int i = 0; i < size; ++i) {  // Correct
        dest[i] = src[i];
    }
}

// C++: Use algorithms
void cpp_safe(int* dest, int* src, int size) {
    std::copy(src, src + size, dest);
}

// Rust: Iterator-based (safe by design)
fn rust_safe(dest: &mut [i32], src: &[i32]) {
    dest.copy_from_slice(src);  // Panics if sizes don't match
}
```

### Bug 3.2: Integer Overflow

**Description**: Arithmetic operation exceeds type's range.

#### Example:
```c
void allocate_buffer(int num_elements) {
    int size = num_elements * sizeof(int);  // BUG: may overflow
    int* buffer = malloc(size);
    // ...
}
```

#### Fix:
```c
// Fix: Check for overflow
#include <limits.h>

void safe_allocate(int num_elements) {
    if (num_elements > INT_MAX / sizeof(int)) {
        // Overflow would occur
        handle_error();
        return;
    }

    int size = num_elements * sizeof(int);
    int* buffer = malloc(size);
    // ...
}

// C++: Use checked arithmetic
#include <stdexcept>

void cpp_safe_allocate(size_t num_elements) {
    if (num_elements > SIZE_MAX / sizeof(int)) {
        throw std::overflow_error("Allocation too large");
    }

    auto buffer = std::make_unique<int[]>(num_elements);
}
```

```rust
// Rust: Checked arithmetic
fn rust_safe_allocate(num_elements: usize) -> Option<Vec<i32>> {
    // checked_mul returns None on overflow
    let size = num_elements.checked_mul(std::mem::size_of::<i32>())?;
    Some(vec![0; num_elements])
}
```

### Bug 3.3: Null Pointer Dereference

**Description**: Dereferencing a null pointer.

#### Example:
```c
void process_user(struct User* user) {
    printf("Name: %s\n", user->name);  // BUG: user might be NULL
}
```

#### Fix:
```c
// Fix: Always check pointers
void safe_process_user(struct User* user) {
    if (user == NULL) {
        fprintf(stderr, "Error: NULL user\n");
        return;
    }
    printf("Name: %s\n", user->name);
}

// C++: Use references (can't be null)
void cpp_safe(const User& user) {
    std::cout << "Name: " << user.name << std::endl;
    // References can't be null
}
```

```rust
// Rust: Option type (compile-time null safety)
fn rust_safe(user: Option<&User>) {
    match user {
        Some(u) => println!("Name: {}", u.name),
        None => eprintln!("Error: No user"),
    }
}

// Or use if-let
fn rust_safe2(user: Option<&User>) {
    if let Some(u) = user {
        println!("Name: {}", u.name);
    }
}
```

## Category 4: Resource Management Bugs

### Bug 4.1: File Descriptor Leak

**Description**: File descriptors not closed, causing resource exhaustion.

#### Example:
```c
void read_config() {
    int fd = open("config.txt", O_RDONLY);
    if (fd == -1) {
        return;
    }

    if (parse_failed()) {
        return;  // BUG: fd leaked
    }

    close(fd);
}
```

#### Fix:
```c
// Fix: Use cleanup attribute (GCC/Clang)
static void close_fd(int* fd_ptr) {
    if (*fd_ptr >= 0) {
        close(*fd_ptr);
    }
}

void safe_read_config() {
    int fd __attribute__((cleanup(close_fd))) = open("config.txt", O_RDONLY);
    if (fd == -1) {
        return;
    }

    if (parse_failed()) {
        return;  // fd automatically closed
    }
}
```

```cpp
// C++: RAII wrapper
class FileDescriptor {
    int fd_;
public:
    explicit FileDescriptor(int fd) : fd_(fd) {}
    ~FileDescriptor() { if (fd_ >= 0) close(fd_); }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    int get() const { return fd_; }
};

void cpp_safe() {
    FileDescriptor fd(open("config.txt", O_RDONLY));
    if (fd.get() == -1) return;

    if (parse_failed()) return;  // Automatic cleanup
}
```

### Bug 4.2: Lock Not Released

**Description**: Mutex acquired but not released on error path.

#### Example:
```c
void process_with_lock() {
    pthread_mutex_lock(&mutex);

    if (error_condition()) {
        return;  // BUG: mutex not released
    }

    // ...
    pthread_mutex_unlock(&mutex);
}
```

#### Fix:
```c
// Fix: Always use lock guards
void safe_process() {
    pthread_mutex_lock(&mutex);

    if (error_condition()) {
        pthread_mutex_unlock(&mutex);
        return;
    }

    // ...
    pthread_mutex_unlock(&mutex);
}
```

```cpp
// C++: std::lock_guard (RAII)
void cpp_safe() {
    std::lock_guard<std::mutex> lock(mutex);

    if (error_condition()) {
        return;  // Automatic unlock
    }

    // ...
}  // Automatic unlock
```

## Debugging Techniques Summary

### Detection Tool Matrix

| Bug Type | Valgrind | ASan | TSan | MSan | UBSan |
|----------|----------|------|------|------|-------|
| Memory leak | ✓ | ✓ | - | - | - |
| Use-after-free | ✓ | ✓ | - | - | - |
| Double free | ✓ | ✓ | - | - | - |
| Buffer overflow | ✓ | ✓ | - | - | - |
| Uninitialized read | ✓ | - | - | ✓ | - |
| Data race | ✓ (Helgrind) | - | ✓ | - | - |
| Deadlock | ✓ (Helgrind) | - | ✓ | - | - |
| Integer overflow | - | - | - | - | ✓ |
| Null deref | ✓ | ✓ | - | - | ✓ |

### Compiler Flags for Bug Detection

```bash
# GCC/Clang - Enable all warnings
gcc -Wall -Wextra -Werror -Wpedantic

# AddressSanitizer
gcc -fsanitize=address -g -O1

# ThreadSanitizer
gcc -fsanitize=thread -g -O1

# UndefinedBehaviorSanitizer
gcc -fsanitize=undefined -g

# MemorySanitizer (Clang only)
clang -fsanitize=memory -g -O1

# Combine sanitizers (not ASan+TSan together)
gcc -fsanitize=address,undefined -g
```

## Prevention Strategies

### 1. Use Modern Language Features

```cpp
// Modern C++: Prefer stack allocation
std::vector<int> data(100);  // Not int* data = new int[100];

// Use smart pointers
auto ptr = std::make_unique<Data>();  // Not Data* ptr = new Data();

// Use RAII for everything
std::lock_guard<std::mutex> lock(mutex);  // Not pthread_mutex_lock
```

### 2. Enable Static Analysis

```bash
# Clang static analyzer
clang --analyze program.c

# Cppcheck
cppcheck --enable=all program.c

# Clang-tidy
clang-tidy program.cpp -- -std=c++17
```

### 3. Write Tests

```cpp
// Unit tests with sanitizers
TEST(MemoryTest, NoLeaks) {
    // Test runs with ASan enabled
    auto ptr = std::make_unique<Data>();
    process(ptr.get());
    // ASan will report any leaks
}
```

### 4. Code Review Checklist

- [ ] All allocations have corresponding deallocations
- [ ] All pointers checked for NULL before use
- [ ] All loops have correct bounds
- [ ] All locks released on all code paths
- [ ] No data shared between threads without synchronization
- [ ] Integer operations checked for overflow
- [ ] Buffer sizes verified before writes

## Summary

**Most Common Bugs:**
1. Memory leaks (all languages except Rust)
2. Data races (all languages)
3. Buffer overflows (C/C++)
4. Use-after-free (C/C++)
5. Null pointer dereference (all languages)

**Best Prevention:**
- Use Rust for memory and thread safety guarantees
- Use modern C++ (RAII, smart pointers, std::thread)
- Enable sanitizers in development and CI
- Write comprehensive tests
- Use static analysis tools
- Conduct thorough code reviews

**Detection Tools:**
- ASan: Memory errors (use-after-free, leaks, overflows)
- TSan: Concurrency bugs (data races, deadlocks)
- Valgrind: Memory errors (slower but comprehensive)
- MSan: Uninitialized reads
- UBSan: Undefined behavior

Regular use of these tools catches bugs before production deployment.
