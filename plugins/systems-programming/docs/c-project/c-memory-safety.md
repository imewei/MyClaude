# C Memory Safety and Validation

Comprehensive guide to memory safety validation with Valgrind, AddressSanitizer, and best practices for preventing memory errors in C programs.

---

## Memory Safety Principles

### Common Memory Errors

1. **Use-After-Free**: Accessing memory after it's been freed
2. **Double Free**: Freeing the same memory twice
3. **Memory Leaks**: Allocated memory never freed
4. **Buffer Overflows**: Writing beyond allocated bounds
5. **Uninitialized Reads**: Reading uninitialized memory
6. **Invalid Frees**: Freeing stack memory or invalid pointers
7. **Stack Buffer Overflow**: Writing beyond stack arrays
8. **Heap Corruption**: Overwriting heap metadata

---

## Valgrind Memory Checker

### Basic Usage

**Detect memory leaks:**
```bash
valgrind --leak-check=full ./program
```

**Comprehensive analysis:**
```bash
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=valgrind-out.txt \
         ./program arg1 arg2
```

**Flags explained:**
- `--leak-check=full`: Detailed leak reporting
- `--show-leak-kinds=all`: Show all types (definite, indirect, possible, reachable)
- `--track-origins=yes`: Track origins of uninitialized values
- `--verbose`: Detailed output
- `--log-file=FILE`: Save output to file

### Interpreting Valgrind Output

**Example: Memory Leak**
```
==12345== 100 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x108678: create_buffer (main.c:23)
==12345==    by 0x108690: main (main.c:45)
```

**Leak types:**
- **Definitely lost**: Memory leak, no references exist
- **Indirectly lost**: Leaked due to containing block being lost
- **Possibly lost**: Pointers to middle of blocks, may be intentional
- **Still reachable**: Not freed but still has references (not a leak)

**Example: Invalid Read**
```
==12345== Invalid read of size 4
==12345==    at 0x108678: process_data (main.c:67)
==12345==    by 0x108690: main (main.c:45)
==12345==  Address 0x52050a8 is 0 bytes after a block of size 40 alloc'd
```

**Example: Use-After-Free**
```
==12345== Invalid read of size 1
==12345==    at 0x108678: use_data (main.c:89)
==12345==  Address 0x52050a8 is 0 bytes inside a block of size 100 free'd
==12345==    at 0x4C30D3B: free (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x108650: cleanup (main.c:78)
```

### Valgrind Suppressions

Create suppression file for known false positives:

**valgrind.supp**
```
{
   <OpenSSL_leak>
   Memcheck:Leak
   fun:malloc
   fun:CRYPTO_malloc
   fun:sk_new
}

{
   <pthread_create>
   Memcheck:Leak
   fun:calloc
   fun:allocate_dtv
   fun:_dl_allocate_tls
   fun:allocate_stack
}
```

**Usage:**
```bash
valgrind --suppressions=valgrind.supp ./program
```

---

## AddressSanitizer (ASan)

### Compilation

**GCC/Clang:**
```bash
gcc -fsanitize=address -fno-omit-frame-pointer -g -O1 program.c -o program
```

**Key flags:**
- `-fsanitize=address`: Enable ASan
- `-fno-omit-frame-pointer`: Better stack traces
- `-g`: Debug symbols
- `-O1`: Some optimization (O0 too slow, O2 may hide bugs)

**Makefile integration:**
```makefile
asan: CFLAGS += -fsanitize=address -fno-omit-frame-pointer -g -O1
asan: LDFLAGS += -fsanitize=address
asan: clean all
```

### Runtime Options

**Environment variables:**
```bash
# Detect leaks (enabled by default on Linux)
export ASAN_OPTIONS=detect_leaks=1

# Continue after first error
export ASAN_OPTIONS=halt_on_error=0

# Custom log path
export ASAN_OPTIONS=log_path=asan.log

# Multiple options
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:log_path=asan.log
```

### ASan Output Examples

**Heap buffer overflow:**
```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000038 at pc 0x000000400b6e bp 0x7ffd12345678 sp 0x7ffd12345670
WRITE of size 4 at 0x602000000038 thread T0
    #0 0x400b6d in process_array /home/user/main.c:45
    #1 0x400c2a in main /home/user/main.c:78
    #2 0x7f1234567b96 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x21b96)

0x602000000038 is located 0 bytes to the right of 40-byte region [0x602000000010,0x602000000038)
allocated by thread T0 here:
    #0 0x7f1234567890 in malloc (/usr/lib/x86_64-linux-gnu/libasan.so.4+0xde890)
    #1 0x400a5d in create_buffer /home/user/main.c:23
    #2 0x400c15 in main /home/user/main.c:75
```

**Use-after-free:**
```
=================================================================
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x60200000eff0 at pc 0x000000400d45 bp 0x7ffd12345678 sp 0x7ffd12345670
READ of size 4 at 0x60200000eff0 thread T0
    #0 0x400d44 in use_data /home/user/main.c:89
    #1 0x400e12 in main /home/user/main.c:105

0x60200000eff0 is located 0 bytes inside of 100-byte region [0x60200000eff0,0x60200000f054)
freed by thread T0 here:
    #0 0x7f1234567890 in free (/usr/lib/x86_64-linux-gnu/libasan.so.4+0xde890)
    #1 0x400cd5 in cleanup /home/user/main.c:78
    #2 0x400e01 in main /home/user/main.c:102

previously allocated by thread T0 here:
    #0 0x7f1234567890 in malloc (/usr/lib/x86_64-linux-gnu/libasan.so.4+0xde890)
    #1 0x400c45 in allocate_data /home/user/main.c:56
    #2 0x400df5 in main /home/user/main.c:98
```

**Memory leak:**
```
=================================================================
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 100 byte(s) in 1 object(s) allocated from:
    #0 0x7f1234567890 in malloc (/usr/lib/x86_64-linux-gnu/libasan.so.4+0xde890)
    #1 0x400a5d in create_buffer /home/user/main.c:23
    #2 0x400c15 in main /home/user/main.c:75

SUMMARY: AddressSanitizer: 100 byte(s) leaked in 1 allocation(s).
```

---

## UndefinedBehaviorSanitizer (UBSan)

### Compilation

```bash
gcc -fsanitize=undefined -g program.c -o program
```

**Detects:**
- Integer overflow/underflow
- Division by zero
- Null pointer dereference
- Unaligned pointer access
- Shift by negative or too large amount
- Invalid casts
- Out-of-bounds array access

**Runtime options:**
```bash
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0
```

**Example output:**
```
main.c:45:12: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
    #0 0x400a5d in calculate /home/user/main.c:45
    #1 0x400c15 in main /home/user/main.c:78
```

---

## MemorySanitizer (MSan)

Detects uninitialized memory reads.

**Compilation (Clang only):**
```bash
clang -fsanitize=memory -fno-omit-frame-pointer -g program.c -o program
```

**Example output:**
```
==12345==WARNING: MemorySanitizer: use-of-uninitialized-value
    #0 0x400a5d in process_value /home/user/main.c:56
    #1 0x400c15 in main /home/user/main.c:78

  Uninitialized value was created by an allocation of 'buffer' in the stack frame of function 'main'
    #0 0x400bf0 in main /home/user/main.c:70
```

---

## ThreadSanitizer (TSan)

Detects data races in multithreaded programs.

**Compilation:**
```bash
gcc -fsanitize=thread -g -O1 program.c -o program -lpthread
```

**Example output:**
```
==================
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 4 at 0x7b0400000010 by thread T2:
    #0 0x400a5d in worker_thread /home/user/main.c:45
    #1 0x7f1234567890 in start_thread (/lib/x86_64-linux-gnu/libpthread.so.0+0x7890)

  Previous write of size 4 at 0x7b0400000010 by thread T1:
    #0 0x400a5d in worker_thread /home/user/main.c:45
    #1 0x7f1234567890 in start_thread (/lib/x86_64-linux-gnu/libpthread.so.0+0x7890)

  Location is global 'shared_counter' of size 4 at 0x7b0400000010 (main+0x000000000010)
```

---

## Memory Safety Best Practices

### 1. Always Initialize Variables

**Bad:**
```c
int* ptr;
*ptr = 10;  // Undefined behavior
```

**Good:**
```c
int* ptr = NULL;
if (ptr == NULL) {
    ptr = malloc(sizeof(int));
    if (ptr) *ptr = 10;
}
```

### 2. Check malloc() Return Values

**Bad:**
```c
int* data = malloc(1000 * sizeof(int));
data[0] = 42;  // Crash if malloc failed
```

**Good:**
```c
int* data = malloc(1000 * sizeof(int));
if (data == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return -1;
}
data[0] = 42;
```

### 3. Free Memory and Set to NULL

**Bad:**
```c
free(ptr);
*ptr = 10;  // Use-after-free
```

**Good:**
```c
free(ptr);
ptr = NULL;
```

### 4. Use Safe String Functions

**Bad:**
```c
char buffer[10];
strcpy(buffer, user_input);  // Buffer overflow risk
```

**Good:**
```c
char buffer[10];
strncpy(buffer, user_input, sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';  // Ensure null termination
```

**Better: Use snprintf**
```c
char buffer[10];
snprintf(buffer, sizeof(buffer), "%s", user_input);
```

### 5. Bounds Checking

**Bad:**
```c
int array[10];
for (int i = 0; i <= 10; i++) {  // Off-by-one error
    array[i] = i;
}
```

**Good:**
```c
int array[10];
for (int i = 0; i < 10; i++) {
    array[i] = i;
}
```

### 6. Avoid Dangling Pointers

**Bad:**
```c
int* get_value(void) {
    int value = 42;
    return &value;  // Returning address of stack variable
}
```

**Good:**
```c
int* get_value(void) {
    int* value = malloc(sizeof(int));
    if (value) *value = 42;
    return value;
}
```

### 7. Use RAII-Style Cleanup Pattern

**Goto cleanup pattern:**
```c
int process_file(const char* filename) {
    int err = 0;
    FILE* fp = NULL;
    char* buffer = NULL;

    fp = fopen(filename, "r");
    if (!fp) {
        err = -1;
        goto cleanup;
    }

    buffer = malloc(1024);
    if (!buffer) {
        err = -1;
        goto cleanup;
    }

    // Process file...

cleanup:
    free(buffer);
    if (fp) fclose(fp);
    return err;
}
```

### 8. Thread-Safe Memory Pools

**Example: Lock-Protected Pool**
```c
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    void** free_list;
    size_t free_count;
    size_t capacity;
    size_t object_size;
    pthread_mutex_t lock;
} pool_t;

pool_t* pool_create(size_t object_size, size_t capacity) {
    pool_t* pool = malloc(sizeof(pool_t));
    if (!pool) return NULL;

    pool->free_list = malloc(capacity * sizeof(void*));
    if (!pool->free_list) {
        free(pool);
        return NULL;
    }

    pool->object_size = object_size;
    pool->capacity = capacity;
    pool->free_count = 0;
    pthread_mutex_init(&pool->lock, NULL);

    return pool;
}

void* pool_alloc(pool_t* pool) {
    void* obj = NULL;

    pthread_mutex_lock(&pool->lock);
    if (pool->free_count > 0) {
        obj = pool->free_list[--pool->free_count];
    } else {
        obj = malloc(pool->object_size);
    }
    pthread_mutex_unlock(&pool->lock);

    return obj;
}

void pool_free(pool_t* pool, void* ptr) {
    if (!ptr) return;

    pthread_mutex_lock(&pool->lock);
    if (pool->free_count < pool->capacity) {
        pool->free_list[pool->free_count++] = ptr;
    } else {
        free(ptr);  // Pool full, free directly
    }
    pthread_mutex_unlock(&pool->lock);
}

void pool_destroy(pool_t* pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);
    for (size_t i = 0; i < pool->free_count; i++) {
        free(pool->free_list[i]);
    }
    free(pool->free_list);
    pthread_mutex_unlock(&pool->lock);

    pthread_mutex_destroy(&pool->lock);
    free(pool);
}
```

---

## Static Analysis Tools

### Clang Static Analyzer

```bash
# Analyze single file
clang --analyze program.c

# With additional checks
clang --analyze -Xanalyzer -analyzer-checker=core,unix,security program.c
```

### Cppcheck

```bash
# Install
sudo apt-get install cppcheck

# Run analysis
cppcheck --enable=all --inconclusive --std=c11 src/

# Generate HTML report
cppcheck --enable=all --xml src/ 2> report.xml
cppcheck-htmlreport --file=report.xml --report-dir=html_report
```

### Splint

```bash
# Install
sudo apt-get install splint

# Run analysis
splint +posixlib -weak program.c
```

---

## Debugging Memory Issues

### GDB with ASan

```bash
# Compile with ASan and debug symbols
gcc -fsanitize=address -g -O0 program.c -o program

# Run under GDB
gdb ./program

# Set breakpoint on ASan error handler
(gdb) break __asan_report_error
(gdb) run

# When ASan detects error, GDB stops
(gdb) backtrace
(gdb) info locals
```

### Heap Profiling with Massif

```bash
# Run with massif
valgrind --tool=massif ./program

# Analyze output
ms_print massif.out.12345

# Visualize with massif-visualizer
massif-visualizer massif.out.12345
```

**Massif output interpretation:**
```
    KB
19.71^                                               #
     |                                               #
     |                                               #
     |                                               #
     |                                      @@@@@@@@@#
     |                                  @@@:#        #
     |                              @@@@:  #        #
     |                          @@@@@   :  #        #
     |                      @@@@:   :  :  #        #
     |                  @@@@@   :   :  :  #        #
     |              @@@@:   :   :   :  :  #        #
     |          @@@@@   :   :   :   :  :  #        #
     |      @@@@:   :   :   :   :   :  :  #        #
   0 +----------------------------------------------------------------------->Mi
     0                                                                   113.1
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Free

**Problem:**
```c
void process_data(void) {
    char* buffer = malloc(1024);
    // Process data...
    // Missing free(buffer)
}
```

**Solution: RAII-style cleanup**
```c
void process_data(void) {
    char* buffer = malloc(1024);
    if (!buffer) return;

    // Process data...

    free(buffer);
}
```

### Pitfall 2: Double Free

**Problem:**
```c
free(ptr);
// ... code ...
free(ptr);  // Double free
```

**Solution:**
```c
free(ptr);
ptr = NULL;
// ... code ...
free(ptr);  // Safe: free(NULL) is a no-op
```

### Pitfall 3: Buffer Overflow in sprintf

**Problem:**
```c
char buffer[10];
sprintf(buffer, "User: %s", username);  // Overflow if username > 4 chars
```

**Solution:**
```c
char buffer[10];
snprintf(buffer, sizeof(buffer), "User: %s", username);
```

### Pitfall 4: Uninitialized Variable

**Problem:**
```c
int calculate(void) {
    int result;
    // ... conditional code that may not set result ...
    return result;  // May be uninitialized
}
```

**Solution:**
```c
int calculate(void) {
    int result = 0;  // Always initialize
    // ... code ...
    return result;
}
```

---

## Continuous Memory Validation

### CI/CD Integration

**GitHub Actions:**
```yaml
name: Memory Safety

on: [push, pull_request]

jobs:
  asan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build with ASan
        run: |
          gcc -fsanitize=address -g program.c -o program
      - name: Run tests
        run: ./program

  valgrind:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Valgrind
        run: sudo apt-get install -y valgrind
      - name: Build
        run: gcc -g program.c -o program
      - name: Run Valgrind
        run: |
          valgrind --leak-check=full --error-exitcode=1 ./program
```

---

## Summary: Memory Safety Checklist

**Development:**
- [ ] Always initialize variables
- [ ] Check all malloc() return values
- [ ] Free memory and set pointers to NULL
- [ ] Use bounds-checked functions (snprintf, strncpy)
- [ ] Avoid returning stack addresses
- [ ] Use goto cleanup pattern for error handling

**Testing:**
- [ ] Run Valgrind on all tests
- [ ] Build with ASan for development
- [ ] Test with UBSan for undefined behavior
- [ ] Use TSan for multithreaded code
- [ ] Run static analysis (clang analyzer, cppcheck)
- [ ] Profile memory usage with massif

**CI/CD:**
- [ ] Automated Valgrind checks
- [ ] ASan builds in CI pipeline
- [ ] Memory leak detection on every commit
- [ ] Coverage analysis with sanitizers
