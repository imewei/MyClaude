# RAII and Resource Management Patterns

## Overview

RAII (Resource Acquisition Is Initialization) is a C++ idiom where resource lifetime is bound to object lifetime. When the object is destroyed, resources are automatically released. This pattern is fundamental to writing exception-safe, leak-free code in C++ and has inspired similar patterns in Rust and other languages.

## Core Principle

**"Resource acquisition IS initialization"**

```cpp
{
    Resource r("file.txt");  // Acquire resource in constructor
    use(r);
    // ...
} // Automatic cleanup in destructor (even if exception thrown)
```

## Pattern 1: Basic RAII Wrapper

### File Handle Wrapper

```cpp
// file_handle.hpp
#pragma once

#include <cstdio>
#include <string>
#include <stdexcept>

class FileHandle {
private:
    FILE* file_ = nullptr;
    std::string filename_;

public:
    // Constructor acquires resource
    explicit FileHandle(const std::string& filename, const char* mode = "r")
        : filename_(filename)
    {
        file_ = std::fopen(filename.c_str(), mode);
        if (!file_) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }

    // Destructor releases resource
    ~FileHandle() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    // Delete copy operations (file handle is unique)
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // Move operations transfer ownership
    FileHandle(FileHandle&& other) noexcept
        : file_(other.file_), filename_(std::move(other.filename_))
    {
        other.file_ = nullptr;
    }

    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            // Close current file
            if (file_) {
                std::fclose(file_);
            }

            // Transfer ownership
            file_ = other.file_;
            filename_ = std::move(other.filename_);
            other.file_ = nullptr;
        }
        return *this;
    }

    // Access underlying resource
    FILE* get() const { return file_; }

    // Explicit release (transfers ownership to caller)
    FILE* release() {
        FILE* result = file_;
        file_ = nullptr;
        return result;
    }

    // Query state
    bool is_open() const { return file_ != nullptr; }
    const std::string& filename() const { return filename_; }
};

// Usage:
void process_file(const std::string& filename) {
    FileHandle file(filename, "r");  // Automatically closed on exception or return

    char buffer[1024];
    while (std::fgets(buffer, sizeof(buffer), file.get())) {
        process(buffer);
        // Even if process() throws, file is automatically closed
    }
}
```

### Memory Buffer Wrapper

```cpp
// buffer.hpp
#pragma once

#include <cstdlib>
#include <cstring>
#include <algorithm>

class Buffer {
private:
    char* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;

public:
    // Allocate buffer
    explicit Buffer(size_t capacity) : size_(0), capacity_(capacity) {
        if (capacity > 0) {
            data_ = static_cast<char*>(std::malloc(capacity));
            if (!data_) {
                throw std::bad_alloc();
            }
        }
    }

    // Destructor frees memory
    ~Buffer() {
        std::free(data_);
        data_ = nullptr;
    }

    // Non-copyable
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Movable
    Buffer(Buffer&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            std::free(data_);

            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;

            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    // Access
    char* data() { return data_; }
    const char* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

    // Modify size (does not reallocate)
    void resize(size_t new_size) {
        if (new_size > capacity_) {
            throw std::length_error("Size exceeds capacity");
        }
        size_ = new_size;
    }

    // Clear content (keeps capacity)
    void clear() { size_ = 0; }
};
```

## Pattern 2: Smart Pointers (C++ Standard Library)

### std::unique_ptr - Unique Ownership

```cpp
#include <memory>
#include <vector>

// Basic usage
std::unique_ptr<int> ptr = std::make_unique<int>(42);
std::cout << *ptr << std::endl;  // 42
// Automatically deleted when ptr goes out of scope

// Array specialization
std::unique_ptr<int[]> array = std::make_unique<int[]>(100);
array[0] = 10;

// Custom deleter
struct FileDeleter {
    void operator()(FILE* f) const {
        if (f) std::fclose(f);
    }
};

std::unique_ptr<FILE, FileDeleter> file(
    std::fopen("data.txt", "r"),
    FileDeleter{}
);

// Lambda deleter
auto lambda_deleter = [](int* p) {
    std::cout << "Deleting " << *p << std::endl;
    delete p;
};

std::unique_ptr<int, decltype(lambda_deleter)> ptr2(
    new int(100),
    lambda_deleter
);

// Transfer ownership
std::unique_ptr<int> ptr3 = std::move(ptr);  // ptr is now nullptr
```

### std::shared_ptr - Shared Ownership

```cpp
#include <memory>

// Reference counted pointer
std::shared_ptr<int> sp1 = std::make_shared<int>(42);
std::shared_ptr<int> sp2 = sp1;  // Reference count: 2

std::cout << "Value: " << *sp1 << std::endl;
std::cout << "Use count: " << sp1.use_count() << std::endl;  // 2

sp1.reset();  // Reference count: 1
// sp2 still valid

sp2.reset();  // Reference count: 0, memory deleted

// Custom deleter
std::shared_ptr<FILE> file(
    std::fopen("data.txt", "r"),
    [](FILE* f) { if (f) std::fclose(f); }
);

// Aliasing constructor (shared ownership, different pointer)
struct Data {
    int x, y, z;
};

std::shared_ptr<Data> data = std::make_shared<Data>();
std::shared_ptr<int> x_ptr(data, &data->x);  // Points to x, but extends data lifetime
```

### std::weak_ptr - Non-Owning Reference

```cpp
#include <memory>

// Solve circular reference problem
class Node {
public:
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    std::weak_ptr<Node> parent;  // Weak to avoid circular reference

    ~Node() {
        std::cout << "Node destroyed" << std::endl;
    }
};

void build_tree() {
    auto root = std::make_shared<Node>();
    root->left = std::make_shared<Node>();
    root->left->parent = root;  // Weak pointer, no circular reference

    // When root goes out of scope, entire tree is properly deleted
}

// Check if referenced object still exists
std::weak_ptr<int> weak;
{
    auto shared = std::make_shared<int>(42);
    weak = shared;

    if (auto locked = weak.lock()) {  // Convert to shared_ptr
        std::cout << *locked << std::endl;  // 42
    }
}

// Object destroyed
if (auto locked = weak.lock()) {
    // Won't execute, object is gone
} else {
    std::cout << "Object no longer exists" << std::endl;
}
```

## Pattern 3: Lock Guards

### std::lock_guard - Basic Scoped Locking

```cpp
#include <mutex>
#include <thread>

std::mutex mtx;
int shared_counter = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mtx);  // Acquires lock
    ++shared_counter;
    // Lock automatically released at end of scope
}

void process() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        // Critical section
        ++shared_counter;
    }  // Lock released here

    // Non-critical section
    do_other_work();
}
```

### std::unique_lock - Flexible Locking

```cpp
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void wait_for_signal() {
    std::unique_lock<std::mutex> lock(mtx);

    // Wait releases lock until notified
    cv.wait(lock, []{ return ready; });

    // Lock reacquired here
    process_data();
}

void send_signal() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        ready = true;
    }  // Release lock before notify

    cv.notify_one();
}

// Deferred locking
void deferred_lock() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);
    // Lock not acquired yet

    if (try_something()) {
        lock.lock();  // Explicitly acquire
        // Critical section
    }
}

// Manual lock/unlock
void manual_control() {
    std::unique_lock<std::mutex> lock(mtx);

    critical_section1();

    lock.unlock();  // Release early
    non_critical_section();
    lock.lock();  // Reacquire

    critical_section2();
}
```

### std::scoped_lock - Multiple Locks (Deadlock-Free)

```cpp
#include <mutex>

std::mutex mtx1, mtx2;

void transfer(int& from, int& to, int amount) {
    // Acquires both locks in deadlock-free manner
    std::scoped_lock lock(mtx1, mtx2);

    from -= amount;
    to += amount;
}

// Equivalent to:
void transfer_manual(int& from, int& to, int amount) {
    std::lock(mtx1, mtx2);  // Deadlock-free acquisition

    std::lock_guard<std::mutex> lock1(mtx1, std::adopt_lock);
    std::lock_guard<std::mutex> lock2(mtx2, std::adopt_lock);

    from -= amount;
    to += amount;
}
```

## Pattern 4: Custom RAII Wrappers

### Scope Guard

```cpp
// scope_guard.hpp
#pragma once

#include <utility>
#include <functional>

template<typename Func>
class ScopeGuard {
private:
    Func cleanup_;
    bool active_ = true;

public:
    explicit ScopeGuard(Func&& cleanup)
        : cleanup_(std::forward<Func>(cleanup)) {}

    ~ScopeGuard() {
        if (active_) {
            cleanup_();
        }
    }

    // Non-copyable
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    // Movable
    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup_(std::move(other.cleanup_)), active_(other.active_)
    {
        other.active_ = false;
    }

    // Dismiss guard (don't run cleanup)
    void dismiss() { active_ = false; }
};

// Helper function
template<typename Func>
ScopeGuard<Func> make_scope_guard(Func&& func) {
    return ScopeGuard<Func>(std::forward<Func>(func));
}

// Usage:
void complex_function() {
    FILE* f = fopen("file.txt", "r");
    auto guard = make_scope_guard([f]{ if (f) fclose(f); });

    // Complex logic with multiple exit points
    if (condition1) return;  // File automatically closed
    if (condition2) throw exception();  // File closed
    process(f);
    // File closed at end
}
```

### Transaction Guard

```cpp
// transaction.hpp
#pragma once

#include <functional>

class Transaction {
private:
    std::function<void()> commit_;
    std::function<void()> rollback_;
    bool committed_ = false;

public:
    Transaction(std::function<void()> commit, std::function<void()> rollback)
        : commit_(std::move(commit)), rollback_(std::move(rollback)) {}

    ~Transaction() {
        if (!committed_) {
            rollback_();
        }
    }

    void commit() {
        if (!committed_) {
            commit_();
            committed_ = true;
        }
    }
};

// Usage:
void database_operation() {
    begin_transaction();

    Transaction tx(
        []{ db_commit(); },
        []{ db_rollback(); }
    );

    insert_record(1);
    insert_record(2);

    if (validate()) {
        tx.commit();  // Commit changes
    }
    // Otherwise, automatic rollback in destructor
}
```

### Thread Pool RAII Wrapper

```cpp
// thread_pool_guard.hpp
#pragma once

#include <thread>
#include <vector>

class ThreadPoolGuard {
private:
    std::vector<std::thread>& threads_;

public:
    explicit ThreadPoolGuard(std::vector<std::thread>& threads)
        : threads_(threads) {}

    ~ThreadPoolGuard() {
        // Join all threads on destruction
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    // Non-copyable, non-movable
    ThreadPoolGuard(const ThreadPoolGuard&) = delete;
    ThreadPoolGuard& operator=(const ThreadPoolGuard&) = delete;
};

// Usage:
void run_parallel_tasks() {
    std::vector<std::thread> threads;
    ThreadPoolGuard guard(threads);  // Ensures all threads joined

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([i]{ process_task(i); });
    }

    // All threads automatically joined when guard destroyed
}
```

## Rust Ownership and RAII

### Basic Ownership

```rust
// Ownership and Drop
struct Resource {
    name: String,
}

impl Resource {
    fn new(name: &str) -> Self {
        println!("Acquiring resource: {}", name);
        Resource { name: name.to_string() }
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        println!("Releasing resource: {}", self.name);
    }
}

fn use_resource() {
    let r = Resource::new("file.txt");
    // Use r
} // r.drop() called automatically
```

### Smart Pointers

```rust
use std::rc::Rc;
use std::sync::Arc;
use std::cell::RefCell;

// Box - Owned heap allocation
let b = Box::new(42);

// Rc - Reference counted (single-threaded)
let rc1 = Rc::new(42);
let rc2 = Rc::clone(&rc1);  // Reference count: 2

// Arc - Atomic reference counted (thread-safe)
let arc1 = Arc::new(42);
let arc2 = Arc::clone(&arc1);

// RefCell - Interior mutability with runtime borrow checking
let cell = RefCell::new(42);
*cell.borrow_mut() = 100;

// Combining Rc and RefCell
let shared_mutable = Rc::new(RefCell::new(vec![1, 2, 3]));
shared_mutable.borrow_mut().push(4);
```

### Mutex and RwLock

```rust
use std::sync::{Mutex, RwLock};

// Mutex automatically releases lock when guard dropped
let data = Mutex::new(0);
{
    let mut guard = data.lock().unwrap();
    *guard += 1;
}  // Lock released

// RwLock for reader-writer access
let data = RwLock::new(vec![1, 2, 3]);

// Multiple readers
{
    let r1 = data.read().unwrap();
    let r2 = data.read().unwrap();
    println!("{:?}", *r1);
}

// Single writer
{
    let mut w = data.write().unwrap();
    w.push(4);
}
```

## Pattern 5: Exception Safety Guarantees

### Basic Exception Safety

```cpp
// Basic guarantee: No resource leaks, invariants preserved
class BasicSafe {
    std::vector<int> data_;

public:
    void operation() {
        data_.push_back(1);  // May throw
        // If exception, data_ still valid (basic guarantee)
    }
};
```

### Strong Exception Safety

```cpp
// Strong guarantee: Operation succeeds or state unchanged
class StrongSafe {
    std::vector<int> data_;

public:
    void operation() {
        std::vector<int> temp = data_;  // Copy
        temp.push_back(1);  // May throw

        // If we reach here, no exception
        data_ = std::move(temp);  // noexcept swap
    }
};
```

### No-throw Guarantee

```cpp
// No-throw guarantee: Operation never throws
class NoThrow {
    std::vector<int> data_;

public:
    void swap(NoThrow& other) noexcept {
        data_.swap(other.data_);  // Never throws
    }

    NoThrow(NoThrow&& other) noexcept
        : data_(std::move(other.data_)) {}
};
```

## Best Practices

### 1. Always Use RAII for Resource Management

```cpp
// Bad
void bad_function() {
    FILE* f = fopen("file.txt", "r");
    process(f);
    fclose(f);  // Leaks if process() throws
}

// Good
void good_function() {
    FileHandle f("file.txt", "r");
    process(f.get());
    // Automatically closed even if process() throws
}
```

### 2. Prefer std::make_unique and std::make_shared

```cpp
// Avoid
std::unique_ptr<Widget> w(new Widget());

// Prefer (exception-safe)
auto w = std::make_unique<Widget>();

// Reason: If constructor throws, make_unique properly handles cleanup
```

### 3. Use Appropriate Ownership Semantics

```cpp
// Unique ownership
std::unique_ptr<Resource> resource;

// Shared ownership
std::shared_ptr<Resource> shared_resource;

// Weak reference (no ownership)
std::weak_ptr<Resource> weak_ref;

// Borrowed reference (no ownership, not null)
Resource& borrowed;

// Optional borrowed reference
Resource* maybe_borrowed;
```

### 4. Avoid Manual Memory Management

```cpp
// Bad
Widget* w = new Widget();
// ... complex logic ...
delete w;  // Easy to forget or miss on exception path

// Good
auto w = std::make_unique<Widget>();
// Automatic cleanup
```

## Common Pitfalls

### 1. Returning References to Temporaries

```cpp
// Bad - returns reference to destroyed object
const std::string& bad_function() {
    std::string temp = "temporary";
    return temp;  // Dangling reference!
}

// Good - return by value (move optimization)
std::string good_function() {
    std::string result = "value";
    return result;  // Move or RVO
}
```

### 2. Circular References with shared_ptr

```cpp
// Bad - memory leak
class Node {
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev;  // Circular reference!
};

// Good - use weak_ptr
class Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // Breaks cycle
};
```

### 3. Exception During Constructor

```cpp
class Resource {
    int* data_;
    FILE* file_;

public:
    // Bad - leaks data_ if fopen throws
    Resource() {
        data_ = new int[100];
        file_ = fopen("file.txt", "r");
        if (!file_) throw std::runtime_error("open failed");
    }

    // Good - use RAII members
    std::unique_ptr<int[]> data_;
    std::unique_ptr<FILE, FileDeleter> file_;

    Resource()
        : data_(std::make_unique<int[]>(100))
        , file_(fopen("file.txt", "r"), FileDeleter{})
    {
        if (!file_) throw std::runtime_error("open failed");
    }
};
```

## Summary

RAII is the cornerstone of safe systems programming in C++ and Rust:

**Key Principles:**
- Acquire resources in constructors
- Release resources in destructors
- Leverage automatic destruction for cleanup
- Make types non-copyable or movable as appropriate
- Use standard library smart pointers

**Benefits:**
- Automatic resource cleanup
- Exception safety
- Prevention of resource leaks
- Clear ownership semantics
- Reduced cognitive load

**Rust Takes It Further:**
- Compile-time ownership verification
- Borrow checker prevents use-after-free
- No data races possible in safe code
- Zero-cost abstractions

RAII transforms error-prone manual resource management into automatic, safe, and efficient patterns that are enforced by the type system.
