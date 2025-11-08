# Debugging Tools Reference

> Comprehensive reference for language-specific debugging tools, IDE configurations, performance profiling, memory leak detection, and tool selection matrices

## Table of Contents

1. [Language-Specific Debugging Tools](#language-specific-debugging-tools)
2. [IDE Debugger Configurations](#ide-debugger-configurations)
3. [Performance Profiling Tools](#performance-profiling-tools)
4. [Memory Leak Detection](#memory-leak-detection)
5. [Network Debugging Tools](#network-debugging-tools)
6. [Tool Selection Matrices](#tool-selection-matrices)

---

## Language-Specific Debugging Tools

### Python Debugging

#### pdb (Python Debugger)

**Basic Usage:**

```python
import pdb

def calculate_total(items):
    total = 0
    for item in items:
        pdb.set_trace()  # Debugger stops here
        total += item['price'] * item['quantity']
    return total

# Breakpoint (Python 3.7+)
def calculate_total_v2(items):
    total = 0
    for item in items:
        breakpoint()  # Modern way
        total += item['price'] * item['quantity']
    return total
```

**Common Commands:**

```
h         # Help
l         # List source code
n         # Next line
s         # Step into function
c         # Continue execution
p var     # Print variable
pp var    # Pretty-print variable
w         # Where (stack trace)
u         # Up stack frame
d         # Down stack frame
b line    # Set breakpoint at line
cl        # Clear all breakpoints
q         # Quit debugger
```

**Advanced Usage:**

```python
# Conditional breakpoint
import pdb

def process_items(items):
    for i, item in enumerate(items):
        if item['price'] > 100:
            pdb.set_trace()  # Only break for expensive items
        process_item(item)

# Post-mortem debugging
try:
    risky_operation()
except Exception:
    import pdb
    pdb.post_mortem()  # Debug at exception point
```

---

#### ipdb (IPython Debugger)

**Installation:**

```bash
pip install ipdb
```

**Usage:**

```python
import ipdb

def complex_function(data):
    ipdb.set_trace()  # Enhanced with IPython features
    # Auto-completion, syntax highlighting, better formatting
    result = process_data(data)
    return result
```

**Features:**
- Tab completion
- Syntax highlighting
- Better object introspection
- History navigation
- Magic commands

---

#### pudb (Visual Debugger)

**Installation:**

```bash
pip install pudb
```

**Usage:**

```python
import pudb

def debug_function():
    pudb.set_trace()  # Opens visual interface
    # Full-screen, terminal-based UI
    # Source code, variables, stack, breakpoints
```

**Features:**
- Full-screen terminal UI
- Variable explorer
- Stack frame navigation
- Breakpoint management
- Watch expressions

---

#### Remote Debugging (ptvsd/debugpy)

**Installation:**

```bash
pip install debugpy
```

**Server Code:**

```python
import debugpy

# Enable debugging
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()  # Blocks until debugger connects

# Your application code
app.run()
```

**VS Code launch.json:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Remote Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

---

### Node.js Debugging

#### node inspector (Built-in)

**Usage:**

```bash
# Start with debugging enabled
node --inspect server.js

# Or with break at start
node --inspect-brk server.js

# Custom port
node --inspect=0.0.0.0:9229 server.js
```

**Chrome DevTools:**
1. Open Chrome: `chrome://inspect`
2. Click "inspect" under Remote Target
3. Full DevTools debugging interface

**Code Integration:**

```javascript
// Programmatic breakpoint
function processPayment(paymentId) {
  debugger;  // Debugger stops here if attached
  const payment = getPayment(paymentId);
  return payment;
}
```

---

#### ndb (Node Debugger)

**Installation:**

```bash
npm install -g ndb
```

**Usage:**

```bash
# Run with ndb
ndb server.js

# Opens Chromium-based debugger UI
# Features: Time-travel debugging, async stack traces
```

**Features:**
- Time-travel debugging
- Async stack traces
- Source maps support
- Black-box scripts
- Conditional breakpoints

---

### Go Debugging

#### delve

**Installation:**

```bash
go install github.com/go-delve/delve/cmd/dlv@latest
```

**Usage:**

```bash
# Debug executable
dlv debug main.go

# Debug test
dlv test

# Attach to running process
dlv attach PID

# Remote debugging
dlv debug --headless --listen=:2345 --api-version=2 main.go
```

**Commands:**

```
break main.go:10    # Set breakpoint
continue            # Continue execution
next                # Next line
step                # Step into
print var           # Print variable
locals              # Print local variables
goroutines          # List goroutines
goroutine 5         # Switch to goroutine 5
```

**Code Integration:**

```go
import "runtime"

func debugFunction() {
    // Programmatic breakpoint
    runtime.Breakpoint()

    // Get stack trace
    buf := make([]byte, 1024)
    n := runtime.Stack(buf, false)
    fmt.Printf("Stack trace:\n%s\n", buf[:n])
}
```

---

### Rust Debugging

#### rust-gdb / rust-lldb

**Usage:**

```bash
# Build with debug info
cargo build

# Debug with gdb
rust-gdb target/debug/myapp

# Debug with lldb
rust-lldb target/debug/myapp
```

**VS Code Configuration:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb",
      "request": "launch",
      "name": "Debug",
      "program": "${workspaceFolder}/target/debug/myapp",
      "args": [],
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

**Code Integration:**

```rust
use std::process;

fn debug_function() {
    // Print to stderr
    eprintln!("Debug: value = {:?}", value);

    // Panic for debugging
    if condition {
        panic!("Debug panic: {:?}", data);
    }

    // Custom debug trait
    #[derive(Debug)]
    struct MyStruct {
        field: i32,
    }
}
```

---

### Java Debugging

#### jdb (Java Debugger)

**Usage:**

```bash
# Start JVM with debugging
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005 MyApp

# Attach jdb
jdb -attach 5005
```

**Commands:**

```
stop at MyClass:123     # Breakpoint at line
stop in MyClass.method  # Breakpoint in method
run                     # Start execution
cont                    # Continue
step                    # Step into
next                    # Next line
print variable          # Print variable
where                   # Stack trace
locals                  # Local variables
```

---

## IDE Debugger Configurations

### VS Code

**Python Configuration:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Django",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/manage.py",
      "args": ["runserver", "--noreload"],
      "django": true,
      "justMyCode": false
    },
    {
      "name": "Python: Flask",
      "type": "python",
      "request": "launch",
      "module": "flask",
      "env": {
        "FLASK_APP": "app.py",
        "FLASK_DEBUG": "1"
      },
      "args": ["run", "--no-debugger", "--no-reload"],
      "jinja": true
    }
  ]
}
```

**Node.js Configuration:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Launch Program",
      "skipFiles": ["<node_internals>/**"],
      "program": "${workspaceFolder}/server.js",
      "env": {
        "NODE_ENV": "development"
      }
    },
    {
      "type": "node",
      "request": "attach",
      "name": "Attach to Process",
      "processId": "${command:PickProcess}",
      "skipFiles": ["<node_internals>/**"]
    }
  ]
}
```

**Go Configuration:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Launch Package",
      "type": "go",
      "request": "launch",
      "mode": "auto",
      "program": "${fileDirname}",
      "env": {},
      "args": []
    },
    {
      "name": "Attach to Process",
      "type": "go",
      "request": "attach",
      "mode": "local",
      "processId": 0
    }
  ]
}
```

---

### JetBrains IDEs (PyCharm, IntelliJ, GoLand)

**Breakpoint Configuration:**
- Right-click breakpoint for conditions
- "Evaluate and log" for non-breaking logging
- "Remove once hit" for one-time breakpoints
- Field watchpoints for attribute changes

**Remote Debugging Setup:**

```python
# PyCharm remote debugging
import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
```

---

## Performance Profiling Tools

### Python Profiling

#### cProfile

**Basic Usage:**

```python
import cProfile
import pstats

# Profile a function
cProfile.run('my_function()', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**Decorator:**

```python
import cProfile
import pstats
from functools import wraps

def profile(func):
    """Profile decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            return func(*args, **kwargs)
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
    return wrapper

@profile
def expensive_function():
    # Function code
    pass
```

---

#### line_profiler

**Installation:**

```bash
pip install line_profiler
```

**Usage:**

```python
from line_profiler import LineProfiler

def profile_function():
    lp = LineProfiler()
    lp.add_function(expensive_function)
    lp.enable()
    expensive_function()
    lp.disable()
    lp.print_stats()

# Or with decorator
@profile  # kernprof will use this
def expensive_function():
    total = 0
    for i in range(1000000):
        total += i
    return total
```

**Command Line:**

```bash
# Run with kernprof
kernprof -l -v script.py

# Output shows time per line
```

---

#### py-spy

**Installation:**

```bash
pip install py-spy
```

**Usage:**

```bash
# Profile running process (no code changes needed!)
py-spy top --pid 12345

# Record flame graph
py-spy record -o profile.svg --pid 12345

# Profile for 60 seconds
py-spy record -o profile.svg --duration 60 -- python script.py
```

**Features:**
- No code modifications required
- Low overhead
- Attach to running processes
- Flame graphs
- Supports native extensions

---

### Node.js Profiling

#### Built-in Profiler

**Usage:**

```bash
# CPU profiling
node --prof server.js

# Process profile
node --prof-process isolate-0x*.log > processed.txt
```

**Code Integration:**

```javascript
const { Session } = require('inspector');
const fs = require('fs');

function profileCPU(duration) {
  const session = new Session();
  session.connect();

  session.post('Profiler.enable', () => {
    session.post('Profiler.start', () => {
      setTimeout(() => {
        session.post('Profiler.stop', (err, { profile }) => {
          fs.writeFileSync('profile.cpuprofile', JSON.stringify(profile));
          session.disconnect();
        });
      }, duration);
    });
  });
}
```

---

#### clinic.js

**Installation:**

```bash
npm install -g clinic
```

**Usage:**

```bash
# Doctor - diagnose performance issues
clinic doctor -- node server.js

# Flame - CPU flame graphs
clinic flame -- node server.js

# Bubbleprof - async operations
clinic bubbleprof -- node server.js

# Heap profiling
clinic heapprofiler -- node server.js
```

---

### Go Profiling

#### pprof (Built-in)

**HTTP Server:**

```go
import (
    "net/http"
    _ "net/http/pprof"
)

func main() {
    // Profiling endpoints automatically registered
    go func() {
        http.ListenAndServe("localhost:6060", nil)
    }()

    // Your application code
    runApp()
}
```

**Access Profiles:**

```bash
# CPU profile
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# Heap profile
go tool pprof http://localhost:6060/debug/pprof/heap

# Goroutine profile
go tool pprof http://localhost:6060/debug/pprof/goroutine

# Interactive mode commands:
(pprof) top10       # Top 10 functions
(pprof) list func   # Source code with annotations
(pprof) web         # Open web UI
```

**Programmatic Profiling:**

```go
import (
    "os"
    "runtime/pprof"
)

func profileCPU() {
    f, _ := os.Create("cpu.prof")
    defer f.Close()

    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()

    // Code to profile
    doWork()
}

func profileMemory() {
    f, _ := os.Create("mem.prof")
    defer f.Close()

    // Your code
    doWork()

    pprof.WriteHeapProfile(f)
}
```

---

## Memory Leak Detection

### Python Memory Profiling

#### memory_profiler

**Installation:**

```bash
pip install memory_profiler
```

**Usage:**

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_list = [i for i in range(1000000)]
    large_dict = {i: i**2 for i in range(100000)}
    return large_list, large_dict

# Run with:
# python -m memory_profiler script.py
```

**Output:**

```
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     3     40.5 MiB     40.5 MiB           1   @profile
     4                                         def memory_intensive_function():
     5     48.3 MiB      7.8 MiB           1       large_list = [i for i in range(1000000)]
     6     56.1 MiB      7.8 MiB           1       large_dict = {i: i**2 for i in range(100000)]
     7     56.1 MiB      0.0 MiB           1       return large_list, large_dict
```

---

#### objgraph

**Installation:**

```bash
pip install objgraph
```

**Usage:**

```python
import objgraph

# Show most common objects
objgraph.show_most_common_types(limit=10)

# Find object growth
objgraph.show_growth()
# ... run your code ...
objgraph.show_growth()  # Shows what increased

# Find references keeping object alive
import gc
objgraph.show_refs([obj], filename='refs.png')

# Find back references
objgraph.show_backrefs([obj], filename='backrefs.png')
```

---

### Node.js Memory Profiling

#### heap snapshots

**Usage:**

```javascript
const v8 = require('v8');
const fs = require('fs');

function takeHeapSnapshot(filename) {
  const snapshot = v8.writeHeapSnapshot(filename);
  console.log(`Heap snapshot written to ${snapshot}`);
}

// Take snapshots at different points
takeHeapSnapshot('before.heapsnapshot');
// ... run your code ...
takeHeapSnapshot('after.heapsnapshot');

// Compare in Chrome DevTools Memory profiler
```

---

#### memwatch-next

**Installation:**

```bash
npm install @airbnb/node-memwatch
```

**Usage:**

```javascript
const memwatch = require('@airbnb/node-memwatch');

// Detect leaks
memwatch.on('leak', (info) => {
  console.error('Memory leak detected:');
  console.error(info);
});

// Monitor stats
memwatch.on('stats', (stats) => {
  console.log('Memory stats:', stats);
});

// Take heap diff
const hd = new memwatch.HeapDiff();
// ... run your code ...
const diff = hd.end();
console.log('Heap diff:', diff);
```

---

### Go Memory Profiling

**Heap Analysis:**

```go
import (
    "os"
    "runtime/pprof"
)

func analyzeMemory() {
    // Force GC to get accurate stats
    runtime.GC()

    f, _ := os.Create("heap.prof")
    defer f.Close()

    pprof.WriteHeapProfile(f)
}

// Analyze:
// go tool pprof heap.prof
// (pprof) top
// (pprof) list functionName
```

**Memory Stats:**

```go
import "runtime"

func printMemStats() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    fmt.Printf("Alloc = %v MiB", bToMb(m.Alloc))
    fmt.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
    fmt.Printf("\tSys = %v MiB", bToMb(m.Sys))
    fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

func bToMb(b uint64) uint64 {
    return b / 1024 / 1024
}
```

---

## Network Debugging Tools

### tcpdump

**Basic Usage:**

```bash
# Capture all traffic on interface
tcpdump -i eth0

# Capture HTTP traffic
tcpdump -i eth0 'tcp port 80'

# Save to file
tcpdump -i eth0 -w capture.pcap

# Read from file
tcpdump -r capture.pcap

# Filter by host
tcpdump -i eth0 host 192.168.1.100

# Capture POST requests
tcpdump -i eth0 -A -s 0 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'
```

---

### Wireshark

**Usage:**
1. Select network interface
2. Start capture
3. Apply display filters
4. Analyze packets

**Common Filters:**

```
http                          # HTTP traffic only
tcp.port == 443              # HTTPS traffic
ip.addr == 192.168.1.100     # Specific IP
http.request.method == POST  # POST requests
tcp.flags.syn == 1           # SYN packets
```

---

### curl with timing

**Detailed Timing:**

```bash
curl -w "\n\
time_namelookup:  %{time_namelookup}s\n\
time_connect:     %{time_connect}s\n\
time_appconnect:  %{time_appconnect}s\n\
time_pretransfer: %{time_pretransfer}s\n\
time_redirect:    %{time_redirect}s\n\
time_starttransfer: %{time_starttransfer}s\n\
time_total:       %{time_total}s\n" \
-o /dev/null -s https://api.example.com/endpoint
```

---

## Tool Selection Matrices

### Debugger Selection Matrix

| Language | Built-in | IDE-Integrated | Visual | Remote | Best Choice |
|----------|----------|----------------|--------|--------|-------------|
| Python | pdb | VS Code, PyCharm | pudb | debugpy | **debugpy + VS Code** |
| Node.js | node --inspect | VS Code, WebStorm | ndb | inspector | **node --inspect + Chrome** |
| Go | delve | VS Code, GoLand | - | delve headless | **delve + VS Code** |
| Rust | rust-gdb | VS Code, CLion | - | gdb remote | **rust-lldb + VS Code** |
| Java | jdb | IntelliJ IDEA | - | JDWP | **IntelliJ IDEA** |

### Profiler Selection Matrix

| Language | CPU Profiling | Memory Profiling | Real-time | Production-Safe | Best Choice |
|----------|---------------|------------------|-----------|----------------|-------------|
| Python | cProfile | memory_profiler | py-spy | py-spy | **py-spy** |
| Node.js | node --prof | heap snapshots | clinic.js | clinic.js | **clinic.js** |
| Go | pprof | pprof heap | pprof | pprof HTTP | **pprof** |
| Rust | cargo flamegraph | valgrind | - | perf | **cargo flamegraph** |
| Java | JFR | JFR | VisualVM | JFR | **Java Flight Recorder** |

### Memory Leak Detection Matrix

| Language | Tool | Heap Snapshots | Live Tracking | Visualization | Complexity |
|----------|------|----------------|---------------|---------------|------------|
| Python | objgraph | ✅ | ✅ | ✅ PNG graphs | ⭐⭐ Medium |
| Python | memory_profiler | ❌ | ✅ | ⚠️ Line-by-line | ⭐ Easy |
| Node.js | heap snapshots | ✅ | ❌ | ✅ Chrome DevTools | ⭐⭐ Medium |
| Node.js | memwatch | ❌ | ✅ | ⚠️ JSON diff | ⭐⭐ Medium |
| Go | pprof heap | ✅ | ✅ | ✅ pprof UI | ⭐⭐⭐ Complex |
| Rust | valgrind | ✅ | ✅ | ⚠️ Text output | ⭐⭐⭐⭐ Complex |

---

## Quick Reference Commands

### Common Debugging Scenarios

| Scenario | Python | Node.js | Go | Rust |
|----------|--------|---------|----|----|
| **Interactive breakpoint** | `breakpoint()` | `debugger;` | `runtime.Breakpoint()` | `println!()` + gdb |
| **Remote debugging** | `debugpy.listen()` | `--inspect` | `dlv --headless` | `gdbserver` |
| **CPU profiling** | `python -m cProfile` | `node --prof` | `go tool pprof` | `cargo flamegraph` |
| **Memory profiling** | `memory_profiler` | `v8.writeHeapSnapshot()` | `pprof.WriteHeapProfile()` | `valgrind` |
| **Attach to process** | `py-spy top --pid` | `node inspect --pid` | `dlv attach PID` | `gdb -p PID` |

---

## Tool Installation Checklist

### Development Environment Setup

**Python:**
- [ ] Install ipdb: `pip install ipdb`
- [ ] Install pudb: `pip install pudb`
- [ ] Install debugpy: `pip install debugpy`
- [ ] Install py-spy: `pip install py-spy`
- [ ] Install memory_profiler: `pip install memory_profiler`
- [ ] Install objgraph: `pip install objgraph`

**Node.js:**
- [ ] Install ndb: `npm install -g ndb`
- [ ] Install clinic.js: `npm install -g clinic`
- [ ] Install memwatch: `npm install @airbnb/node-memwatch`

**Go:**
- [ ] Install delve: `go install github.com/go-delve/delve/cmd/dlv@latest`
- [ ] Enable pprof in application
- [ ] Install graphviz for pprof visualization: `brew install graphviz`

**Rust:**
- [ ] Install rust-gdb/rust-lldb (comes with rustup)
- [ ] Install cargo-flamegraph: `cargo install flamegraph`
- [ ] Install valgrind: `brew install valgrind` (macOS) or `apt install valgrind` (Linux)

**General:**
- [ ] Install Wireshark: https://www.wireshark.org/download.html
- [ ] Configure VS Code debugging
- [ ] Set up IDE remote debugging
- [ ] Install network tools (tcpdump, netstat, etc.)
