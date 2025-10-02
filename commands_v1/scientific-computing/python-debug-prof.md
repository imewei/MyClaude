---
title: "Python Debug Prof"
description: "Profile Python/JAX code for bottlenecks using jax.profiler and cProfile with optimization suggestions"
category: scientific-computing
subcategory: python-performance
complexity: intermediate
argument-hint: "[--jax-profiler] [--cprofile] [--memory] [--suggest-opts] [--agents=auto|python|scientific|ai|optimization|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--monitor]"
allowed-tools: "*"
model: inherit
tags: python, jax, profiling, performance, bottlenecks, optimization
dependencies: []
related: [jax-performance, jax-debug, python-type-hint, debug, optimize]
workflows: [performance-profiling, python-optimization, scientific-computing]
version: "2.1"
last-updated: "2025-09-28"
---

# Python Debug Prof

Profile Python/JAX code for bottlenecks using jax.profiler and cProfile with optimization suggestions.

## Quick Start

```bash
# Basic Python profiling
/python-debug-prof --cprofile --memory

# JAX-specific profiling
/python-debug-prof --jax-profiler --suggest-opts

# Comprehensive profiling with optimization
/python-debug-prof --cprofile --jax-profiler --memory --optimize

# Agent-enhanced profiling
/python-debug-prof --agents=auto --intelligent --monitor
```

## Usage

```bash
/python-debug-prof [options]
```

**Parameters:**
- `options` - Profiling tools, analysis depth, and optimization configuration

## Options

- `--jax-profiler`: Use JAX-specific profiling tools
- `--cprofile`: Use Python cProfile for detailed profiling
- `--memory`: Include memory profiling and analysis
- `--suggest-opts`: Provide optimization suggestions based on results
- `--agents=<agents>`: Agent selection (auto, python, scientific, ai, optimization, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with profiling intelligence
- `--intelligent`: Enable intelligent agent selection based on Python/JAX profiling analysis
- `--breakthrough`: Enable breakthrough performance analysis optimization
- `--optimize`: Apply optimization recommendations with agent coordination
- `--monitor`: Enable continuous performance monitoring with agent intelligence

## What it does

1. **JAX Profiling**: Use jax.profiler for JAX-specific performance analysis
2. **Python Profiling**: Use cProfile for general Python code analysis
3. **Memory Profiling**: Monitor memory usage and identify leaks
4. **Optimization Suggestions**: Provide actionable optimization recommendations
5. **23-Agent Performance Intelligence**: Multi-agent collaboration for optimal Python/JAX profiling
6. **Advanced Bottleneck Analysis**: Agent-driven performance bottleneck detection and resolution
7. **Intelligent Optimization Recommendations**: Agent-coordinated optimization strategy development

## 23-Agent Intelligent Python Profiling System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes Python/JAX profiling requirements, performance patterns, and optimization goals to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Python Profiling Pattern Detection → Agent Selection
- Research Computing → research-intelligence-master + python-expert + optimization-master
- Production Systems → ai-systems-architect + python-expert + systems-architect
- Scientific Computing → scientific-computing-master + python-expert + optimization-master
- ML/AI Development → neural-networks-master + python-expert + jax-pro
- Performance Engineering → optimization-master + python-expert + systems-architect
```

### Core Python Profiling Agents

#### **`python-expert`** - Python Ecosystem Performance Expert
- **Python Profiling**: Deep expertise in cProfile, line_profiler, and Python performance analysis
- **JAX Integration**: Advanced JAX profiling with jax.profiler and TensorBoard integration
- **Memory Analysis**: Python memory profiling with tracemalloc and psutil
- **Package Ecosystem**: Python performance tooling coordination and optimization
- **Cross-Platform Optimization**: Multi-platform Python performance optimization

#### **`optimization-master`** - Performance Analysis & Optimization
- **Bottleneck Detection**: Advanced performance bottleneck identification and analysis
- **Algorithm Optimization**: Computational efficiency analysis and optimization strategies
- **Memory Optimization**: Memory usage pattern analysis and optimization
- **Parallel Computing**: Multi-threading and distributed computing performance analysis
- **Benchmarking Excellence**: Performance measurement and validation strategies

#### **`scientific-computing-master`** - Scientific Python Profiling
- **Scientific Algorithms**: Scientific computing performance analysis in Python/JAX
- **Numerical Methods**: Advanced numerical method performance optimization
- **Research Applications**: Academic and research-grade performance analysis
- **Domain Integration**: Cross-domain scientific computing performance optimization
- **Mathematical Foundation**: Mathematical optimization and algorithmic performance analysis

#### **`jax-pro`** - JAX Performance Specialist
- **JAX Profiling**: Advanced JAX-specific profiling and performance optimization
- **Device Optimization**: GPU/TPU performance analysis and optimization
- **Compilation Analysis**: JAX compilation and transformation performance analysis
- **Memory Management**: JAX device memory profiling and optimization
- **Distributed Computing**: Multi-device JAX performance analysis

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Python Profiling
Automatically analyzes Python profiling requirements and selects optimal agent combinations:
- **Performance Pattern Analysis**: Detects Python/JAX performance patterns and bottlenecks
- **Complexity Assessment**: Evaluates computational complexity and optimization potential
- **Agent Matching**: Maps profiling needs to relevant agent expertise
- **Optimization Strategy**: Balances profiling depth with optimization actionability

#### **`python`** - Python-Specialized Profiling Team
- `python-expert` (Python ecosystem lead)
- `optimization-master` (performance optimization)
- `jax-pro` (JAX integration)
- `scientific-computing-master` (scientific applications)

#### **`scientific`** - Scientific Computing Profiling Team
- `scientific-computing-master` (lead)
- `python-expert` (Python implementation)
- `optimization-master` (performance optimization)
- `research-intelligence-master` (research methodology)

#### **`ai`** - AI/ML Python Profiling Team
- `neural-networks-master` (lead)
- `python-expert` (Python optimization)
- `jax-pro` (JAX performance)
- `ai-systems-architect` (production systems)

#### **`optimization`** - Performance-Focused Profiling Team
- `optimization-master` (lead)
- `python-expert` (Python performance)
- `jax-pro` (JAX optimization)
- `systems-architect` (system performance)

#### **`all`** - Complete 23-Agent Python Profiling Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough Python/JAX performance analysis.

### Advanced 23-Agent Python Profiling Examples

```bash
# Intelligent auto-selection for Python profiling
/python-debug-prof --agents=auto --intelligent --optimize

# Scientific computing profiling with specialized agents
/python-debug-prof --agents=scientific --breakthrough --orchestrate --jax-profiler

# Production system profiling optimization
/python-debug-prof --agents=optimization --optimize --monitor --memory

# AI/ML profiling development
/python-debug-prof --agents=ai --breakthrough --orchestrate --suggest-opts

# Research-grade profiling analysis
/python-debug-prof --agents=all --breakthrough --intelligent --cprofile

# Complete 23-agent profiling ecosystem
/python-debug-prof --agents=all --orchestrate --breakthrough --intelligent --monitor
```

## Example output

```python
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap, pmap
import cProfile
import pstats
import io
import time
import tracemalloc
import psutil
import os

# Initialize PRNG key
key = random.PRNGKey(0)

# ============================================================================
# 1. JAX PROFILER
# ============================================================================

def profile_with_jax_profiler(fn, *args, **kwargs):
    """Profile function using JAX profiler"""

    # Start JAX profiling
    with jax.profiler.trace("/tmp/jax_trace"):
        # Warm up
        _ = fn(*args, **kwargs)

        # Profile execution
        result = fn(*args, **kwargs)
        result.block_until_ready()

    print("JAX profiler trace saved to /tmp/jax_trace")
    print("View with: tensorboard --logdir /tmp/jax_trace")

    return result

# Advanced JAX profiling with custom annotations
def advanced_jax_profiling():
    """Advanced JAX profiling with step annotations"""

    def training_step(params, batch_x, batch_y):
        with jax.profiler.StepTraceAnnotation("forward_pass"):
            predictions = jnp.dot(batch_x, params)

        with jax.profiler.StepTraceAnnotation("loss_computation"):
            loss = jnp.mean((predictions - batch_y) ** 2)

        with jax.profiler.StepTraceAnnotation("backward_pass"):
            gradients = grad(lambda p: jnp.mean((jnp.dot(batch_x, p) - batch_y) ** 2))(params)

        with jax.profiler.StepTraceAnnotation("parameter_update"):
            new_params = params - 0.01 * gradients

        return new_params, loss

    # Generate sample data
    key, subkey = random.split(key)
    params = random.normal(subkey, (100,))
    batch_x = random.normal(key, (32, 100))
    batch_y = random.normal(key, (32,))

    # Profile with step annotations
    with jax.profiler.trace("/tmp/jax_detailed_trace"):
        for step in range(10):
            with jax.profiler.StepTraceAnnotation(f"training_step_{step}"):
                params, loss = training_step(params, batch_x, batch_y)

    print("Detailed JAX trace saved to /tmp/jax_detailed_trace")

# ============================================================================
# 2. PYTHON CPROFILE
# ============================================================================

def profile_with_cprofile(fn, *args, **kwargs):
    """Profile function using Python cProfile"""

    # Create profiler
    profiler = cProfile.Profile()

    # Profile function execution
    profiler.enable()
    result = fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    profiler.disable()

    # Analyze results
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    print("=== cProfile Results ===")
    print(s.getvalue())

    return result, stats

def detailed_cprofile_analysis(stats):
    """Detailed analysis of cProfile results"""

    print("\n=== Detailed cProfile Analysis ===")

    # Most time-consuming functions
    print("\nTop functions by cumulative time:")
    stats.sort_stats('cumulative')
    stats.print_stats(10)

    # Functions with most calls
    print("\nTop functions by call count:")
    stats.sort_stats('ncalls')
    stats.print_stats(10)

    # Functions with highest time per call
    print("\nFunctions with highest time per call:")
    stats.sort_stats('tottime')
    stats.print_stats(10)

# ============================================================================
# 3. MEMORY PROFILING
# ============================================================================

def profile_memory_usage(fn, *args, **kwargs):
    """Profile memory usage during function execution"""

    # Start memory tracing
    tracemalloc.start()

    # Get initial memory stats
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Execute function
    result = fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()

    # Get final memory stats
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_diff = final_memory - initial_memory

    # Get tracemalloc stats
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"=== Memory Profiling Results ===")
    print(f"Process memory change: {memory_diff:.2f} MB")
    print(f"Traced memory current: {current / 1024 / 1024:.2f} MB")
    print(f"Traced memory peak: {peak / 1024 / 1024:.2f} MB")

    return result

def detailed_memory_profiling(fn, *args, **kwargs):
    """Detailed memory profiling with line-by-line analysis"""

    tracemalloc.start()

    # Execute function
    result = fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()

    # Get memory statistics
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("=== Top 10 Memory Allocations ===")
    for index, stat in enumerate(top_stats[:10]):
        print(f"{index + 1}. {stat}")

    tracemalloc.stop()
    return result

# ============================================================================
# 4. JAX DEVICE MEMORY PROFILING
# ============================================================================

def profile_jax_device_memory(fn, *args, **kwargs):
    """Profile JAX device memory usage"""

    devices = jax.devices()

    # Get initial device memory
    initial_memory = {}
    for i, device in enumerate(devices):
        try:
            stats = device.memory_stats()
            initial_memory[i] = stats['bytes_in_use']
        except (AttributeError, KeyError):
            initial_memory[i] = 0

    # Execute function
    result = fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()

    # Get final device memory
    print("=== JAX Device Memory Usage ===")
    for i, device in enumerate(devices):
        try:
            stats = device.memory_stats()
            final_memory = stats['bytes_in_use']
            memory_diff = final_memory - initial_memory[i]
            print(f"Device {i}: {memory_diff / 1024 / 1024:.2f} MB change")
            print(f"  Total: {final_memory / 1024 / 1024:.2f} MB")
        except (AttributeError, KeyError):
            print(f"Device {i}: Memory stats not available")

    return result

# ============================================================================
# 5. COMPREHENSIVE PROFILING SUITE
# ============================================================================

def comprehensive_profiling(fn, *args, **kwargs):
    """Run all profiling tools on a function"""

    print("=== Comprehensive Profiling Suite ===")

    # 1. Basic timing
    print("\n1. Basic Timing:")
    start_time = time.time()
    result = fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f}s")

    # 2. cProfile analysis
    print("\n2. cProfile Analysis:")
    _, stats = profile_with_cprofile(fn, *args, **kwargs)

    # 3. Memory profiling
    print("\n3. Memory Profiling:")
    _ = profile_memory_usage(fn, *args, **kwargs)

    # 4. JAX device memory (if applicable)
    print("\n4. JAX Device Memory:")
    _ = profile_jax_device_memory(fn, *args, **kwargs)

    return result

# ============================================================================
# 6. OPTIMIZATION SUGGESTIONS
# ============================================================================

def analyze_and_suggest_optimizations(fn, *args, **kwargs):
    """Analyze function and suggest optimizations"""

    print("=== Optimization Analysis ===")

    # Profile the function
    result, stats = profile_with_cprofile(fn, *args, **kwargs)

    # Analyze cProfile stats for optimization opportunities
    suggestions = []

    # Get function statistics
    function_stats = stats.get_stats()

    for func_info, (cc, nc, tt, ct, callers) in function_stats.items():
        filename, line_num, func_name = func_info

        # High call count suggests vectorization opportunity
        if nc > 1000:
            suggestions.append(f"High call count for {func_name} ({nc} calls) - consider vectorization")

        # High time per call suggests optimization opportunity
        if nc > 0 and tt / nc > 0.001:  # More than 1ms per call
            suggestions.append(f"Slow function {func_name} ({tt/nc:.4f}s per call) - consider JIT compilation")

    # JAX-specific suggestions
    if hasattr(result, 'shape'):  # JAX array
        if result.size > 10000:
            suggestions.append("Large array detected - consider batch processing or chunking")

    print("\n=== Optimization Suggestions ===")
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    else:
        print("No obvious optimization opportunities detected")

    return result

# ============================================================================
# 7. EXAMPLE FUNCTIONS TO PROFILE
# ============================================================================

# Example 1: Inefficient function
def inefficient_function(x):
    """Example of inefficient code for profiling"""
    result = 0
    for i in range(len(x)):
        for j in range(len(x)):
            result += x[i] * x[j]  # O(n²) complexity
    return result

# Example 2: JAX function that could be optimized
def jax_function_to_optimize(x):
    """JAX function that could benefit from optimization"""
    result = x
    for _ in range(100):  # Inefficient loop
        result = jnp.sin(result) + jnp.cos(result)
    return jnp.sum(result)

# Example 3: Memory-intensive function
def memory_intensive_function(n):
    """Function that uses lots of memory"""
    arrays = []
    for i in range(n):
        arrays.append(jnp.ones((1000, 1000)))  # Creates many large arrays
    return jnp.sum(jnp.stack(arrays))

# ============================================================================
# 8. PROFILING EXAMPLES
# ============================================================================

def run_profiling_examples():
    """Run profiling examples"""

    print("=== Profiling Examples ===")

    # Generate test data
    key, subkey = random.split(key)
    test_array = random.normal(subkey, (100,))

    # Example 1: Profile inefficient function
    print("\n1. Profiling inefficient function:")
    comprehensive_profiling(inefficient_function, test_array)

    # Example 2: Profile JAX function
    print("\n2. Profiling JAX function:")
    comprehensive_profiling(jax_function_to_optimize, test_array)

    # Example 3: Get optimization suggestions
    print("\n3. Optimization suggestions:")
    analyze_and_suggest_optimizations(inefficient_function, test_array)

# ============================================================================
# 9. PERFORMANCE COMPARISON
# ============================================================================

def compare_implementations(implementations, test_data, n_runs=5):
    """Compare performance of different implementations"""

    print("=== Performance Comparison ===")

    results = {}

    for name, fn in implementations.items():
        print(f"\nProfiling {name}:")

        times = []
        for _ in range(n_runs):
            start_time = time.time()
            result = fn(test_data)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            times.append(time.time() - start_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)

        results[name] = {
            'avg_time': avg_time,
            'min_time': min_time,
            'speedup': None
        }

        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Best time: {min_time:.4f}s")

    # Calculate speedups relative to slowest implementation
    slowest_time = max(result['avg_time'] for result in results.values())

    print("\n=== Speedup Analysis ===")
    for name, result in results.items():
        speedup = slowest_time / result['avg_time']
        result['speedup'] = speedup
        print(f"{name}: {speedup:.2f}x speedup")

    return results

# Run profiling examples
print("=== Python/JAX Profiling Examples ===")
run_profiling_examples()

# Example performance comparison
print("\n=== Implementation Comparison ===")
key, subkey = random.split(key)
test_data = random.normal(subkey, (1000,))

# Compare different implementations
implementations = {
    'inefficient': inefficient_function,
    'jax_unoptimized': jax_function_to_optimize,
    'jax_optimized': jit(jax_function_to_optimize)
}

comparison_results = compare_implementations(implementations, test_data)
```

## Profiling Best Practices

### JAX Profiler
- Use `jax.profiler.trace()` for comprehensive JAX analysis
- Add `StepTraceAnnotation` for detailed step tracking
- View results with TensorBoard for visualization
- Profile both compilation and execution phases

### cProfile
- Sort by different metrics (cumulative, tottime, ncalls)
- Focus on functions with high time per call
- Look for functions with excessive call counts
- Use for identifying Python-level bottlenecks

### Memory Profiling
- Use `tracemalloc` for Python memory tracking
- Monitor JAX device memory separately
- Look for memory leaks and excessive allocations
- Profile peak memory usage, not just final usage

### Optimization Suggestions
- Vectorize functions with high call counts
- JIT compile functions with high time per call
- Use batch processing for large datasets
- Consider mixed precision for memory savings

## Common Performance Issues

### Python-Level Issues
- Excessive function calls → Vectorization
- Slow loops → JAX transformations (vmap, scan)
- Memory allocations → Pre-allocation or streaming
- Type conversions → Consistent data types

### JAX-Specific Issues
- Frequent recompilation → Static arguments
- Large intermediate arrays → Memory-efficient operations
- Device transfers → Proper device placement
- Inefficient transformations → Composition optimization

## Agent-Enhanced Python Profiling Integration Patterns

### Complete Python Performance Development Workflow
```bash
# Intelligent Python profiling development pipeline
/python-debug-prof --agents=auto --intelligent --optimize --monitor
/python-type-hint --agents=auto --intelligent --strict
/jax-performance --agents=python --optimization --gpu-accel
```

### Scientific Computing Python Profiling Pipeline
```bash
# High-performance scientific Python profiling workflow
/python-debug-prof --agents=scientific --breakthrough --orchestrate --jax-profiler
/jax-debug --agents=python --intelligent --check-tracers
/optimize --agents=python --category=memory --implement
```

### Production Python Performance Infrastructure
```bash
# Large-scale production Python profiling optimization
/python-debug-prof --agents=ai --optimize --monitor --memory
/run-all-tests --agents=python --performance --profile
/check-code-quality --agents=python --language=python --analysis=scientific
```

## Related Commands

**Python Ecosystem Development**: Enhanced Python performance development with agent intelligence
- `/jax-performance --agents=auto` - JAX-specific performance optimization techniques with agent intelligence
- `/jax-debug --agents=python` - Debug JAX compilation and transformation issues with Python agents
- `/python-type-hint --agents=auto` - Add type hints for better performance analysis with agent optimization

**Cross-Language Performance Computing**: Multi-language profiling integration
- `/julia-jit-like --agents=auto` - Compare with Julia performance optimization
- `/jax-essentials --agents=python` - JAX operations comparison with Python agents
- `/optimize --agents=python` - Python optimization with specialized agents

**Quality Assurance**: Python profiling validation and optimization
- `/generate-tests --agents=python --type=performance` - Generate Python performance tests with agent intelligence
- `/run-all-tests --agents=python --scientific` - Comprehensive Python testing with specialized agents
- `/debug --agents=auto --python` - General debugging with Python profiling integration