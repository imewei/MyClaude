title: "JAX debug"
---
description: Debug JAX code with specialized tools, disable JIT, and check for tracer leaks
subcategory: jax-ecosystem
complexity: intermediate
category: jax-core
argument-hint: "[--disable-jit] [--check-tracers] [--print-values] [--agents=auto|jax|scientific|ai|debug|all] [--orchestrate] [--intelligent] [--breakthrough] [--monitor] [--analyze]"
allowed-tools: "*"
model: inherit
---

# JAX Debug

Debug JAX code with specialized tools, disable JIT compilation, and diagnose common JAX issues.

```bash
/jax-debug [--disable-jit] [--check-tracers] [--print-values] [--agents=auto|jax|scientific|ai|debug|all] [--orchestrate] [--intelligent] [--breakthrough] [--monitor] [--analyze]
```

## Options

- `--disable-jit`: Disable JIT compilation for debugging
- `--check-tracers`: Check for tracer leaks and abstract value errors
- `--print-values`: Add debug prints that work with JAX transformations
- `--agents=<agents>`: Agent selection (auto, jax, scientific, ai, debug, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with debugging intelligence
- `--intelligent`: Enable intelligent agent selection based on error analysis
- `--breakthrough`: Enable breakthrough debugging techniques and error resolution
- `--monitor`: Enable real-time debugging monitoring with agent intelligence
- `--analyze`: Deep analysis of debugging patterns and error root causes

## What it does

1. Disable JIT compilation for debugging
2. Insert debug prints that work with JAX transformations
3. Check for tracer leaks and abstract value errors
4. Diagnose shape and dtype issues
5. **23-Agent Debugging Intelligence**: Multi-agent collaboration for complex error diagnosis
6. **Advanced Error Analysis**: Agent-driven root cause analysis and resolution strategies
7. **Real-Time Monitoring**: Agent-coordinated debugging monitoring and adaptive analysis

## 23-Agent Intelligent Debugging System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes error patterns, debugging complexity, and code characteristics to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Error Type Detection → Agent Selection
- Tracer Leak Errors → jax-pro + code-quality-master + systems-architect
- Performance Issues → jax-pro + systems-architect + ai-systems-architect
- Memory Problems → systems-architect + jax-pro + neural-networks-master
- Scientific Computing Errors → scientific-computing-master + jax-pro + research-intelligence-master
- Complex Debugging Scenarios → multi-agent-orchestrator + code-quality-master + jax-pro
```

### Core JAX Debugging Agents

#### **`jax-pro`** - JAX Debugging & Error Resolution Expert
- **JAX Ecosystem Mastery**: Deep expertise in JAX debugging tools and error patterns
- **Tracer Leak Diagnosis**: Advanced diagnosis and resolution of tracer leak issues
- **Transformation Debugging**: Expert debugging of JIT, grad, vmap, pmap transformations
- **Performance Debugging**: JAX-specific performance issue identification and resolution
- **Device Debugging**: Multi-device debugging and device placement issue resolution

#### **`code-quality-master`** - Code Quality & Debugging Methodology
- **Debugging Strategy**: Systematic debugging methodologies and best practices
- **Error Pattern Recognition**: Advanced error pattern analysis and classification
- **Code Analysis**: Static and dynamic code analysis for bug identification
- **Testing Integration**: Debugging integration with comprehensive testing strategies
- **Quality Assurance**: Debugging workflow integration with quality gates

#### **`systems-architect`** - System-Level Debugging & Performance
- **System Debugging**: System-level debugging and infrastructure issue resolution
- **Resource Debugging**: Memory, CPU, and GPU resource debugging and optimization
- **Performance Analysis**: System performance debugging and bottleneck identification
- **Architecture Debugging**: Debugging complex system architectures and integrations
- **Monitoring Systems**: Real-time debugging monitoring and alerting systems

#### **`research-intelligence-master`** - Advanced Debugging Research & Innovation
- **Research Debugging**: Cutting-edge debugging techniques and methodologies
- **Error Analysis**: Advanced error analysis and pattern recognition research
- **Innovation Synthesis**: Novel debugging approaches and breakthrough techniques
- **Academic Standards**: Research-grade debugging reproducibility and documentation
- **Experimental Debugging**: Advanced experimental debugging for research scenarios

### Specialized Debugging Agents

#### **`scientific-computing-master`** - Scientific Computing Debugging
- **Numerical Debugging**: Debugging numerical algorithms and computational science code
- **Scientific Error Analysis**: Domain-specific error analysis for scientific computing
- **Multi-Scale Debugging**: Debugging strategies for multi-scale scientific problems
- **Research Debugging**: Debugging for research-grade computational standards
- **Domain Integration**: Debugging approaches for specific scientific domains

#### **`ai-systems-architect`** - AI/ML Debugging & System Integration
- **ML Debugging**: Machine learning specific debugging and error resolution
- **Distributed Debugging**: Multi-device and multi-node debugging strategies
- **Production Debugging**: Debugging for production AI systems and deployment
- **Scalability Debugging**: Debugging performance and scalability issues
- **Integration Debugging**: Debugging complex AI system integrations

#### **`neural-networks-master`** - Deep Learning Debugging Expert
- **Training Debugging**: Neural network training debugging and convergence issues
- **Architecture Debugging**: Model architecture debugging and optimization
- **Gradient Debugging**: Gradient computation debugging and numerical stability
- **Large Model Debugging**: Debugging strategies for large-scale neural networks
- **Multi-Modal Debugging**: Debugging complex multi-modal learning systems

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Debugging
Automatically analyzes debugging requirements and selects optimal agent combinations:
- **Error Analysis**: Detects error types, complexity, debugging requirements
- **Code Assessment**: Evaluates code complexity and debugging challenges
- **Agent Matching**: Maps debugging scenarios to relevant agent expertise
- **Efficiency Optimization**: Balances comprehensive debugging with resolution speed

#### **`jax`** - JAX-Specialized Debugging Team
- `jax-pro` (JAX ecosystem lead)
- `code-quality-master` (debugging methodology)
- `systems-architect` (system debugging)
- `ai-systems-architect` (ML debugging)

#### **`scientific`** - Scientific Computing Debugging Team
- `scientific-computing-master` (lead)
- `jax-pro` (JAX implementation)
- `research-intelligence-master` (research methodology)
- `code-quality-master` (debugging standards)
- Domain-specific experts based on scientific application

#### **`ai`** - AI/ML Debugging Team
- `ai-systems-architect` (lead)
- `neural-networks-master` (ML debugging)
- `jax-pro` (JAX optimization)
- `systems-architect` (infrastructure)
- `code-quality-master` (quality standards)

#### **`debug`** - Dedicated Debugging Team
- `code-quality-master` (lead)
- `jax-pro` (JAX debugging)
- `systems-architect` (system debugging)
- `research-intelligence-master` (advanced techniques)
- `ai-systems-architect` (AI debugging)

#### **`all`** - Complete 23-Agent Debugging Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough debugging and error resolution.

### 23-Agent Debugging Orchestration (`--orchestrate`)

#### **Multi-Agent Debugging Pipeline**
1. **Error Analysis Phase**: Multiple agents analyze different error aspects simultaneously
2. **Root Cause Investigation**: Collaborative investigation of error root causes
3. **Resolution Strategy**: Multi-agent development of comprehensive debugging strategies
4. **Implementation Coordination**: Agent-coordinated debugging implementation and testing
5. **Validation Monitoring**: Continuous multi-agent debugging validation and monitoring

#### **Breakthrough Debugging Discovery (`--breakthrough`)**
- **Cross-Domain Innovation**: Debugging techniques from multiple domains and research areas
- **Emergent Resolution**: Novel debugging strategies discovered through agent collaboration
- **Research-Grade Debugging**: Academic and industry-leading debugging standards
- **Adaptive Methodologies**: Dynamic debugging strategy adaptation based on error evolution

### Advanced 23-Agent Debugging Examples

```bash
# Intelligent auto-selection for debugging optimization
/jax-debug --agents=auto --intelligent --check-tracers --analyze

# Scientific computing debugging with specialized agents
/jax-debug --agents=scientific --breakthrough --orchestrate --monitor

# AI/ML debugging with scalability focus
/jax-debug --agents=ai --disable-jit --analyze --breakthrough

# Research-grade debugging development
/jax-debug --agents=all --breakthrough --orchestrate --intelligent

# JAX-specialized debugging optimization
/jax-debug --agents=jax --print-values --monitor --analyze

# Complete 23-agent debugging ecosystem
/jax-debug --agents=all --orchestrate --breakthrough --monitor

# Tracer leak debugging with expert agents
/jax-debug tracer_issues.py --agents=jax --check-tracers --intelligent

# Performance debugging for large models
/jax-debug large_model.py --agents=ai --analyze --breakthrough

# Scientific computation debugging
/jax-debug simulation.py --agents=scientific --orchestrate --monitor

# Production debugging with monitoring
/jax-debug production_code.py --agents=debug --monitor --analyze

# Complex multi-modal debugging
/jax-debug multimodal.py --agents=all --breakthrough --orchestrate

# Research debugging with innovation
/jax-debug research_code.py --agents=all --intelligent --breakthrough
```

### Intelligent Agent Selection Examples

```bash
# Error Type Detection → Intelligent Agent Selection

# Tracer leak error debugging
/jax-debug tracer_leak.py --agents=auto --intelligent
# → Selects: jax-pro + code-quality-master + systems-architect

# Scientific computing numerical errors
/jax-debug numerical_errors.py --agents=auto --intelligent
# → Selects: scientific-computing-master + jax-pro + research-intelligence-master

# Production ML system debugging
/jax-debug production_ml.py --agents=auto --intelligent
# → Selects: ai-systems-architect + jax-pro + systems-architect

# Memory and performance issues
/jax-debug performance_issues.py --agents=auto --intelligent
# → Selects: systems-architect + jax-pro + ai-systems-architect

# Complex research debugging scenario
/jax-debug complex_research.py --agents=auto --intelligent
# → Selects: research-intelligence-master + multi-agent-orchestrator + jax-pro

# Comprehensive debugging audit
/jax-debug complete_system.py --agents=all --orchestrate --breakthrough
# → Activates: All 23 agents with intelligent coordination for maximum debugging effectiveness
```

## Example output

```python
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

# ============================================================================
# 1. DISABLE JIT FOR DEBUGGING
# ============================================================================

# Disable JIT compilation globally
jax.config.update('jax_disable_jit', True)

# Your code runs in Python mode (slower but easier to debug)
def debug_function(x):
    return x ** 2

result = debug_function(jnp.array([1, 2, 3]))
print(f"Result: {result}")

# Re-enable JIT after debugging
jax.config.update('jax_disable_jit', False)

# Alternative: Disable JIT for specific functions
def regular_function(x):
    return x ** 2

# Remove @jit decorator temporarily
# @jit  # Comment out for debugging
def debug_specific(x):
    return x ** 2

# ============================================================================
# 2. DEBUG PRINTS THAT WORK WITH JAX
# ============================================================================

def debug_function_with_prints(x):
    # Use jax.debug.print instead of regular print
    jax.debug.print("x shape: {x_shape}, x dtype: {x_dtype}",
                     x_shape=x.shape, x_dtype=x.dtype)
    jax.debug.print("x value: {x}", x=x)

    result = x ** 2
    jax.debug.print("result: {result}", result=result)
    return result

# Debug prints work with all JAX transformations
debug_fn_jit = jit(debug_function_with_prints)
debug_fn_vmap = vmap(debug_function_with_prints)

# Interactive debugging
def breakpoint_function(x):
    jax.debug.breakpoint()  # Drops into debugger
    return x + 1

# ============================================================================
# 3. TRACER LEAK DETECTION
# ============================================================================

# Common tracer leak patterns and fixes
def check_tracer_leaks():
    """Examples of tracer leaks and how to fix them"""

    # BAD: Using traced values in Python conditionals
    @jit
    def bad_conditional(x):
        if x > 0:  # This will cause a tracer leak!
            return x
        else:
            return -x

    # GOOD: Use JAX conditional operations
    @jit
    def good_conditional(x):
        return jax.lax.cond(x > 0, lambda: x, lambda: -x)

    # BAD: Using traced values for array indexing
    @jit
    def bad_indexing(arr, idx):
        return arr[idx]  # idx cannot be traced

    # GOOD: Use static indices or dynamic_slice
    @jit
    def good_indexing(arr, start_idx, size):
        return jax.lax.dynamic_slice(arr, [start_idx], [size])

    # BAD: Leaking tracers outside function scope
    traced_value = None

    @jit
    def bad_leak(x):
        global traced_value
        traced_value = x  # Don't do this!
        return x

    # GOOD: Keep traced values within function scope
    @jit
    def good_no_leak(x):
        local_value = x
        return local_value

# Try-catch for tracer leak detection
def detect_tracer_leaks(jitted_function, data):
    try:
        result = jitted_function(data)
        return result
    except Exception as e:
        if "tracer" in str(e).lower():
            print(f"Tracer leak detected: {e}")
            print("Common causes:")
            print("- Using traced values in Python conditionals")
            print("- Comparing traced values with Python operators")
            print("- Array indexing with traced values")
            print("- Leaking traced values outside function scope")
        raise e

# ============================================================================
# 4. SHAPE AND DTYPE DEBUGGING
# ============================================================================

def debug_shapes_and_dtypes(x, y):
    """Debug function for shape and dtype issues"""

    print(f"Input x: shape={x.shape}, dtype={x.dtype}, device={x.device()}")
    print(f"Input y: shape={y.shape}, dtype={y.dtype}, device={y.device()}")

    # Check for shape compatibility
    if x.shape[-1] != y.shape[0]:
        print(f"Warning: Shape mismatch for dot product: {x.shape} @ {y.shape}")

    try:
        result = jnp.dot(x, y)
        print(f"Output: shape={result.shape}, dtype={result.dtype}")
        return result
    except Exception as e:
        print(f"Operation failed: {e}")
        raise e

# ============================================================================
# 5. MEMORY AND PERFORMANCE DEBUGGING
# ============================================================================

# Enable NaN and infinity detection
jax.config.update('jax_debug_nans', True)   # Catch NaN values
jax.config.update('jax_debug_infs', True)   # Catch infinity values

def check_device_memory():
    """Check memory usage on available devices"""
    devices = jax.devices()
    for i, device in enumerate(devices):
        try:
            stats = device.memory_stats()
            print(f"Device {i} ({device}): {stats}")
        except AttributeError:
            print(f"Device {i} ({device}): Memory stats not available")

def debug_performance(func, *args, **kwargs):
    """Debug function performance and compilation"""
    import time

    # Time compilation (first call)
    start_time = time.time()
    result = func(*args, **kwargs)
    result.block_until_ready()  # Wait for computation to complete
    compile_time = time.time() - start_time
    print(f"First call (compilation + execution): {compile_time:.4f}s")

    # Time execution (subsequent calls)
    start_time = time.time()
    result = func(*args, **kwargs)
    result.block_until_ready()
    execution_time = time.time() - start_time
    print(f"Subsequent call (execution only): {execution_time:.4f}s")

    return result

# ============================================================================
# 6. COMMON DEBUGGING PATTERNS
# ============================================================================

def debugging_checklist():
    """Common JAX debugging strategies"""

    strategies = [
        "1. Disable JIT with jax.config.update('jax_disable_jit', True)",
        "2. Use jax.debug.print() instead of regular print()",
        "3. Add .block_until_ready() to force computation",
        "4. Check device placement with .device()",
        "5. Use jax.tree_map for nested structure debugging",
        "6. Enable NaN/Inf detection with debug flags",
        "7. Check shapes and dtypes at each step",
        "8. Use try-catch to identify tracer leaks",
        "9. Profile memory usage on devices",
        "10. Use jax.debug.breakpoint() for interactive debugging"
    ]

    print("JAX Debugging Checklist:")
    for strategy in strategies:
        print(f"  {strategy}")

# ============================================================================
# 7. EXAMPLE DEBUGGING SESSION
# ============================================================================

def example_debugging_session():
    """Complete example of debugging a JAX function"""

    # Original problematic function
    @jit
    def problematic_function(x, threshold):
        # Multiple potential issues
        if x.mean() > threshold:  # Tracer leak - Python conditional
            result = x * 2
        else:
            result = x / 2
        return result[0]  # Potential shape issue

    # Step 1: Disable JIT to see the actual error
    jax.config.update('jax_disable_jit', True)

    # Step 2: Add debug prints
    def debug_version(x, threshold):
        jax.debug.print("x: {x}", x=x)
        jax.debug.print("threshold: {threshold}", threshold=threshold)
        jax.debug.print("x.mean(): {mean}", mean=x.mean())

        # Step 3: Fix tracer leak with jax.lax.cond
        def true_fn(x):
            return x * 2

        def false_fn(x):
            return x / 2

        result = jax.lax.cond(x.mean() > threshold, true_fn, false_fn, x)
        jax.debug.print("result shape: {shape}", shape=result.shape)

        # Step 4: Fix indexing if needed
        return result[0] if result.size > 0 else result

    # Step 5: Test the fixed version
    test_data = jnp.array([1.0, 2.0, 3.0])
    test_threshold = 1.5

    try:
        result = debug_version(test_data, test_threshold)
        print(f"Debug success: {result}")
    except Exception as e:
        print(f"Debug failed: {e}")
    finally:
        # Step 6: Re-enable JIT
        jax.config.update('jax_disable_jit', False)

# Run debugging checklist
debugging_checklist()
print("\nExample debugging session:")
example_debugging_session()
```

## Common Issues and Solutions

### Tracer Leaks
**Problem**: Using traced values outside JAX transformations
**Solution**: Use JAX control flow operations (jax.lax.cond, jax.lax.while_loop)

### Shape Errors
**Problem**: Dynamic shapes or shape mismatches
**Solution**: Debug shapes at each step, use static_argnums for constants

### Performance Issues
**Problem**: Unexpected slowdowns or recompilation
**Solution**: Profile with timing, check for dynamic shapes, monitor device memory

### NaN/Infinity Values
**Problem**: Invalid numerical operations
**Solution**: Enable debug flags, add range checks, use stable algorithms

## Agent-Enhanced Debugging Integration Patterns

### Complete Debugging Workflow
```bash
# Intelligent debugging analysis and resolution pipeline
/jax-debug --agents=auto --intelligent --check-tracers --analyze
/jax-performance --agents=jax --technique=profiling --optimization
/generate-tests --agents=auto --type=debug --coverage
```

### Scientific Computing Debugging Pipeline
```bash
# High-performance scientific computing debugging
/jax-debug --agents=scientific --breakthrough --orchestrate
/jax-essentials --agents=scientific --operation=grad --debugging
/run-all-tests --agents=scientific --debug --reproducible
```

### Production Debugging Infrastructure
```bash
# Large-scale production debugging with monitoring
/jax-debug --agents=ai --monitor --analyze --breakthrough
/jax-performance --agents=ai --technique=profiling --gpu-accel
/ci-setup --agents=ai --monitoring --debug-integration
```

## Related Commands

**Prerequisites**: Commands to run before debugging setup
- `/jax-essentials --agents=auto` - Core JAX operations with debugging considerations
- `/jax-init --agents=auto` - JAX project setup with debugging configuration

**Core Workflow**: Debugging development with agent intelligence
- `/jax-performance --agents=jax` - Performance debugging with specialized agents
- `/jax-training --agents=auto` - Training debugging with intelligent assistance
- `/python-debug-prof --agents=debug` - Python-specific debugging with agent intelligence

**Advanced Integration**: Specialized debugging development
- `/jax-models --agents=auto` - Model debugging with architecture analysis
- `/jax-data-load --agents=auto` - Data pipeline debugging and optimization
- `/jax-sparse-ops --agents=scientific` - Sparse operation debugging with scientific agents

**Quality Assurance**: Debugging validation and monitoring
- `/generate-tests --agents=auto --type=debug` - Generate debugging tests with agent intelligence
- `/run-all-tests --agents=debug --coverage` - Comprehensive debugging validation
- `/check-code-quality --agents=debug --auto-fix` - Code quality debugging with agents

**Research & Documentation**: Advanced debugging workflows
- `/update-docs --agents=research --type=debug` - Research-grade debugging documentation
- `/reflection --agents=debug --type=scientific` - Debugging methodology analysis
- `/multi-agent-optimize --agents=all --focus=debugging` - Comprehensive debugging optimization