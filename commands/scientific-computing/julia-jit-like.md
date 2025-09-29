---
title: "Julia JIT-like"
description: "Optimize Julia functions with type stability analysis and precompilation, JAX JIT-like performance using Enzyme.jl"
category: scientific-computing
subcategory: julia-performance
complexity: advanced
argument-hint: "[--type-stability] [--precompile] [--enzyme-ad] [--benchmark] [--agents=auto|julia|scientific|ai|optimization|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--performance]"
allowed-tools: "*"
model: inherit
tags: julia, performance, type-stability, precompilation, enzyme, jit-optimization
dependencies: []
related: [julia-ad-grad, julia-prob-model, optimize, jax-essentials, python-debug-prof]
workflows: [julia-optimization, performance-development, scientific-computing]
version: "2.1"
last-updated: "2025-09-28"
---

# Julia JIT-like Optimization

Optimize Julia functions with type stability analysis and precompilation for JAX JIT-like performance.

## Quick Start

```bash
# Basic optimization with type stability
/julia-jit-like --type-stability --benchmark

# Full optimization with agent orchestration
/julia-jit-like --type-stability --precompile --benchmark --agents=optimization

# Advanced optimization with Enzyme.jl
/julia-jit-like --enzyme-ad --benchmark --agents=all --breakthrough

# High-performance Julia optimization
/julia-jit-like --agents=julia --intelligent --performance --optimize
```

## Usage

```bash
/julia-jit-like [options]
```

**Parameters:**
- `options` - Optimization configuration, agent selection, and performance analysis options

## Options

| Option | Description |
|--------|-------------|
| `--type-stability` | Enable type stability analysis with `@code_warntype` |
| `--precompile` | Use precompilation for faster startup times |
| `--enzyme-ad` | Enable Enzyme.jl for high-performance automatic differentiation |
| `--benchmark` | Include performance benchmarking and before/after comparison |
| `--agents=<agents>` | Agent selection (auto, julia, scientific, ai, optimization, all) |
| `--orchestrate` | Enable advanced 23-agent orchestration with optimization intelligence |
| `--intelligent` | Enable intelligent agent selection based on Julia performance analysis |
| `--breakthrough` | Enable breakthrough Julia optimization discovery |
| `--optimize` | Apply performance optimization to Julia computations |
| `--performance` | Enable performance-focused optimization with agent coordination |

## What it does

1. **Type Stability**: Analyze and optimize functions with @code_warntype
2. **Precompilation**: Implement strategies for faster startup and execution
3. **Enzyme.jl AD**: Use optimized automatic differentiation
4. **Performance Optimization**: Apply Julia-specific optimization patterns
5. **JAX Comparison**: Benchmark against JAX JIT performance
6. **23-Agent Julia Optimization Intelligence**: Multi-agent collaboration for optimal Julia performance
7. **Advanced Compilation Optimization**: Agent-driven Julia compilation and type stability optimization
8. **Intelligent Performance Engineering**: Agent-coordinated high-performance Julia computing

## 23-Agent Intelligent Julia Optimization System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes Julia performance requirements, optimization patterns, and computational goals to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Julia Optimization Pattern Detection → Agent Selection
- Research Computing → research-intelligence-master + optimization-master + julia-pro
- Production Systems → ai-systems-architect + julia-pro + systems-architect
- Scientific Simulation → scientific-computing-master + julia-pro + optimization-master
- High-Performance Computing → optimization-master + julia-pro + neural-networks-master
- Educational Projects → documentation-architect + julia-pro + optimization-master
```

### Core Julia Optimization Agents

#### **`julia-pro`** - Julia Ecosystem Performance Expert
- **Julia Optimization**: Deep expertise in Julia performance optimization and compilation
- **Type System Mastery**: Advanced Julia type system optimization and type stability analysis
- **Package Ecosystem**: Julia performance package coordination (BenchmarkTools, LoopVectorization, etc.)
- **Compilation Optimization**: Julia compilation pipeline optimization and LLVM integration
- **Memory Management**: Julia memory optimization and garbage collection tuning

#### **`optimization-master`** - Advanced Performance Optimization
- **Algorithm Optimization**: High-performance algorithm design and computational efficiency
- **SIMD Optimization**: SIMD vectorization and CPU-level optimization strategies
- **Memory Patterns**: Cache-efficient memory access patterns and optimization
- **Parallel Computing**: Multi-threading and distributed computing optimization
- **Benchmarking Excellence**: Performance measurement and optimization validation

#### **`scientific-computing-master`** - Scientific Julia Computing
- **Scientific Algorithms**: Scientific computing algorithm optimization in Julia
- **Numerical Methods**: Advanced numerical methods and computational mathematics
- **Research Applications**: Academic and research-grade computational optimization
- **Domain Integration**: Cross-domain scientific computing optimization
- **Mathematical Foundation**: Mathematical optimization and algorithmic development

#### **`systems-architect`** - System-Level Julia Optimization
- **System Architecture**: System-level Julia optimization and infrastructure design
- **Resource Management**: Computational resource optimization and system tuning
- **Scalability Engineering**: Large-scale Julia system optimization
- **Infrastructure Integration**: Julia integration with larger system architectures
- **Production Deployment**: Production-grade Julia optimization strategies

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Julia Optimization
Automatically analyzes Julia optimization requirements and selects optimal agent combinations:
- **Performance Analysis**: Detects Julia performance bottlenecks and optimization opportunities
- **Complexity Assessment**: Evaluates computational complexity and resource requirements
- **Agent Matching**: Maps Julia optimization needs to relevant agent expertise
- **Efficiency Balance**: Balances performance gains with development complexity

#### **`julia`** - Julia-Specialized Optimization Team
- `julia-pro` (Julia ecosystem lead)
- `optimization-master` (performance optimization)
- `scientific-computing-master` (scientific applications)
- `systems-architect` (system optimization)

#### **`scientific`** - Scientific Computing Optimization Team
- `scientific-computing-master` (lead)
- `julia-pro` (Julia implementation)
- `optimization-master` (performance optimization)
- `research-intelligence-master` (research methodology)

#### **`ai`** - AI/ML Julia Optimization Team
- `neural-networks-master` (lead)
- `julia-pro` (Julia optimization)
- `optimization-master` (performance)
- `ai-systems-architect` (production systems)

#### **`optimization`** - Performance-Focused Julia Team
- `optimization-master` (lead)
- `julia-pro` (Julia performance)
- `systems-architect` (system optimization)
- `scientific-computing-master` (algorithmic optimization)

#### **`all`** - Complete 23-Agent Julia Optimization Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough Julia performance optimization.

### Advanced 23-Agent Julia Optimization Examples

```bash
# Intelligent auto-selection for Julia optimization
/julia-jit-like --agents=auto --intelligent --optimize

# Scientific computing optimization with specialized agents
/julia-jit-like --agents=scientific --breakthrough --orchestrate --type-stability

# High-performance Julia optimization
/julia-jit-like --agents=optimization --optimize --performance --enzyme-ad

# Production system optimization
/julia-jit-like --agents=ai --breakthrough --orchestrate --benchmark

# Research-grade optimization development
/julia-jit-like --agents=all --breakthrough --intelligent --precompile

# Complete 23-agent Julia optimization ecosystem
/julia-jit-like --agents=all --orchestrate --breakthrough --intelligent --performance
```

## Example output

```julia
using BenchmarkTools
using PrecompileTools
using Enzyme
using StaticArrays
using LoopVectorization
using LinearAlgebra
using Random

# ============================================================================
# 1. TYPE STABILITY ANALYSIS AND OPTIMIZATION
# ============================================================================

# Example of type-unstable function (BAD)
function unstable_function(x)
    if x > 0
        return x
    else
        return "negative"  # Type instability!
    end
end

# Optimized type-stable version (GOOD)
function stable_function(x::T) where T<:Real
    if x > 0
        return x
    else
        return zero(T)  # Always return same type
    end
end

# Function to analyze type stability
function analyze_type_stability(func, args...)
    println("=== Type Stability Analysis ===")
    println("Function: $func")
    println("Arguments: $(typeof.(args))")
    println()

    # Check type stability
    println("@code_warntype output:")
    @code_warntype func(args...)
    println()

    # Check inferred return type
    return_type = Base.return_types(func, typeof.(args))
    println("Inferred return type: $return_type")
    println()
end

# Advanced type-stable scientific function
function type_stable_computation(x::AbstractVector{T}) where T<:AbstractFloat
    result = zero(T)

    # Type-stable loop
    for i in eachindex(x)
        result += sin(x[i]) * cos(x[i])
    end

    return result
end

# Type-stable matrix operations
function stable_matrix_multiply(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    # Ensure type stability
    C = zeros(T, size(A, 1), size(B, 2))

    # Type-stable implementation
    for j in axes(B, 2)
        for i in axes(A, 1)
            temp = zero(T)
            for k in axes(A, 2)
                temp += A[i, k] * B[k, j]
            end
            C[i, j] = temp
        end
    end

    return C
end

# ============================================================================
# 2. PRECOMPILATION STRATEGIES
# ============================================================================

# Create precompilation workload
@setup_workload begin
    # Put setup code here that prepares for precompilation
    using LinearAlgebra
    using StaticArrays

    @compile_workload begin
        # Put workload here that should be precompiled
        x = randn(100)
        A = randn(10, 10)
        B = randn(10, 10)

        # Precompile common operations
        type_stable_computation(x)
        stable_matrix_multiply(A, B)

        # Precompile with different types
        x_f32 = Float32.(x)
        type_stable_computation(x_f32)

        # Static arrays
        sa = @SVector randn(3)
        type_stable_computation(sa)
    end
end

# Function with manual precompilation
function precompile_function(func, types...)
    for T in types
        if T <: AbstractVector
            # Precompile for vectors
            for n in [10, 100, 1000]
                x = zeros(T.parameters[1], n)
                func(x)
            end
        elseif T <: AbstractMatrix
            # Precompile for matrices
            for n in [5, 10, 50]
                A = zeros(T.parameters[1], n, n)
                func(A, A)
            end
        end
    end
end

# Precompile common scientific functions
function setup_precompilation()
    println("Setting up precompilation...")

    # Vector types
    vector_types = [Vector{Float64}, Vector{Float32}]
    matrix_types = [Matrix{Float64}, Matrix{Float32}]

    # Precompile common operations
    precompile_function(type_stable_computation, vector_types...)
    precompile_function(stable_matrix_multiply, matrix_types...)

    println("Precompilation complete!")
end

# ============================================================================
# 3. ENZYME.JL FOR OPTIMIZED AUTOMATIC DIFFERENTIATION
# ============================================================================

# Function for AD optimization
function enzyme_optimized_function(x::Vector{T}) where T
    result = zero(T)

    # Complex computation
    for i in eachindex(x)
        result += x[i]^2 + sin(x[i]) + exp(-x[i]^2)
    end

    return result
end

# Forward mode AD with Enzyme
function enzyme_forward_ad(f, x)
    # Enzyme forward mode
    return autodiff(Forward, f, Duplicated(x, ones(eltype(x), length(x))))
end

# Reverse mode AD with Enzyme
function enzyme_reverse_ad(f, x)
    # Enzyme reverse mode
    dx = zeros(eltype(x), length(x))
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    return dx
end

# Compare AD methods
function compare_ad_methods(f, x)
    println("=== Automatic Differentiation Comparison ===")

    # Enzyme forward mode
    @time "Enzyme Forward" begin
        grad_forward = enzyme_forward_ad(f, copy(x))
    end

    # Enzyme reverse mode
    @time "Enzyme Reverse" begin
        grad_reverse = enzyme_reverse_ad(f, copy(x))
    end

    # Check consistency
    max_diff = maximum(abs.(grad_forward[2] - grad_reverse))
    println("Maximum difference between methods: $max_diff")

    return grad_forward, grad_reverse
end

# ============================================================================
# 4. PERFORMANCE OPTIMIZATION PATTERNS
# ============================================================================

# SIMD-optimized function
function simd_optimized_sum(x::AbstractVector{T}) where T
    # Use @simd for vectorization
    result = zero(T)
    @simd for i in eachindex(x)
        result += x[i]
    end
    return result
end

# Loop vectorization with LoopVectorization.jl
function turbo_optimized_sum(x::AbstractVector{T}) where T
    result = zero(T)
    @turbo for i in eachindex(x)
        result += x[i]
    end
    return result
end

# Memory-efficient operations
function memory_efficient_computation(A::AbstractMatrix{T}) where T
    rows, cols = size(A)
    result = zeros(T, rows)

    # Cache-friendly access pattern
    @inbounds for j in 1:cols
        for i in 1:rows
            result[i] += A[i, j] * A[i, j]
        end
    end

    return result
end

# Static array optimization
function static_array_computation(x::SVector{N, T}) where {N, T}
    # Compile-time optimized operations
    result = zero(T)

    # Unrolled loop at compile time
    @inbounds for i in 1:N
        result += x[i] * x[i] + sin(x[i])
    end

    return result
end

# ============================================================================
# 5. COMPILATION AND OPTIMIZATION ANALYSIS
# ============================================================================

# Function to analyze compilation
function analyze_compilation(func, args...)
    println("=== Compilation Analysis ===")
    println("Function: $func")

    # LLVM IR
    println("\nLLVM IR:")
    @code_llvm func(args...)
    println()

    # Native code
    println("Native assembly:")
    @code_native func(args...)
    println()

    # Method instances
    println("Method instances:")
    methods(func)
end

# Optimization verification
function verify_optimizations(func, args...)
    println("=== Optimization Verification ===")

    # Check for type stability
    rt = Base.return_types(func, typeof.(args))
    if length(rt) == 1 && isconcretetype(rt[1])
        println("✓ Type stable")
    else
        println("✗ Type unstable: $rt")
    end

    # Check for allocations
    allocs = @allocated func(args...)
    if allocs == 0
        println("✓ Zero allocations")
    else
        println("⚠ Allocations: $allocs bytes")
    end

    # Benchmark
    result = @benchmark $func($(args)...)
    println("Performance: $(BenchmarkTools.prettytime(median(result.times)))")

    return result
end

# ============================================================================
# 6. JAX-LIKE OPTIMIZATION PATTERNS
# ============================================================================

# Mimic JAX's jit with precompilation
macro jit_like(expr)
    quote
        # Force compilation and optimization
        let f = $(esc(expr))
            # Precompile with sample arguments
            Base.precompile(f, (Float64,))
            Base.precompile(f, (Vector{Float64},))
            Base.precompile(f, (Matrix{Float64},))
            f
        end
    end
end

# JAX-like vmap equivalent
function vmap_like(f, x::AbstractArray)
    # Vectorized application
    return map(f, x)
end

# JAX-like grad equivalent using Enzyme
function grad_like(f)
    return x -> enzyme_reverse_ad(f, x)
end

# Combined optimization: JIT + grad
function jit_grad_like(f)
    # Optimize both function and gradient
    grad_f = grad_like(f)

    # Precompile both
    @jit_like f
    @jit_like grad_f

    return f, grad_f
end

# ============================================================================
# 7. PERFORMANCE BENCHMARKING
# ============================================================================

# Comprehensive benchmarking suite
function benchmark_julia_optimizations()
    println("=== Julia Optimization Benchmarks ===")

    # Setup test data
    n = 10000
    x = randn(n)
    A = randn(100, 100)
    B = randn(100, 100)

    # Test different optimization levels
    functions_to_test = [
        ("Basic sum", () -> sum(x)),
        ("SIMD sum", () -> simd_optimized_sum(x)),
        ("Turbo sum", () -> turbo_optimized_sum(x)),
        ("Type-stable computation", () -> type_stable_computation(x)),
        ("Matrix multiply (Julia)", () -> A * B),
        ("Matrix multiply (custom)", () -> stable_matrix_multiply(A, B)),
    ]

    results = Dict()

    for (name, func) in functions_to_test
        println("\nBenchmarking: $name")
        result = @benchmark $func()
        results[name] = result

        println("  Time: $(BenchmarkTools.prettytime(median(result.times)))")
        println("  Allocations: $(result.allocs) ($(BenchmarkTools.prettymemory(result.memory)))")
    end

    return results
end

# Compare with reference implementations
function compare_with_reference()
    println("=== Performance Comparison ===")

    n = 1000
    x = randn(n)

    # Julia optimized
    julia_time = @elapsed begin
        for _ in 1:1000
            type_stable_computation(x)
        end
    end

    # Basic implementation
    basic_time = @elapsed begin
        for _ in 1:1000
            sum(sin.(x) .* cos.(x))
        end
    end

    speedup = basic_time / julia_time

    println("Julia optimized: $(julia_time * 1000) ms")
    println("Basic implementation: $(basic_time * 1000) ms")
    println("Speedup: $(round(speedup, digits=2))x")

    return speedup
end

# Memory usage analysis
function analyze_memory_usage(func, args...)
    println("=== Memory Usage Analysis ===")

    # Track allocations
    allocs_before = Base.gc_alloc_count()
    bytes_before = Base.gc_bytes()

    result = func(args...)

    allocs_after = Base.gc_alloc_count()
    bytes_after = Base.gc_bytes()

    println("Allocations: $(allocs_after - allocs_before)")
    println("Memory: $(bytes_after - bytes_before) bytes")

    # Detailed allocation tracking
    allocation_result = @allocated func(args...)
    println("Direct measurement: $allocation_result bytes")

    return result
end

# ============================================================================
# 8. COMPREHENSIVE EXAMPLES
# ============================================================================

function demonstrate_julia_optimizations()
    println("=== Julia JIT-like Optimization Examples ===")

    # Example 1: Type stability analysis
    println("\n1. Type Stability Analysis:")
    x = randn(100)
    analyze_type_stability(type_stable_computation, x)

    # Example 2: Precompilation setup
    println("\n2. Precompilation Setup:")
    setup_precompilation()

    # Example 3: Enzyme AD comparison
    println("\n3. Automatic Differentiation:")
    compare_ad_methods(enzyme_optimized_function, randn(10))

    # Example 4: Optimization verification
    println("\n4. Optimization Verification:")
    verify_optimizations(type_stable_computation, x)

    # Example 5: Static arrays
    println("\n5. Static Array Optimization:")
    static_vec = @SVector randn(3)
    verify_optimizations(static_array_computation, static_vec)

    # Example 6: Performance benchmarking
    println("\n6. Performance Benchmarking:")
    benchmark_results = benchmark_julia_optimizations()

    # Example 7: Memory analysis
    println("\n7. Memory Usage Analysis:")
    analyze_memory_usage(type_stable_computation, x)

    # Example 8: Speedup comparison
    println("\n8. Speedup Comparison:")
    speedup = compare_with_reference()

    println("\n=== Summary ===")
    println("Julia optimization demonstration complete!")
    println("Key techniques: Type stability, precompilation, SIMD, static arrays")
    println("Achieved speedup: $(round(speedup, digits=2))x over basic implementation")
end

# ============================================================================
# 9. REAL-WORLD OPTIMIZATION EXAMPLE
# ============================================================================

# Scientific computing example: numerical integration
function optimized_numerical_integration(f, a::T, b::T, n::Int) where T<:AbstractFloat
    h = (b - a) / n
    result = zero(T)

    # Type-stable Simpson's rule
    result += f(a)
    result += f(b)

    @inbounds @simd for i in 1:(n-1)
        x = a + i * h
        weight = iseven(i) ? T(2) : T(4)
        result += weight * f(x)
    end

    return result * h / T(3)
end

# Test function for integration
test_function(x) = sin(x) * exp(-x^2)

# Demonstrate the optimization
function integration_example()
    println("=== Numerical Integration Example ===")

    # Parameters
    a, b = 0.0, 2π
    n = 100000

    # Benchmark optimized version
    optimized_time = @elapsed begin
        result_opt = optimized_numerical_integration(test_function, a, b, n)
    end

    # Compare with basic implementation
    basic_time = @elapsed begin
        h = (b - a) / n
        result_basic = sum(test_function(a + i * h) for i in 0:n) * h
    end

    println("Optimized result: $result_opt")
    println("Optimized time: $(optimized_time * 1000) ms")
    println("Basic time: $(basic_time * 1000) ms")
    println("Speedup: $(round(basic_time / optimized_time, digits=2))x")

    # Verify optimization
    verify_optimizations(optimized_numerical_integration, test_function, a, b, n)
end

# Run the comprehensive demonstration
demonstrate_julia_optimizations()
integration_example()
```

## Julia Optimization Best Practices

### Type Stability
- **Concrete types**: Always specify concrete types in function signatures
- **Avoid type unions**: Use parametric types instead of Union types
- **Check with @code_warntype**: Regularly analyze type inference
- **Consistent return types**: Ensure functions always return the same type

### Precompilation Strategies
- **@setup_workload**: Use for systematic precompilation
- **Common use cases**: Precompile frequently used type combinations
- **Package structure**: Organize precompilation in package modules
- **Startup time**: Balance precompilation vs startup overhead

### Memory Optimization
- **Avoid allocations**: Use @inbounds and @simd when safe
- **Static arrays**: Use SVector and SMatrix for small, fixed-size arrays
- **In-place operations**: Prefer functions ending with ! for mutation
- **Memory layout**: Consider cache-friendly access patterns

### SIMD and Vectorization
- **@simd**: Enable SIMD for simple loops
- **@turbo**: Use LoopVectorization.jl for complex loops
- **Alignment**: Ensure proper memory alignment for vectors
- **Loop structure**: Write loops that can be vectorized

## Performance Analysis Tools

### Compilation Analysis
- **@code_warntype**: Check type stability
- **@code_llvm**: Examine intermediate representation
- **@code_native**: View generated assembly
- **@benchmark**: Measure performance accurately

### Memory Profiling
- **@allocated**: Track memory allocations
- **Base.gc_alloc_count()**: Monitor garbage collection
- **Profile.jl**: Detailed profiling and flame graphs
- **BenchmarkTools.jl**: Comprehensive benchmarking

### Optimization Verification
- **Zero allocations**: Target for inner loops
- **Type inference**: Ensure concrete types throughout
- **SIMD usage**: Verify vectorization in generated code
- **Cache efficiency**: Consider memory access patterns

## Comparison with JAX JIT

### Similarities
- **Just-in-time optimization**: Both compile functions for optimal performance
- **Type specialization**: Generate specialized code for specific types
- **Vectorization**: Automatic SIMD optimization
- **Memory efficiency**: Minimize allocations and copies

### Differences
- **Compilation timing**: Julia compiles on first call, JAX on explicit jit
- **Type system**: Julia's type system enables more optimizations
- **Ecosystem**: Different tools and libraries for optimization
- **Syntax**: Julia uses macros, JAX uses decorators

### Performance Parity
- **Numerical computing**: Julia can match or exceed JAX performance
- **Memory usage**: Both achieve zero-allocation inner loops
- **Scalability**: Both handle large-scale scientific computing
- **Optimization**: Different approaches, similar end results

## Agent-Enhanced Julia Optimization Integration Patterns

### Complete Julia Performance Development Workflow
```bash
# Intelligent Julia performance development pipeline
/julia-jit-like --agents=auto --intelligent --optimize --performance
/julia-ad-grad --agents=optimization --intelligent --higher-order
/julia-prob-model --agents=julia --breakthrough --orchestrate
```

### Scientific Computing Julia Performance Pipeline
```bash
# High-performance scientific Julia workflow
/julia-jit-like --agents=scientific --breakthrough --orchestrate --type-stability
/jax-essentials --agents=julia --intelligent --operation=jit
/optimize --agents=julia --category=algorithm --implement
```

### Production Julia Performance Infrastructure
```bash
# Large-scale production Julia optimization
/julia-jit-like --agents=ai --optimize --performance --enzyme-ad
/generate-tests --agents=julia --type=performance --coverage=95
/run-all-tests --agents=optimization --scientific --performance
```

## Related Commands

**Julia Ecosystem Development**: Enhanced Julia performance development with agent intelligence
- `/julia-ad-grad --agents=auto` - Automatic differentiation with optimized gradients and agent optimization
- `/julia-prob-model --agents=optimization` - Probabilistic modeling with performance optimization and agents
- `/optimize --agents=julia` - Julia optimization with specialized agents

**Cross-Language Performance Computing**: Multi-language optimization integration
- `/jax-essentials --agents=auto` - Compare with JAX optimization techniques
- `/python-debug-prof --agents=optimization` - Cross-language performance comparison with agents
- `/jax-performance --agents=julia` - JAX performance comparison with Julia agents

**Quality Assurance**: Julia performance validation and optimization
- `/generate-tests --agents=optimization --type=performance` - Generate Julia performance tests with agent intelligence
- `/run-all-tests --agents=julia --scientific` - Comprehensive Julia testing with specialized agents
- `/check-code-quality --agents=auto --language=julia` - Julia performance code quality with agent optimization