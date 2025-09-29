---
title: "Optimize"
description: "Code optimization and performance analysis for Python, Julia, JAX, and scientific computing"
category: optimization
subcategory: performance
complexity: intermediate
argument-hint: "[target] [--language=python|julia|jax|auto] [--category=all|algorithm|memory|io|concurrency] [--format=text|json|html] [--implement] [--agents=auto|scientific|ai|engineering|quantum|all] [--orchestrate] [--intelligent] [--breakthrough]"
allowed-tools: "*"
model: inherit
tags: optimization, performance, python, julia, jax, scientific-computing
dependencies: []
related: [multi-agent-optimize, debug, jax-performance, julia-jit-like, check-code-quality, refactor-clean, generate-tests, run-all-tests, think-ultra]
workflows: [optimize-test-verify, performance-analysis, code-quality-improvement]
version: "2.0"
last-updated: "2025-09-28"
---

# Optimize

Analyze code for performance optimization opportunities across multiple languages and domains.

## Quick Start

```bash
# Basic usage
/optimize src/

# Language-specific optimization
/optimize myfile.py --language=python

# Specific category with JSON output
/optimize algorithm.py --category=algorithm --format=json

# Auto-implement optimizations
/optimize src/ --implement --language=auto
```

## Usage

```bash
/optimize [target] [options]
```

**Parameters:**
- `target` - File, directory, or module to optimize
- `options` - Configuration options for optimization analysis and implementation

## Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--language=<lang>` | python\|julia\|jax\|auto | auto | Target language for optimization analysis |
| `--category=<cat>` | all\|algorithm\|memory\|io\|concurrency | all | Optimization category focus |
| `--format=<fmt>` | text\|json\|html | text | Output format for results |
| `--implement` | - | false | Automatically implement optimization recommendations |
| `--profile` | - | false | Include performance profiling in analysis |
| `--detailed` | - | false | Show detailed analysis and explanations |
| `--agents=<agents>` | auto\|scientific\|ai\|engineering\|quantum\|all | auto | Agent selection for multi-agent analysis |
| `--orchestrate` | - | false | Enable 23-agent orchestration for complex workflows |
| `--intelligent` | - | false | Enable intelligent agent selection based on code analysis |
| `--breakthrough` | - | false | Enable breakthrough optimization discovery across domains |

## 23-Agent Intelligent Optimization System

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection
- **Code Analysis**: Automatically detects optimization opportunities and selects optimal agents
- **Performance Profiling**: Identifies bottlenecks and matches to agent expertise
- **Domain Detection**: Maps code patterns to specialized domain experts
- **Efficiency Optimization**: Balances comprehensive analysis with execution speed

#### **`scientific`** - Scientific Computing Optimization
- `scientific-computing-master` + `jax-pro` + `neural-networks-master` + `research-intelligence-master`
- **Numerical Computing**: High-performance numerical algorithm optimization
- **GPU Acceleration**: CUDA, JAX, CuPy optimization and memory management
- **Research Performance**: Publication-ready computational efficiency

#### **`ai`** - AI/ML Optimization Team
- `ai-systems-architect` + `neural-networks-master` + `data-professional` + `jax-pro`
- **Model Optimization**: Deep learning architecture and training efficiency
- **Data Pipeline**: ETL and data processing optimization
- **ML Infrastructure**: Scalable AI system optimization

#### **`engineering`** - Software Engineering Optimization
- `systems-architect` + `fullstack-developer` + `code-quality-master` + `devops-security-engineer`
- **Architecture Optimization**: System design and scalability improvements
- **Performance Engineering**: Code quality and maintainable optimization
- **Infrastructure**: DevOps and deployment optimization

#### **`quantum`** - Quantum Computing Optimization
- `advanced-quantum-computing-expert` + `scientific-computing-master` + `jax-pro`
- **Quantum Algorithms**: Quantum computing optimization and hybrid systems
- **Quantum-Classical**: Bridge optimization between quantum and classical computing
- **Research Integration**: Cutting-edge quantum optimization techniques

#### **`all`** - Complete 23-Agent Optimization Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough optimization discoveries.

### Breakthrough Optimization (`--breakthrough`)
- **Cross-Domain Innovation**: Optimization techniques from multiple domains
- **Emergent Optimization**: Novel optimization patterns discovered through agent collaboration
- **Research-Grade Performance**: Academic and industry-leading optimization standards

## Languages Supported

### Python
- **Scientific Computing**: NumPy, SciPy, Pandas, Polars, Xarray
- **Machine Learning**: PyTorch, scikit-learn, Hugging Face
- **Performance**: Numba, CuPy, Dask
- **General**: Algorithm optimization, memory management

### Julia
- **Performance**: JIT compilation, type stability, multiple dispatch
- **Scientific**: DifferentialEquations.jl, Flux.jl, MLJ.jl
- **Parallel**: Threading, distributed computing, GPU acceleration
- **Memory**: Allocation optimization, garbage collection

### JAX Ecosystem
- **Core**: XLA compilation, device placement, memory optimization
- **Neural Networks**: Flax model optimization, training efficiency
- **Optimization**: Optax, gradient computation, automatic differentiation
- **Hardware**: GPU/TPU utilization, parallel transformations

## Optimization Categories

### Algorithm Optimization
- Time complexity reduction (O(n²) → O(n log n))
- Space complexity optimization
- Data structure selection
- Search and sort improvements

### Memory Optimization
- Memory leak detection
- Allocation pattern analysis
- Garbage collection optimization
- Memory-mapped I/O opportunities

### I/O Optimization
- Batch I/O operations
- Asynchronous I/O patterns
- Connection pooling
- Caching strategies

### Concurrency Optimization
- Parallel processing opportunities
- Thread safety analysis
- Lock-free data structures
- Async/await improvements

## Analysis Features

### Performance Profiling
- Execution time analysis
- Memory usage tracking
- CPU utilization monitoring
- I/O bottleneck identification

### Code Analysis
- Algorithmic complexity assessment
- Performance pattern detection
- Optimization opportunity identification
- Best practice recommendations

### Optimization Recommendations
- Priority-based suggestions (high/medium/low)
- Before/after code examples
- Estimated performance improvements
- Implementation difficulty ratings

### Automatic Implementation (--implement)
- Apply high-priority optimizations automatically
- Create code backups before modifications
- Execute safe optimizations with minimal risk
- Validate changes through testing and profiling
- Generate implementation reports with performance metrics
- Rollback capability for failed optimizations
- Incremental application with progress tracking

## Output Formats

### Text Format
Human-readable analysis with recommendations and code examples.

### JSON Format
Structured data for programmatic processing and CI/CD integration.

### HTML Format
Interactive report with visualizations and detailed explanations.

### Implementation Results (with --implement)
- **Change Summary** - Applied optimizations and code modifications
- **Performance Metrics** - Before/after performance comparisons
- **Backup Locations** - Paths to original code backups
- **Validation Status** - Test results and verification outcomes
- **Rollback Commands** - Instructions to revert changes if needed

## Examples

### Basic Usage
```bash
# Simple file analysis
/optimize data_analysis.py

# Directory optimization with specific language
/optimize src/ --language=python

# Category-specific optimization
/optimize algorithms/ --category=algorithm
```

### Advanced Usage
```bash
# Complete analysis with profiling and detailed output
/optimize simulation.jl --language=julia --category=memory --profile --detailed

# JAX optimization with automatic implementation
/optimize neural_net.py --language=jax --implement --detailed

# Full project optimization with HTML report
/optimize project/ --implement --format=html --detailed
```

### Multi-Agent Optimization
```bash
# Scientific computing optimization with specialized agents
/optimize research_code/ --agents=scientific --breakthrough

# AI/ML optimization with intelligent agent selection
/optimize ml_pipeline/ --agents=ai --intelligent --orchestrate

# Complete ecosystem optimization
/optimize complex_project/ --agents=all --orchestrate --implement
```

## Domain-Specific Features

### Scientific Computing
- Numerical algorithm optimization
- Vectorization opportunities
- Memory layout optimization
- Parallel computation analysis

### Machine Learning
- Training loop efficiency
- Data loading optimization
- Model architecture analysis
- Memory management for large models

### Data Processing
- DataFrame operation optimization
- Streaming vs batch processing
- Memory-efficient data handling
- Parallel data transformation

## Integration

### CI/CD Pipeline
JSON output format enables integration with continuous integration systems for automated optimization monitoring.

### Performance Tracking
Regular analysis helps track optimization improvements over time and identify performance regressions.

### Development Workflow
Optimization recommendations guide development decisions and code review processes.

## Common Workflows

### Basic Optimization Workflow
```bash
# 1. Analyze current performance
/optimize src/algorithm.py --profile

# 2. Apply optimizations automatically
/optimize src/algorithm.py --implement --language=python

# 3. Test changes
/generate-tests src/algorithm.py --type=performance
/run-all-tests --performance --profile
```

### Scientific Computing Optimization
```bash
# 1. JAX-specific optimization
/optimize neural_net.py --language=jax --category=memory --implement

# 2. Follow with JAX-specific commands
/jax-performance --technique=caching --gpu-accel
/jax-debug --check-tracers

# 3. Validate results
/double-check "JAX optimization results" --deep-analysis
```

### Multi-Language Project Optimization
```bash
# 1. Auto-detect and optimize
/optimize project/ --language=auto --implement

# 2. Language-specific follow-up
/julia-jit-like project/julia_code/ --type-stability
/python-debug-prof project/python_code/ --suggest-opts
```

## Related Commands

**Prerequisites**: Run before optimization
- `/check-code-quality --auto-fix` - Fix quality issues that affect performance
- `/debug --auto-fix` - Resolve runtime issues before optimizing
- `/generate-tests` - Ensure test coverage for optimization validation
- `/run-all-tests --benchmark` - Establish baseline performance metrics

**Alternatives**: Different optimization approaches
- `/multi-agent-optimize` - Multi-agent analysis for complex optimization problems
- `/jax-performance` - JAX-specific performance optimization and GPU acceleration
- `/julia-jit-like` - Julia-specific JIT optimization and type stability
- `/refactor-clean` - Code structure and architectural optimization
- `/think-ultra` - Research-grade optimization analysis with quantum depth
- `/adopt-code` - Technology migration for performance modernization

**Combinations**: Use together for comprehensive optimization
- `/generate-tests --type=performance` - Create performance-specific tests
- `/run-all-tests --benchmark` - Validate optimization results with metrics
- `/double-check` - Verify optimization effectiveness systematically
- `/reflection --type=scientific` - Analyze optimization patterns and improvements
- `/update-docs` - Document optimization strategies and results

**Follow-up**: Next steps after optimization
- `/run-all-tests --benchmark --profile` - Comprehensive performance validation
- `/commit --template=optimization` - Commit with structured optimization message
- `/ci-setup --monitoring` - Automate performance monitoring in CI/CD
- `/reflection --optimize=performance` - Meta-analyze optimization effectiveness

## Integration Patterns

### Performance Testing Integration
```bash
# Optimization → Testing → Validation cycle
/optimize code.py --implement
/generate-tests code.py --type=performance
/run-all-tests --benchmark --profile
```

### Code Quality Workflow
```bash
# Combined quality and performance improvement
/check-code-quality src/ --auto-fix
/optimize src/ --implement --category=all
/refactor-clean src/ --implement
```

### Scientific Computing Pipeline
```bash
# Full scientific code optimization
/adopt-code legacy/ --target=jax --optimize
/optimize modern_code/ --language=jax --implement
/jax-performance --gpu-accel --optimization
```

## Requirements

- Python 3.7+ with standard libraries
- Language-specific tools for Julia and JAX analysis
- Network access for dependency analysis