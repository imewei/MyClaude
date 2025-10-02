---
title: "Debug"
description: "Scientific computing debugging with GPU support and multi-language analysis for Python/Julia"
category: debugging
subcategory: scientific-computing
complexity: intermediate
argument-hint: "[--issue=TYPE] [--gpu] [--julia] [--research] [--jupyter] [--profile] [--monitor] [--logs] [--auto-fix] [--report] [--agents=scientific|quality|orchestrator|all]"
allowed-tools: Bash, Read, Grep, Glob, TodoWrite
model: inherit
tags: debugging, gpu, julia, python, jupyter, scientific-computing
dependencies: []
related: [optimize, check-code-quality, jax-debug, julia-jit-like, run-all-tests, generate-tests, commit, multi-agent-optimize, adopt-code]
workflows: [debug-fix-test, gpu-troubleshooting, research-debugging]
version: "2.0"
last-updated: "2025-09-28"
---

# Debug

Scientific computing debugging engine with AI-powered issue detection, GPU acceleration support, and multi-language analysis for Python/Julia ecosystems.

## Quick Start

```bash
# Basic debugging
/debug --interactive

# GPU debugging with profiling
/debug --gpu --profile --monitor

# Julia ecosystem debugging
/debug --julia --issue=performance --profile

# Auto-fix common issues
/debug --auto-fix --report
```

## Usage

```bash
/debug [options] [issue_description]
```

**Parameters:**
- `options` - Environment and analysis configuration options
- `issue_description` - Optional description of the specific issue to investigate

## Options

### Environment Options
- `--gpu`: GPU computing debugging (CUDA/ROCm errors, memory issues)
- `--julia`: Julia ecosystem analysis and debugging
- `--research`: Research workflow and reproducibility analysis
- `--jupyter`: Jupyter notebook and kernel debugging
- `--interactive`: Interactive debugging workflow (default)

### Issue Types
- `--issue=<type>`: Specify issue type
  - `gpu`: CUDA/ROCm errors, driver problems
  - `memory`: Large array handling, OOM errors, memory leaks
  - `numerical`: NaN/inf values, overflow, precision issues
  - `performance`: Slow computations, vectorization problems
  - `environment`: Package conflicts, version mismatches
  - `jupyter`: Kernel issues, notebook problems
  - `julia`: Package issues, Python interoperability
  - `research`: Reproducibility, experiment tracking
  - `data`: Large datasets, format issues, I/O performance
  - `distributed`: MPI, cluster computing, parallel scaling

### Analysis Modes
- `--profile`: Performance profiling for scientific workloads
- `--monitor`: Real-time monitoring of computing resources
- `--logs`: Intelligent log analysis for scientific applications
- `--auto-fix`: Apply automatic fixes for common issues
- `--report`: Generate comprehensive debugging report

### Filtering Options
- `--time-range=<range>`: Time range for analysis (1h, 2h, 1d, 7d)
- `--process=<pid>`: Target specific process by PID or name
- `--severity=<level>`: Filter by severity (all, info, warning, error, critical)

## Features

### Scientific Computing Focus
- **Python Stack**: NumPy, SciPy, JAX, PyTorch, TensorFlow analysis
- **Julia Ecosystem**: Package debugging, performance analysis
- **GPU Computing**: CUDA, ROCm, JAX, CuPy support
- **Jupyter Environment**: Kernel and notebook optimization
- **Research Tools**: Wandb, MLflow, TensorBoard integration

### AI-Powered Detection
- Intelligent pattern recognition for common scientific computing issues
- Automatic categorization of errors and performance bottlenecks
- Context-aware analysis of log files and error messages
- Smart recommendations for fixes and optimizations

### Performance Analysis
- Real-time resource monitoring (GPU, memory, CPU)
- Performance profiling for scientific workloads
- Memory usage analysis for large datasets
- I/O bottleneck identification

### Issue Resolution
- Automated fixes for common environment issues
- Package conflict resolution suggestions
- Memory optimization recommendations
- Performance tuning guidance

## Examples

### Basic Debugging
```bash
# Quick environment check
/debug --interactive

# Specific issue investigation
/debug --issue=memory --profile
```

### GPU Debugging
```bash
# GPU memory issues
/debug --gpu --issue=memory --monitor

# GPU performance analysis
/debug --gpu --profile --report
```

### Scientific Workflow Analysis
```bash
# Julia performance debugging
/debug --julia --issue=performance --profile

# Research reproducibility check
/debug --research --logs --report

# Jupyter kernel problems
/debug --jupyter --logs --time-range=1h
```

### Comprehensive Analysis
```bash
# Full scientific environment analysis
/debug --gpu --julia --jupyter --research --report

# Memory optimization for large workloads
/debug --issue=memory --profile --auto-fix --report
```

## Output

Debug data is saved in `.debug_cache/`:
- `scientific_environment.json`: Environment analysis and library detection
- `scientific_system_state.json`: System state with scientific computing focus
- `scientific_issue_analysis.json`: Detected issues and recommendations
- `scientific_log_analysis.json`: Log analysis for scientific applications
- `scientific_debug_report_*.json`: Comprehensive debugging reports

## Supported Environments

### Python Scientific Stack
- NumPy, SciPy, Pandas for numerical computing
- JAX, PyTorch, TensorFlow for machine learning
- Matplotlib, Plotly, Seaborn for visualization
- Jupyter, IPython for interactive computing

### Julia Scientific Computing
- DifferentialEquations.jl, MLJ.jl, Flux.jl
- Plots.jl, StatsPlots.jl for visualization
- IJulia for Jupyter integration
- Package manager and environment analysis

### GPU Computing Frameworks
- CUDA and ROCm driver analysis
- JAX, CuPy, PyTorch GPU, TensorFlow GPU
- Memory management and optimization
- Multi-GPU and distributed computing support

### Research and HPC Tools
- SLURM, PBS job scheduler integration
- MPI and distributed computing analysis
- Container support (Docker, Singularity)
- Version control and experiment tracking

## Common Workflows

### Basic Debug Workflow
```bash
# 1. Quick environment check
/debug --interactive

# 2. Fix identified issues
/debug --auto-fix --report

# 3. Verify fixes
/run-all-tests --profile
```

### GPU Debugging Workflow
```bash
# 1. GPU-specific debugging
/debug --gpu --issue=memory --monitor

# 2. JAX GPU optimization
/jax-debug --disable-jit --check-tracers
/jax-performance --gpu-accel

# 3. Validate GPU performance
/run-all-tests --gpu --benchmark
```

### Scientific Computing Debug
```bash
# 1. Research environment analysis
/debug --research --julia --jupyter --logs

# 2. Fix environment issues
/debug --auto-fix --issue=environment

# 3. Test scientific workflows
/generate-tests research/ --type=scientific
/run-all-tests --scientific --gpu
```

## Related Commands

**Prerequisites**: Commands to run first
- Clean working environment - Remove temporary files and clear caches
- `/check-code-quality` - Fix syntax and style issues before debugging runtime problems
- Baseline measurements - Record current performance before debugging

**Alternatives**: Different debugging approaches
- `/jax-debug --disable-jit` - JAX-specific debugging with JIT disabled
- `/python-debug-prof --suggest-opts` - Python performance profiling with optimization suggestions
- `/julia-jit-like --benchmark` - Julia performance debugging with benchmarking
- `/multi-agent-optimize --mode=review` - Multi-agent debugging analysis
- Native debuggers (pdb, gdb, cuda-gdb) for specific use cases

**Combinations**: Use together for comprehensive debugging
- `/optimize --implement` - Apply performance fixes identified by debugging
- `/generate-tests --type=performance` - Create tests for debugged performance issues
- `/double-check --deep-analysis` - Verify debug solutions systematically
- `/adopt-code --optimize` - Modernize legacy code causing issues
- `/reflection --type=scientific` - Analyze debugging process effectiveness

**Follow-up**: Next steps after debugging
- `/optimize --implement` - Apply performance optimizations after fixing bugs
- `/run-all-tests --auto-fix` - Comprehensive testing with automatic fixes
- `/commit --template=fix --ai-message` - Commit debug fixes with proper documentation
- `/ci-setup --monitoring` - Set up monitoring to prevent similar issues

## Integration Patterns

### Debug → Fix → Test Cycle
```bash
# Complete debugging workflow
/debug --gpu --auto-fix --report
/optimize --implement --category=memory
/run-all-tests --performance --gpu
```

### Research Environment Setup
```bash
# Scientific computing environment debugging
/debug --research --jupyter --auto-fix
/julia-jit-like --benchmark --precompile
/jax-essentials --gpu --static-args
```

### Production Debugging
```bash
# Production issue resolution
/debug --logs --monitor --time-range=1h
/check-code-quality --auto-fix --security
/ci-setup --monitoring --security
```

## Exit Codes

- `0`: Debugging completed successfully
- `1`: Issues found requiring attention
- `2`: Critical GPU, memory, or numerical issues detected
- `3`: Environment or configuration errors