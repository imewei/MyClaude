---
title: "Generate Tests"
description: "Generate comprehensive test suites for Python, Julia, and JAX scientific computing projects"
category: testing
subcategory: test-generation
complexity: intermediate
argument-hint: "[target-file-or-module] [--type=all|unit|integration|performance|jax|scientific|gpu] [--framework=auto|pytest|julia|jax] [--coverage=N] [--agents=scientific|quality|orchestrator|all]"
allowed-tools: Read, Write, Edit, Grep, Glob, TodoWrite, Bash, Task
model: inherit
tags: testing, test-generation, python, julia, jax, scientific-computing
dependencies: []
related: [run-all-tests, check-code-quality, debug, optimize, double-check]
workflows: [test-generation, quality-assurance, scientific-testing]
version: "2.1"
last-updated: "2025-09-28"
---

# Generate Tests

Generate comprehensive test suites for Python, Julia, and JAX scientific computing projects.

## Quick Start

```bash
# Generate complete test suite
/generate-tests src/utils.py

# Unit tests with coverage target
/generate-tests src/ --type=unit --coverage=90

# JAX-specific tests
/generate-tests models/ --type=jax --framework=jax

# Performance testing with GPU
/generate-tests simulation.py --type=performance --gpu
```

## Usage

```bash
/generate-tests [target] [options]
```

**Parameters:**
- `target` - File, directory, or module to generate tests for
- `options` - Test type, framework, and configuration options

## Options

| Option | Description | Default |
|--------|-------------|----------|
| `--type=<type>` | Test type: all, unit, integration, performance, jax, scientific, gpu | all |
| `--framework=<framework>` | Testing framework: auto, pytest, julia, jax | auto |
| `--coverage=N` | Target coverage percentage | 85 |
| `--interactive` | Interactive mode with user guidance | false |
| `--security` | Include security testing | false |
| `--gpu` | Include GPU/accelerated testing | false |

## Test Types

- `all` - Complete test suite (default)
- `unit` - Unit tests only
- `integration` - Integration tests
- `performance` - Performance tests
- `jax` - JAX ecosystem testing
- `scientific` - Scientific computing validation
- `gpu` - GPU/accelerated computing tests

## Frameworks

- `auto` - Auto-detect best framework (default)
- `pytest` - Python pytest framework
- `julia` - Julia Test.jl framework
- `jax` - JAX ecosystem (Flax, Optax, Chex)

## Examples

```bash
# Generate complete test suite
/generate-tests src/algorithm.py

# Unit tests with pytest
/generate-tests src/ --type=unit --framework=pytest

# JAX ecosystem tests
/generate-tests models/ --type=jax --framework=jax

# Performance tests with GPU
/generate-tests simulation.py --type=performance --gpu

# Scientific computing validation
/generate-tests numerical/ --type=scientific --coverage=95

# Interactive test generation
/generate-tests complex_module.py --interactive
```

## Features

### Core Testing
- Unit and integration test generation
- Edge case and boundary condition testing
- Performance benchmarking
- Code coverage analysis
- Mock and fixture generation

### Framework-Specific
**JAX Ecosystem:**
- Flax model testing
- Optax optimizer validation
- XLA compilation testing
- GPU/TPU performance testing

**Scientific Computing:**
- Numerical stability testing
- Reproducibility validation
- Cross-platform compatibility
- Research-grade testing standards

**Security Testing:**
- Input validation testing
- Authentication testing
- Encryption function validation

## Common Workflows

### Basic Test Generation
```bash
# 1. Generate complete test suite
/generate-tests src/module.py

# 2. Run tests to verify
/run-all-tests --coverage

# 3. Commit tests
/commit --template=test --ai-message
```

### Performance Testing Workflow
```bash
# 1. Generate performance tests
/generate-tests algorithm.py --type=performance --gpu

# 2. Benchmark performance
/run-all-tests --benchmark --profile

# 3. Optimize based on results
/optimize algorithm.py --implement
```

### Scientific Computing Testing
```bash
# 1. JAX-specific test generation
/generate-tests models/ --type=jax --framework=jax

# 2. Scientific validation tests
/generate-tests simulation.py --type=scientific --coverage=95

# 3. Run comprehensive tests
/run-all-tests --scientific --gpu
```

## Related Commands

**Prerequisites**: Commands to run before generating tests
- `/check-code-quality --auto-fix` - Fix code quality issues first
- `/debug --auto-fix` - Resolve runtime issues before testing
- `/explain-code` - Understand code structure and patterns
- `/optimize` - Optimize performance before performance testing

**Alternatives**: Different testing approaches
- `/run-all-tests` - Execute existing tests instead of generating new ones
- Manual test writing for highly specialized scenarios
- `/multi-agent-optimize --mode=review` - Multi-agent test strategy analysis

**Combinations**: Commands that work with test generation
- `/optimize --implement` - Optimize code then generate performance tests
- `/double-check` - Verify test completeness and coverage
- `/commit --template=test` - Commit generated tests with proper message
- `/refactor-clean` - Refactor code then generate updated tests

**Follow-up**: Commands to run after test generation
- `/run-all-tests --auto-fix` - Execute and fix failing tests
- `/ci-setup --type=basic` - Automate test execution in CI
- `/fix-commit-errors` - Fix CI test failures
- `/update-docs --type=api` - Document testing strategy

## Integration Patterns

### Test-Driven Development
```bash
# TDD workflow
/generate-tests new_feature.py --type=unit
/run-all-tests --coverage=100
/commit --template=test --validate
```

### Code Quality Workflow
```bash
# Quality assurance cycle
/check-code-quality --auto-fix
/generate-tests --type=all --security
/run-all-tests --coverage=90
```

### Scientific Computing Pipeline
```bash
# Research code testing
/generate-tests research/ --type=scientific --framework=auto
/jax-debug --check-tracers
/run-all-tests --gpu --reproducible
```

## Requirements

- Python 3.7+ with standard libraries
- Testing frameworks (pytest, Julia Test.jl, etc.)
- Network access for dependency installation