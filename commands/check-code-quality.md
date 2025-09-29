---
title: "Check Code Quality"
description: "Code quality analysis for Python, Julia, and JAX ecosystems with scientific computing focus"
category: code-quality
subcategory: analysis
complexity: basic
argument-hint: "[--language=python|julia|jax|auto] [--analysis=basic|scientific|gpu] [--auto-fix] [--format=text|json] [--agents=quality|scientific|orchestrator|all] [target-path]"
allowed-tools: Bash, Edit, Read, Glob, MultiEdit, Write, TodoWrite
model: inherit
tags: code-quality, analysis, python, julia, jax, scientific-computing
dependencies: []
related: [optimize, refactor-clean, multi-agent-optimize, debug, generate-tests, run-all-tests, commit, double-check, adopt-code]
workflows: [quality-analysis, code-review, pre-commit-validation]
version: "2.0"
last-updated: "2025-09-28"
---

# Check Code Quality

Analyze code quality across Python, Julia, and JAX ecosystems with scientific computing focus.

```bash
/check-code-quality [options] [target-path]

# Basic usage
/check-code-quality src/
/check-code-quality --language=python myfile.py
/check-code-quality --analysis=scientific --auto-fix .
```

## Options

- `--language=<lang>`: Target language (python, julia, jax, auto)
- `--analysis=<type>`: Analysis type (basic, scientific, gpu)
- `--auto-fix`: Apply automatic fixes where safe
- `--format=<fmt>`: Output format (text, json)
- `--profile`: Include performance profiling
- `--agents=<agents>`: Agent selection (quality, scientific, orchestrator, all)

## Agent Integration

### Quality Agent (`code-quality-master`)
- **Comprehensive Testing**: Multi-level testing strategy and automated quality assurance
- **Code Review**: Static analysis, security scanning, and architecture validation
- **Performance Analysis**: Profiling, optimization, and monitoring integration
- **Accessibility**: WCAG compliance and inclusive design validation
- **Build Optimization**: Development environment and tooling enhancement

### Scientific Computing Agent (`scientific-computing-master`)
- **Scientific Analysis**: Numerical accuracy and reproducibility validation
- **Multi-Language Support**: Python, Julia/SciML, JAX ecosystem analysis
- **GPU Computing**: CUDA, JAX acceleration analysis and optimization
- **Research Validation**: Publication-ready code quality and standards
- **Domain Expertise**: Physics, chemistry, biology computational analysis

### Orchestrator Agent (`multi-agent-orchestrator`)
- **Quality Coordination**: Multi-agent quality assurance workflows
- **Resource Management**: Intelligent allocation of quality analysis tasks
- **Workflow Integration**: Complex quality pipeline orchestration
- **Performance Monitoring**: Quality system efficiency and optimization
- **Automated Validation**: Quality gate coordination and enforcement

## Agent Selection Options

- `quality` - Quality engineering focus for comprehensive code analysis
- `scientific` - Scientific computing focus for numerical and research validation
- `orchestrator` - Multi-agent coordination for complex quality workflows
- `all` - Complete multi-agent quality system with specialized expertise

## Analysis Types

### Basic Analysis
- Code structure and style assessment
- Common bug pattern detection
- Performance anti-pattern identification
- Documentation quality review

### Scientific Analysis
- Numerical stability assessment
- Reproducibility validation
- Research workflow compatibility
- Scientific computing best practices

### GPU Analysis
- GPU acceleration opportunities
- Memory optimization patterns
- Parallel processing candidates
- Hardware utilization assessment

## Language Support

### Python
- **Scientific Stack**: NumPy, SciPy, Pandas optimization
- **Machine Learning**: PyTorch, JAX integration patterns
- **Performance**: JIT compilation opportunities, vectorization
- **Quality**: PEP 8 compliance, type hints, documentation

### Julia
- **Performance**: Type stability, multiple dispatch optimization
- **Scientific Computing**: Best practices for numerical code
- **Memory Management**: Allocation optimization
- **Parallel Computing**: Threading and distributed patterns

### JAX Ecosystem
- **Compilation**: XLA optimization opportunities
- **Memory**: Device placement and memory management
- **Transformations**: vmap, pmap, grad usage patterns
- **Performance**: JIT compilation and optimization

## Code Quality Metrics

### Structure Analysis
- Cyclomatic complexity assessment
- Function and class design patterns
- Module organization and dependencies
- API design quality

### Performance Analysis
- Algorithm complexity evaluation
- Memory usage patterns
- I/O optimization opportunities
- Parallel processing potential

### Scientific Computing Quality
- Numerical algorithm correctness
- Error handling and edge cases
- Reproducibility standards
- Documentation completeness

## Auto-Fix Capabilities

### Safe Transformations
- Import optimization and organization
- Variable naming improvements
- Simple performance optimizations
- Documentation formatting

### Performance Optimizations
- List comprehension conversions
- Vectorization opportunities
- Memory allocation improvements
- Algorithm complexity reductions

### Scientific Computing Fixes
- Numerical stability improvements
- Reproducibility enhancements
- Error handling additions
- Type annotation improvements

## Examples

```bash
# Basic code quality check
/check-code-quality src/

# Scientific computing analysis
/check-code-quality --analysis=scientific research_code/

# GPU optimization analysis
/check-code-quality --analysis=gpu --language=jax neural_net.py

# Auto-fix with profiling
/check-code-quality --auto-fix --profile data_analysis.py

# JSON output for CI/CD
/check-code-quality --format=json project/

# Language-specific analysis with scientific agent
/check-code-quality --language=julia --analysis=scientific --agents=scientific simulation.jl

# Multi-agent quality assurance
/check-code-quality --agents=all --auto-fix --format=json project/
```

## Output Information

### Quality Assessment
- Overall quality score and breakdown
- Issue categorization by severity
- Performance improvement opportunities
- Best practice recommendations

### Scientific Computing Analysis
- Reproducibility assessment
- Numerical stability evaluation
- Research workflow compatibility
- Publication readiness review

### Optimization Opportunities
- JIT compilation candidates
- Vectorization possibilities
- GPU acceleration potential
- Memory optimization targets

## Integration

### CI/CD Pipeline
JSON output format enables integration with continuous integration systems for automated quality monitoring.

### Research Workflow
Scientific analysis helps validate code quality for academic publications and reproducible research.

### Development Process
Quality metrics guide code review processes and development best practices.

## Common Workflows

### Pre-Commit Quality Check
```bash
# 1. Check quality before committing
/check-code-quality src/ --auto-fix

# 2. Apply additional fixes if needed
/refactor-clean src/ --patterns=modern --implement

# 3. Commit quality improvements
/commit --template=refactor --validate
```

### Scientific Code Validation
```bash
# 1. Scientific computing quality analysis
/check-code-quality research/ --analysis=scientific --language=python

# 2. Apply JAX-specific optimizations
/check-code-quality jax_code/ --analysis=gpu --language=jax --auto-fix

# 3. Validate with comprehensive testing
/generate-tests research/ --type=scientific
/run-all-tests --scientific --reproducible
```

### Multi-Language Project Analysis
```bash
# 1. Auto-detect and analyze all languages
/check-code-quality project/ --language=auto --auto-fix

# 2. Language-specific follow-up
/check-code-quality julia_module.jl --language=julia --analysis=scientific
/check-code-quality python_code/ --language=python --analysis=gpu
```

## Related Commands

**Prerequisites**: Commands to run before quality analysis
- `/debug --auto-fix` - Fix runtime issues that affect quality analysis
- Clean working directory - Ensure no temporary files interfere

**Alternatives**: Different quality approaches
- `/multi-agent-optimize --mode=review` - Multi-agent comprehensive code review
- `/refactor-clean --patterns=modern` - Structure-focused modernization
- `/optimize --category=all` - Performance-focused quality analysis
- `/think-ultra` - Research-grade quality analysis with quantum depth
- `/adopt-code --analyze` - Legacy code quality assessment

**Combinations**: Commands that work with quality checks
- `/generate-tests --coverage=95` - Test quality improvements comprehensively
- `/optimize --implement` - Apply performance optimizations after quality fixes
- `/double-check --deep-analysis` - Systematically verify quality improvements
- `/refactor-clean --implement` - Apply structural improvements
- `/commit --template=quality` - Commit quality improvements with proper documentation

**Follow-up**: Commands to run after quality analysis
- `/refactor-clean --implement` - Apply identified structural improvements
- `/optimize --implement` - Apply performance optimizations found
- `/run-all-tests --coverage` - Validate quality changes don't break functionality
- `/commit --validate` - Commit quality improvements with validation
- `/reflection --type=instruction` - Analyze quality improvement effectiveness

## Integration Patterns

### Quality Gate Pipeline
```bash
# Pre-commit quality gate
/check-code-quality --auto-fix
/run-all-tests --coverage=90
/commit --validate
```

### Scientific Computing Quality
```bash
# Research code quality workflow
/check-code-quality research/ --analysis=scientific --auto-fix
/optimize research/ --language=julia --implement
/generate-tests research/ --type=scientific --coverage=95
```

### CI/CD Integration
```bash
# Automated quality monitoring
/check-code-quality --format=json > quality_report.json
/multi-agent-optimize --mode=review --implement
/run-all-tests --coverage --report
```

## Requirements

- Python 3.7+ with standard libraries
- Language-specific tools for Julia and JAX analysis
- Network access for dependency analysis