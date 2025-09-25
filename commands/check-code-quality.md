---
description: Comprehensive code quality analysis and automated fixing for Python and Julia scientific computing projects with CI/CD integration
category: code-analysis-testing
argument-hint: [target-path] [--fix] [--report] [--benchmark] [--ci-mode]
allowed-tools: Bash, Edit, Read, Glob, MultiEdit, Write, TodoWrite
---

# Advanced Code Quality Analyzer & Fixer

Comprehensive code quality analysis, automated fixing, and continuous monitoring for Python and Julia scientific computing projects with emphasis on numerical accuracy, performance, and research reproducibility standards.

## Usage

```bash
# Comprehensive quality check with auto-fixes
/check-code-quality

# Target specific directory with fixes
/check-code-quality src/ --fix

# Generate detailed quality report
/check-code-quality --report

# Include performance benchmarking
/check-code-quality --benchmark

# Interactive quality improvement with TodoWrite tracking
/check-code-quality --interactive

# CI/CD optimized mode for GitHub Actions
/check-code-quality --ci-mode

# Fast feedback mode for development
/check-code-quality --fast
```

## Intelligent Quality Analysis Process

### **Speed Optimization Strategy**

#### **Smart Change Detection**
```bash
# Analyze only changed files (CI/PR mode)
git diff --name-only origin/main...HEAD | grep -E "\.(py|pyx|pyi)$" | \
  xargs -I {} ruff check {} --fix

# Fast syntax check (< 10 seconds)
ruff check . --select=E9,F63,F7,F82 --show-source

# Incremental type checking
mypy . --cache-dir=.mypy_cache --incremental
```

#### **Parallel Execution Optimization**
```bash
# Parallel tool execution (reduce total time by 50-70%)
(
  ruff check . --fix &
  black . --check &
  mypy . --cache-dir=.mypy_cache &
  wait
)

# Smart test selection (only test affected code)
pytest --lf --tb=short --maxfail=5
```

#### **Cache Management**
```bash
# Pre-warm caches for faster subsequent runs
mkdir -p .mypy_cache .pytest_cache .ruff_cache
export RUFF_CACHE_DIR=.ruff_cache
export MYPY_CACHE_DIR=.mypy_cache
```

### 1. **Project Detection & Environment Setup**

#### **Multi-Language Project Detection**
- **Python Projects**: Detect pyproject.toml, setup.py, requirements.txt, conda environment
- **Julia Projects**: Detect Project.toml, Manifest.toml, package structure  
- **Mixed Projects**: Handle polyglot scientific computing environments
- **Framework Recognition**: Identify pytest, Test.jl, Jupyter notebooks, documentation systems

#### **Environment Validation**
- **Virtual Environment**: Verify and activate appropriate environment
- **Dependencies**: Check for required quality tools and scientific packages
- **Configuration**: Parse existing quality configs (pyproject.toml, .pre-commit-config.yaml)
- **CI Integration**: Detect GitHub Actions, GitLab CI, automated quality gates

### 2. **Python Scientific Computing Quality Analysis**

#### **Code Formatting & Style (Auto-Fix)**
```bash
# Modern Python formatting stack (2025 versions)
ruff format .                    # Fast formatting with Rust-based tool (v0.13.1+)
black .                          # PEP 8 compliant code formatting (v25.1.0+)
isort .                          # Import organization and optimization (v5.13.2+)

# CI-optimized formatting (parallel execution)
ruff format . --check            # Fast check mode for CI
black . --check --diff           # Show formatting differences
```

#### **Advanced Linting & Code Quality**
```bash
# Comprehensive linting with scientific focus (latest versions)
ruff check . --fix --unsafe-fixes    # Modern linter with auto-fixes (v0.13.1+)
mypy .                               # Static type checking (v1.14.0+)
bandit -r . --format json            # Security vulnerability scanning (v1.8.0+)
pylint src/ tests/                   # Comprehensive code analysis (v3.4.0+)

# CI-optimized linting (smart change detection)
ruff check . --diff                  # Show only changes for CI
mypy . --cache-dir=.mypy_cache       # Use cache for faster CI runs
bandit -r . --quiet --format txt     # Minimal output for CI

# Scientific computing specific checks
numpy-stubs-check .                  # NumPy type stub validation
scipy-lint .                         # SciPy best practices (if available)
```

#### **Scientific Code Quality Metrics (Enhanced 2025)**
- **Numerical Stability**: Advanced floating-point precision analysis with tolerance checking
- **Algorithm Complexity**: Automated Big-O analysis and performance profiling
- **Memory Efficiency**: Real-time memory leak detection and allocation optimization
- **Vectorization**: ML-powered detection of vectorization opportunities (NumPy 2.1+)
- **GPU Readiness**: JAX/CuPy compatibility analysis with migration suggestions
- **Reproducibility**: Random seed and determinism validation
- **BLAS/LAPACK**: Optimal linear algebra library usage detection
- **Parallel Computing**: Multiprocessing and threading opportunity identification

### 3. **Julia Package Quality Analysis**

#### **Package Quality Checks**
```julia
# Julia quality analysis
using PkgAudit, Aqua, JET

# Static analysis
Aqua.test_all(MyPackage)             # Comprehensive package quality
JET.@test_opt my_function(args...)   # Optimization analysis
JET.@test_call my_function(args...)  # Type inference validation
```

#### **Performance & Type Analysis**
```julia
# Performance and type stability
using BenchmarkTools, ProfileView

# Type stability analysis
@code_warntype my_function(args...)
@code_llvm my_function(args...)

# Memory allocation analysis  
@allocated my_function(args...)
@benchmark my_function($args...)
```

### 4. **Advanced Testing & Coverage Analysis**

#### **Comprehensive Test Quality**
```python
# Testing with scientific computing focus (latest versions)
pytest --cov=. --cov-report=term-missing --cov-report=html \
       --cov-report=xml --cov-branch --cov-fail-under=85  # pytest v8.3.4+

# Advanced coverage analysis
coverage report --show-missing --skip-covered    # coverage v7.6.0+
coverage html --show-contexts --title="Scientific Computing Coverage"

# CI-optimized testing
pytest --maxfail=5 --tb=short               # Fast failure reporting
pytest -x --lf                              # Stop on first failure, run last failed
pytest --durations=10                       # Show slowest tests

# Test quality metrics
pytest-clarity                       # Better test failure reporting
pytest-benchmark                     # Performance regression testing
pytest-xdist -n auto                 # Parallel test execution (auto-detect cores)
```

#### **Statistical & Numerical Test Validation**
- **Reproducibility**: Verify consistent random seed handling
- **Numerical Accuracy**: Test floating-point precision and stability
- **Statistical Power**: Analyze test statistical significance
- **Performance Baselines**: Automated performance regression detection

### 5. **Security & Vulnerability Analysis**

#### **Scientific Computing Security (2025 Enhanced)**
```bash
# Comprehensive security analysis (latest versions)
bandit -r . --format json --confidence-level medium \
       --severity-level medium --exclude tests/      # bandit v1.8.0+

# Modern dependency security scanning
pip-audit --format=json --output=security_report.json  # pip-audit v2.7.0+
safety check --json --full-report                      # safety v3.2.0+

# Scientific package specific security (2025)
numpy-security-scan . --check-dtype-precision          # NumPy 2.1+ specific checks
jupyter-security-check . --scan-notebooks              # Jupyter 7.0+ security
cve-bin-tool --exclude tests/ .                        # CVE scanning for compiled deps

# GPU computing security
jax-security-check . --gpu-memory-safety               # JAX-specific security patterns
cupy-audit . --memory-leaks                            # CuPy memory management check
```

#### **Research Data Security**
- **Data Privacy**: Check for accidental data exposure in code
- **Credential Scanning**: Detect hardcoded API keys, database credentials
- **Model Security**: Validate ML model serialization security
- **Reproducibility Security**: Ensure secure random number generation

### 6. **Documentation & Research Quality**

#### **Scientific Documentation Standards**
```bash
# Documentation quality analysis
interrogate . --verbose --fail-under=80  # Docstring coverage
pydocstyle . --convention=numpy          # NumPy docstring style
darglint . --verbosity=2                 # Docstring argument validation

# Research documentation
sphinx-build -b html docs docs/_build   # Documentation builds
jupyter nbconvert --execute notebooks/  # Notebook execution validation
```

#### **Research Reproducibility Checks**
- **Environment Documentation**: Verify complete dependency specifications
- **Data Provenance**: Check for clear data source documentation
- **Method Documentation**: Validate algorithm and method descriptions
- **Example Validation**: Ensure all code examples execute correctly

#### **Fast Documentation Checks (CI Mode)**
```bash
# Quick docstring coverage check
interrogate . --fail-under=80 --quiet --exclude=tests

# Fast documentation build check
sphinx-build -b html docs docs/_build -W --keep-going
```

### 7. **Performance & Optimization Analysis**

#### **Scientific Computing Performance**
```python
# Performance profiling and analysis
py-spy record --duration 30 --format speedscope -o profile.json -- \
    python -m pytest tests/performance/

# Memory profiling
memory_profiler                      # Line-by-line memory usage
pympler                             # Memory leak detection
tracemalloc                         # Built-in memory tracing
```

#### **Algorithm Optimization Detection**
- **Vectorization Opportunities**: Identify loops that can be vectorized
- **Numerical Library Usage**: Check for optimal BLAS/LAPACK usage
- **Parallel Computing**: Detect opportunities for multiprocessing/threading
- **GPU Acceleration**: Identify code suitable for JAX/CuPy optimization

### 8. **Automated Quality Fixes**

#### **Scientific Computing Auto-Fixes**
- **Import Optimization**: Reorganize scientific library imports
- **Type Annotations**: Add type hints for scientific functions
- **Docstring Enhancement**: Generate NumPy-style docstrings
- **Performance Optimizations**: Suggest vectorization improvements
- **Test Generation**: Auto-generate basic numerical accuracy tests

#### **Code Modernization (Python 3.12+ & 2025 Standards)**
```python
# Automated code modernization (latest tools)
pyupgrade --py312-plus .                    # Upgrade to Python 3.12+ syntax
autoflake --remove-all-unused-imports .     # Remove unused imports
autopep8 --in-place --aggressive .         # Additional PEP 8 compliance

# Scientific computing modernization
numpy-upgrade --numpy2-compat .            # NumPy 2.0+ compatibility
scipy-modernize --latest-api .             # SciPy 1.14+ API updates
pandas-upgrade --pandas2-ready .           # Pandas 2.2+ optimization

# Performance modernization
vectorize-detector . --suggest-fixes       # Auto-detect vectorization opportunities
jax-converter . --highlight-jitable       # Identify JAX-compatible functions
```

### 9. **Quality Metrics & Reporting**

#### **Scientific Computing Quality Metrics (2025 Standards)**
- **Code Coverage**: Target ≥ 85% (≥ 98% for critical numerical functions)
- **Type Coverage**: Target ≥ 92% for scientific computing functions (mypy 1.14+)
- **Documentation Coverage**: Target ≥ 90% for public APIs (NumPy docstring standard)
- **Performance Regression**: 0 regressions > 15% performance degradation
- **Numerical Accuracy**: All tests within ±1e-12 tolerance (IEEE 754 compliant)
- **Security Issues**: 0 high/critical vulnerabilities (NIST standards)
- **Memory Efficiency**: < 5% memory waste in core algorithms
- **GPU Compatibility**: ≥ 80% JAX/CuPy compatible code patterns
- **Reproducibility**: 100% deterministic results across platforms

#### **Research Quality Indicators**
- **Reproducibility Score**: Measure of result reproducibility
- **Method Validation**: Coverage of algorithm validation tests  
- **Data Quality**: Assessment of test data quality and coverage
- **Publication Readiness**: Metrics for research publication standards

### 10. **Comprehensive Quality Report**

#### **Executive Summary Dashboard**
```markdown
## Code Quality Report
### Overall Score: 92/100 ⭐

#### Quality Metrics
- ✅ Code Coverage: 87.3% (Target: ≥85%)
- ✅ Type Coverage: 94.1% (Target: ≥90%)
- ✅ Security Issues: 0 critical, 1 low
- ⚠️ Performance: 2 regressions detected
- ✅ Documentation: 91.2% coverage

#### Scientific Computing Metrics
- ✅ Numerical Accuracy: All tests pass (±1e-10 tolerance)
- ✅ Reproducibility: 100% consistent results
- ⚠️ Vectorization: 3 optimization opportunities identified
- ✅ GPU Readiness: 85% JAX-compatible code
```

#### **Detailed Analysis Sections**
- **Code Quality Trends**: Historical quality improvement tracking
- **Performance Baselines**: Regression analysis with statistical significance
- **Security Posture**: Vulnerability analysis with remediation priorities
- **Research Standards**: Publication readiness assessment
- **Team Collaboration**: Code review readiness metrics

### 11. **Continuous Quality Monitoring**

#### **Pre-commit Integration (2025 Optimized)**
```yaml
# .pre-commit-config.yaml for scientific computing
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.13.1
    hooks:
      - id: ruff
        args: [--fix, --unsafe-fixes]
      - id: ruff-format

  - repo: https://github.com/psf/black
    rev: 25.1.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.0
    hooks:
      - id: mypy
        additional_dependencies: [numpy==2.1.0, scipy==1.14.0, pandas==2.2.3]
        args: [--cache-dir=.mypy_cache]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.8.0
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]

  - repo: local
    hooks:
      - id: numerical-stability-check
        name: Numerical Stability Check
        entry: python scripts/check_numerical_stability.py
        language: system
        pass_filenames: false
        stages: [manual]  # Only run manually, not in CI
```

#### **CI/CD Quality Gates (GitHub Actions Integration)**
```yaml
# GitHub Actions quality workflow integration
jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Code Quality Check
        run: |
          # Use CI-optimized mode
          claude-code check-code-quality --ci-mode

          # Smart change detection
          if [[ "${{ github.event_name }}" == "pull_request" ]]; then
            git diff --name-only origin/main...HEAD | \
              xargs -I {} claude-code check-code-quality {} --fast
          fi
```

**Features:**
- **Automated Quality Checks**: GitHub Actions integration with smart change detection
- **Performance Monitoring**: Continuous performance regression detection
- **Security Scanning**: Automated vulnerability assessment with bandit v1.8.0+
- **Documentation Updates**: Automatic documentation deployment
- **Parallel Execution**: Multi-core CI optimization
- **Cache Management**: Intelligent caching for faster CI runs
- **Three-Tier Architecture**: dev-feedback (3-5min), optimized-ci (8-12min), legacy (15-20min)

## Advanced Features

### **AI-Powered Code Analysis**
- **Code Smell Detection**: ML-based pattern recognition
- **Optimization Suggestions**: AI-powered performance recommendations
- **Documentation Generation**: Automated docstring and comment generation
- **Test Case Suggestions**: AI-generated test scenarios

### **Developer Experience Enhancements**

#### **Fast Feedback Loop**
```bash
# Ultra-fast syntax and import check (< 5 seconds)
ruff check . --select=E9,F63,F7,F82 --no-fix

# Quick type check for changed files only
git diff --name-only HEAD~1 | grep "\.py$" | xargs mypy

# Immediate security scan
bandit -r . --severity-level high --confidence-level high -q
```

#### **Interactive Quality Improvement**
```bash
# Step-by-step quality improvement with TodoWrite
/check-code-quality --interactive
# Creates todo list for each quality issue found
# Guides through fixes with explanations
# Tracks progress with TodoWrite integration
```

#### **Smart Error Reporting**
- **Contextual Fixes**: Show exact file locations with line numbers
- **Priority Ranking**: Critical issues first, then warnings
- **Progress Tracking**: Visual progress bars for long operations
- **Diff Previews**: Show what will change before applying fixes

### **Team Collaboration Features**
- **Quality Dashboards**: Team-wide quality metrics visualization
- **Code Review Integration**: Quality metrics in PR reviews
- **Knowledge Sharing**: Best practices documentation generation
- **Mentoring Support**: Junior developer guidance integration

### **Research Workflow Integration**
- **Publication Pipeline**: Quality checks for research code publication
- **Data Science Workflows**: Jupyter notebook quality analysis
- **Experiment Tracking**: Quality metrics for experimental code
- **Collaboration Standards**: Multi-institution research code quality

## Command Options

### **Core Options**
- `--fix`: Apply all available auto-fixes
- `--report`: Generate comprehensive quality report
- `--benchmark`: Include performance benchmarking analysis
- `--security`: Focus on security vulnerability analysis
- `--interactive`: Interactive quality improvement with TodoWrite

### **CI/CD Integration Options**
- `--ci-mode`: CI-optimized quality checks (GitHub Actions compatible)
- `--fast`: Fast feedback mode for development (3-5 minutes)
- `--changed-only`: Analyze only changed files (smart detection)
- `--parallel`: Enable parallel execution (auto-detect cores)
- `--cache`: Use intelligent caching for faster runs
- `--timeout=300`: Set timeout for CI environments (default: 300s)

### **Specialized Analysis**
- `--research`: Research-focused quality analysis
- `--baseline-update`: Update performance and quality baselines
- `--team`: Generate team collaboration quality metrics
- `--numerical-stability`: Focus on numerical accuracy checks
- `--gpu-ready`: Check for JAX/CuPy optimization opportunities

## Integration Capabilities

### **Development Environment Integration**
- **IDE Support**: VS Code, PyCharm, Vim integration with quality diagnostics
- **Git Hooks**: Pre-commit and pre-push quality gates (latest versions)
- **Container Support**: Docker and Conda environment quality checks
- **Cloud Integration**: Cloud-based quality analysis and reporting
- **GitHub Actions**: Three-tier CI architecture with smart optimization
- **Cache Management**: Intelligent caching for 50-70% faster CI runs
- **Smart Detection**: Analyze only changed files in PR workflows

### **Scientific Computing Ecosystem (2025 Optimized)**
- **Package Managers**: pip 25.0+, conda 24.9+, mamba 1.5+ compatibility
- **Scientific Libraries**: NumPy 2.1+, SciPy 1.14+, PyMC 5.17+, JAX 0.4+ optimization
- **Visualization**: Matplotlib 3.9+, Plotly 5.24+, Bokeh 3.6+ quality patterns
- **Data Processing**: Pandas 2.2+, Dask 2024.11+, Polars 1.15+ best practices
- **GPU Computing**: JAX/CuPy optimization detection and recommendations
- **Performance**: Automatic vectorization opportunity detection

### **Quality Execution Modes**

#### **Development Mode (Default)**
- Full comprehensive analysis
- All auto-fixes applied
- Detailed reporting with explanations
- Interactive improvement suggestions

#### **CI Mode (--ci-mode)**
- Smart change detection (only analyze changed files)
- Parallel execution (50-70% faster)
- Minimal output for CI logs
- Exit codes optimized for CI/CD
- Timeout protection (300s default)

#### **Fast Mode (--fast)**
- Syntax and import checks only (< 30 seconds)
- Critical security issues only
- Type checking for changed files
- Perfect for pre-commit hooks

#### **Research Mode (--research)**
- Enhanced numerical stability checks
- Reproducibility validation
- Publication-ready documentation standards
- Algorithm complexity analysis

### **Performance Metrics & Targets**

#### **Speed Benchmarks (2025 Optimized)**
- **Fast Mode**: < 30 seconds for typical scientific project
- **CI Mode**: < 5 minutes for PR analysis
- **Full Analysis**: < 15 minutes for comprehensive check
- **Cache Hit**: 50-70% faster on subsequent runs

#### **Quality Standards**
- **Code Coverage**: ≥ 85% (≥ 95% for critical numerical functions)
- **Type Coverage**: ≥ 90% for scientific computing functions
- **Security Issues**: 0 high/critical vulnerabilities
- **Performance**: 0 regressions > 20% degradation
- **Numerical Stability**: All tests within ±1e-10 tolerance

**Target: Achieve comprehensive code quality with scientific computing excellence, research reproducibility standards, and CI/CD optimization for fast, error-free GitHub commits.**