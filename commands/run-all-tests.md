---
title: "Run All Tests"
description: "Comprehensive test execution engine with intelligent failure resolution and performance benchmarking"
category: testing
subcategory: test-execution
complexity: basic
argument-hint: "[--scope=all|unit|integration|performance] [--profile] [--benchmark] [--scientific] [--gpu] [--parallel] [--reproducible] [--coverage] [--report] [--implement] [--auto-fix] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--dry-run] [--backup] [--rollback] [--orchestrate] [--intelligent] [--distributed] [--validate]"
allowed-tools: Bash, Read, Write, Glob, MultiEdit, TodoWrite
model: inherit
tags: testing, test-execution, benchmarking, scientific-computing, auto-fix
dependencies: []
related: [generate-tests, check-code-quality, debug, optimize, double-check]
workflows: [test-execution, performance-validation, quality-assurance]
version: "2.1"
last-updated: "2025-09-29"
---

# Run All Tests

Comprehensive test execution engine for multiple frameworks with performance benchmarking and failure resolution.

## Quick Start

```bash
# Run all tests
/run-all-tests

# Unit tests with coverage
/run-all-tests --scope=unit --coverage

# Scientific computing tests
/run-all-tests --scientific --benchmark --reproducible

# Auto-fix failing tests
/run-all-tests --auto-fix --coverage
```

## Usage

```bash
/run-all-tests [options]
```

**Parameters:**
- `options` - Test execution, analysis, and validation configuration

## Options

- `--scope=<scope>`: Test scope (all, unit, integration, performance)
- `--profile`: Enable performance profiling
- `--benchmark`: Run performance benchmarks
- `--scientific`: Scientific computing optimization
- `--gpu`: Enable GPU/TPU testing
- `--parallel`: Run tests in parallel
- `--reproducible`: Ensure reproducible results
- `--coverage`: Generate coverage reports
- `--report`: Generate detailed test reports
- `--implement`: Automatically fix test failures iteratively until 100% pass rate (primary flag)
- `--auto-fix`: Alias for --implement (for backward compatibility)
- `--agents=<agents>`: Agent selection (auto, core, scientific, ai, engineering, domain, quality, research, all)
- `--orchestrate`: Enable advanced 23-agent orchestration with distributed testing
- `--intelligent`: Enable intelligent agent selection based on test analysis
- `--distributed`: Enable distributed testing across multiple agent domains
- `--validate`: Validate test results and coverage metrics

## Supported Frameworks

### Python
- **Framework**: pytest
- **Discovery**: test_*.py, *_test.py files
- **Configuration**: pytest.ini, setup.cfg, pyproject.toml
- **Features**: Coverage, profiling, parallel execution

### Julia
- **Framework**: Pkg.test()
- **Discovery**: test/ directory, *Test.jl files
- **Configuration**: Project.toml
- **Features**: Performance testing, distributed execution

### JavaScript
- **Framework**: Jest
- **Discovery**: *.test.js, *.spec.js files
- **Configuration**: jest.config.js, package.json
- **Features**: Snapshot testing, coverage reports

### Rust
- **Framework**: cargo test
- **Discovery**: Cargo.toml, tests/ directory
- **Configuration**: Cargo.toml
- **Features**: Integration tests, benchmarks

### Go
- **Framework**: go test
- **Discovery**: *_test.go files
- **Configuration**: go.mod
- **Features**: Benchmarks, race detection

### C/C++
- **Framework**: CTest/Google Test
- **Discovery**: CMakeLists.txt, test*.cpp files
- **Configuration**: CMakeLists.txt, Makefile
- **Features**: Memory checking, performance tests

## 23-Agent Distributed Testing Orchestration

### Intelligent Testing Agent Selection (`--intelligent`)
**Auto-Testing Algorithm**: Analyzes codebase, frameworks, and test requirements to automatically deploy optimal agent combinations from the 23-agent library for maximum testing efficiency.

```bash
# Test Type Detection → Agent Selection
- Scientific Computing → scientific-computing-master + jax-pro + neural-networks-master
- AI/ML Pipelines → ai-systems-architect + data-professional + neural-networks-master
- Web Applications → fullstack-developer + systems-architect + code-quality-master
- Quantum Computing → advanced-quantum-computing-expert + scientific-computing-master
- Legacy Modernization → scientific-code-adoptor + systems-architect + devops-security-engineer
```

### Core Testing Agents

#### **`code-quality-master`** - Testing Strategy & Quality Assurance
- **Test Strategy**: Comprehensive testing pyramid with quality gates and validation
- **Automated Quality**: Test coverage analysis, mutation testing, and performance regression
- **Advanced Debugging**: Root cause analysis and systematic test failure investigation
- **Accessibility Testing**: WCAG compliance validation and inclusive design testing
- **Performance Testing**: Load testing, stress testing, and benchmark validation

#### **`scientific-computing-master`** - Scientific & Numerical Testing
- **Scientific Validation**: Numerical accuracy, reproducibility, and scientific method testing
- **Multi-Language Testing**: Python, Julia/SciML, JAX ecosystem comprehensive testing
- **Research Workflow**: Experiment validation and publication-ready testing standards
- **Performance Analysis**: Scientific computing optimization and GPU acceleration testing
- **Domain Integration**: Physics, chemistry, biology simulation validation

#### **`multi-agent-orchestrator`** - Testing Coordination & Resource Management
- **Distributed Testing**: Multi-framework parallel test execution across agent domains
- **Workflow Management**: Complex test pipeline orchestration and dependency management
- **Resource Optimization**: Intelligent load balancing and distributed task allocation
- **Fault Tolerance**: Test failure recovery, resilient execution, and automatic retry
- **Performance Monitoring**: Test system efficiency tracking and bottleneck analysis

### Specialized Testing Agents

#### **Scientific Computing Testing**
- **`jax-pro`**: JAX ecosystem testing with GPU acceleration and scientific ML validation
- **`neural-networks-master`**: Deep learning model testing, architecture validation, and ML pipeline testing
- **`advanced-quantum-computing-expert`**: Quantum computing algorithm testing and quantum-classical hybrid validation
- **`research-intelligence-master`**: Research methodology testing and academic standard validation

#### **Engineering & Architecture Testing**
- **`systems-architect`**: System integration testing, architecture validation, and scalability testing
- **`fullstack-developer`**: Full-stack application testing, UI/UX validation, and end-to-end testing
- **`devops-security-engineer`**: Security testing, infrastructure validation, and DevSecOps pipeline testing
- **`ai-systems-architect`**: AI system testing, model deployment validation, and ML infrastructure testing

#### **Domain-Specific Testing Experts**
- **`data-professional`**: Data pipeline testing, ETL validation, and analytics workflow testing
- **`database-workflow-engineer`**: Database testing, query optimization validation, and data integrity testing
- **`visualization-interface-master`**: UI testing, visualization validation, and user interface testing
- **`command-systems-engineer`**: Command system testing, workflow validation, and CLI testing

#### **Scientific Domain Testing**
- **`correlation-function-expert`**: Statistical testing, correlation analysis validation, and statistical method testing
- **`neutron-soft-matter-expert`**: Neutron scattering simulation testing and experimental data validation
- **`xray-soft-matter-expert`**: X-ray analysis testing, scattering validation, and experimental workflow testing
- **`nonequilibrium-stochastic-expert`**: Stochastic process testing, nonequilibrium system validation
- **`scientific-code-adoptor`**: Legacy code testing, modernization validation, and migration testing

### Advanced Agent Testing Strategies

#### **`auto`** - Intelligent Agent Selection for Testing
Automatically analyzes test requirements and deploys optimal agent combinations:
- **Test Analysis**: Detects test frameworks, languages, domain patterns, and complexity
- **Agent Matching**: Maps detected patterns to relevant testing expertise
- **Efficiency Optimization**: Balances comprehensive testing with execution speed
- **Dynamic Scaling**: Adjusts agent allocation based on test scope and requirements

#### **`scientific`** - Scientific Computing Testing Team
- `scientific-computing-master` (lead testing coordinator)
- `jax-pro` (JAX ecosystem testing)
- `neural-networks-master` (ML testing)
- `research-intelligence-master` (research validation)
- Domain-specific experts based on scientific domain detection

#### **`ai`** - AI/ML Testing Orchestration
- `ai-systems-architect` (AI system testing lead)
- `neural-networks-master` (deep learning testing)
- `data-professional` (data pipeline testing)
- `jax-pro` (scientific ML testing)
- `visualization-interface-master` (ML visualization testing)

#### **`engineering`** - Software Engineering Testing
- `systems-architect` (architecture testing lead)
- `fullstack-developer` (application testing)
- `devops-security-engineer` (infrastructure testing)
- `code-quality-master` (quality testing)
- `database-workflow-engineer` (data system testing)

#### **`domain`** - Specialized Domain Testing
Activates domain-specific testing experts based on codebase analysis:
- `correlation-function-expert` (statistical computing testing)
- `neutron-soft-matter-expert` (neutron scattering testing)
- `xray-soft-matter-expert` (X-ray analysis testing)
- `nonequilibrium-stochastic-expert` (stochastic process testing)
- `scientific-code-adoptor` (legacy code testing)

#### **`all`** - Complete 23-Agent Testing Ecosystem
Activates all relevant agents with intelligent orchestration for comprehensive testing coverage across all domains.

### 23-Agent Distributed Testing (`--orchestrate` + `--distributed`)

#### **Multi-Agent Testing Pipeline**
1. **Test Discovery Phase**: Multiple agents analyze different aspects of the codebase simultaneously
2. **Parallel Test Execution**: Domain experts execute specialized tests in their areas
3. **Cross-Agent Validation**: Tests are validated across multiple agent perspectives
4. **Intelligent Failure Resolution**: Multi-agent collaboration for complex failure analysis
5. **Performance Benchmarking**: Specialized agents provide domain-specific performance validation

#### **Distributed Resource Management**
- **Load Balancing**: Optimal distribution of tests across agents and compute resources
- **Dependency Coordination**: Sequential test execution for dependent test suites
- **Failure Isolation**: Isolated failure analysis prevents cascade failures
- **Performance Optimization**: Real-time optimization of test execution efficiency

## Test Execution Features

### Automatic Discovery
- Detects test frameworks automatically
- Identifies test files and configurations
- Sets up appropriate test environments
- Handles dependencies and requirements

### Failure Resolution
- Analyzes test failures and errors
- Provides diagnostic information
- Suggests fixes for common issues
- Attempts automatic resolution where safe

### Auto-Fix Mode (--auto-fix)
- Iteratively runs tests until 100% pass rate achieved
- Automatically applies fixes for detected issues
- Handles common failure patterns:
  - Import/dependency errors
  - Syntax errors
  - Type mismatches
  - Missing test data/fixtures
  - Configuration issues
- Tracks fix attempts to prevent infinite loops
- Reports all fixes applied during execution
- Stops after maximum fix attempts or when no progress made

### Performance Analysis
- Execution time tracking
- Memory usage monitoring
- Performance regression detection
- Benchmark result comparison

### Scientific Computing Support
- JAX/NumPy/SciPy test optimization
- Numerical accuracy validation
- Reproducibility verification
- GPU acceleration testing

## Execution Modes

### Standard Mode
- Sequential test execution
- Basic failure reporting
- Standard output formatting
- Exit on first failure option

### Parallel Mode
- Multi-process test execution
- Load balancing across workers
- Shared result aggregation
- Resource usage optimization

### Scientific Mode
- Numerical stability testing
- Reproducibility validation
- Performance benchmarking
- Research workflow compatibility

### Coverage Mode
- Code coverage analysis
- Branch coverage tracking
- Missing coverage identification
- Coverage report generation

## Advanced 23-Agent Testing Examples

```bash
# Intelligent auto-selection with test analysis
/run-all-tests --agents=auto --intelligent --coverage --orchestrate

# Scientific computing with distributed testing
/run-all-tests --agents=scientific --scientific --gpu --distributed --benchmark

# AI/ML pipeline testing with specialized agents
/run-all-tests --agents=ai --benchmark --coverage --parallel --orchestrate

# Complete 23-agent testing ecosystem
/run-all-tests --agents=all --orchestrate --distributed --coverage --auto-fix

# Engineering testing with architecture validation
/run-all-tests --agents=engineering --scope=integration --coverage --orchestrate

# Domain-specific testing for specialized code
/run-all-tests --agents=domain --scientific --reproducible --benchmark

# Quantum computing testing with expert agents
/run-all-tests quantum_algorithm/ --agents=scientific --intelligent --gpu

# Legacy code testing with modernization experts
/run-all-tests legacy_tests/ --agents=auto --intelligent --auto-fix

# Cross-domain testing with full orchestration
/run-all-tests complex_project/ --agents=all --orchestrate --distributed --auto-fix

# JAX/Scientific ML testing
/run-all-tests jax_models/ --agents=scientific --gpu --benchmark --reproducible

# Full-stack application testing
/run-all-tests webapp/ --agents=engineering --scope=all --coverage --parallel

# Research validation with publication-ready testing
/run-all-tests research_code/ --agents=scientific --reproducible --benchmark --report
```

### Intelligent Agent Selection Examples

```bash
# Test Type Detection → Intelligent Agent Selection

# Scientific computing project
/run-all-tests simulation.py --agents=auto --intelligent
# → Selects: scientific-computing-master + jax-pro + research-intelligence-master

# Machine learning pipeline
/run-all-tests ml_pipeline/ --agents=auto --intelligent
# → Selects: ai-systems-architect + neural-networks-master + data-professional

# Quantum computing research
/run-all-tests quantum_circuit.py --agents=auto --intelligent
# → Selects: advanced-quantum-computing-expert + scientific-computing-master

# Web application with database
/run-all-tests webapp/ --agents=auto --intelligent
# → Selects: fullstack-developer + database-workflow-engineer + systems-architect

# Statistical analysis code
/run-all-tests stats_analysis/ --agents=auto --intelligent
# → Selects: correlation-function-expert + research-intelligence-master + code-quality-master

# Multi-domain scientific project
/run-all-tests complex_research/ --agents=all --orchestrate --distributed
# → Activates: All 23 agents with intelligent coordination and distributed execution
```

## Output Information

### Test Results
- Total tests run
- Pass/fail counts
- Execution times
- Coverage percentages

### Failure Analysis
- Error messages and stack traces
- Failure categorization
- Root cause identification
- Suggested fixes

### Performance Metrics
- Execution time statistics
- Memory usage analysis
- Performance trends
- Benchmark comparisons

### Reports
- Detailed HTML/XML reports
- Coverage visualization
- Performance dashboards
- CI/CD integration data

## Integration

### CI/CD Pipelines
- Exit codes for automation
- Structured output formats
- Artifact generation
- Performance tracking

### Development Workflow
- Pre-commit hook integration
- IDE test runner compatibility
- Live reload support
- Debug information

### Scientific Computing
- Research workflow integration
- Publication-ready results
- Reproducibility documentation
- Collaboration features

## Common Workflows

### Basic Test Execution
```bash
# 1. Run all tests with coverage
/run-all-tests --coverage --report

# 2. Fix any failures
/run-all-tests --auto-fix

# 3. Validate fix success
/run-all-tests --coverage=95
```

### Performance Testing Workflow
```bash
# 1. Performance benchmarking
/run-all-tests --benchmark --profile --scope=performance

# 2. Optimize based on results
/optimize slow_functions.py --implement

# 3. Validate performance improvements
/run-all-tests --benchmark --scope=performance
```

### Scientific Computing Validation
```bash
# 1. Scientific computing tests with GPU
/run-all-tests --scientific --gpu --reproducible

# 2. Generate missing tests if needed
/generate-tests research/ --type=scientific

# 3. Full validation cycle
/run-all-tests --scientific --coverage=95 --reproducible
```

## Related Commands

**Prerequisites**: Commands to run before testing
- `/generate-tests` - Create tests if missing
- `/check-code-quality --auto-fix` - Fix quality issues first
- `/debug --auto-fix` - Fix runtime issues

**Alternatives**: Different testing approaches
- Manual test execution for specific frameworks
- Individual test file execution

**Combinations**: Commands that work with test execution
- `/optimize` - Optimize after performance testing
- `/double-check` - Verify test completeness
- `/commit` - Commit after successful tests

**Follow-up**: Commands to run after testing
- `/optimize --implement` - Fix performance issues found
- `/generate-tests` - Add tests for uncovered code
- `/commit --validate` - Commit with test validation

## Integration Patterns

### Test-Driven Development
```bash
# TDD cycle
/generate-tests new_feature.py --type=unit
/run-all-tests --scope=unit --auto-fix
/optimize new_feature.py --implement
/run-all-tests --coverage=100
```

### Quality Assurance Pipeline
```bash
# Complete QA workflow
/check-code-quality --auto-fix
/generate-tests --coverage=90
/run-all-tests --auto-fix --coverage --report
/commit --validate
```

### Performance Validation
```bash
# Performance testing and optimization
/run-all-tests --benchmark --profile
/optimize bottlenecks.py --implement
/run-all-tests --benchmark --scope=performance
/double-check "performance improvements" --deep-analysis
```

## Requirements

- Framework-specific test runners (pytest, Jest, etc.)
- Language runtimes and dependencies
- Optional: GPU drivers for accelerated testing
- Optional: Coverage tools and profilers

ARGUMENTS: [--scope=all|unit|integration|performance] [--profile] [--benchmark] [--scientific] [--gpu] [--parallel] [--reproducible] [--coverage] [--report] [--implement] [--auto-fix] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--orchestrate] [--intelligent] [--distributed] [--validate]