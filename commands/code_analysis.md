---
description: Perform comprehensive code analysis with quality metrics and recommendations for Python and Julia codebases
category: code-analysis-testing
argument-hint: [file-or-directory-path]
allowed-tools: Read, Grep, Glob, TodoWrite, Bash
---

Perform a comprehensive code analysis on the specified files or directory. If no path is provided, analyze the current working directory with intelligent language-specific analysis for Python and Julia codebases.

## Analysis Process:

1. **Parse Arguments**:
   - Extract the path from $ARGUMENTS (defaults to current directory if not specified)
   - Determine scope: single file, multiple files, or entire directory
   - Detect project type (Python package, Julia package, mixed)

2. **Language Detection & Project Structure**:
   - **Python Projects**: Detect pyproject.toml, setup.py, requirements.txt, conda envs
   - **Julia Projects**: Detect Project.toml, Manifest.toml, package structure
   - Identify testing frameworks (pytest, unittest, Pkg.test)
   - Find configuration files (tox.ini, .pre-commit-config.yaml, etc.)

3. **Python-Specific Analysis**:
   - **Code Quality**: Use ruff, black, isort for style and linting
   - **Type Checking**: mypy analysis for type hints and annotations  
   - **Security**: bandit security scanning for vulnerabilities
   - **Complexity**: Analyze function complexity, class design, import cycles
   - **Testing**: pytest coverage, test organization, fixture usage
   - **Dependencies**: Check for outdated packages, security advisories
   - **Documentation**: Docstring coverage, README quality, type annotations
   - **Performance**: Identify bottlenecks, async/await usage, memory patterns

4. **Julia-Specific Analysis**:
   - **Code Quality**: Check for Julia style guide compliance
   - **Type Stability**: Analyze type inference and stability
   - **Performance**: Identify type instabilities, allocations, barriers
   - **Package Quality**: Dependency analysis, version bounds, compatibility
   - **Testing**: Test coverage, Pkg.test integration
   - **Documentation**: Docstrings, README, example quality
   - **Best Practices**: Multiple dispatch usage, broadcasting, vectorization

5. **Cross-Language Analysis**:
   - **Architecture**: Module organization, coupling, cohesion
   - **Code Smells**: Long functions, large files, duplicate patterns
   - **Git Integration**: Analyze commit patterns, file change frequency
   - **CI/CD**: Check for GitHub Actions, pre-commit hooks, automated testing

6. **Advanced Metrics**:
   - **Maintainability Index**: Calculate using Halstead metrics and complexity
   - **Technical Debt**: Estimate based on code smells and violations
   - **Test Quality**: Coverage gaps, test isolation, edge case handling
   - **Security Posture**: Vulnerability scanning, dependency security

7. **Interactive Analysis**:
   - **Smart Prioritization**: Focus on files with highest impact/complexity
   - **Diff Analysis**: Compare against git history for regression detection  
   - **Dependency Graph**: Visualize module dependencies and coupling
   - **Performance Hotspots**: Identify optimization opportunities

8. **Generate Comprehensive Report**:
   - **Executive Summary**: Overall health score (0-100), key metrics
   - **Language-Specific Findings**: Python/Julia best practice violations
   - **Priority Matrix**: Critical/High/Medium/Low issues with effort estimates
   - **File-Level Analysis**: Per-file quality scores and recommendations
   - **Dependency Analysis**: Security, updates, license compliance
   - **Test Quality Report**: Coverage, test smells, missing test patterns
   - **Performance Analysis**: Bottlenecks, optimization opportunities
   - **Security Assessment**: Vulnerability scan results, secure coding practices

9. **Actionable Recommendations**:
   - **Quick Wins**: Low-effort, high-impact improvements
   - **Refactoring Targets**: Files/functions needing restructuring  
   - **Testing Gaps**: Missing test scenarios and edge cases
   - **Performance Optimizations**: Specific code improvements
   - **Security Hardening**: Vulnerability fixes and secure patterns
   - **Documentation Improvements**: Missing/outdated documentation

10. **Track with TodoWrite**:
    - Create prioritized todos for critical and high-priority issues
    - Organize by fix complexity, impact, and required effort
    - Group related issues for efficient batch processing

## Python-Specific Checks:

### Code Quality
- **Style**: black, isort, ruff compliance
- **Linting**: flake8, pylint rule violations  
- **Type Hints**: mypy analysis, annotation coverage
- **Imports**: Circular imports, unused imports, import organization

### Performance & Security
- **Security**: bandit vulnerability scanning
- **Performance**: Async patterns, database queries, algorithmic complexity
- **Memory**: Object lifecycle, garbage collection patterns
- **Dependencies**: pip-audit for known vulnerabilities

### Testing & Documentation
- **Coverage**: pytest-cov integration, branch coverage
- **Test Quality**: Fixture design, test isolation, parametrization
- **Docstrings**: Numpy/Google style compliance, coverage metrics
- **Type Safety**: Strict mypy configuration compliance

## Julia-Specific Checks:

### Performance & Type Stability
- **Type Inference**: @code_warntype analysis for type instabilities
- **Allocations**: Memory allocation patterns and optimizations
- **Broadcasting**: Proper vectorization usage
- **Multiple Dispatch**: Effective method design patterns

### Package Quality
- **Project.toml**: Proper versioning, dependency bounds
- **Compatibility**: Julia version compatibility, package ecosystem
- **Testing**: Pkg.test integration, coverage analysis
- **Documentation**: Documenter.jl integration, example quality

## Example Usage:
- `/code_analysis` - Analyze entire current directory with auto-detection
- `/code_analysis src/` - Deep analysis of source directory  
- `/code_analysis mypackage.py` - Single file comprehensive analysis
- `/code_analysis --focus=security` - Security-focused analysis
- `/code_analysis --language=julia src/` - Force Julia-specific analysis

## Report Output:
- **Console Summary**: Key metrics and priority issues
- **Detailed Report**: File-based analysis with specific recommendations
- **Todo Integration**: Actionable items added to task tracking
- **CI Integration**: Exit codes for automated quality gates

Target path: $ARGUMENTS