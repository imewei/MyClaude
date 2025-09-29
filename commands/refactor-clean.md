---
title: "Refactor Clean"
description: "AI-powered code refactoring with multi-language support and modern patterns"
category: refactoring
subcategory: code-modernization
complexity: intermediate
argument-hint: "[target] [--language=python|javascript|typescript|java|julia|auto] [--scope=file|project] [--patterns=modern|performance|security] [--report=summary|detailed] [--implement] [--agents=quality|orchestrator|all]"
allowed-tools: Read, Write, Edit, Glob, MultiEdit, Bash, TodoWrite
model: inherit
tags: refactoring, modernization, multi-language, patterns, code-quality
dependencies: []
related: [optimize, multi-agent-optimize, check-code-quality, clean-codebase, adopt-code, generate-tests, run-all-tests, commit, double-check]
workflows: [refactoring-workflow, modernization-pipeline, code-cleanup]
version: "2.0"
last-updated: "2025-09-28"
---

# Refactor Clean

Code refactoring engine with multi-language support and modern programming patterns.

```bash
/refactor-clean [target] [options]

# Basic usage
/refactor-clean src/main.py
/refactor-clean project/ --scope=project
/refactor-clean --language=python --patterns=modern
```

## Options

- `--language=<lang>`: Target language (python, javascript, typescript, java, julia, auto)
- `--scope=<scope>`: Refactoring scope (file, project)
- `--patterns=<patterns>`: Refactoring patterns to apply (modern, performance, security)
- `--report=<format>`: Output report format (summary, detailed)
- `--dry-run`: Preview changes without applying them
- `--implement`: Automatically apply refactoring recommendations

## Language Support

### Python
- Modern syntax patterns (f-strings, type hints, dataclasses)
- Performance optimizations (list comprehensions, generators)
- Code style improvements (PEP 8 compliance)
- Scientific computing patterns (NumPy, JAX optimization)

### JavaScript/TypeScript
- Modern ES6+ syntax patterns
- Async/await optimization
- Type safety improvements (TypeScript)
- Framework-specific patterns (React, Node.js)

### Java
- Modern Java syntax (streams, lambdas, records)
- Design pattern implementations
- Performance optimizations
- Code organization improvements

### Julia
- Type stability improvements
- Performance optimizations
- Scientific computing patterns
- Memory allocation optimization

## Refactoring Patterns

### Modern Patterns
- Syntax modernization (latest language features)
- Code structure improvements
- Readability enhancements
- Best practice implementations

### Performance Patterns
- Algorithm optimization
- Memory usage improvements
- Parallel processing opportunities
- Caching and memoization

### Security Patterns
- Input validation improvements
- Security best practices
- Vulnerability remediation
- Safe coding patterns

## Refactoring Scope

### File Scope
- Single file analysis and refactoring
- Function-level optimizations
- Class structure improvements
- Import/dependency optimization

### Project Scope
- Multi-file refactoring
- Architecture improvements
- Cross-file dependency optimization
- Consistent pattern application

## Analysis Features

### Code Quality Assessment
- Code smell detection
- Anti-pattern identification
- Complexity analysis
- Maintainability scoring

### Pattern Detection
- Design pattern opportunities
- Framework-specific optimizations
- Performance bottleneck identification
- Security vulnerability scanning

### Refactoring Recommendations
- Priority-based suggestions
- Impact assessment
- Risk analysis
- Implementation guidance

### Implementation Features (--implement)
**Automated Refactoring Process:**
- **Safety Analysis** - Assess refactoring safety and impact before changes
- **Backup Creation** - Create comprehensive backup of all files to be modified
- **Incremental Application** - Apply refactoring patterns step-by-step with validation
- **Test Integration** - Run tests after each refactoring phase to ensure functionality
- **Quality Validation** - Verify code quality improvements through metrics
- **Performance Impact** - Measure performance effects of refactoring changes
- **Rollback System** - Automatic revert capability on test failures or quality degradation
- **Progress Tracking** - Real-time monitoring of refactoring progress and status
- **Implementation Report** - Detailed documentation of all applied changes
- **Verification Pipeline** - Comprehensive validation of refactoring success

## Examples

```bash
# Basic file refactoring
/refactor-clean src/utils.py

# Project-wide modernization
/refactor-clean project/ --scope=project --patterns=modern

# Performance-focused refactoring
/refactor-clean algorithms.py --patterns=performance --report=detailed

# Language-specific refactoring
/refactor-clean app.js --language=javascript --patterns=modern

# Security pattern application
/refactor-clean web_app/ --patterns=security --scope=project

# Preview changes without applying
/refactor-clean src/ --dry-run --report=detailed

# Automatically implement refactoring recommendations
/refactor-clean src/utils.py --implement

# Implement modern patterns across project
/refactor-clean project/ --scope=project --patterns=modern --implement

# Performance refactoring with automatic implementation
/refactor-clean algorithms.py --patterns=performance --implement --report=detailed

# Security pattern implementation
/refactor-clean web_app/ --patterns=security --scope=project --implement
```

## Output Information

### Summary Report
- Files analyzed and modified
- Patterns applied
- Quality improvements
- Performance impact estimates

### Detailed Report
- Line-by-line changes
- Refactoring rationale
- Before/after code examples
- Quality metrics comparison

### Implementation Results (with --implement)
- **Applied Changes** - Summary of refactoring modifications made
- **Backup Locations** - Paths to original file backups
- **Validation Status** - Test results and verification outcomes
- **Performance Impact** - Before/after performance measurements
- **Rollback Instructions** - Steps to revert changes if needed
- **Quality Improvements** - Metrics showing code quality enhancements

## Integration

### Development Workflow
- Code review process enhancement
- Pre-commit hook integration
- CI/CD pipeline integration
- Quality gate validation

### Team Collaboration
- Consistent coding standards
- Knowledge sharing through reports
- Technical debt tracking
- Refactoring progress monitoring

## Common Workflows

### Basic Refactoring Workflow
```bash
# 1. Analyze current code structure
/refactor-clean legacy_code.py --dry-run --report=detailed

# 2. Apply modern patterns
/refactor-clean legacy_code.py --patterns=modern --implement

# 3. Verify refactoring results
/check-code-quality legacy_code.py
/run-all-tests
```

### Project Modernization
```bash
# 1. Project-wide modernization
/refactor-clean project/ --scope=project --patterns=modern --implement

# 2. Performance improvements
/refactor-clean project/ --patterns=performance --implement

# 3. Quality validation
/multi-agent-optimize project/ --mode=review --focus=quality
/double-check "refactoring results" --deep-analysis
```

### Language-Specific Modernization
```bash
# 1. Python scientific computing modernization
/refactor-clean analysis.py --language=python --patterns=modern --implement
/adopt-code analysis.py --target=jax

# 2. JavaScript/TypeScript modernization
/refactor-clean frontend/ --language=typescript --patterns=modern --implement
```

## Related Commands

**Prerequisites**: Commands to run before refactoring
- `/check-code-quality --auto-fix` - Fix basic quality issues first
- `/debug --auto-fix` - Resolve runtime issues before refactoring
- `/generate-tests` - Ensure adequate test coverage for validation
- `/run-all-tests` - Establish baseline functionality and performance
- Version control - Commit current state before major refactoring

**Alternatives**: Different refactoring approaches
- `/optimize --implement` - Performance-focused refactoring with automatic implementation
- `/multi-agent-optimize --mode=review` - Multi-agent architectural review and refactoring
- `/adopt-code` - Language/framework modernization and migration
- `/clean-codebase` - Project-level cleanup and organization
- `/think-ultra` - Research-grade refactoring strategy analysis

**Combinations**: Commands that work with refactor-clean
- `/generate-tests --coverage=95` - Create comprehensive tests for refactored code
- `/optimize --implement` - Apply performance optimizations after refactoring
- `/double-check --deep-analysis` - Verify refactoring quality and completeness
- `/update-docs` - Document refactoring improvements and patterns
- `/reflection --type=instruction` - Analyze refactoring process effectiveness

**Follow-up**: Commands to run after refactoring
- `/run-all-tests --coverage` - Ensure functionality preserved with metrics
- `/check-code-quality` - Validate quality improvements achieved
- `/optimize --implement` - Apply performance optimizations to refactored code
- `/commit --template=refactor --validate` - Commit with comprehensive validation

## Integration Patterns

### Code Quality Improvement Pipeline
```bash
# Comprehensive quality improvement
/check-code-quality --auto-fix
/refactor-clean --patterns=modern --implement
/optimize --implement --category=all
/generate-tests --coverage=95
```

### Legacy Code Modernization
```bash
# Step-by-step modernization
/refactor-clean legacy/ --patterns=modern --dry-run      # Preview changes
/refactor-clean legacy/ --patterns=modern --implement    # Apply modern patterns
/adopt-code legacy/ --target=modern_framework           # Framework migration
/optimize legacy/ --implement                           # Performance optimization
```

### Security Hardening Workflow
```bash
# Security-focused refactoring
/refactor-clean webapp/ --patterns=security --implement
/check-code-quality webapp/ --security
/multi-agent-optimize webapp/ --focus=security --implement
```

## Requirements

- Target language runtime and dependencies
- File system read/write access
- Network access for dependency analysis