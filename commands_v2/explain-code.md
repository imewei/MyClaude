---
title: "Explain Code"
description: "Advanced code analysis and documentation tool with multi-language support"
category: code-analysis
subcategory: analysis
complexity: intermediate
argument-hint: "[--level=basic|advanced|expert] [--focus=AREA] [--docs] [--interactive] [--format=FORMAT] [--export=PATH] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--implement] [--dry-run] [--backup] [--rollback] [--intelligent] [--orchestrate] [--parallel] [--validate] [file/directory]"
allowed-tools: Read, Write, Edit, MultiEdit, Bash, Glob, Grep, TodoWrite, Task
model: inherit
tags: analysis, documentation, patterns, performance, security
dependencies: []
related: [check-code-quality, update-docs, debug, optimize, double-check]
workflows: [code-analysis, documentation-generation, learning-workflow]
version: "2.1"
last-updated: "2025-09-29"
---

# Explain Code

Advanced code analysis and documentation tool with intelligent pattern recognition, multi-language support, and comprehensive reporting capabilities.

## Quick Start

```bash
# Analyze single file
/explain-code src/main.py

# Generate comprehensive documentation
/explain-code --docs --level=advanced src/

# Performance-focused analysis
/explain-code --focus=performance --export=analysis.md algorithm.py

# Interactive exploration
/explain-code --interactive --recursive complex_system/
```

## Options

### Analysis Control
- `--level=<depth>`: Analysis depth
  - `basic`: Surface-level analysis, syntax and structure
  - `advanced`: Deep patterns, algorithms, performance insights
  - `expert`: Comprehensive analysis with optimization recommendations

- `--focus=<area>`: Specialized analysis focus
  - `syntax`: Code structure and style compliance
  - `patterns`: Design patterns and architectural analysis
  - `algorithms`: Algorithmic complexity and efficiency
  - `performance`: Performance bottlenecks and optimization
  - `security`: Vulnerability detection and security best practices
  - `testing`: Test coverage and testing strategies
  - `maintainability`: Code quality and technical debt

### Language and Format
- `--language=<lang>`: Target language (auto-detected)
  - Supported: `python`, `julia`, `javascript`, `typescript`, `rust`, `cpp`, `go`, `java`

- `--format=<type>`: Output format
  - `markdown`: Rich markdown documentation (default)
  - `json`: Structured JSON for integration
  - `html`: Interactive HTML report
  - `text`: Plain text summary

### Output and Integration
- `--docs`: Generate comprehensive documentation suite
- `--export=<path>`: Export analysis to specified file
- `--output=<path>`: Custom output directory
- `--interactive`: Interactive exploration mode
- `--recursive`: Include subdirectories and dependencies
- `--template=<type>`: Documentation template (api, tutorial, reference)
- `--agents=<agents>`: Agent selection (auto, core, scientific, engineering, ai, domain, quality, research, all)
- `--implement`: Automatically implement documentation improvements and code clarifications
- `--intelligent`: Enable intelligent agent selection based on codebase characteristics
- `--orchestrate`: Enable advanced 23-agent orchestration for comprehensive analysis
- `--parallel`: Run analysis in parallel for maximum efficiency

## Core Features

### Intelligent Analysis
- **Multi-language AST parsing**: Deep syntactic and semantic analysis
- **Pattern recognition**: Automatic detection of design patterns and anti-patterns
- **Dependency mapping**: Import/export analysis and dependency visualization
- **Complexity metrics**: Cyclomatic complexity, cognitive load, maintainability index

### Performance Engineering
- **Bottleneck detection**: Identify performance-critical sections
- **Optimization recommendations**: Language-specific performance improvements
- **Big O analysis**: Algorithmic complexity estimation
- **Memory usage patterns**: Stack/heap analysis and memory optimization

### Security and Quality
- **Vulnerability scanning**: Common security issues and OWASP compliance
- **Code smell detection**: Technical debt and maintainability issues
- **Best practices validation**: Language-specific conventions and standards
- **Test coverage analysis**: Test quality and coverage recommendations

### Documentation Generation
- **API documentation**: Auto-generated function and class documentation
- **Architecture diagrams**: Visual representation of code structure
- **Usage examples**: Automatically extracted and generated examples
- **Interactive explanations**: Step-by-step code walkthroughs

## Usage Examples

### Basic Analysis
```bash
# Quick code overview
/explain-code src/main.py

# Detailed analysis with explanations
/explain-code src/algorithm.py --level=advanced --interactive

# Multi-file project analysis
/explain-code src/ --recursive --level=expert
```

### Specialized Analysis
```bash
# Performance optimization focus
/explain-code performance_critical.py --focus=performance --export=perf_analysis.md

# Security audit
/explain-code web_app/ --focus=security --format=json --output=security_report/

# Architecture documentation
/explain-code backend/ --docs --template=api --recursive

# Code quality assessment
/explain-code legacy_code/ --focus=maintainability --level=expert
```

### Integration Workflows
```bash
# CI/CD integration
/explain-code src/ --format=json --export=code_analysis.json

# Documentation pipeline
/explain-code . --docs --template=tutorial --output=docs/

# Code review preparation
/explain-code new_feature/ --level=advanced --focus=patterns,security
```

### Language-Specific Examples
```bash
# Python data science code
/explain-code ml_pipeline.py --focus=performance,algorithms --language=python

# Julia scientific computing
/explain-code simulation.jl --focus=performance --language=julia

# TypeScript web application
/explain-code frontend/ --focus=patterns,security --language=typescript --recursive

# Rust systems programming
/explain-code memory_manager.rs --focus=performance,security --language=rust
```

## Supported Languages

| Language | Features | Specializations |
|----------|----------|----------------|
| **Python** | Full AST analysis, performance profiling | Data science, ML, web frameworks |
| **Julia** | Performance analysis, type inference | Scientific computing, numerical methods |
| **JavaScript/TypeScript** | Module analysis, async patterns | Web development, Node.js, frameworks |
| **Rust** | Memory safety, ownership analysis | Systems programming, performance |
| **C/C++** | Memory management, performance | Systems, embedded, high-performance |
| **Go** | Concurrency patterns, simplicity | Microservices, systems programming |
| **Java** | OOP patterns, enterprise patterns | Enterprise applications, frameworks |

## Output Formats and Files

### Generated Documentation
- **`README.md`**: Project overview and getting started guide
- **`API.md`**: Comprehensive API documentation with examples
- **`ARCHITECTURE.md`**: System architecture and design patterns
- **`PERFORMANCE.md`**: Performance analysis and optimization guide
- **`SECURITY.md`**: Security analysis and recommendations

### Structured Data
- **`analysis.json`**: Complete analysis data for integration
- **`metrics.json`**: Code quality and complexity metrics
- **`dependencies.json`**: Dependency graph and analysis
- **`patterns.json`**: Detected patterns and anti-patterns

### Interactive Reports
- **`report.html`**: Interactive HTML report with charts and navigation
- **`coverage.html`**: Test coverage visualization
- **`performance.html`**: Performance bottleneck visualization

## Integration with Other Commands

```bash
# Comprehensive analysis workflow
/explain-code src/ --level=expert --docs
/optimize src/ --category=all
/check-code-quality src/ --auto-fix

# Documentation generation pipeline
/explain-code . --docs --template=api
/update-docs --type=api --interactive

# Performance optimization workflow
/explain-code algorithm.py --focus=performance
/optimize algorithm.py --category=algorithm,memory
/generate-tests algorithm.py --type=performance
```

## Advanced Features

### AI-Powered Insights
- **Code intelligence**: Context-aware explanations and suggestions
- **Pattern learning**: Learns from codebase patterns for better analysis
- **Predictive analysis**: Identifies potential issues before they occur

### Collaboration Features
- **Team insights**: Multi-developer contribution analysis
- **Knowledge transfer**: Automated onboarding documentation
- **Code reviews**: AI-assisted review comments and suggestions

### Extensibility
- **Custom analyzers**: Plugin system for domain-specific analysis
- **Template system**: Customizable documentation templates
- **API integration**: REST API for CI/CD and tool integration

## Common Workflows

### Code Understanding Workflow
```bash
# 1. Quick code overview
/explain-code complex_module.py --level=basic

# 2. Deep analysis for understanding
/explain-code complex_module.py --level=expert --interactive

# 3. Generate documentation
/explain-code complex_module.py --docs --export=module_docs.md
```

### Performance Analysis Workflow
```bash
# 1. Performance-focused analysis
/explain-code slow_algorithm.py --focus=performance --level=expert

# 2. Apply optimization recommendations
/optimize slow_algorithm.py --implement

# 3. Verify improvements
/run-all-tests --benchmark --profile
```

### Learning and Onboarding
```bash
# 1. Understand unfamiliar codebase
/explain-code legacy_system/ --recursive --level=advanced --interactive

# 2. Generate learning materials
/explain-code legacy_system/ --docs --template=tutorial

# 3. Create comprehensive documentation
/update-docs --type=all --interactive
```

## Related Commands

**Prerequisites**: Commands that provide context for analysis
- Clean, working code (no syntax errors)

**Alternatives**: Different analysis approaches
- `/check-code-quality` - Quality-focused analysis
- `/debug` - Runtime behavior analysis
- `/multi-agent-optimize --mode=review` - Multi-perspective code review

**Combinations**: Commands that work with explain-code
- `/optimize` - Apply performance insights from explanation
- `/update-docs` - Use analysis for documentation generation
- `/generate-tests` - Create tests based on code understanding

**Follow-up**: Commands to run after code explanation
- `/optimize --implement` - Apply performance recommendations
- `/refactor-clean --implement` - Apply structural improvements
- `/update-docs` - Generate comprehensive documentation

## Integration Patterns

### Code Review Preparation
```bash
# Comprehensive analysis for code review
/explain-code new_feature/ --level=expert --focus=patterns,security,performance
/check-code-quality new_feature/ --auto-fix
/generate-tests new_feature/ --coverage=90
```

### Documentation Generation Pipeline
```bash
# Complete documentation workflow
/explain-code project/ --docs --recursive --level=advanced
/update-docs --type=api --optimize
/double-check "documentation completeness" --deep-analysis
```

### Learning and Knowledge Transfer
```bash
# Onboarding workflow for complex systems
/explain-code complex_system/ --interactive --level=expert --focus=architecture
/update-docs --type=readme --collaborative
/generate-tests complex_system/ --interactive
```

ARGUMENTS: [--level=basic|advanced|expert] [--focus=AREA] [--docs] [--interactive] [--format=FORMAT] [--export=PATH] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--implement] [--intelligent] [--orchestrate] [--parallel] [file/directory]