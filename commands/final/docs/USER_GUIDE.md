# Claude Code Command Executor Framework - Complete User Guide

> Comprehensive documentation for all features and capabilities

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [14 Commands Reference](#14-commands-reference)
4. [23-Agent System](#23-agent-system)
5. [Workflow Framework](#workflow-framework)
6. [Plugin System](#plugin-system)
7. [Configuration](#configuration)
8. [Best Practices](#best-practices)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is this system?

The Claude Code Command Executor Framework is a production-ready AI-powered development automation system that provides:

- **14 specialized commands** covering analysis, quality, testing, performance, and automation
- **23 AI agents** with expertise in different domains, working independently or together
- **Workflow framework** for complex multi-step automation
- **Plugin system** for extensibility and customization
- **Multi-language support** - Python, Julia, JAX, JavaScript, TypeScript, and more
- **Enterprise features** - CI/CD integration, security scanning, audit trails

### Who is it for?

- **Developers** - Automate repetitive tasks, improve code quality, optimize performance
- **Teams** - Standardize workflows, enforce quality gates, setup CI/CD
- **Researchers** - Scientific computing, reproducible research, legacy code adoption
- **Enterprises** - Compliance, security, scalability, team collaboration

### Key Benefits

- **10x productivity** - Automate manual tasks
- **Higher quality** - AI-powered analysis and improvements
- **Faster performance** - Intelligent optimization
- **Better documentation** - Auto-generated, always up-to-date
- **Reduced errors** - Automated testing and validation
- **Team consistency** - Standardized workflows

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code CLI                          │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│              Command Executor Framework                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     14 Specialized Commands                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │     23-Agent System (Orchestrated)                  │   │
│  │  • Core • Scientific • AI/ML • Engineering • Domain │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌──────────────┬──────────────┬─────────────────────┐    │
│  │  Workflow    │   Plugin     │    UX System        │    │
│  │  Framework   │   System     │    (Phase 6)        │    │
│  └──────────────┴──────────────┴─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│        Integration Layer (Git, CI/CD, IDEs)                 │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Command Layer
- 14 specialized commands for different development tasks
- Each command has specific purpose and capabilities
- Commands can work independently or in workflows
- Extensive flag support for customization

#### 2. Agent System
- 23 specialized AI agents with domain expertise
- Automatic agent selection based on task
- Multi-agent coordination through Orchestrator
- Parallel execution for performance

#### 3. Workflow Framework
- Pre-built workflows for common tasks
- YAML-based workflow definition
- Conditional execution and branching
- Error handling and recovery

#### 4. Plugin System
- Extend functionality with plugins
- Custom commands, agents, workflows
- Third-party tool integration
- Easy installation and management

#### 5. UX System (Phase 6)
- Rich console output with animations
- Progress tracking and status updates
- Interactive prompts and confirmations
- Accessibility features

---

## 14 Commands Reference

### Analysis & Planning

#### /think-ultra
**Advanced analytical thinking engine with multi-agent collaboration**

```bash
/think-ultra [--depth=auto|comprehensive|ultra|quantum] \
             [--mode=auto|systematic|discovery|hybrid] \
             [--paradigm=auto|multi|cross|meta] \
             [--agents=auto|core|engineering|domain-specific|all] \
             [--priority=auto|implementation] \
             [--recursive=false|true] \
             [--export-insights] \
             [--auto-fix=false|true] \
             [--orchestrate] \
             [--intelligent] \
             [--breakthrough] \
             [problem]
```

**Use cases:**
- Complex problem analysis
- Architectural decisions
- Research questions
- System design
- Performance investigations

**Examples:**
```bash
# Deep problem analysis
/think-ultra --depth=ultra --mode=systematic "Why is my application slow?"

# Breakthrough thinking
/think-ultra --depth=quantum --breakthrough "Novel approaches to distributed caching"

# Multi-paradigm analysis
/think-ultra --paradigm=meta --agents=all "Refactor monolith to microservices"
```

**Agents used:** Orchestrator, domain-specific agents based on problem

**Output:** Detailed analysis with insights, recommendations, and action plan

---

#### /reflection
**Reflection engine with advanced AI reasoning and session analysis**

```bash
/reflection [--type=comprehensive|focused|scientific|instruction|session] \
            [--analysis=deep|surface|meta] \
            [--optimize=performance|accuracy|collaboration|innovation] \
            [--export-insights] \
            [--breakthrough-mode] \
            [--implement] \
            [--agents=orchestrator|scientific|quality|research|all]
```

**Use cases:**
- Session analysis and optimization
- Learn from past work
- Improve collaboration
- Meta-cognitive analysis
- Continuous improvement

**Examples:**
```bash
# Session reflection
/reflection --type=session --analysis=deep

# Scientific workflow optimization
/reflection --type=scientific --optimize=accuracy --agents=scientific

# Breakthrough insights
/reflection --breakthrough-mode --export-insights
```

**Agents used:** Orchestrator, Quality Assurance, Research Scientist

**Output:** Reflection report with insights and improvement suggestions

---

#### /double-check
**Verification and auto-completion engine**

```bash
/double-check [--interactive] \
              [--auto-complete] \
              [--deep-analysis] \
              [--report] \
              [--agents=auto|core|engineering|domain-specific|all] \
              [--orchestrate] \
              [--intelligent] \
              [--breakthrough] \
              ["task/problem description"]
```

**Use cases:**
- Verify work completeness
- Auto-complete missing work
- Quality validation
- Task verification
- Comprehensive review

**Examples:**
```bash
# Verify quality improvements
/double-check "all code quality issues resolved"

# Deep verification with auto-complete
/double-check --deep-analysis --auto-complete "tests cover all edge cases"

# Interactive verification
/double-check --interactive --report "documentation is complete and accurate"
```

**Agents used:** Quality Assurance, relevant domain agents

**Output:** Verification report with completeness assessment and auto-completion

---

### Code Quality

#### /check-code-quality
**Code quality analysis for multiple languages**

```bash
/check-code-quality [--language=python|julia|jax|auto] \
                    [--analysis=basic|scientific|gpu] \
                    [--auto-fix] \
                    [--format=text|json] \
                    [--agents=quality|scientific|orchestrator|all] \
                    [target-path]
```

**Use cases:**
- Code quality assessment
- Identify issues and anti-patterns
- Auto-fix common problems
- Generate quality reports
- Pre-commit validation

**Examples:**
```bash
# Python quality check with auto-fix
/check-code-quality --language=python --auto-fix .

# Scientific code analysis
/check-code-quality --analysis=scientific --agents=scientific src/

# Generate JSON report
/check-code-quality --format=json --auto-fix > quality-report.json
```

**Agents used:** Quality Assurance, language experts, Scientific Computing (if applicable)

**Output:** Quality report with score, issues, and fixes applied

---

#### /refactor-clean
**AI-powered code refactoring with modern patterns**

```bash
/refactor-clean [--language=python|javascript|typescript|java|julia|auto] \
                [--scope=file|project] \
                [--patterns=modern|performance|security] \
                [--report=summary|detailed] \
                [--implement] \
                [--agents=quality|orchestrator|all] \
                [target]
```

**Use cases:**
- Modernize legacy code
- Apply design patterns
- Improve code structure
- Security refactoring
- Performance refactoring

**Examples:**
```bash
# Refactor with modern patterns
/refactor-clean --patterns=modern --implement src/

# Security-focused refactoring
/refactor-clean --patterns=security --scope=project --implement

# Performance refactoring with report
/refactor-clean --patterns=performance --report=detailed --implement
```

**Agents used:** Refactoring, Quality Assurance, language experts

**Output:** Refactoring report and implemented changes

---

#### /clean-codebase
**Advanced codebase cleanup with AST-based analysis**

```bash
/clean-codebase [--dry-run] \
                [--analysis=basic|thorough|comprehensive|ultrathink] \
                [--agents=auto|core|scientific|engineering|domain-specific|all] \
                [--imports] \
                [--dead-code] \
                [--duplicates] \
                [--ast-deep] \
                [--orchestrate] \
                [--intelligent] \
                [--breakthrough] \
                [--parallel] \
                [path]
```

**Use cases:**
- Remove unused imports
- Eliminate dead code
- Find and remove duplicates
- Deep AST analysis
- Codebase maintenance

**Examples:**
```bash
# Clean unused imports
/clean-codebase --imports --dry-run .

# Complete cleanup
/clean-codebase --analysis=comprehensive --imports --dead-code --duplicates

# Deep AST analysis with ultrathink
/clean-codebase --ast-deep --analysis=ultrathink --orchestrate --parallel
```

**Agents used:** Quality Assurance, Code Reviewer, language experts

**Output:** Cleanup report with changes made and files modified

---

### Testing

#### /generate-tests
**Generate comprehensive test suites**

```bash
/generate-tests [--type=all|unit|integration|performance|jax|scientific|gpu] \
                [--framework=auto|pytest|julia|jax] \
                [--coverage=N] \
                [--agents=scientific|quality|orchestrator|all] \
                [target-file-or-module]
```

**Use cases:**
- Generate unit tests
- Create integration tests
- Performance benchmarks
- Scientific test suites
- GPU test generation

**Examples:**
```bash
# Generate unit tests with 90% coverage
/generate-tests --type=unit --coverage=90 src/

# Scientific tests for GPU code
/generate-tests --type=scientific --gpu --framework=pytest

# Complete test suite
/generate-tests --type=all --coverage=95 --agents=all
```

**Agents used:** Testing, Quality Assurance, Scientific Computing (if applicable)

**Output:** Generated test files with specified coverage

---

#### /run-all-tests
**Comprehensive test execution with intelligent failure resolution**

```bash
/run-all-tests [--scope=all|unit|integration|performance] \
               [--profile] \
               [--benchmark] \
               [--scientific] \
               [--gpu] \
               [--parallel] \
               [--reproducible] \
               [--coverage] \
               [--report] \
               [--auto-fix] \
               [--agents=auto|scientific|ai|engineering|domain|all] \
               [--orchestrate] \
               [--intelligent] \
               [--distributed]
```

**Use cases:**
- Run test suites
- Performance benchmarking
- Coverage reporting
- Auto-fix failing tests
- Parallel test execution

**Examples:**
```bash
# Run all tests with auto-fix
/run-all-tests --scope=all --auto-fix --coverage

# Performance benchmarking
/run-all-tests --benchmark --profile --report

# Scientific tests with GPU
/run-all-tests --scientific --gpu --reproducible

# Parallel execution
/run-all-tests --parallel --orchestrate --intelligent
```

**Agents used:** Testing, Quality Assurance, Performance Engineer

**Output:** Test results, coverage report, benchmarks, auto-fixes applied

---

#### /debug
**Scientific computing debugging with GPU support**

```bash
/debug [--issue=TYPE] \
       [--gpu] \
       [--julia] \
       [--research] \
       [--jupyter] \
       [--profile] \
       [--monitor] \
       [--logs] \
       [--auto-fix] \
       [--report] \
       [--agents=scientific|quality|orchestrator|all]
```

**Use cases:**
- Debug performance issues
- GPU debugging
- Memory issues
- Julia/Python debugging
- Research code debugging

**Examples:**
```bash
# Debug performance issue
/debug --issue=performance --profile --auto-fix

# GPU debugging
/debug --gpu --monitor --report

# Julia debugging
/debug --julia --research --profile

# Jupyter notebook debugging
/debug --jupyter --auto-fix
```

**Agents used:** Scientific Computing, GPU Specialist, Performance Engineer

**Output:** Debug report with issue identification and fixes

---

### Performance

#### /optimize
**Code optimization and performance analysis**

```bash
/optimize [--language=python|julia|jax|auto] \
          [--category=all|algorithm|memory|io|concurrency] \
          [--format=text|json|html] \
          [--implement] \
          [--agents=auto|scientific|ai|engineering|quantum|all] \
          [--orchestrate] \
          [--intelligent] \
          [--breakthrough] \
          [target]
```

**Use cases:**
- Performance optimization
- Algorithm improvements
- Memory optimization
- I/O optimization
- Concurrency optimization

**Examples:**
```bash
# Complete optimization
/optimize --category=all --implement src/

# Algorithm optimization
/optimize --category=algorithm --agents=scientific --breakthrough

# Memory optimization with report
/optimize --category=memory --format=html --implement > report.html

# JAX optimization
/optimize --language=jax --agents=ai --orchestrate
```

**Agents used:** Performance Engineer, Scientific Computing, AI/ML Engineer

**Output:** Optimization report with improvements and performance gains

---

### Development Workflow

#### /commit
**Git commit engine with AI message generation**

```bash
/commit [--all] \
        [--staged] \
        [--amend] \
        [--interactive] \
        [--split] \
        [--template=TYPE] \
        [--ai-message] \
        [--validate] \
        [--push] \
        [--agents=quality|devops|orchestrator|all]
```

**Use cases:**
- Smart commit messages
- Pre-commit validation
- Conventional commits
- Quality-gated commits
- Automated commits

**Examples:**
```bash
# AI-generated commit message
/commit --all --ai-message --validate

# Interactive commit
/commit --interactive --template=conventional

# Validated commit with push
/commit --staged --validate --push --agents=quality
```

**Agents used:** DevOps, Quality Assurance

**Output:** Commit created with generated message and validation

---

#### /fix-commit-errors
**GitHub Actions error analysis and automated resolution**

```bash
/fix-commit-errors [--auto-fix] \
                   [--debug] \
                   [--emergency] \
                   [--interactive] \
                   [--max-cycles=N] \
                   [--agents=devops|quality|orchestrator|all] \
                   [--learn] \
                   [--batch] \
                   [--correlate] \
                   [commit-hash-or-pr-number]
```

**Use cases:**
- Fix CI/CD failures
- Auto-resolve build errors
- Pattern-based fixes
- Emergency fixes
- Batch error resolution

**Examples:**
```bash
# Auto-fix commit errors
/fix-commit-errors --auto-fix abc1234

# Emergency fix with learning
/fix-commit-errors --emergency --learn --auto-fix

# Batch fix multiple commits
/fix-commit-errors --batch --correlate --max-cycles=5
```

**Agents used:** DevOps, Quality Assurance, language experts

**Output:** Fixed errors with commits/PRs created

---

#### /fix-github-issue
**GitHub issue analysis and automated fixing**

```bash
/fix-github-issue [--auto-fix] \
                  [--draft] \
                  [--interactive] \
                  [--emergency] \
                  [--branch=name] \
                  [--agents=quality|devops|orchestrator|all] \
                  [issue-number-or-url]
```

**Use cases:**
- Auto-fix GitHub issues
- Create fix PRs
- Emergency issue resolution
- Batch issue fixing
- Interactive fixing

**Examples:**
```bash
# Auto-fix issue and create PR
/fix-github-issue --auto-fix 123

# Emergency fix
/fix-github-issue --emergency --auto-fix --branch=hotfix/issue-123 456

# Interactive fixing
/fix-github-issue --interactive --draft 789
```

**Agents used:** DevOps, Quality Assurance, domain experts

**Output:** Issue fixed with PR created

---

### CI/CD & Documentation

#### /ci-setup
**CI/CD pipeline setup and automation**

```bash
/ci-setup [--platform=github|gitlab|jenkins] \
          [--type=basic|security|enterprise] \
          [--deploy=staging|production|both] \
          [--monitoring] \
          [--security] \
          [--agents=devops|quality|orchestrator|all]
```

**Use cases:**
- Setup CI/CD pipelines
- Add security scanning
- Configure deployments
- Add monitoring
- Enterprise CI/CD

**Examples:**
```bash
# Basic GitHub Actions
/ci-setup --platform=github --type=basic

# Enterprise setup with security
/ci-setup --platform=github --type=enterprise --security --monitoring

# Multi-environment deployment
/ci-setup --deploy=both --security --agents=devops
```

**Agents used:** DevOps, Security Engineer, Quality Assurance

**Output:** CI/CD configuration files created and configured

---

#### /update-docs
**Documentation generation with AST-based extraction**

```bash
/update-docs [--type=readme|api|research|all] \
             [--format=markdown|html|latex] \
             [--interactive] \
             [--collaborative] \
             [--publish] \
             [--optimize] \
             [--agents=auto|documentation|scientific|ai|engineering|research|all] \
             [--orchestrate] \
             [--parallel] \
             [--intelligent]
```

**Use cases:**
- Generate README
- API documentation
- Research papers
- User guides
- Multi-format docs

**Examples:**
```bash
# Generate complete documentation
/update-docs --type=all --format=markdown

# Research paper
/update-docs --type=research --format=latex --agents=scientific

# API docs with publishing
/update-docs --type=api --format=html --publish --optimize
```

**Agents used:** Documentation, domain experts, Research Scientist

**Output:** Generated documentation files in specified formats

---

### Multi-Agent & Integration

#### /multi-agent-optimize
**Multi-agent system for optimization and review**

```bash
/multi-agent-optimize [--mode=optimize|review|hybrid|research] \
                      [--agents=all|core|scientific|ai|engineering|domain-specific] \
                      [--focus=performance|security|quality|architecture|research|innovation] \
                      [--implement] \
                      [--orchestrate] \
                      [target]
```

**Use cases:**
- Multi-agent optimization
- Comprehensive code review
- Research workflows
- Architecture review
- Security audit

**Examples:**
```bash
# Quality-focused review
/multi-agent-optimize --mode=review --focus=quality --implement

# Performance optimization
/multi-agent-optimize --mode=optimize --focus=performance --agents=scientific

# Research workflow
/multi-agent-optimize --mode=research --focus=innovation --orchestrate

# Complete optimization
/multi-agent-optimize --mode=hybrid --agents=all --implement
```

**Agents used:** All agents coordinated by Orchestrator

**Output:** Comprehensive analysis and improvements across all dimensions

---

#### /adopt-code
**Analyze, integrate, and optimize scientific computing codebases**

```bash
/adopt-code [--analyze] \
            [--integrate] \
            [--optimize] \
            [--language=fortran|c|cpp|python|julia|mixed] \
            [--target=python|jax|julia] \
            [--parallel=mpi|openmp|cuda|jax] \
            [--agents=scientific|quality|orchestrator|all] \
            [codebase-path]
```

**Use cases:**
- Adopt legacy scientific code
- Fortran to Python conversion
- C/C++ integration
- Modernize parallel code
- Scientific code analysis

**Examples:**
```bash
# Analyze Fortran code
/adopt-code --analyze --language=fortran legacy/

# Convert Fortran to Python
/adopt-code --integrate --language=fortran --target=python --parallel=mpi

# Complete adoption workflow
/adopt-code --analyze --integrate --optimize --language=c --target=jax
```

**Agents used:** Scientific Computing, Performance Engineer, language experts

**Output:** Analysis report, integrated code, optimized implementation

---

#### /explain-code
**Advanced code analysis and documentation**

```bash
/explain-code [--level=basic|advanced|expert] \
              [--focus=AREA] \
              [--docs] \
              [--interactive] \
              [--format=FORMAT] \
              [--export=PATH] \
              [--agents=documentation|quality|scientific|all] \
              [file/directory]
```

**Use cases:**
- Code explanation
- Onboarding documentation
- Architecture documentation
- Educational materials
- Code review support

**Examples:**
```bash
# Basic explanation
/explain-code --level=basic src/main.py

# Expert-level analysis
/explain-code --level=expert --focus=algorithms --docs src/

# Interactive explanation
/explain-code --interactive --export=explanation.md
```

**Agents used:** Documentation, domain experts, Code Reviewer

**Output:** Detailed code explanation and documentation

---

## 23-Agent System

### Overview

The 23-agent system provides specialized expertise across different domains. Agents work independently or collaboratively, coordinated by the Orchestrator.

### Agent Categories

#### Core Agents (3)

**1. Orchestrator Agent**
- **Role:** Multi-agent coordination and workflow management
- **Expertise:** Task decomposition, agent selection, result synthesis
- **When used:** All multi-agent operations, workflow execution
- **Capabilities:**
  - Coordinate multiple agents
  - Optimize execution order
  - Handle agent communication
  - Synthesize results
  - Manage resources

**2. Quality Assurance Agent**
- **Role:** Code quality, testing, and standards enforcement
- **Expertise:** Quality metrics, testing strategies, best practices
- **When used:** Quality checks, test generation, code review
- **Capabilities:**
  - Quality analysis
  - Test strategy
  - Standards enforcement
  - Code review
  - Quality gates

**3. DevOps Agent**
- **Role:** CI/CD, deployment, and infrastructure
- **Expertise:** Pipeline design, automation, monitoring
- **When used:** CI/CD setup, deployment, infrastructure
- **Capabilities:**
  - Pipeline configuration
  - Deployment automation
  - Infrastructure as code
  - Monitoring setup
  - Container orchestration

#### Scientific Computing Agents (4)

**4. Scientific Computing Agent**
- **Role:** Scientific algorithms and numerical methods
- **Expertise:** Numerical analysis, scientific algorithms, HPC
- **When used:** Scientific code analysis, optimization
- **Capabilities:**
  - Numerical algorithm optimization
  - Scientific code review
  - Mathematical correctness
  - Precision analysis
  - Domain-specific patterns

**5. Performance Engineer Agent**
- **Role:** Optimization and profiling
- **Expertise:** Performance analysis, profiling, optimization techniques
- **When used:** Performance optimization, profiling
- **Capabilities:**
  - Performance profiling
  - Bottleneck identification
  - Optimization strategies
  - Benchmark design
  - Resource optimization

**6. GPU Specialist Agent**
- **Role:** GPU computing and CUDA/JAX
- **Expertise:** GPU programming, CUDA, JAX, parallel computing
- **When used:** GPU code, CUDA optimization, JAX development
- **Capabilities:**
  - GPU code optimization
  - CUDA debugging
  - JAX transformation
  - Memory management
  - Kernel optimization

**7. Research Scientist Agent**
- **Role:** Research workflows and experimentation
- **Expertise:** Experimental design, reproducibility, scientific method
- **When used:** Research projects, experiments, papers
- **Capabilities:**
  - Experimental design
  - Reproducibility setup
  - Paper writing
  - Literature review
  - Hypothesis testing

#### AI/ML Agents (3)

**8. AI/ML Engineer Agent**
- **Role:** Machine learning and neural networks
- **Expertise:** ML algorithms, model training, deployment
- **When used:** ML projects, model development
- **Capabilities:**
  - Model architecture design
  - Training pipeline setup
  - Hyperparameter tuning
  - Model evaluation
  - ML deployment

**9. JAX Specialist Agent**
- **Role:** JAX framework expertise
- **Expertise:** JAX transformations, XLA, auto-differentiation
- **When used:** JAX code development, optimization
- **Capabilities:**
  - JAX transformation
  - XLA optimization
  - Auto-diff debugging
  - JAX patterns
  - Performance tuning

**10. Model Optimization Agent**
- **Role:** Model performance and efficiency
- **Expertise:** Model compression, quantization, optimization
- **When used:** Model optimization, deployment
- **Capabilities:**
  - Model compression
  - Quantization
  - Pruning
  - Knowledge distillation
  - Inference optimization

#### Engineering Agents (5)

**11. Backend Engineer Agent**
- **Role:** Server-side architecture
- **Expertise:** APIs, databases, microservices, scalability
- **When used:** Backend development, API design
- **Capabilities:**
  - API design
  - Database optimization
  - Microservices architecture
  - Scalability patterns
  - Backend testing

**12. Frontend Engineer Agent**
- **Role:** UI/UX development
- **Expertise:** React, Vue, Angular, responsive design, accessibility
- **When used:** Frontend development, UI work
- **Capabilities:**
  - Component architecture
  - State management
  - Responsive design
  - Accessibility
  - Performance optimization

**13. Security Engineer Agent**
- **Role:** Security and vulnerability analysis
- **Expertise:** Security best practices, vulnerability detection, compliance
- **When used:** Security audits, vulnerability fixes
- **Capabilities:**
  - Security scanning
  - Vulnerability assessment
  - Secure coding practices
  - Compliance checking
  - Penetration testing

**14. Database Engineer Agent**
- **Role:** Database design and optimization
- **Expertise:** SQL, NoSQL, query optimization, data modeling
- **When used:** Database work, query optimization
- **Capabilities:**
  - Schema design
  - Query optimization
  - Index strategy
  - Data migration
  - Database tuning

**15. Cloud Architect Agent**
- **Role:** Cloud infrastructure and scaling
- **Expertise:** AWS, GCP, Azure, serverless, Kubernetes
- **When used:** Cloud deployment, infrastructure
- **Capabilities:**
  - Infrastructure design
  - Cloud migration
  - Kubernetes orchestration
  - Serverless architecture
  - Cost optimization

#### Domain-Specific Agents (8)

**16. Python Expert Agent**
- **Role:** Python best practices
- **Expertise:** Python idioms, libraries, performance, typing
- **When used:** Python development
- **Capabilities:**
  - Pythonic code
  - Library selection
  - Type hints
  - Performance patterns
  - Python tooling

**17. Julia Expert Agent**
- **Role:** Julia language expertise
- **Expertise:** Julia patterns, performance, ecosystem
- **When used:** Julia development
- **Capabilities:**
  - Julia idioms
  - Type system
  - Multiple dispatch
  - Performance optimization
  - Package development

**18. JavaScript Expert Agent**
- **Role:** JavaScript/TypeScript
- **Expertise:** Modern JS/TS, Node.js, frameworks
- **When used:** JavaScript/TypeScript development
- **Capabilities:**
  - Modern JavaScript
  - TypeScript types
  - Async patterns
  - Framework knowledge
  - Node.js expertise

**19. Documentation Agent**
- **Role:** Technical writing
- **Expertise:** Documentation, API docs, tutorials
- **When used:** Documentation generation
- **Capabilities:**
  - Clear writing
  - API documentation
  - Tutorial creation
  - README generation
  - Documentation structure

**20. Code Reviewer Agent**
- **Role:** Code review and standards
- **Expertise:** Code review, best practices, patterns
- **When used:** Code review, refactoring
- **Capabilities:**
  - Code review
  - Pattern recognition
  - Best practice enforcement
  - Architecture review
  - Maintainability assessment

**21. Refactoring Agent**
- **Role:** Code refactoring patterns
- **Expertise:** Refactoring techniques, design patterns
- **When used:** Refactoring, modernization
- **Capabilities:**
  - Refactoring patterns
  - Design patterns
  - Code smells detection
  - Legacy code modernization
  - Structure improvement

**22. Testing Agent**
- **Role:** Test strategy and implementation
- **Expertise:** Testing patterns, frameworks, coverage
- **When used:** Test generation, test strategy
- **Capabilities:**
  - Test strategy
  - Test generation
  - Coverage analysis
  - Test frameworks
  - Testing patterns

**23. Quantum Computing Agent**
- **Role:** Quantum algorithms
- **Expertise:** Quantum computing, Qiskit, Cirq
- **When used:** Quantum computing projects
- **Capabilities:**
  - Quantum algorithms
  - Circuit design
  - Quantum optimization
  - Framework selection
  - Hybrid classical-quantum

### Agent Selection Strategies

#### Automatic Selection (--agents=auto)
Default behavior. System automatically selects appropriate agents based on:
- Task type
- File types in codebase
- Command being executed
- Explicit requirements

**Example:**
```bash
/optimize --agents=auto src/
# Automatically selects: Performance Engineer, language expert, Scientific Computing (if applicable)
```

#### Explicit Selection
Choose specific agent categories:

```bash
# Core agents only
/multi-agent-optimize --agents=core

# Scientific computing agents
/optimize --agents=scientific

# All AI/ML agents
/optimize --agents=ai

# All engineering agents
/multi-agent-optimize --agents=engineering

# Domain-specific agents
/clean-codebase --agents=domain-specific

# All agents
/multi-agent-optimize --agents=all
```

#### Intelligent Selection (--intelligent)
Advanced agent selection using:
- Codebase analysis
- Historical patterns
- Performance metrics
- Resource availability

**Example:**
```bash
/multi-agent-optimize --orchestrate --intelligent
# Uses intelligent selection with orchestration
```

### Agent Coordination

#### Sequential Execution
Agents work one after another, each building on previous results:

```bash
/multi-agent-optimize --mode=optimize --focus=quality
# 1. Quality Assurance analyzes
# 2. Code Reviewer reviews
# 3. Refactoring Agent suggests improvements
# 4. Testing Agent adds tests
```

#### Parallel Execution (--parallel)
Multiple agents work simultaneously:

```bash
/multi-agent-optimize --parallel --orchestrate
# Multiple agents analyze different aspects simultaneously
# Orchestrator coordinates and synthesizes results
```

#### Orchestrated Execution (--orchestrate)
Orchestrator manages complex multi-agent workflows:

```bash
/multi-agent-optimize --orchestrate --intelligent --agents=all
# Orchestrator:
# - Selects optimal agents
# - Determines execution order
# - Manages dependencies
# - Synthesizes results
# - Handles conflicts
```

---

## Workflow Framework

### Overview

The workflow framework enables complex multi-step automation through YAML-based workflow definitions.

### Pre-built Workflows

Located in `/Users/b80985/.claude/commands/workflows/`

#### Quality Workflows

**quality-gate.yml**
```yaml
name: Complete Quality Gate
steps:
  - check-code-quality: {auto-fix: true}
  - generate-tests: {coverage: 90}
  - run-all-tests: {auto-fix: true}
  - double-check: "all quality criteria met"
```

**auto-fix.yml**
```yaml
name: Automated Quality Improvement
steps:
  - check-code-quality: {auto-fix: true}
  - refactor-clean: {implement: true}
  - clean-codebase: {imports: true, dead-code: true}
  - run-all-tests: {auto-fix: true}
```

#### Performance Workflows

**performance-audit.yml**
```yaml
name: Complete Performance Audit
steps:
  - optimize: {category: all, profile: true}
  - run-all-tests: {benchmark: true}
  - generate-report: {type: performance}
  - double-check: "performance targets achieved"
```

#### CI/CD Workflows

**ci-pipeline.yml**
```yaml
name: CI/CD Pipeline Setup
steps:
  - ci-setup: {platform: github, type: enterprise}
  - generate-tests: {type: integration}
  - update-docs: {type: all}
  - commit: {ai-message: true, validate: true}
```

#### Scientific Computing Workflows

**research-pipeline.yml**
```yaml
name: Research Workflow
steps:
  - adopt-code: {analyze: true, integrate: true}
  - optimize: {agents: scientific, category: algorithm}
  - generate-tests: {type: scientific, reproducible: true}
  - update-docs: {type: research, format: latex}
  - run-all-tests: {scientific: true, gpu: true}
```

### Using Workflows

```bash
# Execute a workflow
/multi-agent-optimize --mode=optimize --focus=quality --implement
# This executes the quality optimization workflow

# Custom workflow
# Create workflow YAML file and execute
```

### Creating Custom Workflows

**workflow-template.yml**
```yaml
name: My Custom Workflow
description: Description of what this workflow does
version: 1.0.0

# Define workflow parameters
parameters:
  target_coverage:
    type: integer
    default: 90
  language:
    type: string
    default: python

# Define workflow steps
steps:
  # Step 1: Quality check
  - name: quality-check
    command: check-code-quality
    args:
      language: ${language}
      auto-fix: true
    on_failure: continue  # or abort, skip

  # Step 2: Test generation
  - name: generate-tests
    command: generate-tests
    args:
      coverage: ${target_coverage}
      type: unit
    depends_on: [quality-check]

  # Step 3: Run tests
  - name: run-tests
    command: run-all-tests
    args:
      auto-fix: true
      coverage: true
    depends_on: [generate-tests]

  # Step 4: Verification
  - name: verify
    command: double-check
    args:
      description: "all quality improvements complete"
    depends_on: [run-tests]

# Define success criteria
success_criteria:
  - tests_passing: true
  - coverage_met: true
  - quality_improved: true

# Define outputs
outputs:
  quality_report: results/quality-report.json
  coverage_report: results/coverage-report.html
  test_results: results/test-results.xml
```

### Workflow Features

#### Conditional Execution
```yaml
steps:
  - name: optimize
    command: optimize
    condition: ${performance_issues} > 0
```

#### Error Handling
```yaml
steps:
  - name: risky-step
    command: some-command
    on_failure: continue
    retry: 3
    timeout: 300
```

#### Parallel Steps
```yaml
parallel:
  - name: quality-check
    command: check-code-quality
  - name: security-scan
    command: check-code-quality
    args:
      focus: security
```

#### Resource Management
```yaml
resources:
  agents: scientific
  parallel: true
  max_concurrent: 5
```

---

## Plugin System

### Overview

The plugin system allows you to extend the framework with custom functionality.

### Available Plugins

See **[Plugin Index](../../plugins/PLUGIN_INDEX.md)** for complete list.

### Using Plugins

#### Install Plugin
```bash
# Install from registry
claude-commands install plugin-name

# Install from file
claude-commands install ./my-plugin.zip

# Install from git
claude-commands install github:user/plugin-repo
```

#### List Plugins
```bash
# List installed plugins
claude-commands list-plugins

# Show plugin info
claude-commands info plugin-name
```

#### Enable/Disable Plugins
```bash
# Enable plugin
claude-commands enable plugin-name

# Disable plugin
claude-commands disable plugin-name

# Update plugin
claude-commands update plugin-name
```

### Creating Plugins

See **[Plugin Development Guide](../../plugins/docs/PLUGIN_DEVELOPMENT_GUIDE.md)** for complete guide.

#### Basic Plugin Structure

```python
# my_plugin/__init__.py
from claude_commands.plugin import Plugin, command, agent

class MyPlugin(Plugin):
    name = "my-plugin"
    version = "1.0.0"
    description = "My custom plugin"

    @command("my-command")
    def my_command(self, args):
        """Custom command implementation"""
        return {"status": "success"}

    @agent("MyAgent")
    def my_agent(self):
        """Custom agent implementation"""
        return {
            "expertise": ["custom-domain"],
            "capabilities": ["custom-capability"]
        }
```

#### Plugin Metadata

```yaml
# plugin.yml
name: my-plugin
version: 1.0.0
description: My custom plugin
author: Your Name
license: MIT

dependencies:
  - claude-commands>=1.0.0
  - some-library>=2.0.0

commands:
  - my-command

agents:
  - MyAgent

workflows:
  - my-workflow.yml
```

---

## Configuration

### Configuration Files

#### User Configuration

`~/.claude-commands/config.yml`

```yaml
# Default settings
defaults:
  agents: auto
  auto_fix: true
  parallel: true
  cache_enabled: true

# Agent preferences
agents:
  selection_strategy: intelligent
  max_concurrent: 10
  timeout: 300

# Performance settings
performance:
  parallel_execution: true
  cache_size: 1000
  max_memory: 8GB

# Quality settings
quality:
  min_coverage: 80
  strict_mode: false
  auto_fix: true

# Reporting
reporting:
  format: text
  export_path: ./reports
  detailed: true

# Integrations
integrations:
  git:
    enabled: true
    auto_commit: false
  ci_cd:
    platform: github
    auto_setup: false
```

#### Project Configuration

`.claude-commands.yml` (project root)

```yaml
project:
  name: my-project
  language: python
  type: scientific

quality:
  min_coverage: 90
  style_guide: pep8

agents:
  preferred: [scientific, quality, devops]

workflows:
  default: quality-gate

plugins:
  - scientific-computing-enhanced
  - custom-linters
```

### Environment Variables

```bash
# Enable/disable features
export CLAUDE_COMMANDS_AUTO_FIX=true
export CLAUDE_COMMANDS_PARALLEL=true
export CLAUDE_COMMANDS_CACHE=true

# Agent configuration
export CLAUDE_COMMANDS_AGENTS=auto
export CLAUDE_COMMANDS_MAX_AGENTS=10

# Performance tuning
export CLAUDE_COMMANDS_TIMEOUT=300
export CLAUDE_COMMANDS_MEMORY_LIMIT=8G

# Output configuration
export CLAUDE_COMMANDS_OUTPUT_FORMAT=json
export CLAUDE_COMMANDS_VERBOSE=true
```

### Command-Line Configuration

Override defaults with command-line flags:

```bash
# Override agent selection
/optimize --agents=scientific

# Override auto-fix
/check-code-quality --auto-fix=false

# Override parallel execution
/multi-agent-optimize --parallel=false

# Override output format
/check-code-quality --format=json
```

---

## Best Practices

### Code Quality

1. **Run quality checks frequently**
```bash
# Before commits
/check-code-quality --auto-fix
```

2. **Maintain high test coverage**
```bash
# Aim for 90%+ coverage
/generate-tests --coverage=90
/run-all-tests --coverage
```

3. **Use quality gates in CI/CD**
```bash
# Setup quality gates
/ci-setup --type=enterprise --security
```

4. **Regular codebase cleanup**
```bash
# Monthly cleanup
/clean-codebase --analysis=comprehensive --imports --dead-code --duplicates
```

### Performance

1. **Profile before optimizing**
```bash
# Always profile first
/optimize --profile
/run-all-tests --benchmark
```

2. **Use appropriate agents**
```bash
# For scientific code
/optimize --agents=scientific

# For web applications
/optimize --agents=engineering
```

3. **Benchmark improvements**
```bash
# Before and after
/run-all-tests --benchmark > before.txt
/optimize --implement
/run-all-tests --benchmark > after.txt
```

### Testing

1. **Comprehensive test suites**
```bash
# Generate all test types
/generate-tests --type=all --coverage=95
```

2. **Use auto-fix for failing tests**
```bash
# Auto-fix test failures
/run-all-tests --auto-fix
```

3. **Scientific testing best practices**
```bash
# Reproducible scientific tests
/generate-tests --type=scientific --reproducible
/run-all-tests --scientific --gpu
```

### Workflows

1. **Use pre-built workflows**
```bash
# Quality workflow
/multi-agent-optimize --mode=review --focus=quality --implement
```

2. **Create custom workflows for common tasks**
- Define workflows in YAML
- Version control workflows
- Share across team

3. **Combine commands for complex tasks**
```bash
# Complex workflow
/check-code-quality --auto-fix && \
/generate-tests --coverage=90 && \
/run-all-tests --auto-fix && \
/optimize --implement && \
/double-check "all improvements complete"
```

### Team Collaboration

1. **Standardize on configurations**
- Use project `.claude-commands.yml`
- Version control configuration
- Document team workflows

2. **Setup CI/CD early**
```bash
/ci-setup --platform=github --type=enterprise --security --monitoring
```

3. **Use quality gates**
```bash
# Pre-commit hook
/check-code-quality --auto-fix
/run-all-tests
```

4. **Generate comprehensive documentation**
```bash
/update-docs --type=all --format=markdown
```

### Scientific Computing

1. **Adopt legacy code carefully**
```bash
# Analyze first, then integrate
/adopt-code --analyze legacy/
# Review analysis
/adopt-code --integrate --optimize
```

2. **Ensure reproducibility**
```bash
/generate-tests --type=scientific --reproducible
/run-all-tests --scientific --reproducible
```

3. **Optimize for your hardware**
```bash
# GPU optimization
/optimize --agents=scientific --gpu

# Parallel optimization
/optimize --parallel=mpi
```

### Enterprise

1. **Security first**
```bash
/ci-setup --security
/multi-agent-optimize --focus=security --implement
```

2. **Compliance and audit trails**
```bash
# Generate audit reports
/multi-agent-optimize --report --export=audit/
```

3. **Scalability testing**
```bash
/run-all-tests --scope=performance --distributed
```

---

## Performance Optimization

### System Performance

#### Caching
- Automatic caching of analysis results
- Cache invalidation on file changes
- Configurable cache size

```bash
# Check cache status
claude-commands cache-status

# Clear cache
claude-commands cache-clear

# Configure cache
export CLAUDE_COMMANDS_CACHE_SIZE=2000
```

#### Parallel Execution
```bash
# Enable parallel execution
/multi-agent-optimize --parallel --orchestrate

# Configure max concurrent
export CLAUDE_COMMANDS_MAX_CONCURRENT=10
```

#### Resource Management
```bash
# Monitor resource usage
/multi-agent-optimize --monitor

# Set resource limits
export CLAUDE_COMMANDS_MEMORY_LIMIT=8G
export CLAUDE_COMMANDS_TIMEOUT=600
```

### Code Performance

#### Algorithm Optimization
```bash
# Focus on algorithms
/optimize --category=algorithm --agents=scientific
```

#### Memory Optimization
```bash
# Memory profiling and optimization
/optimize --category=memory --profile
```

#### I/O Optimization
```bash
# I/O optimization
/optimize --category=io --implement
```

#### Concurrency Optimization
```bash
# Parallel and concurrent optimization
/optimize --category=concurrency --parallel=mpi
```

### Benchmark ing

```bash
# Performance benchmarking
/run-all-tests --benchmark --profile

# Compare benchmarks
/run-all-tests --benchmark > before.json
/optimize --implement
/run-all-tests --benchmark > after.json
claude-commands compare-benchmarks before.json after.json
```

---

## Troubleshooting

See **[Troubleshooting Guide](TROUBLESHOOTING.md)** for comprehensive troubleshooting.

### Common Issues

#### Command Not Found
**Problem:** Command not recognized
**Solution:** Ensure you're using slash command format: `/command-name`

#### Permission Errors
**Problem:** Cannot write files
**Solution:** Check directory permissions, run from project root

#### Import Errors
**Problem:** Missing dependencies
**Solution:**
```bash
pip install -r requirements.txt
```

#### Test Failures
**Problem:** Tests failing after changes
**Solution:**
```bash
/run-all-tests --auto-fix
/debug --auto-fix
```

#### Performance Issues
**Problem:** Commands running slowly
**Solution:**
```bash
# Enable parallel execution
/multi-agent-optimize --parallel

# Clear cache
claude-commands cache-clear

# Check system resources
/debug --monitor
```

#### Agent Selection Issues
**Problem:** Wrong agents selected
**Solution:**
```bash
# Explicit agent selection
/optimize --agents=scientific

# Use intelligent selection
/multi-agent-optimize --intelligent
```

### Getting Help

1. **Check documentation**
   - This user guide
   - Command-specific docs
   - Troubleshooting guide

2. **Use debug command**
```bash
/debug --issue=TYPE --report
```

3. **Check system status**
```bash
claude-commands status
claude-commands diagnostics
```

4. **Community support**
   - GitHub Issues
   - Documentation
   - FAQ

---

## Next Steps

### For New Users
1. Complete [Tutorial 01: Introduction](../tutorials/tutorial-01-introduction.md)
2. Try [Tutorial 02: Code Quality](../tutorials/tutorial-02-code-quality.md)
3. Explore [Workflow Framework](../../workflows/INDEX.md)

### For Developers
1. Read [Developer Guide](DEVELOPER_GUIDE.md)
2. Review [API Reference](API_REFERENCE.md)
3. Check [Architecture](../ARCHITECTURE.md)

### For Teams
1. Review [Tutorial 08: Enterprise](../tutorials/tutorial-08-enterprise.md)
2. Setup CI/CD: `/ci-setup`
3. Define team workflows

### For Researchers
1. Complete [Tutorial 07: Scientific Computing](../tutorials/tutorial-07-scientific-computing.md)
2. Try `/adopt-code` for legacy code
3. Explore scientific workflows

---

## Additional Resources

- **[Master Index](MASTER_INDEX.md)** - Complete documentation index
- **[Getting Started](GETTING_STARTED.md)** - Quick start guide
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Development documentation
- **[API Reference](API_REFERENCE.md)** - API documentation
- **[Troubleshooting](TROUBLESHOOTING.md)** - Problem solving
- **[FAQ](FAQ.md)** - Frequently asked questions
- **[Tutorials](../tutorials/)** - Hands-on learning
- **[Workflows](../../workflows/INDEX.md)** - Workflow documentation
- **[Plugins](../../plugins/PLUGIN_INDEX.md)** - Plugin system

---

**Version**: 1.0.0 | **Last Updated**: September 2025 | **Status**: Production Ready

Need help? Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or [FAQ](FAQ.md)