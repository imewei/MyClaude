---
title: "Commit"
description: "Git commit engine with AI message generation and automated quality validation"
category: git-workflow
subcategory: version-control
complexity: basic
argument-hint: "[--all] [--staged] [--amend] [--interactive] [--split] [--template=TYPE] [--ai-message] [--validate] [--push] [--agents=quality|devops|orchestrator|all]"
allowed-tools: Bash, Read, Grep, Glob, TodoWrite
model: inherit
tags: git, commit, ai-message, validation, version-control
dependencies: []
related: [check-code-quality, run-all-tests, ci-setup, fix-commit-errors, optimize, generate-tests, refactor-clean, double-check]
workflows: [commit-workflow, quality-validation, code-review]
version: "2.0"
last-updated: "2025-09-28"
---

# Commit

Git commit engine with AI message generation, change analysis, and quality validation.

## Quick Start

```bash
# Basic commit with AI message
/commit --ai-message

# Commit all changes with validation
/commit --all --ai-message --validate

# Interactive commit with template
/commit --interactive --template=feat

# Quick commit and push
/commit --all --ai-message --push
```

## Usage

```bash
/commit [options]
```

**Parameters:**
- `options` - Commit configuration, message generation, and validation options

## Options

### Staging Options
- `--all`: Add all changed files before committing
- `--staged`: Only commit currently staged changes
- `--split`: Split large changes into multiple logical commits

### Commit Options
- `--amend`: Amend the previous commit instead of creating new one
- `--template=<type>`: Use commit template (feat, fix, docs, refactor, test, chore)
- `--ai-message`: Generate commit message using AI analysis of changes
- `--interactive`: Interactive commit creation with interactive workflow

### Quality Options
- `--validate`: Run pre-commit checks (linting, tests, security)
- `--push`: Push to remote after successful commit
- `--agents=<agents>`: Agent selection (quality, devops, orchestrator, all)

## Agent Integration

### Quality Agent (`code-quality-master`)
- **Pre-Commit Validation**: Comprehensive code quality analysis and automated fixes
- **Test Integration**: Automated testing and coverage validation before commits
- **Code Review**: Static analysis, security scanning, and best practice validation
- **Performance Analysis**: Build optimization and deployment readiness checks
- **Documentation Validation**: Documentation quality and completeness verification

### DevOps Security Agent (`devops-security-engineer`)
- **Secure Commit Practices**: Security validation and vulnerability scanning
- **Compliance Checks**: Regulatory compliance and audit trail validation
- **Secret Detection**: Automated scanning for exposed credentials and sensitive data
- **Pipeline Integration**: CI/CD pipeline preparation and validation
- **Infrastructure Security**: Infrastructure-as-code security and compliance validation

### Multi-Agent Orchestrator (`multi-agent-orchestrator`)
- **Workflow Coordination**: Complex commit workflow orchestration and automation
- **Quality Gate Management**: Multi-stage validation and approval processes
- **Resource Optimization**: Intelligent allocation of validation and testing tasks
- **Failure Recovery**: Automated issue resolution and commit retry mechanisms
- **Process Monitoring**: Commit workflow efficiency and performance tracking

## Agent Selection Options

- `quality` - Quality engineering focus for code validation and testing
- `devops` - DevSecOps focus for security and compliance validation
- `orchestrator` - Multi-agent coordination for complex commit workflows
- `all` - Complete multi-agent commit system with comprehensive validation

## Features

### AI Message Generation
Analyzes your changes to generate commit messages:
- Detects change types (features, fixes, refactoring)
- Identifies affected components and scope
- Follows conventional commit format
- Includes breaking change detection

### Change Analysis
- Categorizes files by type (source, test, docs, config)
- Assesses change complexity and impact
- Suggests logical commit groupings
- Detects project patterns and conventions

### Quality Validation
- Pre-commit hook integration
- Syntax and style checking
- Security scan for exposed secrets
- Test execution verification

### Interactive Workflow
- Interactive commit creation process
- File selection with previews
- Message editing and refinement
- Template-based commit structure

## Commit Templates

### Feature (`--template=feat`)
```
feat(scope): add [feature]

- Change summary
- Breaking changes
- Related issue references

Closes #123
```

### Bug Fix (`--template=fix`)
```
fix(scope): fix [issue]

- Root cause
- Solution
- Test added

Fixes #456
```

### Documentation (`--template=docs`)
```
docs(scope): update documentation

- What was changed
- Reason
- User changes
```

## Examples

### Basic Usage
```bash
# Simple commit with AI message
/commit --ai-message

# Commit staged changes only
/commit --staged --template=fix
```

### Complex Workflows
```bash
# Full validation workflow with quality agent
/commit --all --ai-message --validate --push --agents=quality

# Interactive feature commit with security validation
/commit --interactive --template=feat --validate --agents=devops

# Multi-agent comprehensive commit validation
/commit --all --ai-message --validate --agents=all --push

# Split complex changes
/commit --split --ai-message
```

### Template-Based Commits
```bash
# Feature development
/commit --template=feat --ai-message --validate

# Bug fix with validation
/commit --template=fix --staged --validate

# Documentation update
/commit --template=docs --all
```

## Integration

Works with existing git workflows and tools:
- Pre-commit hooks
- Conventional commit standards
- CI/CD pipelines
- Code review processes
- Issue tracking systems

## Best Practices

**Use AI messages for:**
- Complex multi-file changes
- Feature development
- Refactoring work
- For complex changes

**Use templates for:**
- Team consistency
- Conventional commit compliance
- Structured workflows
- Documentation requirements

**Use validation for:**
- Production branches
- Shared repositories
- Quality requirements
- Security compliance

## AI Message Generation Guidelines

## Common Workflows

### Basic Commit Workflow
```bash
# 1. Quick commit with AI message
/commit --ai-message --validate

# 2. Review and push
/commit --staged --push
```

### Feature Development Workflow
```bash
# 1. Feature commit with template
/commit --template=feat --ai-message --validate

# 2. Run tests before pushing
/run-all-tests --coverage
/commit --push
```

### Code Quality Workflow
```bash
# 1. Check quality before commit
/check-code-quality --auto-fix

# 2. Commit quality improvements
/commit --all --template=refactor --validate

# 3. Push with CI validation
/commit --push
```

## Related Commands

**Prerequisites**: Commands to run before committing
- `/check-code-quality --auto-fix` - Fix quality issues that could cause CI failures
- `/run-all-tests --auto-fix` - Ensure all tests pass before committing
- `/debug --auto-fix` - Fix runtime issues that affect functionality
- Clean working directory - Ensure no untracked temporary files

**Alternatives**: Different commit approaches
- Manual git commands for simple single-file changes
- IDE git integration for basic workflows
- Git CLI for complex merge scenarios

**Combinations**: Commands that work with commit
- `/generate-tests --coverage=90` - Add comprehensive tests before committing
- `/optimize --implement` - Apply performance optimizations before committing
- `/double-check --deep-analysis` - Verify changes comprehensively before commit
- `/refactor-clean --implement` - Apply structural improvements before committing
- `/reflection --type=session` - Analyze work session before committing

**Follow-up**: Commands to run after committing
- `/ci-setup --monitoring` - Set up automated testing and monitoring
- `/fix-commit-errors` - Monitor and fix CI/CD issues automatically
- `/update-docs --type=api` - Update documentation to reflect changes
- `/reflection --type=instruction` - Analyze commit effectiveness

## Integration Patterns

### Development Workflow
```bash
# Complete development cycle
/debug --auto-fix                    # Fix issues
/optimize --implement                # Optimize code
/generate-tests --coverage=90        # Add tests
/commit --ai-message --validate --push  # Commit changes
```

### Quality Assurance Workflow
```bash
# Quality-focused commits
/check-code-quality --auto-fix
/refactor-clean --implement
/commit --template=refactor --validate
```

### Feature Development
```bash
# Feature development cycle
/commit --template=feat --interactive  # Initial feature commit
/commit --template=test --staged       # Add tests
/commit --template=docs --all          # Update documentation
```

**Message Generation Guidelines**

**CRITICAL**: When generating commit messages, use direct technical language:
- Use factual descriptions of changes made
- Avoid promotional words like "enhance", "improve", "optimize", "streamline", "powerful", "robust"
- Write concise, technical descriptions of what was changed
- Focus on specific actions: "add", "fix", "remove", "update", "refactor"
- Use imperative mood with concrete technical details
- Example: "add user authentication" not "enhance user experience with robust authentication"