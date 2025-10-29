# Quality Engineering Plugin

Comprehensive quality assurance, validation, and correctness verification tools for software development.

## Overview

This plugin provides systematic quality engineering tools that go beyond traditional testing. It includes comprehensive validation frameworks, automated quality checks, and specialized validators for ensuring code correctness, security, and maintainability.

## Features

- **Multi-Dimensional Validation**: 10-dimension quality checks
- **Automated Security Scanning**: Dependency audits, SAST, secret detection
- **Plugin Syntax Validation**: Prevent runtime failures in Claude Code plugins
- **Comprehensive Checklists**: Code quality, security, performance, accessibility
- **Cross-Reference Validation**: Ensure consistency across systems

## Commands

### /double-check

Comprehensive multi-dimensional validation with automated testing, security scanning, and ultra-think reasoning.

**Usage:**
```bash
/double-check [work-to-validate] [--deep] [--security] [--performance]
```

**10 Validation Dimensions:**
1. **Scope & Requirements**: Verify all requirements met
2. **Functional Correctness**: Test happy paths and edge cases
3. **Code Quality**: Clean, readable, maintainable code
4. **Security**: Vulnerability scanning and security best practices
5. **Performance**: Profiling, optimization, load testing
6. **Accessibility & UX**: Usability and accessibility compliance
7. **Testing Coverage**: Adequate test coverage and quality
8. **Breaking Changes**: Backward compatibility verification
9. **Deployment Readiness**: Configuration, observability, CI/CD
10. **Documentation**: Complete and clear documentation

**Examples:**
```bash
# Basic validation
/double-check

# Deep analysis with extended checks
/double-check --deep

# Security-focused validation
/double-check --security

# Performance-focused validation
/double-check --performance
```

**Automated Checks:**
- Linting and formatting
- All tests with coverage
- Security scanning (npm audit, pip-audit, semgrep, gitleaks)
- Build verification
- Type checking
- Accessibility testing

### /lint-plugins

Validate Claude Code plugin syntax, structure, and cross-references. Prevents runtime failures by catching issues before deployment.

**Usage:**
```bash
/lint-plugins [--fix] [--plugin=name] [--report] [--analyze-deps]
```

**Validation Rules:**
- ✅ Agent reference format (`plugin:agent` vs `plugin::agent`)
- ✅ Skill reference format
- ✅ File existence for all referenced agents/skills
- ✅ plugin.json structure and metadata
- ✅ Cross-plugin dependencies
- ✅ Circular dependency detection
- ✅ Unused agent/skill identification

**Examples:**
```bash
# Basic validation
/lint-plugins

# Auto-fix syntax errors
/lint-plugins --fix

# Validate specific plugin
/lint-plugins --plugin=backend-development

# Generate detailed report
/lint-plugins --report

# Analyze cross-plugin dependencies
/lint-plugins --analyze-deps
```

**Auto-Fixable Issues:**
- ✅ Double colons (`::` → `:`)
- ✅ Whitespace in references
- ❌ Missing namespaces (requires manual mapping)
- ❌ Non-existent agents (requires creating or fixing)

## Skills

### comprehensive-validation-framework

Framework for comprehensive multi-dimensional validation and verification:

**Validation Layers:**
1. Requirements verification
2. Functional correctness testing
3. Code quality assessment
4. Security vulnerability scanning
5. Performance profiling
6. User experience validation
7. Test coverage analysis
8. Compatibility verification
9. Operations readiness
10. Documentation completeness

**Automated Tools Integration:**
- Linters: eslint, ruff, clippy
- Test Runners: pytest, jest, cargo test
- Security: semgrep, bandit, gitleaks, npm audit
- Performance: profilers, load testers
- Accessibility: pa11y, lighthouse

### plugin-syntax-validator

Plugin syntax and structure validation utilities for Claude Code plugins:

**Validation Capabilities:**
- Agent reference syntax validation
- Skill reference syntax validation
- File existence verification
- plugin.json schema validation
- Cross-plugin dependency analysis
- Circular dependency detection
- Unused component identification
- Dependency graph visualization

**Integration:**
- Pre-commit hooks
- CI/CD pipelines (GitHub Actions, GitLab CI)
- Command-line validation scripts

## Use Cases

### Pre-Commit Quality Gates
Use /double-check before commits to ensure code meets quality standards across all dimensions.

### Plugin Development
Use /lint-plugins during plugin development to catch syntax errors and structural issues before deployment.

### Pull Request Reviews
Use /double-check with --deep flag for comprehensive validation before merging pull requests.

### Security Audits
Use /double-check --security for focused security vulnerability scanning and best practices verification.

### Performance Optimization
Use /double-check --performance to identify bottlenecks and optimization opportunities.

### CI/CD Integration
Integrate both commands into CI/CD pipelines for automated quality assurance.

## Installation

### From GitHub Marketplace

```bash
/plugin marketplace add <your-username>/scientific-computing-workflows
/plugin install quality-engineering
```

### Local Installation

```bash
/plugin add ./plugins/quality-engineering
```

## Requirements

- Claude Code
- Optional: Security scanning tools (semgrep, gitleaks, npm audit, etc.)
- Optional: Testing frameworks (pytest, jest, etc.)
- Optional: Performance profiling tools

## Best Practices

1. **Use Early and Often**: Run validation checks frequently, not just before release
2. **Start with Auto-Fix**: Use --fix flag to automatically correct common issues
3. **Integrate with CI/CD**: Add validation to automated pipelines
4. **Customize Depth**: Adjust validation depth based on importance (--deep for critical changes)
5. **Review Auto-Fixes**: Always review automatically fixed issues before committing
6. **Document Exceptions**: Document why certain validation failures are acceptable
7. **Trend Analysis**: Track validation metrics over time to identify quality trends

## Advanced Features

### Multi-Agent Orchestration
Both commands orchestrate specialized agents:
- code-quality agent
- code-reviewer agent
- security-auditor agent
- performance-engineer agent
- test-automator agent

### Validation Reports
Generate comprehensive validation reports with:
- Summary of all checks performed
- Issues found with severity levels
- Actionable recommendations
- Evidence and verification data

### Dependency Analysis
Analyze cross-plugin dependencies:
- Build dependency graphs
- Detect circular dependencies
- Identify unused components
- Generate visualizations (DOT/PNG)

## Integration Examples

### Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "🔍 Running quality checks..."

/double-check

if [ $? -ne 0 ]; then
    echo "❌ Quality check failed"
    exit 1
fi

/lint-plugins

if [ $? -ne 0 ]; then
    echo "❌ Plugin validation failed"
    exit 1
fi

echo "✅ Quality checks passed"
```

### GitHub Actions

```yaml
name: Quality Engineering

on: [push, pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run double-check
        run: /double-check --deep

      - name: Validate plugins
        run: /lint-plugins --report
```

## Output Formats

### Summary Mode
```
✅ Validation complete
├─ Dimensions checked: 10
├─ Tests passed: 247/250
├─ Coverage: 87%
├─ Security issues: 0
└─ Status: READY FOR DEPLOYMENT
```

### Detailed Report Mode
```
📊 Quality Engineering Report

Overall Assessment: ⚠️ Needs work (7.5/10)

Critical Issues (Must Fix):
  [SECURITY] Dependency vulnerability in package X
  [TESTING] Coverage below threshold (65% vs 80%)

Important Issues (Should Fix):
  [PERFORMANCE] N+1 query in UserService
  [CODE_QUALITY] High cyclomatic complexity in processData()

Recommendations:
  1. Update dependency X to version Y
  2. Add tests for edge cases in feature Z
  3. Refactor processData() to reduce complexity
```

## License

MIT

## Authors

Wei Chen & Seth Hobson
