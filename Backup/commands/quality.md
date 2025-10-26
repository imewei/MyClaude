---
description: Comprehensive code quality analysis - audit, optimize, refactor, and review
allowed-tools: Bash(find:*), Bash(grep:*), Bash(git:*), Bash(npm:*), Bash(pylint:*), Bash(cargo:*)
argument-hint: <target-path> [--audit] [--optimize] [--refactor]
color: green
agents:
  primary:
    - code-quality
  conditional:
    - agent: devops-security-engineer
      trigger: flag "--audit" OR pattern "security|vulnerability"
    - agent: systems-architect
      trigger: flag "--optimize" OR complexity > 10
    - agent: hpc-numerical-coordinator
      trigger: pattern "numpy|scipy|pandas|matplotlib|scientific.*computing"
  orchestrated: true
---

# Code Quality Suite

## Context Discovery
- Target: @$ARGUMENTS
- Git status: !`git status --short 2>/dev/null | head -5`
- File type: !`file $ARGUMENTS 2>/dev/null || echo "Directory or pattern"`
- Size: !`du -h $ARGUMENTS 2>/dev/null | tail -1 || echo "Multiple files"`
- Recent changes: !`git log --oneline $ARGUMENTS -5 2>/dev/null || echo "Not in git"`

## Your Task

Execute comprehensive code quality analysis covering:

### 1. Security Audit
**Focus**:
- Dependency vulnerabilities
- Authentication/authorization issues
- Input validation gaps
- Sensitive data exposure
- Security misconfigurations

**Quick Checks**:
```bash
# Dependencies
npm audit || pip-audit || cargo audit

# Secrets
grep -r "API_KEY\|SECRET\|PASSWORD" --exclude-dir=node_modules

# Common vulnerabilities
grep -r "eval(\|exec(\|__import__" --include="*.py"
grep -r "dangerouslySetInnerHTML\|eval(" --include="*.{js,jsx,ts,tsx}"
```

### 2. Performance Optimization
**Analyze**:
- Algorithm complexity (target: O(n log n) or better)
- Memory usage patterns
- I/O operations efficiency
- Caching opportunities
- Lazy loading potential

**Profile First**: Use appropriate profiler (cProfile, Chrome DevTools, cargo flamegraph)

### 3. Code Refactoring
**Targets**:
- Functions >50 lines → Extract smaller units
- Cyclomatic complexity >10 → Simplify logic
- Code duplication >5% → DRY principle
- Magic numbers → Named constants
- Deep nesting → Early returns, guard clauses

### 4. Code Review
**Check**:
- Readability & maintainability
- Best practices adherence
- Test coverage (target: >80%)
- Documentation completeness
- Error handling robustness

## Execution Strategy

**Mode 1: Quick Audit** (5 minutes)
```bash
# Security
npm audit --audit-level=high
grep -r "TODO\|FIXME\|HACK\|XXX" $ARGUMENTS

# Quality metrics
find $ARGUMENTS -name "*.py" -exec pylint {} \; 2>/dev/null | grep "rated at"
find $ARGUMENTS -name "*.js" -exec eslint {} \; 2>/dev/null | tail -5
```

**Mode 2: Deep Analysis** (30 minutes)
- Profile hotspots → optimization targets
- Calculate complexity metrics → refactoring candidates
- Security scan → vulnerability remediation
- Test coverage → gap identification

**Mode 3: Full Review** (2 hours)
- Line-by-line code review
- Architecture assessment
- Documentation review
- Comprehensive test suite evaluation

## Provide actionable recommendations prioritized by: Impact (High/Med/Low) × Effort (Hours)
