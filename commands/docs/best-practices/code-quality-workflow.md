# Code Quality Workflow Best Practices

Comprehensive best practices for maintaining high code quality using the Claude Code Command Executor.

## Overview

Code quality is foundational to maintainable, performant, and secure software. This guide provides battle-tested workflows for achieving and maintaining excellent code quality.

## Quick Reference

### Daily Quality Workflow
```bash
# Morning check (30 seconds)
/check-code-quality --agents=core --quick
```

### Pre-Commit Quality Workflow
```bash
# Before committing (2 minutes)
/check-code-quality --auto-fix && \
/clean-codebase --imports && \
/run-all-tests && \
/commit --ai-message --validate
```

### Weekly Quality Workflow
```bash
# Deep dive (10 minutes)
/check-code-quality --agents=all --orchestrate --detailed --report
```

## Core Principles

### 1. Quality First, Always

**Principle:** Never compromise on code quality for speed.

**Why:** Technical debt compounds exponentially.

**Practice:**
```bash
# Always check quality before new feature
/check-code-quality --auto-fix
# Then proceed with development
```

### 2. Continuous Improvement

**Principle:** Quality improves incrementally, not all at once.

**Why:** Small, consistent improvements are sustainable.

**Practice:**
```bash
# Fix one category at a time
/check-code-quality --category=imports --auto-fix
/check-code-quality --category=complexity --auto-fix
/check-code-quality --category=documentation --auto-fix
```

### 3. Measure Everything

**Principle:** You can't improve what you don't measure.

**Why:** Metrics provide objective quality assessment.

**Practice:**
```bash
# Regular quality reports
/check-code-quality --report --format=json > quality-$(date +%Y%m%d).json
```

### 4. Automate All Checks

**Principle:** Manual quality checks are unreliable.

**Why:** Humans forget; automation doesn't.

**Practice:**
```bash
# Setup CI/CD for automatic checks
/ci-setup --platform=github --type=enterprise
```

## Comprehensive Quality Workflow

### Phase 1: Assessment (5 minutes)

```bash
# Step 1: Initial quality check
/check-code-quality --dry-run --report

# Step 2: Identify priority areas
# Review report and note HIGH priority issues

# Step 3: Estimate effort
# Use report to plan improvement phases
```

**Success Criteria:**
- Quality score calculated
- Issues categorized by severity
- Improvement plan created

### Phase 2: Quick Wins (10 minutes)

```bash
# Step 1: Fix automatic issues
/check-code-quality --auto-fix

# Step 2: Clean unused code
/clean-codebase --imports --backup

# Step 3: Verify no breakage
/run-all-tests

# Step 4: Commit improvements
/commit --template=quality --ai-message
```

**Expected Improvements:**
- +5 to +15 quality score points
- Unused code removed
- Formatting standardized
- Basic issues resolved

### Phase 3: Complexity Reduction (20 minutes)

```bash
# Step 1: Identify complex code
/check-code-quality --detailed --report | grep "complexity"

# Step 2: Refactor to simpler patterns
/refactor-clean --patterns=modern --interactive

# Step 3: Validate with tests
/run-all-tests --coverage

# Step 4: Measure improvement
/check-code-quality --report
```

**Expected Improvements:**
- Reduced cyclomatic complexity
- Better code organization
- Improved readability

### Phase 4: Documentation (15 minutes)

```bash
# Step 1: Add missing docstrings
/check-code-quality --category=documentation --auto-fix

# Step 2: Generate API docs
/update-docs --type=api

# Step 3: Add examples
/explain-code --level=basic --docs src/

# Step 4: Generate README
/update-docs --type=readme
```

**Expected Improvements:**
- 100% docstring coverage
- Complete API documentation
- Usage examples
- Updated README

### Phase 5: Testing (20 minutes)

```bash
# Step 1: Check current coverage
/run-all-tests --coverage --report

# Step 2: Generate missing tests
/generate-tests src/ --coverage=90

# Step 3: Run full test suite
/run-all-tests --coverage --auto-fix

# Step 4: Validate quality
/check-code-quality
```

**Expected Improvements:**
- 90%+ test coverage
- All tests passing
- Quality score boost from testing

### Phase 6: Performance (30 minutes)

```bash
# Step 1: Profile performance
/optimize --profile src/

# Step 2: Apply optimizations
/optimize --implement --category=all src/

# Step 3: Benchmark improvements
/run-all-tests --benchmark --profile

# Step 4: Validate quality maintained
/check-code-quality
```

**Expected Improvements:**
- Performance bottlenecks resolved
- Optimized algorithms
- Better resource usage
- Quality maintained or improved

### Phase 7: Finalization (10 minutes)

```bash
# Step 1: Final comprehensive check
/check-code-quality --agents=all --orchestrate --detailed

# Step 2: Setup CI/CD
/ci-setup --platform=github --type=enterprise

# Step 3: Generate final report
/check-code-quality --report --format=html > quality-report.html

# Step 4: Commit everything
/commit --template=quality --ai-message --validate
```

**Success Criteria:**
- Quality score 85+ (excellent)
- 90%+ test coverage
- Complete documentation
- CI/CD configured
- All issues resolved

## Quality Metrics

### Target Metrics

| Metric | Good | Excellent | Target |
|--------|------|-----------|--------|
| Quality Score | 75-84 | 85-100 | 90+ |
| Test Coverage | 80-89% | 90-100% | 95%+ |
| Documentation | 80-89% | 90-100% | 100% |
| Complexity | < 15 | < 10 | < 10 |
| Duplication | < 5% | < 2% | < 2% |
| Technical Debt | Medium | Low | Low |

### Tracking Progress

```bash
# Weekly quality snapshot
cat > track-quality.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
/check-code-quality --report --format=json > "quality-${DATE}.json"
echo "Quality tracked: quality-${DATE}.json"
EOF

chmod +x track-quality.sh
./track-quality.sh
```

## Common Issues and Solutions

### Issue 1: Low Quality Score

**Problem:** Quality score below 70

**Diagnosis:**
```bash
/check-code-quality --detailed --report
```

**Solution:**
```bash
# Phase 1: Auto-fix
/check-code-quality --auto-fix

# Phase 2: Clean code
/clean-codebase --imports --dead-code

# Phase 3: Refactor
/refactor-clean --patterns=modern --implement

# Phase 4: Document
/update-docs --type=all

# Phase 5: Test
/generate-tests --coverage=90
/run-all-tests
```

### Issue 2: High Complexity

**Problem:** Cyclomatic complexity > 15

**Diagnosis:**
```bash
/check-code-quality --detailed | grep "complexity"
```

**Solution:**
```bash
# Extract functions
/refactor-clean --patterns=extract-method --interactive

# Simplify conditionals
/refactor-clean --patterns=simplify-conditionals --interactive

# Use early returns
/refactor-clean --patterns=early-return --implement
```

### Issue 3: Low Test Coverage

**Problem:** Coverage below 80%

**Diagnosis:**
```bash
/run-all-tests --coverage --report
```

**Solution:**
```bash
# Generate missing tests
/generate-tests src/ --coverage=90

# Focus on untested modules
/generate-tests src/untested_module.py --type=all

# Run and validate
/run-all-tests --coverage --auto-fix
```

### Issue 4: Outdated Documentation

**Problem:** Docs don't match code

**Diagnosis:**
```bash
/update-docs --dry-run --report
```

**Solution:**
```bash
# Regenerate all docs
/update-docs --type=all

# Add usage examples
/explain-code --level=basic --docs src/

# Validate with code review
git diff docs/
```

## Team Practices

### For Individual Developers

```bash
# Before starting work
/check-code-quality --quick

# Before committing
/check-code-quality --auto-fix && \
/run-all-tests && \
/commit --ai-message

# Before push
/run-all-tests --coverage
```

### For Team Leads

```bash
# Weekly quality review
/check-code-quality --agents=all --orchestrate --report

# Monthly deep dive
/multi-agent-optimize --mode=review --agents=all

# Setup automation
/ci-setup --platform=github --type=enterprise --monitoring
```

### For Code Reviewers

```bash
# Before reviewing PR
/check-code-quality pr-branch/

# Check test coverage
/run-all-tests --coverage --report

# Validate docs updated
/update-docs --dry-run
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/quality.yml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Quality Check
        run: /check-code-quality --format=json --report

      - name: Test Coverage
        run: /run-all-tests --coverage

      - name: Quality Gate
        run: |
          SCORE=$(jq '.quality_score' quality-report.json)
          if [ $SCORE -lt 80 ]; then
            echo "Quality score $SCORE below threshold 80"
            exit 1
          fi
```

### Quality Gates

```bash
# Enforce minimum quality
cat > quality-gate.sh << 'EOF'
#!/bin/bash
/check-code-quality --format=json > report.json
SCORE=$(jq '.quality_score' report.json)

if [ $SCORE -lt 85 ]; then
    echo "❌ Quality gate failed: Score $SCORE < 85"
    exit 1
fi
echo "✅ Quality gate passed: Score $SCORE"
EOF
```

## Quick Fixes

### Remove Unused Imports
```bash
/clean-codebase --imports
```

### Fix Formatting
```bash
/check-code-quality --auto-fix --category=formatting
```

### Add Type Hints
```bash
/refactor-clean --patterns=add-types --implement
```

### Add Docstrings
```bash
/update-docs --type=api --auto-add-missing
```

### Reduce Complexity
```bash
/refactor-clean --patterns=reduce-complexity --interactive
```

## Monitoring Quality

### Dashboard Metrics

Track these metrics weekly:
1. Quality score trend
2. Test coverage trend
3. Issue count by severity
4. Technical debt ratio
5. Documentation coverage

### Alerting

```bash
# Alert if quality drops
cat > quality-alert.sh << 'EOF'
#!/bin/bash
/check-code-quality --format=json > current.json
SCORE=$(jq '.quality_score' current.json)
PREV_SCORE=$(jq '.quality_score' previous.json 2>/dev/null || echo "0")

if [ $SCORE -lt $((PREV_SCORE - 5)) ]; then
    echo "⚠️  Quality dropped by $((PREV_SCORE - SCORE)) points!"
    # Send notification
fi

mv current.json previous.json
EOF
```

## Summary

Quality workflow essentials:

1. **Check daily** - Quick morning check
2. **Fix immediately** - Don't accumulate debt
3. **Test everything** - 90%+ coverage
4. **Document thoroughly** - Future you will thank you
5. **Automate checks** - CI/CD enforcement
6. **Measure progress** - Track metrics
7. **Improve continuously** - Small, steady gains

## See Also

- **[Common Workflows](../getting-started/common-workflows.md)** - Standard patterns
- **[Tutorial 01](../tutorials/tutorial-01-code-quality.md)** - Hands-on quality tutorial
- **[Command Reference](../guides/command-reference.md)** - All commands

---

**Ready to improve quality?** → [Tutorial 01: Code Quality](../tutorials/tutorial-01-code-quality.md)