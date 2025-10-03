# Common Workflows

Real-world workflows and patterns for typical development tasks using the Claude Code Command Executor.

## Table of Contents

- [Code Quality Improvement](#code-quality-improvement)
- [Performance Optimization](#performance-optimization)
- [Codebase Cleanup](#codebase-cleanup)
- [Testing Pipeline](#testing-pipeline)
- [Documentation Generation](#documentation-generation)
- [Scientific Computing](#scientific-computing)
- [Web Development](#web-development)
- [CI/CD Integration](#cicd-integration)
- [Legacy Code Modernization](#legacy-code-modernization)
- [Research Project](#research-project)

## Code Quality Improvement

**Goal:** Improve code quality, fix issues, and enforce standards.

### Workflow

```bash
# 1. Analyze current quality
/check-code-quality --dry-run --report

# 2. Fix automatic issues
/check-code-quality --auto-fix

# 3. Clean unused code
/clean-codebase --imports --dead-code --backup

# 4. Refactor to modern patterns
/refactor-clean --patterns=modern --implement

# 5. Verify with tests
/run-all-tests --coverage

# 6. Update documentation
/update-docs --type=all

# 7. Commit changes
/commit --template=refactor --ai-message --validate
```

### Expected Outcomes

- Code quality score improved by 20-40%
- All auto-fixable issues resolved
- Unused code removed
- Modern patterns applied
- Tests passing with good coverage
- Documentation updated

## Performance Optimization

**Goal:** Improve code performance and identify bottlenecks.

### Workflow

```bash
# 1. Profile current performance
/optimize --profile --detailed src/

# 2. Preview optimizations
/optimize --implement --dry-run src/

# 3. Apply high-priority optimizations
/optimize --implement --category=algorithm src/

# 4. Benchmark results
/run-all-tests --benchmark --profile

# 5. Apply memory optimizations
/optimize --implement --category=memory src/

# 6. Final validation
/run-all-tests --benchmark --coverage

# 7. Document improvements
/update-docs --type=api

# 8. Commit with metrics
/commit --template=optimization --ai-message
```

### Expected Outcomes

- 2-10x performance improvement
- Memory usage reduced
- Algorithmic complexity improved
- Benchmarks showing gains
- Performance tests passing

## Codebase Cleanup

**Goal:** Remove unused code, imports, and duplicates.

### Workflow

```bash
# 1. Analyze cleanup opportunities
/clean-codebase --analysis=ultrathink --dry-run --report

# 2. Remove unused imports (safest)
/clean-codebase --imports --backup

# 3. Verify no breakage
/run-all-tests

# 4. Remove dead code
/clean-codebase --dead-code --ast-deep --interactive

# 5. Find and remove duplicates
/clean-codebase --duplicates --interactive

# 6. Final verification
/run-all-tests --coverage

# 7. Check quality improved
/check-code-quality

# 8. Commit cleanup
/commit --template=cleanup --ai-message
```

### Expected Outcomes

- 10-30% reduction in codebase size
- All unused imports removed
- Dead code eliminated
- Duplicates consolidated
- Tests still passing
- Improved maintainability

## Testing Pipeline

**Goal:** Achieve comprehensive test coverage.

### Workflow

```bash
# 1. Check current coverage
/run-all-tests --coverage --report

# 2. Generate missing unit tests
/generate-tests src/ --type=unit --coverage=90

# 3. Generate integration tests
/generate-tests src/ --type=integration

# 4. Generate performance tests
/generate-tests src/ --type=performance

# 5. Run all tests
/run-all-tests --coverage --auto-fix

# 6. Fix any failures
/debug --auto-fix

# 7. Achieve target coverage
/generate-tests src/ --coverage=95

# 8. Set up CI/CD
/ci-setup --platform=github --type=basic

# 9. Commit test suite
/commit --template=testing --ai-message
```

### Expected Outcomes

- 90%+ test coverage
- Unit, integration, and performance tests
- All tests passing
- CI/CD configured
- Automated test execution

## Documentation Generation

**Goal:** Create comprehensive, up-to-date documentation.

### Workflow

```bash
# 1. Generate README
/update-docs --type=readme

# 2. Generate API documentation
/update-docs --type=api --format=markdown

# 3. Generate research documentation (if applicable)
/update-docs --type=research --format=latex

# 4. Create architecture docs
/explain-code --level=expert --docs src/

# 5. Review and edit
# Manually review generated docs

# 6. Build documentation site (if using sphinx/mkdocs)
cd docs && make html

# 7. Commit documentation
/commit --template=docs --ai-message
```

### Expected Outcomes

- Complete README
- API documentation
- Architecture guides
- Research documentation (if applicable)
- Examples and tutorials
- Published documentation site

## Scientific Computing

**Goal:** Optimize scientific code for performance and accuracy.

### Workflow

```bash
# 1. Analyze scientific code
/optimize --agents=scientific --language=python --profile

# 2. Check for numerical issues
/debug --issue=numerical --scientific

# 3. Optimize algorithms
/optimize --agents=scientific --category=algorithm --implement

# 4. Optimize memory usage
/optimize --category=memory --implement

# 5. Generate scientific tests
/generate-tests src/ --type=scientific --framework=pytest

# 6. Run tests with profiling
/run-all-tests --profile --scientific

# 7. Generate research docs
/update-docs --type=research --format=latex

# 8. Validate numerical accuracy
/double-check "numerical accuracy" --deep-analysis

# 9. Commit improvements
/commit --template=scientific --ai-message
```

### Expected Outcomes

- Optimized numerical algorithms
- GPU acceleration where applicable
- Comprehensive scientific tests
- Research-quality documentation
- Validated numerical accuracy
- Publication-ready code

## Web Development

**Goal:** Build and optimize web applications.

### Workflow

```bash
# 1. Check code quality (backend and frontend)
/check-code-quality --agents=engineering

# 2. Optimize backend performance
/optimize --agents=engineering backend/

# 3. Optimize frontend performance
/optimize --agents=engineering frontend/

# 4. Generate API tests
/generate-tests backend/api/ --type=integration

# 5. Generate frontend tests
/generate-tests frontend/ --type=unit --framework=jest

# 6. Set up CI/CD
/ci-setup --platform=github --type=enterprise --deploy=production

# 7. Run all tests
/run-all-tests --coverage

# 8. Generate API documentation
/update-docs --type=api --format=markdown

# 9. Commit changes
/commit --template=feature --ai-message --validate
```

### Expected Outcomes

- Optimized backend and frontend
- Comprehensive test coverage
- API documentation
- CI/CD pipeline configured
- Production deployment ready

## CI/CD Integration

**Goal:** Automate testing and deployment.

### Workflow

```bash
# 1. Set up basic CI/CD
/ci-setup --platform=github --type=basic

# 2. Add security scanning
/ci-setup --platform=github --security

# 3. Add monitoring
/ci-setup --platform=github --monitoring

# 4. Configure deployment
/ci-setup --platform=github --deploy=staging

# 5. Test CI/CD locally
/run-all-tests --ci-mode

# 6. Commit CI/CD configuration
/commit --template=ci --ai-message

# 7. Monitor first run
# Check GitHub Actions dashboard

# 8. Fix any issues
/fix-commit-errors --auto-fix
```

### Expected Outcomes

- Automated testing on every push
- Security scanning enabled
- Deployment automation
- Monitoring configured
- CI/CD pipeline working

## Legacy Code Modernization

**Goal:** Modernize legacy codebase.

### Workflow

```bash
# 1. Analyze legacy code
/adopt-code --analyze legacy_code/

# 2. Plan modernization
/think-ultra "modernization strategy" --depth=ultra --agents=all

# 3. Modernize to modern framework
/adopt-code --integrate --target=python legacy_code/

# 4. Refactor to modern patterns
/refactor-clean --patterns=modern --implement

# 5. Optimize performance
/optimize --implement --language=auto

# 6. Generate tests for new code
/generate-tests src/ --type=all --coverage=90

# 7. Clean up old patterns
/clean-codebase --dead-code --duplicates

# 8. Update documentation
/update-docs --type=all

# 9. Verify everything works
/run-all-tests --coverage

# 10. Commit modernization
/commit --template=refactor --ai-message
```

### Expected Outcomes

- Modern framework integration
- Legacy patterns eliminated
- Comprehensive test coverage
- Updated documentation
- Improved performance
- Maintainable codebase

## Research Project

**Goal:** Organize and optimize research code.

### Workflow

```bash
# 1. Analyze research code
/optimize --agents=research --profile

# 2. Clean up experimental code
/clean-codebase --dead-code --interactive

# 3. Optimize computations
/optimize --agents=scientific --category=all --implement

# 4. Generate research tests
/generate-tests src/ --type=scientific --reproducible

# 5. Run tests with profiling
/run-all-tests --scientific --profile --reproducible

# 6. Generate research documentation
/update-docs --type=research --format=latex

# 7. Validate reproducibility
/double-check "reproducibility" --deep-analysis

# 8. Commit research code
/commit --template=research --ai-message

# 9. Prepare for publication
/update-docs --type=research --format=latex --publish
```

### Expected Outcomes

- Optimized research code
- Reproducible results
- Research-quality tests
- LaTeX documentation
- Publication-ready codebase
- Validated numerical accuracy

## Workflow Combinations

### Quick Daily Workflow

```bash
# Morning code health check
/check-code-quality --agents=core --quick && \
/run-all-tests --quick
```

### Pre-Commit Workflow

```bash
# Before committing code
/check-code-quality --auto-fix && \
/clean-codebase --imports && \
/run-all-tests && \
/commit --ai-message --validate
```

### Weekly Maintenance Workflow

```bash
# Weekly codebase maintenance
/check-code-quality --detailed --report && \
/clean-codebase --analysis=comprehensive --dry-run && \
/optimize --profile --report && \
/update-docs --type=all
```

### Pre-Release Workflow

```bash
# Before releasing new version
/check-code-quality --agents=all --orchestrate && \
/optimize --agents=all --implement && \
/generate-tests --coverage=95 && \
/run-all-tests --coverage --benchmark && \
/update-docs --type=all && \
/commit --template=release --ai-message --validate
```

## Workflow Tips

### Tip 1: Always Start with Dry-Run

```bash
# Preview before executing
command --dry-run
# Review output
command  # Execute
```

### Tip 2: Use Agents Appropriately

```bash
# Quick checks
--agents=core

# Domain-specific
--agents=scientific  # or engineering, ai, etc.

# Comprehensive
--agents=all --orchestrate
```

### Tip 3: Chain Commands with &&

```bash
# Only continue if previous succeeds
/check-code-quality --auto-fix && \
/run-all-tests && \
/commit --ai-message
```

### Tip 4: Use Reports for Analysis

```bash
# Generate detailed reports
command --report --format=json > report.json
# Analyze programmatically
```

### Tip 5: Enable Safety Features

```bash
# Maximum safety
command --backup --rollback --interactive --dry-run
```

## Next Steps

- **[Tutorials](../tutorials/)** - Step-by-step guides
- **[Command Reference](../guides/command-reference.md)** - Complete command docs
- **[Examples](../examples/)** - Real-world examples
- **[Best Practices](../best-practices/)** - Recommended patterns

---

**Ready for more?** → [Tutorials](../tutorials/)