# Tutorial 04: Workflow Automation

**Duration**: 45 minutes | **Level**: Intermediate | **Prerequisites**: Tutorials 01-03

---

## Learning Objectives

- Master pre-built workflow templates
- Create custom YAML workflows
- Implement conditional and parallel execution
- Build multi-stage pipelines
- Automate common development tasks

---

## Overview

Workflows automate sequences of commands with dependencies, conditions, and error handling. This tutorial teaches workflow creation and automation.

**What You'll Build**:
- Quality improvement pipeline
- Automated deployment workflow
- Multi-environment testing workflow
- Error recovery system

---

## Part 1: Using Pre-Built Workflows (10 minutes)

### Step 1: Discover Available Workflows

```bash
# List workflow templates
ls /workflows/templates/

# Output:
# quality-improvement.yaml
# performance-optimization.yaml
# deployment-pipeline.yaml
# test-and-deploy.yaml
# refactoring-workflow.yaml
# documentation-generation.yaml
# security-scan.yaml
# code-review.yaml
# ci-cd-full.yaml
```

### Step 2: Run Quality Improvement Workflow

```bash
# Execute pre-built workflow
/workflow run quality-improvement

# Workflow executes automatically:
# Step 1/5: Checking code quality... ✅ (Quality score: 78)
# Step 2/5: Running tests... ✅ (96% passing)
# Step 3/5: Optimizing performance... ✅ (3.2x faster)
# Step 4/5: Updating documentation... ✅ (Docs generated)
# Step 5/5: Committing changes... ✅ (Committed)
#
# ✅ Workflow completed successfully in 4m 32s
```

### Step 3: Understand Workflow Structure

```bash
# View workflow definition
cat /workflows/templates/quality-improvement.yaml
```

**Workflow Structure**:
```yaml
workflow:
  name: quality-improvement
  description: Complete quality improvement pipeline

steps:
  - id: check-quality
    command: /check-code-quality
    flags: [--auto-fix]

  - id: run-tests
    command: /run-all-tests
    flags: [--auto-fix, --coverage]
    depends_on: [check-quality]

  - id: optimize
    command: /optimize
    flags: [--implement]
    depends_on: [run-tests]

  - id: update-docs
    command: /update-docs
    depends_on: [optimize]

  - id: commit
    command: /commit
    flags: [--ai-message, --validate]
    depends_on: [update-docs]
```

---

## Part 2: Creating Custom Workflows (15 minutes)

### Step 4: Create Deployment Workflow

**Create** `my-deployment.yaml`:
```yaml
workflow:
  name: deploy-to-production
  description: Safe production deployment workflow

steps:
  # Stage 1: Pre-deployment validation
  - id: quality-gate
    command: /check-code-quality
    flags: [--validate]
    failure: abort  # Stop if quality check fails

  - id: test-suite
    command: /run-all-tests
    flags: [--coverage, --report]
    depends_on: [quality-gate]
    failure: abort

  - id: security-scan
    command: /check-code-quality
    flags: [--security]
    depends_on: [quality-gate]
    failure: abort

  # Stage 2: Staging deployment
  - id: deploy-staging
    command: /ci-setup
    flags: [--platform=github, --deploy=staging]
    depends_on: [test-suite, security-scan]

  - id: test-staging
    command: /run-all-tests
    flags: [--environment=staging, --integration]
    depends_on: [deploy-staging]
    failure: rollback

  # Stage 3: Production deployment
  - id: deploy-production
    command: /ci-setup
    flags: [--platform=github, --deploy=production]
    depends_on: [test-staging]
    conditional: "staging_tests_passed == true"

  - id: smoke-test
    command: /run-all-tests
    flags: [--environment=production, --smoke]
    depends_on: [deploy-production]
    failure: rollback

on_failure:
  - notify: ["slack", "email"]
  - create_issue: true
  - rollback: true
```

### Step 5: Execute Custom Workflow

```bash
# Run with dry-run first
/workflow run my-deployment.yaml --dry-run

# Review the execution plan, then run for real
/workflow run my-deployment.yaml

# Output shows parallel execution:
# ⚡ Executing in parallel:
#   - [quality-gate] Checking quality...
#
# ✅ quality-gate completed (23s)
#
# ⚡ Executing in parallel:
#   - [test-suite] Running tests...
#   - [security-scan] Scanning security...
#
# ✅ test-suite completed (45s)
# ✅ security-scan completed (31s)
#
# → [deploy-staging] Deploying to staging...
# ✅ deploy-staging completed (1m 12s)
```

---

## Part 3: Advanced Workflow Features (10 minutes)

### Step 6: Conditional Execution

```yaml
# Workflow with conditions
steps:
  - id: check-branch
    command: /bash
    script: "git branch --show-current"
    output: current_branch

  - id: deploy-prod
    command: /ci-setup
    conditional: "current_branch == 'main'"

  - id: deploy-staging
    command: /ci-setup
    conditional: "current_branch != 'main'"
```

### Step 7: Parallel Execution

```yaml
# Run multiple steps in parallel
steps:
  - id: parallel-tests
    parallel:
      - command: /run-all-tests
        flags: [--scope=unit]
      - command: /run-all-tests
        flags: [--scope=integration]
      - command: /check-code-quality
        flags: [--analysis=security]
```

### Step 8: Error Handling and Recovery

```yaml
steps:
  - id: risky-operation
    command: /optimize
    flags: [--implement, --aggressive]
    on_error:
      - rollback: true
      - retry:
          max_attempts: 3
          backoff: exponential
      - fallback:
          command: /optimize
          flags: [--implement, --safe]
```

---

## Part 4: Real-World Workflows (10 minutes)

### Step 9: Multi-Environment Testing

```yaml
workflow:
  name: multi-environment-test

environments:
  - name: unit
    commands:
      - /run-all-tests --scope=unit

  - name: integration
    commands:
      - /run-all-tests --scope=integration
    depends_on: [unit]

  - name: e2e
    commands:
      - /run-all-tests --scope=e2e
    depends_on: [integration]

  - name: performance
    commands:
      - /run-all-tests --benchmark --validate
    depends_on: [e2e]
```

### Step 10: CI/CD Integration

```bash
# Generate GitHub Actions workflow
/ci-setup --platform=github --workflows=all

# Creates .github/workflows/main.yml:
```

**Generated**:
```yaml
name: CI/CD Pipeline
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Quality Check
        run: /check-code-quality --auto-fix

  test:
    needs: quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: /run-all-tests --coverage

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy
        run: /ci-setup --deploy=production
```

---

## Practice Projects

### Project 1: Refactoring Workflow (15 min)
Create a workflow that:
1. Analyzes code quality
2. Refactors code automatically
3. Generates tests for refactored code
4. Validates changes
5. Creates PR

### Project 2: Documentation Pipeline (10 min)
Build workflow to:
1. Generate API docs
2. Create README
3. Build tutorials
4. Publish to GitHub Pages

### Project 3: Release Workflow (20 min)
Automate release process:
1. Version bump
2. Changelog generation
3. Tag creation
4. Build artifacts
5. Deploy to registries

---

## Summary

**Completed**: ✅ Workflow automation mastery
**Time**: 45 minutes
**Next**: [Tutorial 05: Plugin Development →](tutorial-05-plugin-development.md)