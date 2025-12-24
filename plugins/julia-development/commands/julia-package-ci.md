---
version: "1.0.6"
category: julia-development
command: /julia-package-ci
description: Generate GitHub Actions CI/CD workflows for Julia packages
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: ""
color: blue
execution_modes:
  quick: "5-10 minutes"
  standard: "15-20 minutes"
  comprehensive: "25-35 minutes"
agents:
  primary:
    - julia-developer
  conditional:
    - agent: julia-pro
      trigger: pattern "performance|benchmark|optimization"
  orchestrated: false
---

# Generate CI/CD Workflows for Julia Packages

Generate production-ready GitHub Actions workflows with test matrices, coverage, and automation.

---

## Mode Selection

| Mode | Duration | Output |
|------|----------|--------|
| Quick | 5-10 min | CI.yml (Linux + latest), coverage, badge |
| Standard (default) | 15-20 min | + cross-platform, version matrix, docs, automation |
| Comprehensive | 25-35 min | + quality checks, formatting, security, benchmarks |

---

## External Documentation

| Topic | Reference | Lines |
|-------|-----------|-------|
| Complete Workflows | [ci-cd-workflows.md](../docs/ci-cd-workflows.md) | ~400 |
| Test Matrices | [ci-cd-workflows.md#test-matrices](../docs/ci-cd-workflows.md#test-matrices) | ~100 |
| Coverage Setup | [ci-cd-workflows.md#coverage-reporting](../docs/ci-cd-workflows.md#coverage-reporting) | ~80 |
| Docs Deployment | [ci-cd-workflows.md#documentation-deployment](../docs/ci-cd-workflows.md#documentation-deployment) | ~100 |
| Automation Tools | [ci-cd-workflows.md#automation-tools](../docs/ci-cd-workflows.md#automation-tools) | ~120 |

---

## Phase 1: Detect Existing CI

Check for existing workflows and offer to update vs replace.

---

## Phase 2: Configuration Selection

### Platform Coverage

| Platform | When to Include |
|----------|-----------------|
| Linux (ubuntu-latest) | Always |
| macOS | Platform-specific code |
| Windows | Windows compatibility needed |

### Julia Version Matrix

| Version | Purpose |
|---------|---------|
| LTS (1.6) | Long-term support |
| Latest (1) | Current release |
| Nightly | Future compatibility |

### Features

| Feature | Purpose |
|---------|---------|
| Coverage | Codecov or Coveralls reporting |
| Documentation | Documenter.jl deployment |
| CompatHelper | Automated dependency updates |
| TagBot | Release automation |
| JuliaFormatter | Code formatting check |
| Quality checks | Aqua.jl, JET.jl |

**Quick mode:** Auto-select Linux + Latest + Coverage

---

## Phase 3: Generate CI Workflow

### Workflow Structure

| Component | Purpose |
|-----------|---------|
| actions/checkout@v4 | Checkout code |
| julia-actions/setup-julia@v1 | Install Julia |
| julia-actions/cache@v1 | Cache packages |
| julia-actions/julia-buildpkg@v1 | Build package |
| julia-actions/julia-runtest@v1 | Run tests |

**Full templates:** [ci-cd-workflows.md#github-actions-workflows](../docs/ci-cd-workflows.md#github-actions-workflows)

---

## Phase 4: Coverage Setup

| Step | Action |
|------|--------|
| 1 | Add julia-processcoverage action |
| 2 | Add codecov-action with lcov.info |
| 3 | Create .codecov.yml (optional, target 80%) |
| 4 | Add badge to README |

---

## Phase 5: Documentation Workflow

Generate Documentation.yml with:
- Trigger on push to main, tags, PRs
- Install docs dependencies
- Build and deploy with DOCUMENTER_KEY

---

## Phase 6: Automation Tools (Comprehensive)

| Workflow | Purpose |
|----------|---------|
| CompatHelper.yml | Automated dependency updates |
| TagBot.yml | Automatic release creation |
| Format.yml | JuliaFormatter checks |
| Quality.yml | Aqua + JET checks |

---

## Generated Files by Mode

| Mode | Files |
|------|-------|
| Quick | CI.yml |
| Standard | + Documentation.yml, CompatHelper.yml, TagBot.yml |
| Comprehensive | + Format.yml, Quality.yml |

---

## Common Configurations

| Package Type | Platforms | Versions | Features |
|--------------|-----------|----------|----------|
| Academic/Research | Linux | Latest + nightly | Docs, coverage |
| Open Source Library | All 3 | LTS + latest | All features |
| Internal Tool | Linux | Latest | CI only |
| Performance-Critical | Linux + macOS | Latest | CI, benchmarks, quality |

---

## Post-Generation Steps

| Step | Command |
|------|---------|
| 1. Commit | `git add .github/workflows/ && git commit` |
| 2. Setup secrets | `DocumenterTools.genkeys()` |
| 3. Push | `git push` |
| 4. Add badges | Update README.md |
| 5. Monitor | Check GitHub Actions tab |

---

## Success Criteria

- [ ] Workflows generated in `.github/workflows/`
- [ ] Test matrix configured
- [ ] Caching enabled
- [ ] Coverage reporting set up
- [ ] Documentation deployment configured
- [ ] Automation tools added
- [ ] README badges provided

---

## Related Commands

- `/julia-scaffold` - Generate package structure with CI
