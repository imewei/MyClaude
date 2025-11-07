---
version: "1.0.3"
category: "julia-development"
command: "/julia-package-ci"
description: Generate comprehensive GitHub Actions CI/CD workflows for Julia packages with test matrices, coverage reporting, documentation deployment, and automation
allowed-tools: Bash(find:*), Bash(git:*)
argument-hint: ""
color: blue
execution_modes:
  quick: "5-10 minutes - Generate basic CI workflow for Linux + latest Julia"
  standard: "15-20 minutes - Generate comprehensive cross-platform CI with coverage and docs"
  comprehensive: "25-35 minutes - Generate full CI/CD suite with security scanning, benchmarks, and automation"
agents:
  primary:
    - julia-developer
  conditional:
    - agent: julia-pro
      trigger: pattern "performance|benchmark|optimization"
  orchestrated: false
---

# Generate CI/CD Workflows for Julia Packages

Generate production-ready GitHub Actions workflows with test matrices, coverage reporting, documentation deployment, and automation tools (CompatHelper, TagBot).

## Quick Reference

| Topic | External Documentation | Lines |
|-------|------------------------|-------|
| **Complete Workflows** | [ci-cd-workflows.md](../docs/ci-cd-workflows.md) | ~400 |
| **Test Matrices** | [ci-cd-workflows.md#test-matrices](../docs/ci-cd-workflows.md#test-matrices) | ~100 |
| **Coverage Setup** | [ci-cd-workflows.md#coverage-reporting](../docs/ci-cd-workflows.md#coverage-reporting) | ~80 |
| **Docs Deployment** | [ci-cd-workflows.md#documentation-deployment](../docs/ci-cd-workflows.md#documentation-deployment) | ~100 |
| **Automation Tools** | [ci-cd-workflows.md#automation-tools](../docs/ci-cd-workflows.md#automation-tools) | ~120 |

**Total External Documentation**: ~400 lines of workflow templates and guides

## Core Workflow

### Phase 1: Detect Existing CI

**Check for existing workflows**:

```bash
ls -la .github/workflows/
```

**If exists**: Offer to update vs replace
**If missing**: Proceed with generation

### Phase 2: Configuration Selection

**Prompt for CI scope** (standard & comprehensive modes):

**Platform Coverage**:
- [ ] Linux (ubuntu-latest) - Recommended, always include
- [ ] macOS (macOS-latest) - For platform-specific code
- [ ] Windows (windows-latest) - For Windows compatibility

**Julia Version Matrix**:
- [ ] LTS (1.6) - Long-term support
- [ ] Latest stable (1) - Current release
- [ ] Nightly - Test future compatibility

**Features**:
- [ ] Coverage reporting (Codecov or Coveralls)
- [ ] Documentation deployment (Documenter.jl)
- [ ] CompatHelper (dependency updates)
- [ ] TagBot (release automation)
- [ ] JuliaFormatter (code formatting check)
- [ ] Quality checks (Aqua.jl, JET.jl)

**Quick mode**: Auto-select Linux + Latest + Coverage

### Phase 3: Generate CI Workflow

**Create `.github/workflows/CI.yml`**:

**Minimal CI** (Quick mode):
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: '1'
      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
```

**Comprehensive CI** (Standard mode):
- Cross-platform matrix (Linux/macOS/Windows)
- Julia version matrix (1.6, 1)
- Coverage reporting
- Caching for faster builds
- fail-fast: false (see all failures)

**Full template**: [ci-cd-workflows.md#github-actions-workflows](../docs/ci-cd-workflows.md#github-actions-workflows)

### Phase 4: Coverage Setup (if selected)

**Generate coverage workflow**:

1. **Add coverage step** to CI.yml:
```yaml
- uses: julia-actions/julia-processcoverage@v1
- uses: codecov/codecov-action@v3
  with:
    files: lcov.info
```

2. **Create `.codecov.yml`** (optional):
```yaml
coverage:
  status:
    project:
      default:
        target: 80%
        threshold: 1%
```

3. **Add badge** to README.md:
```markdown
[![Coverage](https://codecov.io/gh/username/Package.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/username/Package.jl)
```

**Details**: [ci-cd-workflows.md#coverage-reporting](../docs/ci-cd-workflows.md#coverage-reporting)

### Phase 5: Documentation Workflow (if selected)

**Create `.github/workflows/Documentation.yml`**:

```yaml
name: Documentation
on:
  push:
    branches: [main]
    tags: '*'
  pull_request:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
      - name: Install dependencies
        run: julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
      - name: Build and deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
        run: julia --project=docs docs/make.jl
```

**Setup instructions**: [ci-cd-workflows.md#documentation-deployment](../docs/ci-cd-workflows.md#documentation-deployment)

### Phase 6: Automation Tools (comprehensive mode)

**Generate additional workflows**:

1. **CompatHelper.yml**: Automated dependency updates
2. **TagBot.yml**: Automatic release creation
3. **Format.yml**: Code formatting checks (if JuliaFormatter selected)
4. **Quality.yml**: Aqua and JET checks (if quality checks selected)

**Templates**: [ci-cd-workflows.md#automation-tools](../docs/ci-cd-workflows.md#automation-tools)

### Phase 7: Setup Instructions

**Provide step-by-step setup**:

1. **Commit and push** workflows
2. **Set up secrets** (if docs deployment enabled):
   - Generate DOCUMENTER_KEY
   - Add to GitHub secrets
3. **Enable GitHub Actions** (should auto-enable)
4. **Create first PR** to test CI
5. **Add badges** to README.md

## Mode-Specific Execution

### Quick Mode (5-10 minutes)

**Generated**:
- Basic CI.yml (Linux + latest Julia)
- Coverage with Codecov
- README badge

**Skip**: Cross-platform, documentation, automation tools

### Standard Mode (15-20 minutes) - DEFAULT

**Generated**:
- Comprehensive CI.yml (cross-platform, version matrix)
- Coverage reporting (user choice of Codecov/Coveralls)
- Documentation.yml (if docs exist)
- CompatHelper.yml
- TagBot.yml
- README badges
- Setup instructions

### Comprehensive Mode (25-35 minutes)

**Generated**:
- All from standard mode
- Quality.yml (Aqua + JET checks)
- Format.yml (JuliaFormatter)
- Security scanning (if requested)
- Performance benchmarking workflow (if requested)
- Detailed optimization guide
- Troubleshooting section

## Generated Files

### Basic Package (Quick)

```
.github/workflows/
└── CI.yml
```

### Standard Package (Standard)

```
.github/workflows/
├── CI.yml
├── Documentation.yml
├── CompatHelper.yml
└── TagBot.yml
```

### Production Package (Comprehensive)

```
.github/workflows/
├── CI.yml
├── Documentation.yml
├── CompatHelper.yml
├── TagBot.yml
├── Format.yml
└── Quality.yml
```

## Common Configurations

### Academic/Research Package
- **Platforms**: Linux only
- **Versions**: Latest + nightly
- **Features**: Docs, coverage
- **Why**: Cost-effective, docs important for citations

### Open Source Library
- **Platforms**: Linux + macOS + Windows
- **Versions**: LTS + latest
- **Features**: All (coverage, docs, automation, quality)
- **Why**: Broad compatibility, professional quality

### Internal Tool
- **Platforms**: Linux only
- **Versions**: Latest
- **Features**: CI only (minimal)
- **Why**: Fast feedback, no public docs needed

### Performance-Critical Package
- **Platforms**: Linux + macOS
- **Versions**: Latest
- **Features**: CI, benchmarking, quality checks
- **Why**: Performance regression detection critical

## Success Criteria

✅ Workflows generated in `.github/workflows/`
✅ Test matrix configured (platforms + Julia versions)
✅ Caching enabled for faster CI
✅ Coverage reporting set up (if selected)
✅ Documentation deployment configured (if selected)
✅ Automation tools added (if selected)
✅ README badges provided
✅ Setup instructions clear
✅ External documentation referenced

## Agent Integration

- **julia-developer**: Primary agent for CI/CD workflow generation and package best practices
- **julia-pro**: Triggered for performance benchmarking configuration

## Post-Generation

After workflows are generated:

1. **Commit workflows**:
   ```bash
   git add .github/workflows/
   git commit -m "Add CI/CD workflows"
   ```

2. **Set up secrets** (if docs enabled):
   ```bash
   julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="username", repo="Package.jl")'
   ```

3. **Push and verify**:
   ```bash
   git push
   # Check GitHub Actions tab
   ```

4. **Add badges** to README.md

5. **Monitor first run** and fix any issues

**Troubleshooting**: [ci-cd-workflows.md#troubleshooting](../docs/ci-cd-workflows.md#troubleshooting)

**See Also**:
- `/julia-scaffold` - Generate package structure with CI
- [ci-cd-workflows.md](../docs/ci-cd-workflows.md) - Complete workflow library
- [package-scaffolding.md](../docs/package-scaffolding.md) - Package best practices

---

Focus on **appropriate coverage** for package type, **caching for speed**, and **automation to reduce maintenance**.
