---
description: Generate GitHub Actions CI/CD workflows for Julia packages
triggers:
- /julia-package-ci
- generate github actions ci/cd
allowed-tools: [Read, Task, Bash]
version: 1.0.0
---



# Generate CI/CD Workflows for Julia Packages

**Docs**: [ci-cd-workflows.md](../../plugins/julia-development/docs/ci-cd-workflows.md) (~400 lines)

## Workflow Structure

| Component | Purpose |
|-----------|---------|
| actions/checkout@v4 | Checkout code |
| julia-actions/setup-julia@v1 | Install Julia |
| julia-actions/cache@v1 | Cache packages |
| julia-actions/julia-buildpkg@v1 | Build |
| julia-actions/julia-runtest@v1 | Test |

## Configuration

### Platform
- Linux (ubuntu-latest): Always
- macOS: Platform-specific code
- Windows: Windows compatibility

### Julia Versions
- LTS (1.6): Long-term support
- Latest (1): Current
- Nightly: Future compatibility

### Features
- Coverage: Codecov/Coveralls
- Documentation: Documenter.jl
- CompatHelper: Dependency updates
- TagBot: Release automation
- JuliaFormatter: Code formatting
- Quality: Aqua.jl, JET.jl

## Coverage Setup

1. Add julia-processcoverage action
2. Add codecov-action with lcov.info
3. Create .codecov.yml (target 80%)
4. Add badge to README

## Documentation

Generate Documentation.yml:
- Trigger on main, tags, PRs
- Install docs dependencies
- Deploy with DOCUMENTER_KEY

## Post-Generation

1. `git add .github/workflows/ && git commit`
2. `DocumenterTools.genkeys()` (setup secrets)
3. `git push`
4. Add badges to README
5. Monitor GitHub Actions

**Outcome**: Production-ready CI/CD with tests, coverage, docs, and automation
