---
name: ci-cd-patterns
description: Master CI/CD with GitHub Actions for Julia packages including test matrices, CompatHelper.jl, TagBot.jl, and documentation deployment for automated testing and releases. Use when creating or editing GitHub Actions workflows (.github/workflows/*.yml files), setting up test matrices across Julia versions and OS platforms, implementing CompatHelper.jl for automatic dependency updates, configuring TagBot.jl for automated release tagging, deploying documentation with Documenter.jl, running tests on multiple platforms (Linux, macOS, Windows), automating code coverage reporting (Codecov, Coveralls), setting up continuous integration, or managing package releases. Foundation for /julia-package-ci command and essential for modern Julia package development workflows.
---

# CI/CD Patterns

Master GitHub Actions workflows for Julia packages.

## When to use this skill

- Creating or editing GitHub Actions workflows (.github/workflows/*.yml)
- Setting up test matrices across Julia versions (1.6, 1, nightly)
- Testing on multiple platforms (Linux, macOS, Windows)
- Implementing CompatHelper.jl for automatic dependency updates
- Configuring TagBot.jl for automated release tagging
- Deploying documentation with Documenter.jl to GitHub Pages
- Running automated tests with julia-actions
- Implementing code coverage reporting (Codecov, Coveralls)
- Setting up continuous integration for Julia packages
- Managing package releases and versioning
- Automating quality checks (Aqua.jl, JET.jl) in CI
- Configuring nightly builds and scheduled workflows

## Basic CI Workflow
```yaml
# .github/workflows/CI.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        julia-version: ['1.6', '1', 'nightly']
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v3
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v3
```

## Documentation Deployment
```yaml
# .github/workflows/Documentation.yml
name: Documentation
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: julia-actions/setup-julia@v1
      - run: julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
      - run: julia --project=docs/ docs/make.jl
```

## Resources
- **Julia Actions**: https://github.com/julia-actions
