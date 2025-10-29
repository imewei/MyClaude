---
name: ci-cd-patterns
description: GitHub Actions for Julia, test matrices, CompatHelper.jl, TagBot.jl, and documentation deployment. Foundation for /julia-package-ci command. Use for automating testing, dependency updates, and releases.
---

# CI/CD Patterns

Master GitHub Actions workflows for Julia packages.

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
