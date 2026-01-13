---
name: ci-cd-patterns
version: "1.0.7"
maturity: "5-Expert"
specialization: Julia CI/CD
description: Master GitHub Actions for Julia packages with test matrices, CompatHelper, TagBot, and documentation deployment. Use when setting up CI workflows for Julia packages.
---

# Julia CI/CD Patterns

GitHub Actions workflows for Julia packages.

---

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
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v4
```

---

## Documentation Deployment

```yaml
name: Documentation
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
      - run: |
          julia --project=docs/ -e '
            using Pkg
            Pkg.develop(PackageSpec(path=pwd()))
            Pkg.instantiate()
          '
      - run: julia --project=docs/ docs/make.jl
```

---

## Automation Tools

| Tool | Purpose |
|------|---------|
| CompatHelper.jl | Auto dependency updates |
| TagBot.jl | Automated release tagging |
| Documenter.jl | Documentation deployment |
| Codecov | Coverage reporting |

---

## Checklist

- [ ] Test matrix covers Julia versions
- [ ] Multi-platform testing (Linux, macOS, Windows)
- [ ] Coverage reporting configured
- [ ] Documentation auto-deploys
- [ ] CompatHelper/TagBot enabled

---

**Version**: 1.0.5
