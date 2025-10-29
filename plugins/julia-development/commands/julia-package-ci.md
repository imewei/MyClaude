# /julia-package-ci - Generate CI/CD Workflows

**Priority**: 4
**Agent**: julia-developer
**Description**: Generate GitHub Actions CI/CD workflows for Julia packages with test matrices, coverage, and documentation deployment.

## Usage
```
/julia-package-ci
```

## Generated Workflows
- `.github/workflows/CI.yml`: Test matrix (Julia versions, OS)
- `.github/workflows/Documentation.yml`: Docs deployment
- `.github/workflows/CompatHelper.yml`: Dependency updates
- `.github/workflows/TagBot.yml`: Release automation

## Features
- Cross-platform testing (Linux, macOS, Windows)
- Multiple Julia versions (1.6, stable, nightly)
- Code coverage with Codecov
- Documentation deployment to GitHub Pages
