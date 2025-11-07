# Julia CI/CD Workflows

Comprehensive guide to setting up continuous integration and deployment for Julia packages.

## Table of Contents

- [GitHub Actions Workflows](#github-actions-workflows)
- [Test Matrices](#test-matrices)
- [Coverage Reporting](#coverage-reporting)
- [Documentation Deployment](#documentation-deployment)
- [Automation Tools](#automation-tools)

---

## GitHub Actions Workflows

### Basic CI Workflow

**File**: `.github/workflows/CI.yml`

```yaml
name: CI
on:
  push:
    branches:
      - main
      - master
  pull_request:
    branches:
      - main
      - master

jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }} - ${{ matrix.arch }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1.6'    # LTS
          - '1'      # Latest stable
          - 'nightly'
        os:
          - ubuntu-latest
          - macOS-latest
          - windows-latest
        arch:
          - x64
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}
      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v3
        with:
          files: lcov.info
```

### Minimal CI (Fast Feedback)

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: '1'  # Latest stable only
      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
```

### Advanced CI with Quality Checks

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        version: ['1.6', '1']
        os: [ubuntu-latest, macOS-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.version }}
      - uses: julia-actions/cache@v1
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v3

  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
      - name: Install dependencies
        run: |
          julia --project=@. -e 'using Pkg; Pkg.add(["Aqua", "JET"])'
      - name: Run Aqua tests
        run: |
          julia --project=@. -e 'using Aqua, YourPackage; Aqua.test_all(YourPackage)'
      - name: Run JET analysis
        run: |
          julia --project=@. -e 'using JET, YourPackage; JET.report_package(YourPackage)'
```

---

## Test Matrices

### Platform Coverage

#### Linux Only (Default)

```yaml
runs-on: ubuntu-latest
```

**Use when**:
- Package has no platform-specific code
- Fast CI is priority
- Cost-conscious

#### Cross-Platform

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macOS-latest, windows-latest]
runs-on: ${{ matrix.os }}
```

**Use when**:
- Platform-specific code (file paths, system calls)
- Binary dependencies
- Production package

#### Linux + One Other

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest]  # Or macOS-latest
```

**Use when**:
- Budget for 2 platforms
- Want some cross-platform validation

### Julia Version Coverage

#### LTS + Latest

```yaml
strategy:
  matrix:
    version: ['1.6', '1']  # Recommended
```

**Use when**:
- Standard package
- Support current LTS and latest

#### Latest Only

```yaml
setup-julia:
  version: '1'
```

**Use when**:
- New package
- Rapid development
- Using latest features

#### Full Range

```yaml
strategy:
  matrix:
    version: ['1.6', '1.9', '1.10', '1', 'nightly']

allow-failure:
  - version: 'nightly'  # Optional
```

**Use when**:
- Core ecosystem package
- Broad compatibility needed
- Testing future Julia versions

### Architecture Coverage

#### x64 Only (Default)

```yaml
arch: x64
```

**Use when**:
- Standard case
- No ARM-specific code

#### Multi-Architecture

```yaml
strategy:
  matrix:
    arch: [x64, x86, aarch64]
```

**Use when**:
- ARM deployment (Apple Silicon, servers)
- 32-bit support needed

---

## Coverage Reporting

### Codecov Integration

```yaml
- uses: julia-actions/julia-processcoverage@v1
- uses: codecov/codecov-action@v3
  with:
    files: lcov.info
    fail_ci_if_error: true  # Optional: fail if upload fails
```

**Setup**: Create account at [codecov.io](https://codecov.io), add repository.

### Coveralls Integration

```yaml
- uses: julia-actions/julia-processcoverage@v1
- uses: coverallsapp/github-action@v2
  with:
    github-token: ${{ secrets.GITHUB_TOKEN }}
    path-to-lcov: lcov.info
```

### Coverage Badge

Add to `README.md`:

```markdown
[![Coverage](https://codecov.io/gh/username/Package.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/username/Package.jl)
```

### Coverage Configuration

Create `.codecov.yml`:

```yaml
coverage:
  status:
    project:
      default:
        target: 80%        # Minimum coverage
        threshold: 1%      # Allow 1% drop
    patch:
      default:
        target: 90%        # New code coverage

ignore:
  - "test"               # Ignore test files
  - "examples"           # Ignore examples
  - "docs"               # Ignore docs
```

---

## Documentation Deployment

### Documenter.jl Workflow

**File**: `.github/workflows/Documentation.yml`

```yaml
name: Documentation

on:
  push:
    branches:
      - main
    tags: '*'
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: '1'
      - name: Install dependencies
        run: |
          julia --project=docs -e '
            using Pkg
            Pkg.develop(PackageSpec(path=pwd()))
            Pkg.instantiate()'
      - name: Build and deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
        run: julia --project=docs docs/make.jl
```

### Documenter Setup

1. **Generate SSH key**:
   ```bash
   julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="username", repo="Package.jl")'
   ```

2. **Add keys**:
   - Private key → GitHub Secrets as `DOCUMENTER_KEY`
   - Public key → Deploy keys (with write access)

3. **Create `docs/make.jl`**:
   ```julia
   using Documenter, YourPackage

   makedocs(
       sitename = "YourPackage.jl",
       modules = [YourPackage],
       pages = [
           "Home" => "index.md",
           "API" => "api.md",
       ]
   )

   deploydocs(
       repo = "github.com/username/YourPackage.jl.git",
       devbranch = "main"
   )
   ```

### Documentation Badge

```markdown
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://username.github.io/Package.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://username.github.io/Package.jl/dev)
```

---

## Automation Tools

### CompatHelper.jl

Automatically creates PRs to update package versions in `[compat]`.

**File**: `.github/workflows/CompatHelper.yml`

```yaml
name: CompatHelper
on:
  schedule:
    - cron: '00 00 * * *'  # Daily at midnight
  workflow_dispatch:

jobs:
  CompatHelper:
    runs-on: ubuntu-latest
    steps:
      - name: Check if Julia is already available
        id: julia_exists
        run: echo "julia=$(which julia)" >> $GITHUB_OUTPUT
      - name: Install Julia
        uses: julia-actions/setup-julia@v1
        if: steps.julia_exists.outputs.julia == ''
        with:
          version: '1'
      - name: Install CompatHelper
        run: julia -e 'using Pkg; Pkg.add("CompatHelper")'
      - name: CompatHelper
        run: julia -e 'using CompatHelper; CompatHelper.main()'
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          COMPATHELPER_PRIV: ${{ secrets.DOCUMENTER_KEY }}
```

### TagBot.jl

Automatically creates GitHub releases when JuliaRegistrator registers a new version.

**File**: `.github/workflows/TagBot.yml`

```yaml
name: TagBot
on:
  issue_comment:
    types:
      - created
  workflow_dispatch:
    inputs:
      lookback:
        default: '3'
permissions:
  actions: read
  checks: read
  contents: write
  deployments: read
  issues: read
  discussions: read
  packages: read
  pages: read
  pull-requests: read
  repository-projects: read
  security-events: read
  statuses: read
jobs:
  TagBot:
    if: github.event_name == 'workflow_dispatch' || github.actor == 'JuliaTagBot'
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          ssh: ${{ secrets.DOCUMENTER_KEY }}
```

### JuliaFormatter.jl

Automatically format code on PR.

**File**: `.github/workflows/Format.yml`

```yaml
name: Format Check

on:
  push:
    branches:
      - main
  pull_request:

jobs:
  format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v1
        with:
          version: '1'
      - name: Install JuliaFormatter
        run: julia -e 'using Pkg; Pkg.add("JuliaFormatter")'
      - name: Format code
        run: julia -e 'using JuliaFormatter; format(".", verbose=true)'
      - name: Check for changes
        run: |
          git diff --exit-code || (echo "Code needs formatting. Run 'julia -e \"using JuliaFormatter; format(\\\".\\\")'\"" && exit 1)
```

---

## Security Scanning

### Dependency Scanning

```yaml
name: Security Scan
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
```

### Secret Scanning

GitHub automatically scans for exposed secrets. Configure in Settings → Code security and analysis.

---

## Performance Benchmarking

### PkgBenchmark.jl Integration

```yaml
name: Benchmarks
on:
  pull_request:
    types: [labeled]

jobs:
  benchmark:
    if: contains(github.event.pull_request.labels.*.name, 'benchmark')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: julia-actions/setup-julia@v1
      - name: Run benchmarks
        run: |
          julia --project -e '
            using Pkg
            Pkg.add("PkgBenchmark")
            using PkgBenchmark
            results = benchmarkpkg(".", "HEAD^")  # Compare with parent
            export_markdown("benchmark_results.md", results)
          '
      - name: Comment results
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = fs.readFileSync('benchmark_results.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.name,
              body: results
            });
```

---

## Workflow Optimization

### Caching

```yaml
- uses: julia-actions/cache@v1  # Caches compiled packages
```

**Impact**: 2-5x faster CI runs.

### Conditional Execution

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'test/**'
      - 'Project.toml'
  # Skip if only docs changed
```

### Matrix Optimization

```yaml
strategy:
  fail-fast: false  # Continue other jobs if one fails
  matrix:
    version: ['1.6', '1']
    os: [ubuntu-latest]
    include:
      - version: '1'
        os: macOS-latest  # Only test latest on macOS
      - version: '1'
        os: windows-latest
```

**Benefit**: Faster feedback (Linux first), comprehensive coverage.

---

## Best Practices

### DO ✅

1. **Test LTS + Latest** Julia versions
2. **Use caching** for faster CI
3. **Enable CompatHelper** for dependency updates
4. **Set up TagBot** for automatic releases
5. **Include coverage** reporting
6. **Test cross-platform** if relevant
7. **Use fail-fast: false** to see all failures

### DON'T ❌

1. **Don't test every minor version** (slow, unnecessary)
2. **Don't skip caching** (wastes CI time)
3. **Don't ignore failing nightly** (future compatibility)
4. **Don't hardcode versions** (use matrix variables)
5. **Don't forget documentation** deployment
6. **Don't skip quality checks** (Aqua, JET)

---

## Troubleshooting

### Problem: CI Fails on Windows

**Cause**: Path separator differences (`/` vs `\`)

**Solution**: Use `joinpath` or `@__DIR__` for paths.

### Problem: Slow CI

**Solutions**:
- Enable caching
- Reduce test matrix
- Use `fail-fast: true` for quick feedback
- Parallelize independent jobs

### Problem: Coverage Not Uploading

**Solutions**:
- Check `julia-processcoverage` ran
- Verify `lcov.info` exists
- Check Codecov/Coveralls tokens

### Problem: Documentation Not Deploying

**Solutions**:
- Verify `DOCUMENTER_KEY` secret exists
- Check deploy key has write access
- Ensure `deploydocs` in `make.jl`

---

## CI Configuration Checklist

- [ ] `.github/workflows/CI.yml` created
- [ ] Test matrix includes LTS + latest Julia
- [ ] Cross-platform testing configured (if needed)
- [ ] Caching enabled (`julia-actions/cache@v1`)
- [ ] Coverage reporting set up (Codecov/Coveralls)
- [ ] Documentation workflow created
- [ ] CompatHelper.yml added
- [ ] TagBot.yml added
- [ ] Quality checks included (Aqua, JET)
- [ ] Badges added to README.md

---

**Version**: 1.0.3
**Last Updated**: 2025-11-07
**Plugin**: julia-development
