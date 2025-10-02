# CI/CD Automation System

Comprehensive CI/CD automation for the Claude Commands framework with automated testing, deployment, and monitoring.

## Overview

This CI/CD system provides:

- **Continuous Integration** - Automated testing on every commit
- **Continuous Deployment** - Automated deployment to PyPI, Docker Hub, and documentation
- **Quality Gates** - Enforced code quality, coverage, and security thresholds
- **Performance Monitoring** - Continuous performance benchmarking
- **Security Scanning** - Automated vulnerability and SAST scanning
- **Multi-Platform Support** - GitHub Actions and GitLab CI configurations

## Directory Structure

```
cicd/
├── testing/              # Testing automation
│   ├── run_all_tests.sh
│   ├── run_integration_tests.sh
│   ├── run_performance_tests.sh
│   ├── run_smoke_tests.sh
│   └── test_matrix_generator.py
├── build/                # Build and packaging
│   ├── build.py
│   ├── version_bump.py
│   └── changelog_generator.py
├── deploy/               # Deployment scripts
│   ├── deploy_pypi.sh
│   └── smoke_test_prod.sh
├── quality/              # Quality gates
│   ├── quality_gate.py
│   ├── coverage_gate.py
│   ├── security_gate.py
│   └── performance_gate.py
├── monitoring/           # Monitoring and alerting
│   ├── health_check.py
│   ├── performance_monitor.py
│   └── error_tracker.py
├── release/              # Release management
│   ├── release_notes.py
│   └── announcement_generator.py
└── docs/                 # Documentation deployment
    ├── build_docs.sh
    ├── link_checker.py
    └── validate_docs.py
```

## GitHub Actions Workflows

### CI Pipeline (`.github/workflows/ci.yml`)

Runs on every push and pull request:

1. **Test Matrix** - Python 3.10, 3.11, 3.12
   - Unit tests with coverage
   - Integration tests
   - Performance benchmarks

2. **Code Quality**
   - Linting (ruff, black, mypy, pylint)
   - Type checking
   - Import sorting

3. **Security Scanning**
   - Dependency vulnerabilities (Safety, pip-audit)
   - SAST scanning (Bandit, Semgrep)
   - Secret detection

4. **Build Validation**
   - Package building
   - Distribution validation
   - Installation testing

5. **Quality Gates**
   - Coverage threshold (90%+)
   - Complexity limits
   - Security thresholds

### CD Pipeline (`.github/workflows/cd.yml`)

Runs on releases:

1. **PyPI Deployment**
   - Build package
   - Validate package
   - Upload to PyPI
   - Smoke tests

2. **Documentation Deployment**
   - Build documentation
   - Validate links
   - Deploy to GitHub Pages

3. **Docker Deployment**
   - Build container images
   - Security scanning
   - Push to Docker Hub

4. **Release Management**
   - Generate release notes
   - Update changelog
   - Create announcements

### Quality Checks (`.github/workflows/quality.yml`)

Weekly comprehensive quality analysis:

1. **Code Quality Metrics**
   - Cyclomatic complexity
   - Maintainability index
   - Code duplication

2. **Documentation Quality**
   - Docstring coverage
   - Documentation completeness
   - Link validation

3. **Dependency Management**
   - Dependency auditing
   - License compliance
   - Outdated packages

### Performance Monitoring (`.github/workflows/performance.yml`)

Continuous performance tracking:

1. **Benchmarking**
   - Automated benchmarks
   - Baseline comparison
   - Regression detection

2. **Profiling**
   - Memory profiling
   - CPU profiling
   - Load testing

3. **Performance Gates**
   - Threshold enforcement
   - Regression alerts

### Security Scanning (`.github/workflows/security.yml`)

Daily security checks:

1. **Dependency Scanning**
   - Known vulnerabilities
   - Supply chain security
   - SBOM generation

2. **Code Analysis**
   - SAST scanning
   - Secret detection
   - Container scanning

3. **Security Gates**
   - Zero critical vulnerabilities
   - Zero high vulnerabilities
   - Limited medium vulnerabilities

## GitLab CI Pipeline

The `.gitlab-ci.yml` configuration provides equivalent functionality:

- **Stages**: test, quality, security, build, deploy, monitor
- **Parallel Execution**: Multiple Python versions and OS
- **Caching**: pip cache for faster builds
- **Artifacts**: Test reports, coverage, benchmarks

## Pre-commit Hooks

Install pre-commit hooks to catch issues before commit:

```bash
pip install pre-commit
pre-commit install
```

The hooks will automatically:

1. **Format Code** - black, isort
2. **Lint Code** - ruff, flake8, pylint
3. **Type Check** - mypy
4. **Security Scan** - bandit, detect-secrets
5. **Check Documentation** - pydocstyle
6. **Run Tests** - pytest with coverage
7. **Check Complexity** - radon

## Usage

### Running Tests Locally

```bash
# Run all tests
./cicd/testing/run_all_tests.sh

# Run integration tests only
./cicd/testing/run_integration_tests.sh

# Run performance benchmarks
./cicd/testing/run_performance_tests.sh

# Run quick smoke tests
./cicd/testing/run_smoke_tests.sh
```

### Building Package

```bash
# Clean build with validation
python cicd/build/build.py --clean --checksums

# Test installation
python cicd/build/build.py --test-install
```

### Version Management

```bash
# Bump patch version
python cicd/build/version_bump.py patch

# Bump minor version
python cicd/build/version_bump.py minor

# Bump major version
python cicd/build/version_bump.py major

# Create prerelease
python cicd/build/version_bump.py prerelease --prerelease beta.1
```

### Generating Changelog

```bash
# Generate changelog for new version
python cicd/build/changelog_generator.py --version 1.2.3
```

### Quality Gates

```bash
# Check code quality
python cicd/quality/quality_gate.py \
  --complexity complexity.json \
  --maintainability maintainability.json

# Check coverage
python cicd/quality/coverage_gate.py \
  --report coverage.xml \
  --threshold 90

# Check security
python cicd/quality/security_gate.py \
  --vulnerability-report safety-report.json \
  --sast-report bandit-report.json
```

### Health Monitoring

```bash
# Check system health
python cicd/monitoring/health_check.py

# Check documentation links
python cicd/docs/link_checker.py --fail-on-error
```

## Configuration

### Quality Thresholds

Edit `cicd/quality/*_thresholds.yaml` files to configure:

- Coverage requirements (default: 90%)
- Complexity limits (default: max 10)
- Security issue limits (default: 0 critical, 0 high)
- Performance regression limits (default: 150%)

### Environment Variables

Required for deployments:

```bash
# PyPI
export PYPI_API_TOKEN="your-token"
export TEST_PYPI_API_TOKEN="your-test-token"

# Docker Hub
export DOCKER_USERNAME="your-username"
export DOCKER_PASSWORD="your-password"

# GitHub
export GITHUB_TOKEN="your-token"
```

## Pipeline Stages

### 1. Test Stage

- Run unit tests with coverage
- Run integration tests
- Run performance benchmarks
- Generate test reports

**Success Criteria:**
- All tests pass
- Coverage >= 90%
- No performance regressions

### 2. Quality Stage

- Code linting and formatting
- Type checking
- Documentation validation
- Complexity analysis

**Success Criteria:**
- No linting errors
- Type checking passes
- Documentation coverage >= 80%
- Complexity within limits

### 3. Security Stage

- Dependency vulnerability scanning
- SAST analysis
- Secret detection
- License compliance

**Success Criteria:**
- Zero critical/high vulnerabilities
- No secrets detected
- All licenses compatible

### 4. Build Stage

- Build package distributions
- Generate checksums
- Validate package metadata
- Test installation

**Success Criteria:**
- Package builds successfully
- Metadata validation passes
- Installation test passes

### 5. Deploy Stage

- Deploy to PyPI
- Deploy documentation
- Deploy Docker images
- Create GitHub release

**Success Criteria:**
- All deployments successful
- Smoke tests pass
- Services healthy

### 6. Monitor Stage

- Run production smoke tests
- Health checks
- Performance monitoring
- Error tracking

**Success Criteria:**
- All health checks pass
- No errors detected
- Performance within limits

## Automated Checks

### On Every Commit

- Unit tests
- Integration tests
- Code linting
- Type checking
- Security scanning (basic)

### On Pull Request

- All commit checks
- Performance benchmarks
- Coverage reporting
- Quality gates
- Documentation validation

### On Release

- Full test suite
- Security scanning (comprehensive)
- Package building
- Deployment to PyPI
- Documentation deployment
- Docker image deployment
- Release notes generation

### Scheduled

- **Daily**: Security vulnerability scanning
- **Weekly**: Code quality analysis, dependency audits
- **Every 6 hours**: Performance monitoring

## Quality Metrics

The CI/CD system tracks and enforces:

1. **Test Coverage**: >= 90%
2. **Code Complexity**: <= 10 per function
3. **Maintainability Index**: >= 20
4. **Documentation Coverage**: >= 80%
5. **Security Issues**: 0 critical, 0 high
6. **Performance Regression**: <= 150% baseline

## Notifications

Notifications are sent for:

- Failed CI builds
- Security vulnerabilities
- Performance regressions
- Failed deployments
- Health check failures

## Troubleshooting

### CI Build Failures

1. Check the workflow logs in GitHub Actions
2. Run the failing command locally
3. Check for recent dependency updates
4. Verify environment variables

### Deployment Failures

1. Verify API tokens are set correctly
2. Check package version is unique
3. Verify all tests passed
4. Check network connectivity

### Quality Gate Failures

1. Review the quality report
2. Fix reported issues
3. Run quality checks locally
4. Commit fixes and re-run

## Best Practices

1. **Always run tests locally** before pushing
2. **Use pre-commit hooks** to catch issues early
3. **Keep dependencies updated** regularly
4. **Monitor CI/CD metrics** for trends
5. **Review security reports** promptly
6. **Document CI/CD changes** in this README

## Contributing

When modifying CI/CD:

1. Test changes in a feature branch
2. Use workflow dispatch for manual testing
3. Document configuration changes
4. Update this README
5. Get CI/CD changes reviewed

## Support

For CI/CD issues:

1. Check workflow logs
2. Review troubleshooting section
3. Check GitHub Actions status page
4. Contact DevOps team

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI Documentation](https://docs.gitlab.com/ee/ci/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)