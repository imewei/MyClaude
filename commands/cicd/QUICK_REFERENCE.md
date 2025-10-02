# CI/CD Quick Reference Guide

## Common Commands

### Testing
```bash
# Run all tests
./cicd/testing/run_all_tests.sh

# Run integration tests only
./cicd/testing/run_integration_tests.sh

# Run performance benchmarks
./cicd/testing/run_performance_tests.sh

# Quick smoke tests
./cicd/testing/run_smoke_tests.sh
```

### Building
```bash
# Clean build
python cicd/build/build.py --clean

# Build with validation
python cicd/build/build.py --clean --checksums --test-install

# Build specific type
python cicd/build/build.py --type wheel
```

### Version Management
```bash
# Bump patch version (1.0.0 → 1.0.1)
python cicd/build/version_bump.py patch

# Bump minor version (1.0.0 → 1.1.0)
python cicd/build/version_bump.py minor

# Bump major version (1.0.0 → 2.0.0)
python cicd/build/version_bump.py major

# Create prerelease (1.0.0 → 1.0.0-beta.1)
python cicd/build/version_bump.py prerelease --prerelease beta.1
```

### Quality Checks
```bash
# Check code quality
python cicd/quality/quality_gate.py \
  --complexity complexity.json \
  --maintainability maintainability.json

# Check test coverage
python cicd/quality/coverage_gate.py \
  --report coverage.xml \
  --threshold 90

# Check security
python cicd/quality/security_gate.py \
  --vulnerability-report safety-report.json \
  --sast-report bandit-report.json
```

### Deployment
```bash
# Deploy to PyPI
export PYPI_API_TOKEN="your-token"
bash cicd/deploy/deploy_pypi.sh

# Run smoke tests
bash cicd/deploy/smoke_test_prod.sh
```

### Monitoring
```bash
# Check system health
python cicd/monitoring/health_check.py

# Check documentation links
python cicd/docs/link_checker.py --fail-on-error

# Build documentation
bash cicd/docs/build_docs.sh
```

### Pre-commit
```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Update hooks
pre-commit autoupdate
```

## Release Process

### 1. Prepare Release
```bash
# Update version
python cicd/build/version_bump.py minor

# Generate changelog
python cicd/build/changelog_generator.py --version X.Y.Z

# Commit changes
git add .
git commit -m "chore: prepare release X.Y.Z"
```

### 2. Create Tag
```bash
# Create and push tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
```

### 3. Monitor Deployment
```bash
# Watch GitHub Actions
# Go to: https://github.com/{owner}/{repo}/actions

# After deployment completes
bash cicd/deploy/smoke_test_prod.sh
python cicd/monitoring/health_check.py
```

## Workflow Triggers

| Workflow | Trigger | When |
|----------|---------|------|
| CI | Push, PR | Every commit |
| CD | Release | Manual tag creation |
| Quality | Schedule | Weekly (Sunday) |
| Performance | Schedule | Every 6 hours |
| Security | Schedule | Daily (midnight) |

## Quality Thresholds

| Metric | Threshold | Blocking |
|--------|-----------|----------|
| Test Coverage | ≥ 90% | Yes |
| Cyclomatic Complexity | ≤ 10 | Yes |
| Maintainability Index | ≥ 20 | No (warning) |
| Documentation Coverage | ≥ 80% | Yes |
| Critical Vulnerabilities | 0 | Yes |
| High Vulnerabilities | 0 | Yes |
| Medium Vulnerabilities | ≤ 5 | No (warning) |
| Performance Regression | ≤ 150% | No (warning) |

## File Locations

### Configurations
- GitHub Actions: `.github/workflows/*.yml`
- GitLab CI: `.gitlab-ci.yml`
- Pre-commit: `.pre-commit-config.yaml`

### Scripts
- Testing: `cicd/testing/`
- Building: `cicd/build/`
- Quality: `cicd/quality/`
- Deployment: `cicd/deploy/`
- Monitoring: `cicd/monitoring/`
- Documentation: `cicd/docs/`

### Documentation
- Main Guide: `cicd/README.md`
- Deployment: `cicd/DEPLOYMENT.md`
- Summary: `cicd/CICD_SUMMARY.md`
- Overview: `cicd/OVERVIEW.md`
- This Guide: `cicd/QUICK_REFERENCE.md`

## Environment Variables

### Required for Deployment
```bash
export PYPI_API_TOKEN="pypi-..."           # PyPI production
export TEST_PYPI_API_TOKEN="pypi-..."     # PyPI staging
export DOCKER_USERNAME="username"          # Docker Hub
export DOCKER_PASSWORD="password"          # Docker Hub
export GITHUB_TOKEN="${GITHUB_TOKEN}"      # Auto-provided
```

## Troubleshooting

### Tests Failing
```bash
# Run tests locally
./cicd/testing/run_all_tests.sh

# Check specific test
pytest tests/path/to/test.py -v

# Check coverage
pytest --cov=claude_commands --cov-report=html
```

### Quality Gate Failing
```bash
# Check what's failing
python cicd/quality/quality_gate.py

# Fix complexity
# Refactor complex functions

# Fix coverage
# Add more tests
pytest --cov=claude_commands --cov-report=term-missing
```

### Pre-commit Failing
```bash
# Run specific hook
pre-commit run black --all-files

# Skip hooks (not recommended)
git commit --no-verify

# Update hooks
pre-commit autoupdate
```

### Deployment Failing
```bash
# Check GitHub Actions logs
# Go to Actions tab in GitHub

# Verify credentials
echo $PYPI_API_TOKEN | wc -c

# Test build locally
python cicd/build/build.py --clean --test-install
```

## GitHub Actions Secrets

Set these in: Repository → Settings → Secrets → Actions

| Secret | Purpose |
|--------|---------|
| `PYPI_API_TOKEN` | PyPI package publishing |
| `TEST_PYPI_API_TOKEN` | Test PyPI publishing |
| `DOCKER_USERNAME` | Docker Hub login |
| `DOCKER_PASSWORD` | Docker Hub login |

## Useful Links

- **GitHub Actions**: https://github.com/{owner}/{repo}/actions
- **PyPI Package**: https://pypi.org/project/claude-commands
- **Documentation**: https://docs.claude-commands.dev
- **Docker Hub**: https://hub.docker.com/r/claudecommands/executor

## Tips

- Always run tests locally before pushing
- Use pre-commit hooks to catch issues early
- Check workflow logs for detailed error messages
- Monitor performance benchmarks in PR comments
- Review security scan results promptly
- Keep dependencies updated regularly

## Getting Help

1. Check workflow logs in GitHub Actions
2. Review documentation in `cicd/`
3. Run commands locally to reproduce issues
4. Check this quick reference guide
5. Consult the full guides:
   - `cicd/README.md` - Comprehensive guide
   - `cicd/DEPLOYMENT.md` - Deployment procedures
   - `cicd/OVERVIEW.md` - System architecture