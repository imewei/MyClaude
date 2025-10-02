# Deployment Procedures

This document outlines the deployment procedures for the Claude Commands framework.

## Deployment Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Security scan passed
- [ ] Performance benchmarks acceptable

### Deployment Steps

1. **Prepare Release**
   ```bash
   # Bump version
   python cicd/build/version_bump.py [major|minor|patch]

   # Generate changelog
   python cicd/build/changelog_generator.py --version X.Y.Z

   # Commit changes
   git add .
   git commit -m "chore: prepare release X.Y.Z"
   git push
   ```

2. **Create Release**
   ```bash
   # Create and push tag
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

3. **Automated Deployment**
   - GitHub Actions will automatically:
     - Build package
     - Run full test suite
     - Deploy to PyPI
     - Deploy documentation
     - Build and push Docker images
     - Create GitHub release

4. **Verify Deployment**
   ```bash
   # Wait 2 minutes for PyPI propagation
   sleep 120

   # Run smoke tests
   bash cicd/deploy/smoke_test_prod.sh

   # Check health
   python cicd/monitoring/health_check.py
   ```

### Post-Deployment

- [ ] Verify package on PyPI
- [ ] Check documentation site
- [ ] Test Docker image
- [ ] Monitor for errors
- [ ] Announce release

## Rollback Procedures

If deployment fails or critical issues are found:

1. **Immediate Actions**
   ```bash
   # Yank bad release from PyPI (doesn't delete)
   twine yank -u __token__ -p $PYPI_API_TOKEN claude-commands X.Y.Z
   ```

2. **Deploy Hotfix**
   ```bash
   # Create hotfix branch from previous version
   git checkout vX.Y.Z-1
   git checkout -b hotfix/X.Y.Z+1

   # Apply fixes
   # ... make fixes ...

   # Deploy hotfix
   python cicd/build/version_bump.py patch
   git add .
   git commit -m "fix: critical issue"
   git push
   git tag -a vX.Y.Z+1 -m "Hotfix X.Y.Z+1"
   git push origin vX.Y.Z+1
   ```

3. **Communication**
   - Notify users of issue
   - Provide workaround if available
   - Announce hotfix release

## Environment-Specific Deployments

### Test PyPI (Staging)

Deploy to Test PyPI for validation:

```bash
# Set test token
export TEST_PYPI_API_TOKEN="your-test-token"

# Build package
python cicd/build/build.py --clean

# Deploy to Test PyPI
twine upload --repository testpypi dist/* \
  -u __token__ -p $TEST_PYPI_API_TOKEN

# Test installation
pip install --index-url https://test.pypi.org/simple/ claude-commands
```

### Production PyPI

Automated via GitHub Actions on release, or manually:

```bash
# Set production token
export PYPI_API_TOKEN="your-production-token"

# Deploy
bash cicd/deploy/deploy_pypi.sh
```

### Documentation

Deploy documentation to GitHub Pages:

```bash
# Build docs
bash cicd/docs/build_docs.sh

# Deploy manually (or let GitHub Actions handle it)
mkdocs gh-deploy
```

### Docker Hub

Build and push Docker images:

```bash
# Build image
docker build -t claudecommands/executor:X.Y.Z .
docker tag claudecommands/executor:X.Y.Z claudecommands/executor:latest

# Login
echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin

# Push
docker push claudecommands/executor:X.Y.Z
docker push claudecommands/executor:latest
```

## Release Types

### Patch Release (X.Y.Z -> X.Y.Z+1)

For bug fixes and minor updates:

```bash
python cicd/build/version_bump.py patch
```

### Minor Release (X.Y.Z -> X.Y+1.0)

For new features (backward compatible):

```bash
python cicd/build/version_bump.py minor
```

### Major Release (X.Y.Z -> X+1.0.0)

For breaking changes:

```bash
python cicd/build/version_bump.py major
```

### Pre-Release

For alpha, beta, or release candidates:

```bash
python cicd/build/version_bump.py prerelease --prerelease alpha.1
python cicd/build/version_bump.py prerelease --prerelease beta.1
python cicd/build/version_bump.py prerelease --prerelease rc.1
```

## Deployment Validation

After deployment, validate:

### 1. Package Availability

```bash
# Check PyPI
curl https://pypi.org/pypi/claude-commands/json | jq '.info.version'

# Test installation
pip install --upgrade claude-commands
python -c "import claude_commands; print(claude_commands.__version__)"
```

### 2. Documentation

```bash
# Check docs site
curl -I https://docs.claude-commands.dev

# Validate links
python cicd/docs/link_checker.py
```

### 3. Docker Image

```bash
# Pull and test
docker pull claudecommands/executor:latest
docker run claudecommands/executor:latest --version
```

### 4. Functionality

```bash
# Run smoke tests
bash cicd/deploy/smoke_test_prod.sh

# Run integration tests against deployed version
pytest tests/integration/ --deployed
```

## Monitoring Post-Deployment

Monitor for 24 hours after deployment:

1. **Error Tracking**
   - Check error rates
   - Monitor exception reports
   - Review user feedback

2. **Performance**
   - Monitor response times
   - Check resource usage
   - Verify benchmarks

3. **Usage**
   - Track download statistics
   - Monitor API calls
   - Check feature usage

## Emergency Procedures

### Critical Bug Found

1. **Assess Severity**
   - Security vulnerability: Immediate hotfix
   - Data loss risk: Immediate hotfix
   - Feature broken: Hotfix or next release

2. **Deploy Hotfix**
   - Follow hotfix procedure above
   - Expedite testing
   - Deploy ASAP

3. **Communication**
   - Security advisory if needed
   - Release announcement
   - User notification

### Service Outage

1. **Identify Issue**
   - Check PyPI status
   - Check GitHub status
   - Check Docker Hub status

2. **Mitigation**
   - Use mirrors if available
   - Provide alternative download
   - Update documentation

3. **Resolution**
   - Wait for service recovery
   - Verify functionality
   - Notify users

## Deployment Schedule

### Regular Releases

- **Patch releases**: As needed for bug fixes
- **Minor releases**: Monthly (new features)
- **Major releases**: Annually (breaking changes)

### Maintenance Windows

- **Preferred**: Tuesday-Thursday, 10 AM - 2 PM UTC
- **Avoid**: Fridays, weekends, holidays

## Automation

The CI/CD pipeline automates:

✅ Testing
✅ Building
✅ Security scanning
✅ Quality checks
✅ PyPI deployment
✅ Documentation deployment
✅ Docker image deployment
✅ Release notes generation
✅ Changelog updates

Manual steps required:

❌ Version bumping
❌ Git tagging
❌ Release announcements
❌ Critical decision making

## Security Considerations

### API Tokens

- Store in GitHub Secrets
- Rotate regularly
- Use scoped tokens
- Never commit to repository

### Signing

- Sign git tags: `git tag -s`
- Sign commits: `git commit -S`
- Sign packages: `twine upload --sign`

### Verification

- Verify package checksums
- Check GPG signatures
- Validate Docker image digests

## Documentation

Update before release:

- [ ] README.md
- [ ] CHANGELOG.md
- [ ] API documentation
- [ ] Migration guides (breaking changes)
- [ ] Example code
- [ ] Version compatibility matrix

## Support

For deployment issues:

- **CI/CD**: Check workflow logs
- **PyPI**: Contact PyPI support
- **Docker Hub**: Check Docker Hub status
- **Emergency**: Contact on-call engineer