# Troubleshooting Guide

> Solutions to common issues and problems

## Quick Diagnostics

```bash
# Check system status
claude-commands status

# Run diagnostics
claude-commands diagnostics

# Check cache
claude-commands cache-status

# View logs
claude-commands logs --tail=100
```

---

## Common Issues

### Installation & Setup

#### Command Not Found
**Symptom:** `command not found: /command-name`

**Solutions:**
1. Commands are slash commands in Claude Code CLI - use `/command-name` format
2. Verify installation: check that commands exist in `~/.claude/commands/`
3. Restart Claude Code CLI

#### Permission Errors
**Symptom:** `Permission denied` when running commands

**Solutions:**
```bash
# Check directory permissions
ls -la ~/.claude/commands/

# Fix permissions
chmod +x ~/.claude/commands/*.sh
chmod +x ~/.claude/commands/*.py

# Ensure you're in a writable directory
cd /path/to/your/project
```

#### Import Errors
**Symptom:** `ModuleNotFoundError` or `ImportError`

**Solutions:**
```bash
# Install requirements
pip install -r ~/.claude/commands/requirements.txt

# Check Python version (3.9+ required)
python --version

# Verify pip installation
pip list | grep claude

# Reinstall if needed
pip install --upgrade --force-reinstall -r requirements.txt
```

---

### Command Execution

#### Command Hangs or Timeout
**Symptom:** Command runs indefinitely or times out

**Solutions:**
```bash
# Increase timeout
/command-name --timeout=600

# Check system resources
top
htop

# Clear cache
claude-commands cache-clear

# Run with debug mode
/command-name --debug
```

#### Auto-Fix Not Working
**Symptom:** `--auto-fix` flag doesn't fix issues

**Possible causes:**
1. Issues require manual intervention
2. Permission issues
3. Complex refactoring needed

**Solutions:**
```bash
# Check what was attempted
/check-code-quality --auto-fix --verbose

# Review unfixed issues
/double-check --report "review unfixed issues"

# Manual fix with guidance
/check-code-quality --interactive
```

#### Agent Selection Issues
**Symptom:** Wrong agents selected or agent errors

**Solutions:**
```bash
# Explicit agent selection
/command-name --agents=scientific

# Use intelligent selection
/multi-agent-optimize --intelligent

# Check available agents
claude-commands list-agents

# Debug agent selection
/command-name --agents=auto --debug
```

---

### Testing Issues

#### Tests Failing After Generation
**Symptom:** Generated tests fail immediately

**Solutions:**
```bash
# Auto-fix test failures
/run-all-tests --auto-fix

# Regenerate with more context
/generate-tests --type=unit --coverage=90 --context

# Debug specific test
/debug --issue=test --auto-fix

# Check test configuration
pytest --collect-only
```

#### Coverage Not Meeting Target
**Symptom:** Coverage below expected threshold

**Solutions:**
```bash
# Generate missing tests
/generate-tests --coverage=90 --focus=uncovered

# Check current coverage
/run-all-tests --coverage --report

# Identify uncovered code
pytest --cov --cov-report=html
# Open htmlcov/index.html

# Generate targeted tests
/generate-tests --target=path/to/uncovered/file.py
```

#### Test Performance Issues
**Symptom:** Tests run very slowly

**Solutions:**
```bash
# Profile test execution
/run-all-tests --profile

# Run parallel
/run-all-tests --parallel --workers=4

# Run specific scope
/run-all-tests --scope=unit

# Skip slow tests
pytest -m "not slow"
```

---

### Performance Issues

#### Slow Command Execution
**Symptom:** Commands take too long to execute

**Solutions:**
```bash
# Enable parallel execution
/command-name --parallel

# Check cache status
claude-commands cache-status

# Clear stale cache
claude-commands cache-clear

# Profile execution
/command-name --profile --report

# Reduce scope
/command-name --scope=specific/directory
```

#### High Memory Usage
**Symptom:** System runs out of memory

**Solutions:**
```bash
# Limit memory
export CLAUDE_COMMANDS_MEMORY_LIMIT=4G

# Process in batches
/command-name --batch-size=100

# Clear cache
claude-commands cache-clear

# Reduce parallelism
/command-name --parallel=false

# Monitor memory
/command-name --monitor
```

#### Cache Issues
**Symptom:** Stale results or cache errors

**Solutions:**
```bash
# Clear cache
claude-commands cache-clear

# Disable cache temporarily
/command-name --no-cache

# Check cache stats
claude-commands cache-stats

# Rebuild cache
claude-commands cache-rebuild
```

---

### Workflow Issues

#### Workflow Fails Mid-Execution
**Symptom:** Workflow stops at specific step

**Solutions:**
```bash
# Check step configuration
cat workflow.yml

# Run with error handling
# Edit workflow.yml:
steps:
  - name: problematic-step
    on_failure: continue  # or retry

# Debug specific step
/command-name --debug

# Skip failed step
# Edit workflow to add condition
```

#### Workflow Parameter Issues
**Symptom:** Parameters not being passed correctly

**Solutions:**
```yaml
# Check parameter syntax in workflow
parameters:
  target:
    type: string
    required: true

steps:
  - name: step1
    args:
      target: ${target}  # Use ${} for parameters
```

#### Circular Dependencies
**Symptom:** Workflow detects circular dependencies

**Solutions:**
1. Review `depends_on` relationships
2. Remove circular references
3. Reorganize workflow steps
```yaml
# Bad
- name: step1
  depends_on: [step2]
- name: step2
  depends_on: [step1]

# Good
- name: step1
- name: step2
  depends_on: [step1]
```

---

### Plugin Issues

#### Plugin Won't Load
**Symptom:** Plugin fails to load

**Solutions:**
```bash
# Check plugin structure
ls -la plugins/my-plugin/

# Verify plugin.yml
cat plugins/my-plugin/plugin.yml

# Check dependencies
pip install -r plugins/my-plugin/requirements.txt

# Load with debug
claude-commands load-plugin my-plugin --debug

# Check logs
claude-commands logs --grep="plugin"
```

#### Plugin Command Not Available
**Symptom:** Plugin command not recognized

**Solutions:**
```bash
# Enable plugin
claude-commands enable-plugin my-plugin

# Verify registration
claude-commands list-commands | grep my-command

# Reload plugins
claude-commands reload-plugins

# Check plugin status
claude-commands plugin-status my-plugin
```

#### Plugin Conflicts
**Symptom:** Multiple plugins interfere with each other

**Solutions:**
```bash
# Disable conflicting plugin
claude-commands disable-plugin conflicting-plugin

# Check plugin priority
claude-commands plugin-priority

# Adjust load order
claude-commands set-plugin-priority my-plugin 10

# Uninstall problematic plugin
claude-commands uninstall-plugin conflicting-plugin
```

---

### CI/CD Issues

#### GitHub Actions Failing
**Symptom:** CI pipeline fails

**Solutions:**
```bash
# Analyze errors
/fix-commit-errors --auto-fix <commit-hash>

# Check workflow file
cat .github/workflows/ci.yml

# Test locally
act  # GitHub Actions local testing

# Debug specific job
/fix-commit-errors --debug --emergency

# Review logs
gh run view <run-id> --log
```

#### Pre-commit Hooks Failing
**Symptom:** Commits blocked by pre-commit

**Solutions:**
```bash
# Fix issues
/check-code-quality --auto-fix

# Run pre-commit
pre-commit run --all-files

# Skip if necessary (not recommended)
git commit --no-verify

# Update hooks
pre-commit autoupdate
```

#### Deployment Failures
**Symptom:** Deployment step fails

**Solutions:**
```bash
# Check deployment config
/ci-setup --platform=github --deploy=production

# Review deployment logs
gh run view <run-id> --log

# Emergency fix
/fix-commit-errors --emergency --auto-fix

# Rollback if needed
git revert HEAD
git push
```

---

### Scientific Computing Issues

#### GPU Not Detected
**Symptom:** GPU commands fail or don't detect GPU

**Solutions:**
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Install GPU support
pip install jax[cuda11] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Test GPU
python -c "import jax; print(jax.devices())"

# Force GPU usage
/optimize --gpu --force
```

#### Fortran Code Adoption Failing
**Symptom:** `/adopt-code` fails on Fortran code

**Solutions:**
```bash
# Check Fortran compiler
gfortran --version

# Install dependencies
pip install numpy f2py

# Analyze first
/adopt-code --analyze --language=fortran legacy/

# Review analysis before integration
/adopt-code --integrate --language=fortran --target=python

# Manual intervention may be needed
/adopt-code --interactive
```

#### Numerical Precision Issues
**Symptom:** Scientific calculations produce incorrect results

**Solutions:**
```bash
# Check precision settings
/optimize --category=algorithm --precision=double

# Use arbitrary precision
pip install mpmath

# Verify numerical stability
/debug --issue=numerical --profile

# Generate numerical tests
/generate-tests --type=scientific --reproducible
```

---

### Language-Specific Issues

#### Python Type Errors
**Symptom:** Type checking fails

**Solutions:**
```bash
# Install type stubs
pip install types-all

# Run type checker
mypy .

# Fix type issues
/check-code-quality --auto-fix --focus=types

# Add type ignores if needed (last resort)
# Add # type: ignore to line
```

#### Julia Package Issues
**Symptom:** Julia commands fail

**Solutions:**
```bash
# Update Julia packages
julia -e 'using Pkg; Pkg.update()'

# Rebuild packages
julia -e 'using Pkg; Pkg.build()'

# Check Julia version
julia --version

# Install required packages
julia -e 'using Pkg; Pkg.add(["Package1", "Package2"])'
```

#### JavaScript/TypeScript Issues
**Symptom:** JS/TS analysis fails

**Solutions:**
```bash
# Install dependencies
npm install

# Check Node version
node --version

# Install TypeScript
npm install -g typescript

# Check configuration
cat tsconfig.json

# Fix JS issues
/check-code-quality --language=javascript --auto-fix
```

---

## Error Messages

### Common Error Messages and Solutions

#### "No agents available for task"
**Solution:**
```bash
# Install required plugins
claude-commands install scientific-computing-plugin

# Use different agent selection
/command-name --agents=core

# Check agent availability
claude-commands list-agents
```

#### "Execution timeout exceeded"
**Solution:**
```bash
# Increase timeout
/command-name --timeout=600

# Reduce scope
/command-name --scope=smaller/directory

# Use parallel execution
/command-name --parallel
```

#### "Insufficient permissions"
**Solution:**
```bash
# Check file permissions
ls -la file

# Change permissions
chmod u+w file

# Run from correct directory
cd /correct/directory
```

#### "Cache corruption detected"
**Solution:**
```bash
# Clear cache
claude-commands cache-clear

# Rebuild cache
claude-commands cache-rebuild

# Disable cache if persistent
export CLAUDE_COMMANDS_CACHE=false
```

---

## Debug Mode

### Enable Debug Output

```bash
# Global debug
export CLAUDE_COMMANDS_DEBUG=true

# Command-specific debug
/command-name --debug

# Verbose output
/command-name --verbose

# Debug with logging
/command-name --debug --log-file=debug.log
```

### Debug Information

```bash
# System info
claude-commands system-info

# Diagnostics
claude-commands diagnostics

# Agent status
claude-commands agent-status

# Cache status
claude-commands cache-status

# Plugin status
claude-commands plugin-status
```

---

## Getting Help

### Documentation
- **[User Guide](USER_GUIDE.md)** - Complete user documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Development docs
- **[API Reference](API_REFERENCE.md)** - API documentation
- **[FAQ](FAQ.md)** - Frequently asked questions

### Support Channels
1. Check documentation first
2. Search existing issues on GitHub
3. Run diagnostics: `claude-commands diagnostics`
4. Create GitHub issue with:
   - Error message
   - Command used
   - System info
   - Debug output

### Diagnostic Information to Include

When reporting issues, include:

```bash
# System information
claude-commands system-info

# Diagnostics
claude-commands diagnostics

# Error logs
claude-commands logs --tail=100

# Command output
/command-name --debug 2>&1 | tee debug.log

# Environment
env | grep CLAUDE
```

---

## Performance Tuning

### Optimization Tips

```bash
# Enable caching
export CLAUDE_COMMANDS_CACHE=true

# Parallel execution
/command-name --parallel --workers=8

# Selective processing
/command-name --target=specific/path

# Batch processing
/command-name --batch-size=100

# Memory limit
export CLAUDE_COMMANDS_MEMORY_LIMIT=8G
```

---

## Emergency Procedures

### System Unresponsive

```bash
# Kill hung processes
ps aux | grep claude-commands
kill -9 <PID>

# Clear cache
rm -rf ~/.claude-commands/cache/

# Reset configuration
mv ~/.claude-commands/config.yml ~/.claude-commands/config.yml.bak
```

### Corrupted State

```bash
# Reset system
claude-commands reset

# Reinstall
pip uninstall claude-commands
pip install claude-commands

# Clear all data
rm -rf ~/.claude-commands/
```

### Emergency Rollback

```bash
# Git rollback
git reset --hard HEAD~1
git push --force

# Restore from backup
cp backup/code/* .
```

---

**Version**: 1.0.0 | **Last Updated**: September 2025

Need more help? Check the [FAQ](FAQ.md) or [User Guide](USER_GUIDE.md)