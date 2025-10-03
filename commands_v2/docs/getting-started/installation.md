# Installation Guide

Complete installation and configuration instructions for the Claude Code Command Executor Framework.

## System Requirements

### Minimum Requirements
- **Python**: 3.7 or higher
- **Memory**: 4GB RAM
- **Disk Space**: 500MB
- **Operating System**: Linux, macOS, or Windows (with WSL)

### Recommended Requirements
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM
- **Disk Space**: 2GB (for cache)
- **Operating System**: Linux or macOS

## Installation Steps

### 1. Install Claude Code CLI

The command executor framework is part of Claude Code CLI. Install it following the official Claude Code documentation.

### 2. Verify Installation

```bash
# Check Claude Code version
claude --version

# Verify commands are available
/check-code-quality --help
```

### 3. Initialize Framework

```bash
# Framework initializes automatically on first use
# Creates ~/.claude directory with cache and logs
/check-code-quality --dry-run .
```

## Configuration

### Directory Structure

The framework creates the following structure:

```
~/.claude/
├── cache/           # Multi-level cache
│   ├── ast/        # AST cache (24-hour TTL)
│   ├── analysis/   # Analysis cache (7-day TTL)
│   └── agent/      # Agent cache (7-day TTL)
├── logs/           # Command execution logs
├── backups/        # Automatic backups
└── config/         # Configuration files
```

### Configuration File (Optional)

Create `~/.claude/config/executor.json` for custom settings:

```json
{
  "cache": {
    "enabled": true,
    "ast_ttl_hours": 24,
    "analysis_ttl_days": 7,
    "max_size_mb": 1000
  },
  "agents": {
    "default_selection": "auto",
    "max_parallel": 4
  },
  "execution": {
    "dry_run_default": false,
    "backup_enabled": true,
    "parallel_workers": 4
  },
  "logging": {
    "level": "INFO",
    "max_log_size_mb": 100,
    "keep_days": 30
  }
}
```

## Verification

### Test Basic Functionality

```bash
# Test code quality command
cd /path/to/project
/check-code-quality --dry-run

# Test optimization command
/optimize --profile --dry-run src/

# Test documentation command
/update-docs --type=readme --dry-run
```

### Test Agent System

```bash
# Test auto agent selection
/optimize --agents=auto --dry-run

# Test specific agent groups
/optimize --agents=scientific --dry-run
/optimize --agents=engineering --dry-run
```

### Test Safety Features

```bash
# Test dry-run
/clean-codebase --imports --dry-run

# Test backup
/refactor-clean --patterns=modern --backup --dry-run

# Test interactive
/clean-codebase --imports --interactive --dry-run
```

## Troubleshooting

### Issue: Command not found

**Symptom:** `/check-code-quality: command not found`

**Solution:**
```bash
# Verify Claude Code is installed
claude --version

# Check PATH
echo $PATH

# Reinstall if necessary
```

### Issue: Permission denied

**Symptom:** `Permission denied: ~/.claude/cache`

**Solution:**
```bash
# Fix permissions
chmod -R 755 ~/.claude

# Or recreate directory
rm -rf ~/.claude
mkdir -p ~/.claude
```

### Issue: Cache errors

**Symptom:** `Cache read error` or `Cache write error`

**Solution:**
```bash
# Clear cache
rm -rf ~/.claude/cache/*

# Framework will rebuild cache automatically
```

### Issue: Out of memory

**Symptom:** Process killed or memory errors

**Solution:**
```bash
# Reduce parallel workers
/optimize --parallel=false

# Process smaller directories
/optimize src/specific-module/

# Disable cache temporarily
# Edit ~/.claude/config/executor.json:
# "cache": {"enabled": false}
```

## Optional Dependencies

### For Python Projects

```bash
# Code quality tools
pip install pylint black mypy flake8

# Testing tools
pip install pytest pytest-cov

# Performance tools
pip install py-spy memory-profiler
```

### For Scientific Computing

```bash
# Scientific packages
pip install numpy scipy pandas

# JAX ecosystem
pip install jax jaxlib flax optax

# Visualization
pip install matplotlib plotly
```

### For Web Development

```bash
# Node.js and npm
# Install from: https://nodejs.org/

# Linting tools
npm install -g eslint prettier
```

## Performance Tuning

### Cache Configuration

Adjust cache settings in `~/.claude/config/executor.json`:

```json
{
  "cache": {
    "ast_ttl_hours": 48,        # Increase for stable code
    "analysis_ttl_days": 14,    # Increase for slow analysis
    "max_size_mb": 2000         # Increase for large projects
  }
}
```

### Parallel Execution

Configure parallelism:

```json
{
  "execution": {
    "parallel_workers": 8       # Adjust based on CPU cores
  }
}
```

### Memory Optimization

For large projects:

```json
{
  "execution": {
    "chunk_size": 100,          # Process files in chunks
    "memory_limit_mb": 4000     # Set memory limit
  }
}
```

## Updating

### Update Framework

Framework updates are included in Claude Code CLI updates:

```bash
# Update Claude Code CLI
# Follow official Claude Code update instructions

# Verify new version
claude --version
/check-code-quality --version
```

### Migrate Configuration

After major updates:

```bash
# Backup current config
cp -r ~/.claude ~/.claude.backup

# Framework will create new config on first run
# Merge your settings if needed
```

## Uninstallation

### Remove Framework Data

```bash
# Remove all framework data
rm -rf ~/.claude

# Keep logs and backups, remove cache only
rm -rf ~/.claude/cache
```

### Remove Configuration

```bash
# Remove configuration
rm -rf ~/.claude/config
```

## Next Steps

- **[Quick Start Guide](README.md)** - Get started in 5 minutes
- **[First Commands](first-commands.md)** - Try essential commands
- **[Understanding Agents](understanding-agents.md)** - Learn the agent system
- **[Common Workflows](common-workflows.md)** - Typical usage patterns

---

**Installation complete!** → [Quick Start Guide](README.md)