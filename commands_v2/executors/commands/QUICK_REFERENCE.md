# Command Executor Quick Reference

## Command List

### Documentation & Analysis
| Command | Purpose | Key Options |
|---------|---------|-------------|
| `update-docs` | Generate documentation | `--type`, `--format`, `--publish` |
| `explain-code` | Explain code structure | `--level`, `--focus`, `--export` |
| `reflection` | Project analysis & insights | `--type`, `--export-insights` |

### Code Quality & Refactoring
| Command | Purpose | Key Options |
|---------|---------|-------------|
| `refactor-clean` | AI-powered refactoring | `--patterns`, `--implement` |
| `clean-codebase` | Remove dead code/imports | `--imports`, `--dead-code`, `--duplicates` |
| `check-code-quality` | Quality analysis & scoring | `--analysis`, `--auto-fix` |

### Performance & Optimization
| Command | Purpose | Key Options |
|---------|---------|-------------|
| `optimize` | Performance optimization | `--category`, `--implement` |
| `multi-agent-optimize` | Multi-agent optimization | `--mode`, `--focus`, `--agents` |

### Testing & CI/CD
| Command | Purpose | Key Options |
|---------|---------|-------------|
| `generate-tests` | Generate test suites | `--type`, `--coverage`, `--framework` |
| `run-all-tests` | Execute all tests | `--scope`, `--auto-fix`, `--coverage` |
| `ci-setup` | Setup CI/CD pipeline | `--platform`, `--type`, `--security` |

### Git & GitHub
| Command | Purpose | Key Options |
|---------|---------|-------------|
| `commit` | Smart git commits | `--all`, `--ai-message`, `--push` |
| `fix-github-issue` | Auto-fix GitHub issues | `--auto-fix`, `--draft` |

### Debugging
| Command | Purpose | Key Options |
|---------|---------|-------------|
| `debug` | Debug analysis | `--issue`, `--gpu`, `--auto-fix` |

---

## Common Usage Patterns

### Quick Documentation
```bash
# Generate README
update-docs --type=readme

# Full documentation suite
update-docs --type=all --format=markdown

# API documentation
update-docs --type=api --export
```

### Code Improvement Workflow
```bash
# 1. Check quality
check-code-quality src/ --analysis=basic

# 2. Clean codebase
clean-codebase src/ --imports --dead-code

# 3. Refactor
refactor-clean src/ --patterns=modern --implement

# 4. Optimize
optimize src/ --category=all --implement

# 5. Verify
run-all-tests --coverage
```

### Multi-Agent Analysis
```bash
# Comprehensive optimization
multi-agent-optimize src/ --mode=hybrid --agents=all --implement

# Focused performance review
multi-agent-optimize src/ --focus=performance --agents=scientific

# Security-focused review
multi-agent-optimize src/ --focus=security --agents=engineering
```

### Test Generation & Execution
```bash
# Generate tests
generate-tests mymodule.py --type=all --coverage=80

# Run tests with auto-fix
run-all-tests --auto-fix --coverage

# Generate and run
generate-tests src/ && run-all-tests --coverage
```

---

## Argument Reference

### Common Arguments

#### Agent Selection
```
--agents=auto          # Automatic selection
--agents=core          # Core agents only
--agents=scientific    # Scientific computing agents
--agents=all           # All available agents
```

#### Analysis Depth
```
--analysis=basic       # Basic analysis
--analysis=thorough    # Thorough analysis
--analysis=comprehensive  # Full analysis
```

#### Implementation
```
--implement            # Apply changes
--dry-run             # Preview only
--auto-fix            # Automatic fixes
```

#### Output Format
```
--format=text         # Plain text
--format=markdown     # Markdown format
--format=json         # JSON output
--format=html         # HTML format
```

---

## Result Structure

All executors return:
```python
{
    'success': bool,           # Execution status
    'summary': str,            # Brief summary
    'details': str,            # Detailed information
    # Command-specific fields...
}
```

---

## Exit Codes

- `0` - Success
- `1` - General error
- `130` - User interrupt (Ctrl+C)

---

## File Locations

### Configuration
- Command executors: `/Users/b80985/.claude/commands/executors/commands/`
- Base framework: `/Users/b80985/.claude/commands/executors/`

### Output
- Documentation: `./docs/`
- Test files: `./tests/`
- Reports: `./.claude/reports/`
- Insights: `./.claude/insights/`

---

## Best Practices

### 1. Always Preview First
```bash
# Use --dry-run to preview changes
clean-codebase --dry-run
refactor-clean --report=detailed
```

### 2. Incremental Improvements
```bash
# Start small, verify, then expand
check-code-quality file.py
refactor-clean file.py --implement
run-all-tests file.py
```

### 3. Use Multi-Agent for Complex Tasks
```bash
# Let multiple agents analyze
multi-agent-optimize --mode=hybrid --agents=all
```

### 4. Combine Commands
```bash
# Chain commands for workflows
generate-tests && run-all-tests --coverage && commit --ai-message
```

### 5. Save Important Analysis
```bash
# Export for later review
reflection --export-insights
optimize --format=json --export=report.json
```

---

## Troubleshooting

### Import Errors
- Ensure you're in the correct directory
- Check Python path includes executor directories

### Permission Errors
- Check file permissions
- Ensure write access to target directories

### Out of Memory
- Reduce scope (target fewer files)
- Use `--parallel=false` to reduce memory usage

### Slow Execution
- Use `--agents=core` instead of `--agents=all`
- Limit file count with specific targets
- Use `--analysis=basic` for faster results

---

## Performance Tips

### 1. Target Specific Files
```bash
# Better: Target specific files
optimize src/core/engine.py

# Slower: Process entire directory
optimize src/
```

### 2. Use Appropriate Analysis Level
```bash
# For quick checks
check-code-quality --analysis=basic

# For comprehensive review
check-code-quality --analysis=comprehensive
```

### 3. Parallel Processing
```bash
# Enable parallel execution where supported
clean-codebase --parallel
run-all-tests --parallel
```

### 4. Cache Results
- Results are cached automatically
- Rerun on same files is faster

---

## Integration Examples

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
check-code-quality --auto-fix
run-all-tests
```

### CI Pipeline
```yaml
# .github/workflows/quality.yml
- name: Quality Check
  run: |
    check-code-quality src/ --analysis=thorough
    run-all-tests --coverage
```

### Development Workflow
```bash
# Start of day
reflection --type=session

# During development
refactor-clean . --dry-run
optimize . --category=performance

# End of day
update-docs --type=all
commit --ai-message --push
```

---

## Quick Tips

💡 Use `--help` with any command for detailed options
💡 Start with `--dry-run` to preview changes
💡 Use `--report=detailed` for comprehensive output
💡 Combine `--implement` with caution (creates backups)
💡 Export important analyses with `--export`
💡 Use multi-agent mode for complex analyses
💡 Chain commands with `&&` for workflows

---

## Getting Help

```bash
# List all commands
cli.py --list

# Show by category
cli.py --categories

# Command-specific help
cli.py <command> --help

# Examples
cli.py optimize --help
cli.py multi-agent-optimize --help
```

---

## Summary

This reference covers the 14 implemented command executors. Each provides specialized functionality while maintaining consistent interfaces and integration with the unified framework.

For detailed implementation information, see `IMPLEMENTATION_SUMMARY.md`.