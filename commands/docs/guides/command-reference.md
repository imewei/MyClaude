# Command Reference

Complete reference for all 14 commands in the Claude Code Command Executor Framework.

## Quick Command Index

| Command | Purpose | Quick Example |
|---------|---------|---------------|
| [check-code-quality](#check-code-quality) | Code quality analysis and fixing | `/check-code-quality --auto-fix` |
| [optimize](#optimize) | Performance optimization | `/optimize --implement src/` |
| [clean-codebase](#clean-codebase) | Remove unused code | `/clean-codebase --imports --dead-code` |
| [generate-tests](#generate-tests) | Generate test suites | `/generate-tests src/ --type=all` |
| [update-docs](#update-docs) | Generate documentation | `/update-docs --type=readme` |
| [refactor-clean](#refactor-clean) | Code refactoring | `/refactor-clean --patterns=modern` |
| [run-all-tests](#run-all-tests) | Execute test suite | `/run-all-tests --coverage` |
| [commit](#commit) | Smart git commits | `/commit --ai-message` |
| [fix-commit-errors](#fix-commit-errors) | Fix CI/CD errors | `/fix-commit-errors --auto-fix` |
| [fix-github-issue](#fix-github-issue) | Fix GitHub issues | `/fix-github-issue 123` |
| [ci-setup](#ci-setup) | Setup CI/CD | `/ci-setup --platform=github` |
| [debug](#debug) | Debug code | `/debug --auto-fix` |
| [multi-agent-optimize](#multi-agent-optimize) | Multi-agent optimization | `/multi-agent-optimize --mode=review` |
| [think-ultra](#think-ultra) | Advanced analysis | `/think-ultra "problem" --depth=ultra` |

## Common Flags

These flags work across most commands:

| Flag | Purpose | Commands |
|------|---------|----------|
| `--dry-run` | Preview without changes | All |
| `--agents=<type>` | Select agent group | All |
| `--backup` | Create backup | Modification commands |
| `--rollback` | Enable rollback | Modification commands |
| `--interactive` | Confirm each change | Modification commands |
| `--orchestrate` | Enable orchestration | Multi-agent commands |
| `--intelligent` | Smart agent selection | All |
| `--parallel` | Parallel execution | Analysis commands |
| `--implement` | Apply changes | Analysis commands |
| `--auto-fix` | Automatic fixes | Quality commands |

## check-code-quality

**Purpose:** Analyze and improve code quality.

### Syntax

```bash
/check-code-quality [options] [target]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--language` | python\|javascript\|java\|auto | auto | Target language |
| `--analysis` | basic\|thorough\|comprehensive | thorough | Analysis depth |
| `--auto-fix` | - | false | Apply automatic fixes |
| `--agents` | auto\|core\|quality\|all | auto | Agent selection |
| `--format` | text\|json\|html | text | Output format |

### Examples

```bash
# Basic quality check
/check-code-quality src/

# Fix issues automatically
/check-code-quality --auto-fix

# Comprehensive analysis
/check-code-quality --analysis=comprehensive --agents=all
```

### See Also
- [Full Guide](../commands/check-quality-guide.md)
- [Tutorial](../tutorials/tutorial-01-code-quality.md)

## optimize

**Purpose:** Optimize code performance.

### Syntax

```bash
/optimize [options] [target]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--language` | python\|julia\|jax\|auto | auto | Target language |
| `--category` | all\|algorithm\|memory\|io | all | Optimization category |
| `--profile` | - | false | Include profiling |
| `--implement` | - | false | Apply optimizations |
| `--agents` | auto\|scientific\|ai\|all | auto | Agent selection |

### Examples

```bash
# Profile and analyze
/optimize --profile src/

# Apply optimizations
/optimize --implement --category=algorithm

# Scientific code optimization
/optimize --agents=scientific --language=python
```

### See Also
- [Full Guide](../commands/optimize-guide.md)
- [Tutorial](../tutorials/tutorial-02-optimization.md)

## clean-codebase

**Purpose:** Remove unused code, imports, and duplicates.

### Syntax

```bash
/clean-codebase [options] [path]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--analysis` | basic\|thorough\|comprehensive\|ultrathink | thorough | Analysis depth |
| `--imports` | - | false | Remove unused imports |
| `--dead-code` | - | false | Remove dead code |
| `--duplicates` | - | false | Remove duplicates |
| `--ast-deep` | - | false | Deep AST analysis |
| `--agents` | auto\|core\|quality\|all | auto | Agent selection |

### Examples

```bash
# Remove unused imports
/clean-codebase --imports --dry-run

# Complete cleanup
/clean-codebase --imports --dead-code --duplicates --backup

# Ultrathink analysis
/clean-codebase --analysis=ultrathink --agents=all
```

### See Also
- [Full Guide](../commands/clean-codebase-guide.md)
- [Tutorial](../tutorials/tutorial-05-refactoring.md)

## generate-tests

**Purpose:** Generate comprehensive test suites.

### Syntax

```bash
/generate-tests [options] [target]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--type` | all\|unit\|integration\|performance | all | Test type |
| `--framework` | auto\|pytest\|jest\|junit | auto | Test framework |
| `--coverage` | 0-100 | 90 | Target coverage |
| `--agents` | auto\|quality\|scientific\|all | auto | Agent selection |

### Examples

```bash
# Generate unit tests
/generate-tests src/ --type=unit

# High coverage
/generate-tests src/ --coverage=95

# Scientific tests
/generate-tests src/ --type=scientific --agents=scientific
```

### See Also
- [Full Guide](../commands/generate-tests-guide.md)
- [Tutorial](../tutorials/tutorial-04-testing.md)

## update-docs

**Purpose:** Generate comprehensive documentation.

### Syntax

```bash
/update-docs [options]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--type` | readme\|api\|research\|all | readme | Documentation type |
| `--format` | markdown\|html\|latex | markdown | Output format |
| `--agents` | auto\|documentation\|all | auto | Agent selection |

### Examples

```bash
# Generate README
/update-docs --type=readme

# Complete documentation
/update-docs --type=all

# Research docs
/update-docs --type=research --format=latex
```

### See Also
- [Full Guide](../commands/update-docs-guide.md)
- [Tutorial](../tutorials/tutorial-03-documentation.md)

## run-all-tests

**Purpose:** Execute comprehensive test suite.

### Syntax

```bash
/run-all-tests [options]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--scope` | all\|unit\|integration | all | Test scope |
| `--coverage` | - | false | Generate coverage |
| `--benchmark` | - | false | Run benchmarks |
| `--auto-fix` | - | false | Fix failing tests |
| `--profile` | - | false | Profile performance |

### Examples

```bash
# Run with coverage
/run-all-tests --coverage

# Benchmark performance
/run-all-tests --benchmark --profile

# Auto-fix failures
/run-all-tests --auto-fix
```

### See Also
- [Full Guide](../commands/run-all-tests-guide.md)
- [Tutorial](../tutorials/tutorial-04-testing.md)

## Additional Commands

For detailed documentation on the remaining commands:

- **[refactor-clean](../commands/refactor-clean-guide.md)** - Code refactoring
- **[commit](../commands/commit-guide.md)** - Smart git commits
- **[fix-commit-errors](../commands/fix-commit-errors-guide.md)** - Fix CI/CD
- **[fix-github-issue](../commands/fix-github-issue-guide.md)** - Fix issues
- **[ci-setup](../commands/ci-setup-guide.md)** - Setup CI/CD
- **[debug](../commands/debug-guide.md)** - Debug code
- **[multi-agent-optimize](../commands/multi-agent-optimize-guide.md)** - Multi-agent
- **[think-ultra](../commands/think-ultra-guide.md)** - Advanced analysis

## Universal Patterns

### Standard Workflow

```bash
# 1. Preview
command --dry-run

# 2. Review output

# 3. Execute
command
```

### Safety-First Approach

```bash
# Maximum safety
command --dry-run --backup --rollback --interactive
```

### Agent Selection Pattern

```bash
# Auto (recommended)
command --agents=auto

# Specific domain
command --agents=scientific  # or engineering, ai, etc.

# Comprehensive
command --agents=all --orchestrate
```

### Combining Commands

```bash
# Sequential pipeline
/check-code-quality --auto-fix && \
/optimize --implement && \
/run-all-tests
```

## Flag Combinations

### Recommended Combinations

```bash
# Safe modification
--dry-run --backup --interactive

# Comprehensive analysis
--agents=all --orchestrate --intelligent

# Quick check
--agents=core --quick

# Scientific code
--agents=scientific --language=python

# Production-safe
--backup --rollback --validate
```

## Output Formats

### Text Format (Default)

Human-readable output with formatting.

### JSON Format

```bash
command --format=json > output.json
```

Structured data for programmatic processing.

### HTML Format

```bash
command --format=html > report.html
```

Interactive reports with visualizations.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Validation error |
| 130 | Interrupted (Ctrl+C) |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `DEBUG=1` | Enable debug output |
| `CLAUDE_CONFIG` | Config file path |
| `CLAUDE_CACHE_DIR` | Cache directory |

## Next Steps

- **[Agent Selection Guide](agent-selection-guide.md)** - Choose agents
- **[Workflow Patterns](workflow-patterns.md)** - Best practices
- **[Troubleshooting](troubleshooting.md)** - Common issues
- **[Examples](../examples/)** - Real-world examples

---

**Need help?** → [Troubleshooting Guide](troubleshooting.md)