# Quick Start Guide

Get up and running with the Claude Code Command Executor in 5 minutes!

## Welcome!

The Claude Code Command Executor is a powerful framework that uses a 23-agent personal agent system to help you analyze, optimize, and maintain your code. This quick start guide will get you productive immediately.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.7+** installed
- **Git** installed (for version control commands)
- **Claude Code CLI** installed
- Basic familiarity with command line

## Your First Command in 30 Seconds

Let's run your first command to check your code quality:

```bash
# Navigate to your project
cd /path/to/your/project

# Run code quality check (dry-run mode - safe, no changes)
/check-code-quality --dry-run
```

That's it! You've just run your first command with intelligent agent selection.

## What Just Happened?

The command:
1. **Analyzed your codebase** to determine languages and frameworks
2. **Selected optimal agents** from the 23-agent system automatically
3. **Performed quality analysis** across multiple dimensions
4. **Generated a report** with actionable recommendations

All without making any changes to your code (thanks to `--dry-run`).

## Try 3 More Essential Commands

### 1. Optimize Performance

```bash
# Analyze performance and get optimization suggestions
/optimize --profile src/
```

### 2. Generate Documentation

```bash
# Generate comprehensive documentation
/update-docs --type=readme
```

### 3. Clean Your Codebase

```bash
# Remove unused imports and dead code (preview only)
/clean-codebase --imports --dead-code --dry-run
```

## Understanding the Output

Each command provides structured output:

```
✅ Command completed successfully

Summary:
  Files analyzed: 42
  Issues found: 15
  Agents used: 5 (auto-selected)

Recommendations:
  1. [HIGH] Remove 23 unused imports
  2. [MEDIUM] Optimize algorithm complexity in utils.py
  3. [LOW] Add docstrings to 8 functions

Duration: 3.2s
```

## Key Concepts in 2 Minutes

### 1. Agent System

The framework has **23 specialized agents**:
- **Core agents** (5): Essential for any task
- **Scientific agents** (8): For research and scientific computing
- **Engineering agents** (4): For production software
- **Quality agents** (2): For code quality
- **Domain specialists** (4): For specific domains

### 2. Agent Selection

You can select agents explicitly or let the system decide:

```bash
--agents=auto      # Intelligent auto-selection (recommended)
--agents=core      # Essential 5-agent team
--agents=scientific # 8 scientific computing specialists
--agents=all       # All 23 agents with orchestration
```

### 3. Safety Features

Always safe by default:

```bash
--dry-run          # Preview without making changes
--backup           # Create backup before modifications
--interactive      # Confirm each change
--rollback         # Enable rollback capability
```

## Common Workflows

### Workflow 1: Code Quality Improvement

```bash
# Step 1: Check quality
/check-code-quality --auto-fix --dry-run

# Step 2: Apply fixes
/check-code-quality --auto-fix

# Step 3: Verify
/run-all-tests
```

### Workflow 2: Performance Optimization

```bash
# Step 1: Profile and analyze
/optimize --profile --language=python src/

# Step 2: Apply optimizations
/optimize --implement src/

# Step 3: Test and validate
/run-all-tests --benchmark
```

### Workflow 3: Documentation Generation

```bash
# Step 1: Generate all docs
/update-docs --type=all --format=markdown

# Step 2: Review changes
git diff

# Step 3: Commit
/commit --template=docs
```

## Quick Reference Card

### Most Used Commands

| Command | Purpose | Quick Example |
|---------|---------|---------------|
| `/check-code-quality` | Check and fix code quality | `/check-code-quality --auto-fix` |
| `/optimize` | Optimize performance | `/optimize --implement src/` |
| `/clean-codebase` | Remove unused code | `/clean-codebase --imports --dry-run` |
| `/generate-tests` | Create test suites | `/generate-tests --type=unit src/` |
| `/update-docs` | Generate documentation | `/update-docs --type=readme` |
| `/run-all-tests` | Run test suite | `/run-all-tests --coverage` |

### Most Used Flags

| Flag | Purpose | When to Use |
|------|---------|-------------|
| `--dry-run` | Preview only | Always use first |
| `--agents=auto` | Smart agent selection | Most of the time |
| `--implement` | Apply changes | After dry-run review |
| `--backup` | Create backup | Before big changes |
| `--interactive` | Confirm each change | Important files |

## Next Steps

Now that you've run your first commands, here's what to explore next:

### 1. Learn About Agents
- Read: [Understanding Agents](understanding-agents.md)
- Learn how the 23-agent system works
- Understand when to use which agents

### 2. Explore Common Workflows
- Read: [Common Workflows](common-workflows.md)
- Learn patterns for typical tasks
- See real-world examples

### 3. Try a Tutorial
- Start with: [Tutorial 01: Code Quality](../tutorials/tutorial-01-code-quality.md)
- Follow step-by-step guides
- Build hands-on experience

### 4. Deep Dive into Commands
- Explore: [Command Reference](../guides/command-reference.md)
- Learn all options and features
- Discover advanced capabilities

## Quick Tips for Success

### Tip 1: Always Dry-Run First
```bash
# Good practice
/clean-codebase --imports --dry-run    # Preview first
/clean-codebase --imports              # Then execute

# Risky
/clean-codebase --imports              # Direct execution
```

### Tip 2: Use Auto Agent Selection
```bash
# Let the system choose optimal agents
/optimize --agents=auto src/

# The system analyzes your code and selects:
# - Python project → scientific-computing-master + code-quality-master
# - Web app → fullstack-developer + systems-architect
# - Research code → research-intelligence-master + scientific agents
```

### Tip 3: Combine Commands
```bash
# Quality improvement pipeline
/check-code-quality --auto-fix && \
/optimize --implement && \
/run-all-tests --coverage
```

### Tip 4: Use Interactive Mode for Important Files
```bash
# Review each change before applying
/refactor-clean --patterns=modern --interactive
```

### Tip 5: Enable Backup for Major Changes
```bash
# Safe refactoring with backup
/refactor-clean --patterns=modern --backup --rollback
```

## Common Questions

### Q: Is it safe to use on my production code?
**A:** Yes! The framework is designed with safety in mind:
- Always use `--dry-run` first to preview changes
- Enable `--backup` for automatic backups
- Use `--rollback` capability for reversing changes
- Start with `--interactive` mode for critical files

### Q: Which agents should I use?
**A:** Start with `--agents=auto` for intelligent selection:
- The system analyzes your codebase
- Selects optimal agents based on languages, frameworks, and patterns
- You can override with specific agent groups when needed

### Q: How long do commands take?
**A:** Depends on codebase size:
- Small project (< 1000 files): 1-5 seconds
- Medium project (1000-10000 files): 5-30 seconds
- Large project (> 10000 files): 30-120 seconds
- Use `--parallel` flag to speed up large projects

### Q: Can I use this in CI/CD?
**A:** Absolutely! Many commands support CI/CD integration:
```bash
# Example GitHub Actions integration
- name: Code Quality Check
  run: /check-code-quality --format=json --report
```

### Q: What if something goes wrong?
**A:** Multiple safety nets:
1. Use `--dry-run` to preview (no changes made)
2. Automatic backups with `--backup` flag
3. Rollback capability with `--rollback` flag
4. All changes logged in `~/.claude/logs/`

## Troubleshooting

### Issue: Command not found
**Solution:** Ensure Claude Code CLI is installed and in your PATH.

### Issue: Permission denied
**Solution:** Check file permissions or run with appropriate privileges.

### Issue: Agent selection takes too long
**Solution:** Use `--agents=core` for faster analysis with essential agents.

### Issue: Out of memory
**Solution:** Process smaller directories or use `--parallel=false` to reduce memory usage.

## Getting Help

If you need assistance:

1. **Check the Troubleshooting Guide**: [../guides/troubleshooting.md](../guides/troubleshooting.md)
2. **Read Command Documentation**: [../guides/command-reference.md](../guides/command-reference.md)
3. **View Examples**: [../examples/](../examples/)
4. **Consult the Glossary**: [../reference/glossary.md](../reference/glossary.md)

## What's Next?

You're now ready to explore the full power of the framework:

- **[Installation Guide](installation.md)** - Complete installation and configuration
- **[First Commands](first-commands.md)** - Detailed walkthrough of essential commands
- **[Understanding Agents](understanding-agents.md)** - Deep dive into the 23-agent system
- **[Common Workflows](common-workflows.md)** - Patterns for typical development tasks
- **[Tutorials](../tutorials/)** - Step-by-step guides for specific tasks

---

**Congratulations!** You've completed the quick start guide. You're now ready to leverage the power of the 23-agent personal agent system for your development workflows.

Happy coding! 🚀