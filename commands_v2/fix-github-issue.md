---
title: "Fix GitHub Issue"
description: "GitHub issue analysis and automated fixing tool with PR creation"
category: github-workflow
subcategory: issue-automation
complexity: advanced
argument-hint: "[--implement] [--auto-fix] [--draft] [--interactive] [--emergency] [--branch=name] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--dry-run] [--backup] [--rollback] [--intelligent] [--orchestrate] [--parallel] [--validate] [issue-number-or-url]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, TodoWrite, MultiEdit, WebSearch, WebFetch
model: inherit
tags: github-issues, automation, pull-requests, issue-resolution
dependencies: []
related: [fix-commit-errors, commit, debug, check-code-quality, generate-tests, run-all-tests, multi-agent-optimize]
workflows: [issue-resolution, automated-pr-creation, bug-fixing]
version: "2.1"
last-updated: "2025-09-29"
---

# Fix GitHub Issue

Analyze GitHub issues and apply automated fixes with pull request creation.

## Quick Start

```bash
# Analyze and fix issue automatically
/fix-github-issue --auto-fix 123

# Create draft PR for review
/fix-github-issue --auto-fix --draft 123

# Interactive mode with user guidance
/fix-github-issue --interactive 123

# Emergency mode for urgent issues
/fix-github-issue --emergency 123
```

## Options

| Option | Description |
|--------|-------------|
| `--implement` | Apply fixes automatically and create PR (primary flag) |
| `--auto-fix` | Alias for --implement (for backward compatibility) |
| `--draft` | Create draft PR instead of regular PR |
| `--branch=<name>` | Specify custom branch name for fixes |
| `--interactive` | Interactive mode with step-by-step guidance |
| `--emergency` | Emergency rapid resolution mode |
| `--debug` | Enable verbose debugging output |
| `--agents=<agents>` | Agent selection (auto, core, scientific, engineering, ai, domain, quality, research, all) |
| `--dry-run` | Preview fixes without applying them |
| `--intelligent` | Enable intelligent agent selection based on issue analysis |
| `--orchestrate` | Enable advanced 23-agent orchestration for complex issues |
| `--parallel` | Run analysis in parallel for maximum efficiency |
| `--validate` | Run tests and validation before creating PR |
| `--backup` | Create backup before applying fixes |
| `--rollback` | Enable rollback capability for failed fixes |

## Agent Integration

### Quality Agent (`code-quality-master`)
- **Issue Analysis**: Comprehensive code quality analysis and problem identification
- **Root Cause Investigation**: Systematic debugging and issue pattern analysis
- **Code Fix Implementation**: Quality-focused code changes and improvements
- **Testing Integration**: Automated test generation and validation for fixes
- **Performance Analysis**: Performance issue identification and optimization

### DevOps Security Agent (`devops-security-engineer`)
- **Security Issue Resolution**: Security vulnerability analysis and automated fixes
- **CI/CD Integration**: Pipeline issue diagnosis and resolution
- **Infrastructure Issues**: Infrastructure-as-code problem resolution
- **Compliance Fixes**: Regulatory compliance and audit issue resolution
- **Deployment Issues**: Deployment and release problem analysis and fixes

### Multi-Agent Orchestrator (`multi-agent-orchestrator`)
- **Issue Resolution Workflow**: Complex issue resolution pipeline coordination
- **Resource Management**: Intelligent allocation of issue resolution tasks
- **Multi-Stage Resolution**: Coordinated fix implementation and validation
- **Quality Assurance**: Automated testing and verification of issue fixes
- **Process Monitoring**: Issue resolution efficiency and success tracking

## Agent Selection Options

- `auto` - Intelligent agent selection based on issue type and complexity
- `core` - Essential multi-agent team for common issues
- `scientific` - Scientific computing and research code issues
- `engineering` - Software engineering and architecture issues
- `ai` - AI/ML system and pipeline issues
- `domain` - Domain-specific specialized issues
- `quality` - Quality engineering focus for code quality and testing issues
- `research` - Research methodology and innovation issues
- `all` - Complete 23-agent issue resolution system with comprehensive expertise

## Usage Examples

```bash
# Basic issue fixing
/fix-github-issue --auto-fix 42

# Using issue URL
/fix-github-issue --auto-fix https://github.com/user/repo/issues/42

# Create draft PR for review
/fix-github-issue --auto-fix --draft 42

# Custom branch name
/fix-github-issue --auto-fix --branch=hotfix-memory-leak 42

# Interactive mode for complex issues
/fix-github-issue --interactive 42

# Emergency production fix with DevOps agent
/fix-github-issue --emergency --agents=devops 42

# Multi-agent comprehensive issue resolution
/fix-github-issue --auto-fix --agents=all --draft 42

# Quality-focused issue analysis and fix
/fix-github-issue --interactive --agents=quality --branch=quality-fix 42
```

## Core Features

- **Issue Analysis**: Automatic categorization and root cause investigation
- **Fix Discovery**: Codebase analysis to identify potential solutions
- **Automated Resolution**: Apply fixes with validation and testing
- **PR Creation**: Generate pull requests with proper descriptions
- **Branch Management**: Custom branch creation and cleanup
- **Multiple Modes**: Interactive, automatic, and emergency operation

## Operation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Analysis** | Investigate issue without applying fixes | Understanding problem scope |
| **Interactive** | Step-by-step guidance with user confirmation | Complex issues requiring review |
| **Automatic** | Full automated analysis and fixing | Standard bug fixes |
| **Emergency** | Rapid resolution with minimal prompts | Production incidents |

## Related Commands

**Prerequisites**: Commands to run before fixing GitHub issues
- `/debug --auto-fix` - Fix runtime issues before addressing GitHub issues
- `/check-code-quality --auto-fix` - Fix quality issues that may be related
- `/explain-code` - Understand codebase context for issue resolution

**Alternatives**: Different issue resolution approaches
- Manual issue analysis and fixing
- `/multi-agent-optimize --mode=review` - Multi-agent issue analysis
- Platform-specific issue management tools
- Traditional development workflow without automation

**Combinations**: Commands that work with GitHub issue fixing
- `/generate-tests --coverage=95` - Add tests to prevent similar issues
- `/optimize --implement` - Optimize code while fixing performance issues
- `/double-check --deep-analysis` - Verify issue resolution completeness
- `/commit --template=fix` - Commit fixes with proper issue linking

**Follow-up**: Commands to run after fixing GitHub issues
- `/run-all-tests --auto-fix` - Ensure fixes don't break existing functionality
- `/ci-setup --monitoring` - Set up monitoring to prevent similar issues
- `/reflection --type=instruction` - Analyze issue resolution effectiveness
- `/update-docs` - Document fixes and prevention strategies

## Requirements

- GitHub CLI (`gh`) installed and authenticated
- Git repository with issue tracking enabled
- Write permissions to repository (for creating branches and PRs)
- Network access to GitHub API
- Valid GitHub token with appropriate scopes

ARGUMENTS: [--implement] [--auto-fix] [--draft] [--interactive] [--emergency] [--branch=name] [--agents=auto|core|scientific|engineering|ai|domain|quality|research|all] [--dry-run] [--intelligent] [--orchestrate] [--parallel] [issue-number-or-url]