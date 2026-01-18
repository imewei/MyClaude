---
version: "2.1.0"
category: "code-documentation"
command: "/update-claudemd"
description: Automatically update CLAUDE.md file based on recent code changes
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(git status:*), Bash(find:*), Bash(grep:*), Bash(wc:*), Bash(ls:*)
argument-hint: [--force] [--summary]
color: cyan
execution_modes:
  quick: "5-10 minutes - Essential updates only (last 5 commits)"
  standard: "10-15 minutes - Comprehensive analysis (last 10 commits)"
  force: "15-20 minutes - Complete rebuild with full history"
agents:
  primary:
    - research-intelligence
  conditional:
    - agent: systems-architect
      trigger: pattern "architecture|structure" OR files > 50
  orchestrated: false
---

# Update CLAUDE.md File

Automatically analyze recent code changes and intelligently update CLAUDE.md to keep project documentation synchronized.

## Execution Modes

| Mode | Time | Scope | Use Case |
|------|------|-------|----------|
| **quick** | 5-10 min | Last 5 commits | Quick updates after small changes |
| **standard** (default) | 10-15 min | Last 10 commits | Regular weekly updates |
| **force** (--force flag) | 15-20 min | Full rebuild | Major refactoring or outdated docs |

## Arguments

$ARGUMENTS

**Flags**:
- `--force`: Complete rebuild analyzing entire git history
- `--summary`: Only show summary of changes without updating file

## Current State Analysis

### Current CLAUDE.md
@CLAUDE.md

### Git Analysis

**Repository status**:
!`git status --porcelain`

**Recent commits** (mode-dependent):
- Quick: `git log --oneline -5`
- Standard: `git log --oneline -10`
- Force: `git log --oneline -20`

!`git log --oneline -10`

**Detailed changes**:
!`git log --since="1 week ago" --pretty=format:"%h - %an, %ar : %s" --stat`

**Changed files**:
!`git diff HEAD~5 --name-only | head -20`

**Code changes in key files**:
!`git diff HEAD~5 -- "*.js" "*.ts" "*.jsx" "*.tsx" "*.py" "*.md" "*.json" | head -200`

**New files added**:
!`git diff --name-status HEAD~10 | grep "^A" | head -15`

**Deleted files**:
!`git diff --name-status HEAD~10 | grep "^D" | head -10`

**Modified core files**:
!`git diff --name-status HEAD~10 | grep "^M" | grep -E "(package\.json|README|config|main|index|app)" | head -10`

**Configuration changes**:
!`git diff HEAD~10 -- package.json tsconfig.json webpack.config.js next.config.js .env* | head -100`

**API/Route changes**:
!`git diff HEAD~10 -- "**/routes/**" "**/api/**" "**/controllers/**" | head -150`

**Database changes**:
!`git diff HEAD~10 -- "**/models/**" "**/schemas/**" "**/migrations/**" | head -100`

## Update Strategy

### 1. Preserve Core Content
- ✓ Core project description and architecture
- ✓ Essential setup instructions
- ✓ Key architectural decisions
- ✓ Development workflow fundamentals

### 2. Integrate Recent Changes

Analyze git diff/logs to identify:

**New Features**: Functionality added
**API Changes**: Endpoints, routes, parameters
**Config Updates**: Build tools, dependencies, env vars
**File Structure**: New directories, moved files
**Database**: Models, schemas, migrations
**Bug Fixes**: Important behavioral fixes
**Refactoring**: Architectural changes

### 3. Update Sections Intelligently

**Project Overview**: Scope, technologies, version
**Architecture**: Patterns, structure, components
**Setup Instructions**: Environment, dependencies, config
**API Documentation**: Endpoints, auth, parameters
**Development Workflow**: Scripts, tools, testing
**Recent Updates**: Timestamped change summary
**File Structure**: Directory organization

### 4. Smart Content Management

- **Don't duplicate**: Avoid repeating existing docs
- **Prioritize relevance**: Focus on developer-impacting changes
- **Keep concise**: Summarize, don't list every change
- **Maintain structure**: Follow existing organization
- **Add timestamps**: Note major update dates

## Output Format

```markdown
# Project Name

## Overview
[Updated description with new scope/tech]

## Architecture
[Updated patterns and structure]

## Setup & Installation
[Updated with new dependencies/env vars]

## Development Workflow
[Updated scripts and processes]

## API Documentation
[New/updated endpoints]

## File Structure
[Updated directory organization]

## Recent Updates (Updated: YYYY-MM-DD)
### Major Changes
- [Feature/change 1]
- [Feature/change 2]

### Breaking Changes
- [If any]

### Bug Fixes
- [Important fixes]

## Important Notes
[Key developer information]
```

## Success Criteria

✅ All significant changes from git history reflected
✅ New features documented
✅ API changes noted
✅ Configuration updates included
✅ Existing structure preserved
✅ Concise and developer-focused
✅ Timestamp added for update tracking
✅ No duplicate information

Focus on **keeping documentation synchronized** with code through intelligent git analysis and targeted updates.
