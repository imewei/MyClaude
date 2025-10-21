---
description: Clean up development artifacts while preserving working code
argument-hint: "[path] [--auto-fix] [--aggressive]"
allowed-tools: Glob, Grep, Read, Edit, Bash, TodoWrite
---

# Clean Project

Clean up development artifacts, temporary files, debug statements, and unnecessary code while preserving your working codebase.

## Usage:

`/clean-project [path] [--auto-fix] [--aggressive]`

**Arguments:**
- `path` (optional): Specific directory to clean (defaults to current directory)
- `--auto-fix`: Automatically remove safe artifacts without confirmation
- `--aggressive`: Include more aggressive cleanup (remove commented code, unused imports, etc.)

## Strategic Thinking Process

Before cleaning, carefully consider:

1. **Artifact Identification**
   - What patterns indicate temporary/debug files?
   - Which files might look temporary but are actually important?
   - Are there project-specific conventions for temp files?
   - What about generated files that should be kept?

2. **Safety Analysis**
   - Which deletions are definitely safe?
   - Which require more careful inspection?
   - Are there active processes using these files?
   - Could removing these break the development environment?

3. **Common Pitfalls**
   - .env files might look like artifacts but contain config
   - .cache directories might be needed for performance
   - Some .tmp files might be active session data
   - Debug logs might contain important error information

4. **Cleanup Strategy**
   - Start with obvious artifacts (*.log, *.tmp, *~)
   - Check file age - older files are usually safer to remove
   - Verify with git status what's tracked vs untracked
   - Group similar files for batch decision making

## Process:

### 1. Create Safety Checkpoint

**CRITICAL**: First, create a git checkpoint for safety:
```bash
git add -A
git commit -m "Pre-cleanup checkpoint" || echo "No changes to commit"
```

**Important Git Rules**:
- NEVER add "Co-authored-by" or any Claude signatures
- NEVER include "Generated with Claude Code" or similar messages
- NEVER modify git config or user credentials
- NEVER add any AI/assistant attribution to commits
- NEVER use emojis in commits, PRs, or git-related content

### 2. Identify Cleanup Targets

Use native tools to identify artifacts:
- **Glob tool** to find temporary and debug files
- **Grep tool** to detect debug statements in code
- **Read tool** to verify file contents before removal

Common patterns to search for:
- Temporary files: `**/*.log`, `**/*.tmp`, `**/*~`, `**/*.bak`
- Debug files: `**/debug*.py`, `**/test_*.log`, `**/.DS_Store`
- Cache directories: `**/__pycache__`, `**/.pytest_cache`, `**/.mypy_cache`
- Build artifacts: `**/dist`, `**/build`, `**/*.egg-info`
- Debug statements in code: `console.log`, `print("debug")`, `debugger`, etc.

### 3. Protected Directories

The following are AUTOMATICALLY protected and should NEVER be removed:
- `.claude/` directory (commands and configurations)
- `.git/` directory (version control)
- `node_modules/`, `vendor/` (dependency directories)
- Essential configuration files (`.env`, `package.json`, `pyproject.toml`, etc.)

### 4. Create Todo List

When you find multiple items to clean, use the TodoWrite tool to process them systematically:
- Group similar files together
- Prioritize by safety (safest first)
- Track progress as items are cleaned

### 5. Present Findings

Show the user what will be removed and why:
- **Debug/log files**: Temporary output files
- **Failed implementations**: Backup or old code files
- **Development-only files**: Test outputs, debug scripts
- **Debug statements**: Console logs, print statements in production code

### 6. Execute Cleanup

Based on flags:
- **Default**: Ask for confirmation before removing each category
- **--auto-fix**: Automatically remove safe artifacts (logs, tmp files, cache)
- **--aggressive**: Also remove commented code, unused imports, dead code

### 7. Verify and Report

After cleanup:
- Verify project integrity (run tests if available)
- Report what was cleaned and disk space saved
- Remind about the git checkpoint for easy restoration

## Examples:

**Basic cleanup with confirmation:**
```
/clean-project
```

**Auto-clean current directory:**
```
/clean-project --auto-fix
```

**Aggressive cleanup of specific path:**
```
/clean-project src/legacy --aggressive
```

**Clean entire project automatically:**
```
/clean-project . --auto-fix --aggressive
```

## Recovery

If any issues occur after cleanup:
```bash
git reset --hard HEAD~1  # Restore from checkpoint
```

## Notes:

- Always creates a git checkpoint before making changes
- Uses TodoWrite to track cleanup progress for multiple items
- Preserves all working code and essential files
- Reports disk space savings after cleanup
- Can be safely interrupted - checkpoint remains for recovery
- More conservative by default; use flags for automated cleanup
