---
name: plugin-syntax-validator
description: Comprehensive validation framework for Claude Code plugin syntax, structure, and cross-references with automated scripts/validate_plugin_syntax.py script for detecting and auto-fixing common errors. Use when validating plugin command files (*.md) for correct agent/skill namespace syntax (plugin:agent single colon format, not plugin::agent double colon), checking agent file existence in plugins/*/agents/*.md, detecting syntax errors (double colons, missing namespaces, invalid references), auto-fixing common issues with --fix flag, verifying all referenced agents/skills exist across all plugins, preparing plugins for distribution or pull request submission, setting up CI/CD validation pipelines with GitHub Actions or pre-commit hooks, resolving validation errors (SYNTAX errors for double colons, REFERENCE errors for missing agents, namespace format issues), generating comprehensive validation reports with file:line error locations and actionable suggestions, building complete agent/skill maps across plugin ecosystems, ensuring plugin.json structure consistency, validating subagent_type references in command files, and maintaining plugin quality standards before deployment or marketplace submission.
---

# Plugin Syntax Validator

## When to use this skill

- Validating Claude Code plugin command files (plugins/*/commands/*.md) before committing changes or submitting pull requests
- Running scripts/validate_plugin_syntax.py to check agent and skill reference syntax across all plugins in the codebase
- Auto-fixing common syntax errors like double colons (plugin::agent) by running scripts/validate_plugin_syntax.py --fix to convert to single colon format (plugin:agent)
- Checking that all subagent_type references use correct namespace format (plugin:agent) instead of missing namespace (agent only) or incorrect format
- Verifying that all referenced agents exist as actual files (plugins/*/agents/*.md) to prevent runtime errors when agents are invoked
- Detecting SYNTAX errors for double colon format violations and getting auto-fix suggestions with file:line locations
- Detecting REFERENCE errors when agents are referenced but don't exist in the plugin directory structure
- Preparing plugins for distribution, marketplace submission, or production deployment by ensuring all references are valid
- Setting up CI/CD validation in GitHub Actions workflows or GitLab CI to automatically check plugin syntax on push or pull request events
- Creating pre-commit hooks to validate plugin syntax before allowing commits to ensure only valid plugin files enter the repository
- Generating comprehensive validation reports with statistics (plugins scanned, files scanned, agent refs checked) and detailed error listings
- Building complete agent/skill maps across all plugins to understand cross-plugin dependencies and reference patterns
- Validating specific plugins using --plugin flag to focus validation on a single plugin namespace during development
- Resolving validation errors by reviewing suggestions for each error (e.g., "Change to: plugin:agent" for double colon errors, "Available agents in plugin: agent1, agent2" for missing references)
- Ensuring plugin.json structure consistency and completeness before plugin releases
- Maintaining plugin quality standards by catching syntax errors before they reach production or cause runtime failures
- Validating namespace consistency across command files to ensure all agent references follow the same plugin:agent convention
- Debugging agent reference issues when commands fail to find or invoke agents correctly
- Reviewing validation output for warnings and errors to prioritize fixes (errors block deployment, warnings are improvements)
- Integrating validation into development workflows with --verbose flag for detailed step-by-step validation progress
- Validating plugin ecosystems with hundreds of agent references in ~30 seconds for rapid feedback during development

## Overview

This skill provides systematic validation of Claude Code plugin structure, syntax, and cross-references. It validates agent/skill references, checks file existence, detects syntax errors, and auto-fixes common issues like double colons.

---

## Quick Start

```bash
# Validate all plugins
python scripts/validate_plugin_syntax.py

# Auto-fix common issues
python scripts/validate_plugin_syntax.py --fix

# Validate specific plugin
python scripts/validate_plugin_syntax.py --plugin backend-development

# Generate report
python scripts/validate_plugin_syntax.py --report validation-report.md
```

---

## Validation Rules

### Rule 1: Single Colon Format

✅ **VALID**: `subagent_type="comprehensive-review:code-reviewer"`
❌ **INVALID**: `subagent_type="comprehensive-review:code-reviewer"`
🔧 **Auto-fixable**: Yes

### Rule 2: Namespace Required

✅ **VALID**: `subagent_type="backend-development:backend-architect"`
❌ **INVALID**: `subagent_type="backend-architect"` (missing plugin namespace)
🔧 **Auto-fixable**: No (requires knowing which plugin)

### Rule 3: Agent Must Exist

✅ **VALID**: Agent file exists at `plugins/backend-development/agents/backend-architect.md`
❌ **INVALID**: Agent file not found
🔧 **Auto-fixable**: No (requires creating agent or fixing reference)

---

## Validation Script

### `scripts/validate_plugin_syntax.py`

**Purpose**: Validates plugin syntax, agent/skill references, and file existence.

**Features**:
- Builds complete agent/skill map across all plugins
- Validates namespace format (plugin:agent)
- Checks file existence
- Auto-fixes double colons
- Provides file:line error locations
- Generates detailed reports

**Usage**:

```bash
# Basic validation
python scripts/validate_plugin_syntax.py

# Specify plugins directory
python scripts/validate_plugin_syntax.py --plugins-dir /path/to/plugins

# Auto-fix and re-validate
python scripts/validate_plugin_syntax.py --fix --verbose

# Validate specific plugin
python scripts/validate_plugin_syntax.py --plugin backend-development
```

**Exit Codes**:
- `0`: All validations passed
- `1`: Errors found

**Output Example**:
```
🔍 Validating all plugins

================================================================================
PLUGIN SYNTAX VALIDATION REPORT
================================================================================

📊 Statistics:
  Plugins scanned:      17
  Files scanned:        154
  Agent refs checked:   247

📈 Results:
  🔴 Errors:   3
  🟡 Warnings: 2

────────────────────────────────────────────────────────────────────────────────
🔴 ERRORS (Must Fix)
────────────────────────────────────────────────────────────────────────────────

  [SYNTAX] backend-development/commands/feature-development.md:29
  Double colon (::) in agent reference: 'comprehensive-review:code-reviewer'
  💡 Suggestion: Change to: comprehensive-review:code-reviewer

  [REFERENCE] custom-commands/commands/smart-fix.md:92
  Agent not found: 'debugger' in plugin 'incident-response'
  💡 Suggestion: Available agents in incident-response: incident-responder
```

---

## Common Validation Errors

### Error 1: Double Colon
**Fix**: Run `--fix` flag
**Example**: `plugin::agent` → `plugin:agent`

### Error 2: Missing Namespace
**Fix**: Add plugin prefix manually
**Example**: `code-reviewer` → `comprehensive-review:code-reviewer`

### Error 3: Agent Not Found
**Fix**: Create agent file or fix reference
**Example**: Create `plugins/plugin-name/agents/agent-name.md`

---

## CI/CD Integration

### Pre-commit Hook

```bash
#!/bin/bash
if git diff --cached --name-only | grep -q "plugins/"; then
  python scripts/validate_plugin_syntax.py || exit 1
fi
```

### GitHub Actions

```yaml
name: Validate Plugins
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
      - run: python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py
```

---

## Best Practices

1. Validate before every commit (use pre-commit hooks)
2. Run `--fix` first to auto-correct obvious issues
3. Review changes before committing
4. Keep agent namespace mappings consistent
5. Add CI validation to catch errors in PRs

---

## Summary

Provides automated validation of plugin syntax with auto-fixing capabilities. Validates 200+ agent references in ~30 seconds. Essential for maintaining plugin quality and preventing runtime errors.
