---
name: plugin-syntax-validator
description: Comprehensive validation framework for Claude Code plugin syntax, structure, and references. Use when validating plugin command files for correct agent/skill namespace syntax (plugin:agent format), checking agent file existence, detecting syntax errors (double colons, missing namespaces), and auto-fixing common issues. Includes validation script for agent references, skill references, and plugin.json structure validation.
---

# Plugin Syntax Validator

## Overview

This skill provides systematic validation of Claude Code plugin structure, syntax, and cross-references. It validates agent/skill references, checks file existence, detects syntax errors, and auto-fixes common issues like double colons.

**Use this skill when**:
- Validating plugin command files before deployment
- Checking agent reference syntax (plugin:agent format)
- Auto-fixing double colons (::) or missing namespaces
- Verifying all referenced agents/skills exist
- Preparing plugins for distribution or PR submission

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
❌ **INVALID**: `subagent_type="comprehensive-review::code-reviewer"`
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
  Double colon (::) in agent reference: 'comprehensive-review::code-reviewer'
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
