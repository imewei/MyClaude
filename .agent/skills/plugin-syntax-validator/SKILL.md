---
name: plugin-syntax-validator
version: "1.0.7"
maturity: "5-Expert"
specialization: Plugin Validation
description: Validate Claude Code plugin syntax, structure, and cross-references with automated detection and auto-fixing of common errors. Use when validating agent/skill references, checking namespace syntax, or setting up CI validation pipelines.
---

# Plugin Syntax Validator

Comprehensive validation for Claude Code plugin syntax and references.

---

## Quick Start

```bash
# Validate all plugins
python .agent/scripts/validate_plugin_syntax.py

# Auto-fix common issues
python .agent/scripts/validate_plugin_syntax.py --fix

# Validate specific plugin
python .agent/scripts/validate_plugin_syntax.py --plugin backend-development
```

---

## Validation Rules

| Rule | Valid | Invalid | Auto-Fix |
|------|-------|---------|----------|
| Single colon | `plugin:agent` | `plugin::agent` | Yes |
| Namespace required | `backend:architect` | `architect` | No |
| Agent exists | File at `plugins/*/agents/*.md` | Missing file | No |

---

## Error Types

| Type | Description | Fix |
|------|-------------|-----|
| SYNTAX | Double colon format | Run `--fix` |
| REFERENCE | Agent not found | Create agent or fix ref |
| NAMESPACE | Missing plugin prefix | Add plugin prefix |

---

## CI/CD Integration

### Pre-commit Hook

```bash
#!/bin/bash
if git diff --cached --name-only | grep -q "plugins/"; then
  python .agent/scripts/validate_plugin_syntax.py || exit 1
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
      - run: python .agent/scripts/validate_plugin_syntax.py
```

---

## Output Example

```
PLUGIN SYNTAX VALIDATION REPORT

Statistics:
  Plugins scanned:      17
  Files scanned:        154
  Agent refs checked:   247

Results:
  Errors:   3
  Warnings: 2

[SYNTAX] backend/commands/feature.md:29
Double colon (::) in agent reference
Suggestion: Change to: plugin:agent

[REFERENCE] custom/commands/smart-fix.md:92
Agent not found: 'debugger' in plugin 'incident-response'
Suggestion: Available agents: incident-responder
```

---

## Best Practices

| Practice | Implementation |
|----------|----------------|
| Validate before commit | Use pre-commit hooks |
| Run --fix first | Auto-correct obvious issues |
| Review changes | Before committing fixes |
| Add CI validation | Catch errors in PRs |
| Document suppressions | Track intentional deviations |

---

## Checklist

- [ ] Run validation before committing
- [ ] Use --fix for auto-corrections
- [ ] Verify agent files exist
- [ ] Check namespace format
- [ ] Integrate with CI/CD

---

**Version**: 1.0.5
