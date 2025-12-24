---
version: "1.0.6"
command: /lint-plugins
description: Validate Claude Code plugin syntax, structure, and cross-references
argument-hint: [--fix] [--plugin=name] [--report] [--analyze-deps]
execution_modes:
  quick:
    duration: "30 seconds"
    scope: "Single plugin syntax checks"
  standard:
    duration: "1-2 minutes"
    scope: "All plugins + file existence + plugin.json"
  enterprise:
    duration: "3-5 minutes"
    scope: "Full + cross-plugin deps + circular detection"
color: cyan
allowed-tools: Read, Write, Bash, Glob, Grep
agents:
  primary:
    - comprehensive-review:code-reviewer
  conditional:
    - agent: comprehensive-review:architect-review
      trigger: pattern "architecture|dependency.*graph|circular"
  orchestrated: false
---

# Comprehensive Plugin Linter

Validate all plugin files for correct syntax, structure, and cross-plugin references.

## Target

$ARGUMENTS

---

## Mode Selection

| Mode | Duration | Scope |
|------|----------|-------|
| Quick | 30 sec | Single plugin, syntax only |
| Standard (default) | 1-2 min | All plugins, file existence, plugin.json |
| Enterprise | 3-5 min | All + dependency graph, circular detection, unused agents |

---

## Phase 1: Syntax Validation

### Validation Script

```bash
python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py \
  --plugins-dir plugins [--plugin <name>] [--verbose]
```

### What Is Checked

| Rule | Description | Auto-fixable |
|------|-------------|--------------|
| SYNTAX_001 | Single colon format (`plugin:agent` not `plugin::agent`) | ✅ |
| SYNTAX_002 | Namespace required (no bare `agent` names) | Partial |
| REFERENCE_001 | Agent file exists at expected path | ❌ |
| METADATA_001 | plugin.json has required fields | ❌ |

---

## Phase 2: Auto-Fix (--fix)

```bash
python validate_plugin_syntax.py --fix --verbose
```

| Issue | Fix |
|-------|-----|
| Double colons | `::` → `:` |
| Whitespace | Trim from references |
| Missing namespaces | Requires mapping file |

**After auto-fix:** Review changes → Test commands → Commit separately

---

## Phase 3: Dependency Analysis (Enterprise)

### Available Commands

| Command | Purpose |
|---------|---------|
| `--analyze-deps` | Cross-plugin dependency graph |
| `--check-circular` | Detect circular dependencies |
| `--find-unused` | Identify unreferenced agents |
| `--output-graph deps.dot` | Generate DOT visualization |

### Dependency Metrics
- Efferent/afferent coupling
- Most-used agents across plugins
- Dependency patterns

---

## Integration

### Pre-Commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: lint-plugins
        name: Validate Plugin Syntax
        entry: python validate_plugin_syntax.py
        files: '^plugins/.*\.(md|json)$'
```

### GitHub Actions

```yaml
# .github/workflows/lint-plugins.yml
on:
  push:
    paths: ['plugins/**/*.md', 'plugins/**/plugin.json']
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python validate_plugin_syntax.py
```

---

## Output Format

```
PLUGIN SYNTAX VALIDATION REPORT

📊 Statistics:
  Plugins scanned:      17
  Agent refs checked:   247

📈 Results:
  🔴 Errors:   3
  🟡 Warnings: 2

🔴 ERRORS
  [SYNTAX_001] file.md:29 - Double colon in 'comprehensive-review::code-reviewer'
  💡 Auto-fix: Change to 'comprehensive-review:code-reviewer'
```

---

## Success Criteria

| Mode | Criteria |
|------|----------|
| Quick | Plugin validated, syntax errors identified |
| Standard | All plugins valid, file existence verified, plugin.json correct |
| Enterprise | + No circular deps, unused agents identified, graph generated |

---

## External Documentation

- `plugin-validation-rules.md` - All rules with examples
- `dependency-analysis-guide.md` - Cross-plugin analysis
- `plugin-development-workflow.md` - CI/CD integration
