---
version: "2.1.0"
command: /lint-plugins
description: Validate Claude Code plugin syntax, structure, cross-references
argument-hint: "[--fix] [--plugin=name] [--report] [--analyze-deps]"
execution_modes: {quick: "30sec", standard: "1-2min", enterprise: "3-5min"}
color: cyan
---

# Plugin Linter

$ARGUMENTS

## Modes

| Mode | Time | Scope |
|------|------|-------|
| Quick | 30sec | Single plugin, syntax only |
| Standard | 1-2min | All plugins, file existence, plugin.json |
| Enterprise | 3-5min | All + dependency graph, circular detection, unused agents |

## Validation

```bash
python .agent/scripts/validate_plugin_syntax.py \
  --plugins-dir plugins [--plugin <name>] [--verbose]
```

### Rules

| Rule | Description | Auto-fix |
|------|-------------|----------|
| SYNTAX_001 | Single colon (`plugin:agent` not `plugin::agent`) | ✅ |
| SYNTAX_002 | Namespace required (no bare `agent`) | Partial |
| REFERENCE_001 | Agent file exists at expected path | ❌ |
| METADATA_001 | plugin.json has required fields | ❌ |

## Auto-Fix (--fix)

```bash
python validate_plugin_syntax.py --fix --verbose
```

| Issue | Fix |
|-------|-----|
| Double colons | `::` → `:` |
| Whitespace | Trim |
| Missing namespaces | Requires mapping file |

After: Review changes → Test commands → Commit separately

## Dependency Analysis (Enterprise)

| Command | Purpose |
|---------|---------|
| `--analyze-deps` | Cross-plugin dependency graph |
| `--check-circular` | Detect circular dependencies |
| `--find-unused` | Unreferenced agents |
| `--output-graph deps.dot` | DOT visualization |

**Metrics**: Efferent/afferent coupling, most-used agents, dependency patterns

## Integration

### Pre-Commit
```yaml
repos:
  - repo: local
    hooks:
      - id: lint-plugins
        name: Validate Plugin Syntax
        entry: python validate_plugin_syntax.py
        files: '^plugins/.*\\.(md|json)$'
```

### GitHub Actions
```yaml
on: {push: {paths: ['plugins/**/*.md', 'plugins/**/plugin.json']}}
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python validate_plugin_syntax.py
```

## Output

```
PLUGIN SYNTAX VALIDATION

📊 Statistics:
  Plugins: 17
  Agent refs: 247

📈 Results:
  🔴 Errors: 3
  🟡 Warnings: 2

🔴 ERRORS
  [SYNTAX_001] file.md:29 - Double colon in 'comprehensive-review::code-reviewer'
  💡 Fix: 'comprehensive-review:code-reviewer'
```

## Success

| Mode | Criteria |
|------|----------|
| Quick | Plugin validated, syntax errors identified |
| Standard | All valid, file existence verified, plugin.json correct |
| Enterprise | + No circular deps, unused agents identified, graph generated |

## External Docs

- `plugin-validation-rules.md` - All rules with examples
- `dependency-analysis-guide.md` - Cross-plugin analysis
- `plugin-development-workflow.md` - CI/CD integration
