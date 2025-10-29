---
description: Comprehensively validate Claude Code plugin syntax, structure, and cross-references
argument-hint: [--fix] [--plugin=name] [--report] [--analyze-deps]
allowed-tools: Read, Write, Bash, Glob, Grep, TodoWrite, Skill
color: cyan
skills:
  - plugin-syntax-validator
agents:
  primary:
    - comprehensive-review:code-reviewer
  conditional:
    - agent: comprehensive-review:architect-review
      trigger: pattern "architecture|dependency.*graph|circular|cross-plugin"
    - agent: debugging-toolkit:dx-optimizer
      trigger: pattern "tooling|workflow|setup|automation|pre-commit|ci"
---

# Comprehensive Plugin Linter

Validate all plugin files for correct syntax, structure, and cross-plugin references. Prevent agent loading errors by catching issues before deployment.

## Purpose

**Prevents runtime failures** by validating:
- ✅ Agent/skill reference syntax (`plugin:agent` format)
- ✅ File existence (agents, skills, commands)
- ✅ plugin.json structure and metadata
- ✅ Cross-plugin dependencies
- ✅ Circular dependency detection
- ✅ Unused agent/skill identification

---

## Arguments

```bash
# Basic validation (read-only)
/lint-plugins

# Auto-fix syntax errors
/lint-plugins --fix

# Validate specific plugin
/lint-plugins --plugin=backend-development

# Generate detailed report
/lint-plugins --report

# Analyze cross-plugin dependencies
/lint-plugins --analyze-deps
```

---

## Workflow

Execute validation using the **plugin-syntax-validator** skill:

### Phase 1: Syntax Validation (30 seconds)

Run automated validation using the skill's validation script:

```bash
# Invoke plugin-syntax-validator skill
python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py \
  --plugins-dir plugins \
  --verbose
```

**What it checks**:
1. Agent reference format (`plugin:agent` vs `plugin::agent` or bare `agent`)
2. Skill reference format
3. File existence for all referenced agents/skills
4. plugin.json structure and metadata
5. SKILL.md frontmatter validation

**Output**: Detailed report with file:line locations for all issues

### Phase 2: Auto-Fix (if --fix flag)

```bash
# Auto-fix common syntax errors
python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py \
  --fix \
  --verbose
```

**Auto-fixable issues**:
- ✅ Double colons (`::` → `:`)
- ✅ Whitespace in references
- ❌ Missing namespaces (requires manual mapping)
- ❌ Non-existent agents (requires creating or fixing)

### Phase 3: Dependency Analysis (if --analyze-deps flag)

Analyze cross-plugin dependencies and interactions:

1. **Build Dependency Graph**
   - Map which plugins reference agents from other plugins
   - Identify cross-plugin skill usage
   - Track command cross-references

2. **Detect Issues**
   - Circular dependencies (A→B→C→A)
   - Missing dependencies
   - Unused agents/skills
   - Orphaned files

3. **Generate Visualization**
   - Dependency graph in DOT format
   - Convert to PNG/SVG with Graphviz

---

## Validation Rules

### Rule 1: Single Colon Format

**✅ VALID**:
```markdown
Use Task tool with subagent_type="comprehensive-review:code-reviewer"
Invoke Skill: "backend-development:api-design-principles"
```

**❌ INVALID**:
```markdown
Use Task tool with subagent_type="comprehensive-review:code-reviewer"  # Double colon
```

**Auto-fixable**: ✅ Yes

---

### Rule 2: Namespace Required

**✅ VALID**:
```markdown
subagent_type="unit-testing:test-automator"
```

**❌ INVALID**:
```markdown
subagent_type="test-automator"  # Missing plugin namespace
```

**✅ VALID (after fix)**:
```markdown
subagent_type="unit-testing:test-automator"  # With plugin namespace
```

**Auto-fixable**: ✅ Yes

**Manual fix**: Add correct plugin namespace from this mapping:

| Bare Name | Correct Reference |
|-----------|-------------------|
| `code-reviewer` | `comprehensive-review:code-reviewer` |
| `backend-architect` | `backend-development:backend-architect` |
| `performance-engineer` | `full-stack-orchestration:performance-engineer` |
| `test-automator` | `unit-testing:test-automator` |
| `debugger` | `debugging-toolkit:debugger` |

---

### Rule 3: Agent File Exists

**✅ VALID**:
```markdown
"backend-development:backend-architect"
→ File exists: plugins/backend-development/agents/backend-architect.md
```

**❌ INVALID**:
```markdown
"backend-development:nonexistent-agent"
→ File not found
```

**Auto-fixable**: ❌ No

**Manual fix**: Either:
1. Create the agent file: `plugins/plugin-name/agents/agent-name.md`
2. Fix the reference to an existing agent

---

### Rule 4: plugin.json Structure

**✅ VALID**:
```json
{
  "name": "backend-development",
  "version": "1.0.0",
  "description": "Backend development workflows",
  "agents": [
    {
      "name": "backend-architect",
      "description": "...",
      "status": "active"
    }
  ],
  "keywords": ["backend", "api", "architecture"]
}
```

**❌ INVALID**:
- Missing required fields (`name`, `version`, `description`)
- Invalid JSON syntax
- Agent listed but file missing
- Duplicate agent names

---

## Usage Examples

### Example 1: Basic Validation

```bash
/lint-plugins

# Output:
# 🔍 Validating all plugins in plugins/
#
# ================================================================================
# PLUGIN SYNTAX VALIDATION REPORT
# ================================================================================
#
# 📊 Statistics:
#   Plugins scanned:      17
#   Files scanned:        154
#   Agent refs checked:   247
#   Skill refs checked:   89
#
# 📈 Results:
#   🔴 Errors:   0
#   🟡 Warnings: 0
#
# ✅ All validations passed!
```

### Example 2: Validation with Errors

```bash
/lint-plugins

# Output:
# 🔍 Validating all plugins
#
# 📈 Results:
#   🔴 Errors:   3
#   🟡 Warnings: 2
#
# ────────────────────────────────────────────────────────────────────────────────
# 🔴 ERRORS (Must Fix)
# ────────────────────────────────────────────────────────────────────────────────
#
#   [SYNTAX] backend-development/commands/feature-development.md:29
#   Double colon (::) in agent reference: 'comprehensive-review::code-reviewer'
#   💡 Suggestion: Change to: comprehensive-review:code-reviewer
#
#   [REFERENCE] custom-commands/commands/smart-fix.md:92
#   Agent not found: 'debugger' in plugin 'incident-response'
#   💡 Suggestion: Available agents in incident-response: incident-responder
#
# ⚠️  Found 3 error(s) that must be fixed.
```

### Example 3: Auto-Fix Mode

```bash
/lint-plugins --fix

# Output:
# 🔍 Validating all plugins
# 🔧 Attempting to auto-fix issues...
# ✅ Fixed 2 issue(s)
#
# 🔍 Re-validating...
# ✅ All validations passed!
```

### Example 4: Specific Plugin

```bash
/lint-plugins --plugin=backend-development

# Output:
# 🔍 Validating plugin: backend-development
# ✅ 12/12 agent references valid
# ✅ 3/3 skill references valid
```

### Example 5: Dependency Analysis

```bash
/lint-plugins --analyze-deps

# Output:
# 🔍 Analyzing cross-plugin dependencies...
#
# Cross-Plugin Dependencies:
# ├─ backend-development
# │  ├─ Uses: comprehensive-review:code-reviewer (3×)
# │  ├─ Uses: unit-testing:test-automator (2×)
# │  └─ Uses: full-stack-orchestration:deployment-engineer (1×)
# ├─ custom-commands
# │  ├─ Uses: debugging-toolkit:debugger (5×)
# │  └─ Uses: comprehensive-review:code-reviewer (2×)
# └─ ...
#
# Circular Dependencies: None detected ✅
#
# Unused Agents:
# └─ backend-development:legacy-adapter (never referenced)
#
# Dependency Graph: dependency-graph.dot
# Run: dot -Tpng dependency-graph.dot -o dependency-graph.png
```

---

## Integration with Development Workflow

### Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Validate plugins before commit

if git diff --cached --name-only | grep -q "plugins/"; then
  echo "🔍 Validating plugin syntax..."

  /lint-plugins

  if [ $? -ne 0 ]; then
    echo "❌ Plugin validation failed"
    echo "Run '/lint-plugins --fix' to auto-correct issues"
    exit 1
  fi

  echo "✅ Plugin validation passed"
fi
```

### CI/CD Pipeline

**GitHub Actions** (`.github/workflows/lint-plugins.yml`):

```yaml
name: Lint Plugins

on:
  push:
    paths:
      - 'plugins/**/*.md'
      - 'plugins/**/plugin.json'
  pull_request:
    paths:
      - 'plugins/**/*.md'
      - 'plugins/**/plugin.json'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Validate plugin syntax
        run: |
          python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py

      - name: Upload report on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: validation-report.md
```

---

## Advanced Features

### 1. Cross-Plugin Dependency Detection

Identifies which plugins depend on agents from other plugins:

```
backend-development → comprehensive-review (code-reviewer)
backend-development → unit-testing (test-automator)
custom-commands → debugging-toolkit (debugger)
```

**Use case**: Understand plugin coupling before refactoring

### 2. Circular Dependency Detection

Detects invalid circular references:

```
❌ Circular dependency detected:
   plugin-a → plugin-b → plugin-c → plugin-a
```

**Use case**: Prevent infinite loops in agent orchestration

### 3. Unused Agent Identification

Finds agents that are never referenced:

```
⚠️  Unused agents:
   - backend-development:legacy-adapter
   - data-engineering:deprecated-transformer
```

**Use case**: Clean up unused code, reduce maintenance burden

### 4. Dependency Visualization

Generates visual dependency graph:

```bash
# Generate DOT file
/lint-plugins --analyze-deps

# Convert to image
dot -Tpng dependency-graph.dot -o dependency-graph.png
```

**Use case**: Visualize plugin architecture

---

## Common Issues and Solutions

### Issue 1: "Plugin not found" error

**Cause**: Plugin directory doesn't exist or is misnamed

**Solution**:
```bash
# Check plugin directory exists
ls plugins/your-plugin/

# Ensure it contains agents/ or skills/ directory
ls plugins/your-plugin/agents/
```

### Issue 2: Auto-fix doesn't fix all errors

**Explanation**: Only syntax errors are auto-fixable:
- ✅ Double colons (`::` → `:`)
- ❌ Missing namespaces (ambiguous)
- ❌ Non-existent agents (requires creation)

**Solution**: Manually fix remaining issues using suggestions

### Issue 3: False positives for custom plugins

**Solution**: Ensure custom plugins follow structure:
```
plugins/your-custom-plugin/
├── plugin.json
├── agents/
│   └── agent-name.md
└── commands/
    └── command-name.md
```

---

## Best Practices

1. **Validate before every commit**: Use pre-commit hooks
2. **Run --fix first**: Auto-correct obvious syntax errors
3. **Review changes**: Always verify auto-fixes before committing
4. **CI integration**: Catch errors in pull requests automatically
5. **Keep namespace consistent**: Use official plugin:agent format
6. **Document custom agents**: Update namespace mapping table
7. **Analyze dependencies**: Understand plugin coupling before refactoring
8. **Clean unused agents**: Remove agents that are never referenced

---

## Troubleshooting

### Slow validation

**Solution**: Validate specific plugin:
```bash
/lint-plugins --plugin=your-plugin
```

### Permission errors

**Solution**: Check file permissions:
```bash
chmod -R u+r plugins/
```

### Script not found

**Solution**: Ensure skill is installed:
```bash
ls plugins/custom-commands/skills/plugin-syntax-validator/scripts/
```

---

## Output Format

### Summary Mode (Default)

```
✅ Validation complete
├─ Plugins scanned: 17
├─ References checked: 247
├─ Errors: 0
├─ Warnings: 0
└─ Pass rate: 100%
```

### Detailed Mode (--report)

```
📊 Plugin Lint Report

Scan Date: 2025-10-27 14:30:00
Duration: 28.3s

Plugins: 17 scanned
└─ backend-development         ✅ 12/12 valid
└─ comprehensive-review        ✅ 8/8 valid
└─ debugging-toolkit           ✅ 5/5 valid
└─ (14 more...)                ✅ 222/222 valid

Agent References: 247 total
└─ comprehensive-review:code-reviewer        (15 uses)
└─ backend-development:backend-architect     (12 uses)
└─ unit-testing:test-automator               (11 uses)
└─ (20 more distinct agents)

Validation Results: ✅ PASSED
└─ All agent references use correct plugin:agent format
└─ All agent files exist and are accessible
└─ No syntax violations detected
```

---

## Error Severity Levels

- **ERROR** (Exit code 1):
  - Double colons in references
  - Agent/skill file does not exist
  - Malformed syntax
  - Invalid plugin.json

- **WARNING** (Exit code 0):
  - Missing namespace (bare agent name)
  - Deprecated agent reference
  - Unused agents

- **INFO** (Exit code 0):
  - Statistics
  - Suggestions for optimization

---

## See Also

- `/command-creator` - Create new custom commands
- `/quality` - Comprehensive code quality analysis
- Plugin development documentation
- Agent namespace reference guide

---

**Version**: 2.0.0
**Updated**: 2025-10-27
**Skill**: plugin-syntax-validator
**Maintainer**: Claude Code Workflows Team
