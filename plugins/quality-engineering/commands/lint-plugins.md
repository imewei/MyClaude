---
version: 1.0.3
command: /lint-plugins
description: Comprehensively validate Claude Code plugin syntax, structure, and cross-references across 3 execution modes
argument-hint: [--fix] [--plugin=name] [--report] [--analyze-deps]
execution_modes:
  quick:
    duration: "30 seconds"
    description: "Fast syntax validation for single plugin"
    agents: ["code-reviewer"]
    scope: "Basic syntax checks (colon format, namespaces)"
    checks: "Syntax validation only"
  standard:
    duration: "1-2 minutes"
    description: "Full validation for all plugins"
    agents: ["code-reviewer", "comprehensive-review:architect-review"]
    scope: "All syntax + file existence + plugin.json validation"
    checks: "All validation rules"
  enterprise:
    duration: "3-5 minutes"
    description: "Deep analysis with dependency graph and architecture review"
    agents: ["code-reviewer", "comprehensive-review:architect-review", "debugging-toolkit:dx-optimizer"]
    scope: "Full validation + cross-plugin deps + circular detection + unused agents"
    checks: "All rules + dependency analysis + visualization"
workflow_type: "sequential"
interactive_mode: true
color: cyan
allowed-tools: Read, Write, Bash, Glob, Grep, TodoWrite, Skill
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

## Context

The user needs plugin validation for: $ARGUMENTS

## Execution Mode Selection

<AskUserQuestion>
questions:
  - question: "Which validation depth do you need?"
    header: "Validation Mode"
    multiSelect: false
    options:
      - label: "Quick (30 seconds)"
        description: "Fast syntax validation for single plugin. Basic syntax checks only (colon format, namespaces)."

      - label: "Standard (1-2 minutes)"
        description: "Full validation for all plugins. All syntax + file existence + plugin.json validation. All validation rules enforced."

      - label: "Enterprise (3-5 minutes)"
        description: "Deep analysis with dependency graph and architecture review. Full validation + cross-plugin deps + circular detection + unused agents + visualization."
</AskUserQuestion>

## Instructions

### Phase 1: Syntax Validation

Run automated validation using the **plugin-syntax-validator** skill:

#### Quick Mode: Single Plugin

```bash
# Validate specific plugin only
python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py \
  --plugins-dir plugins \
  --plugin <plugin-name> \
  --verbose
```

#### Standard/Enterprise Mode: All Plugins

```bash
# Validate all plugins
python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py \
  --plugins-dir plugins \
  --verbose
```

**What is checked**:
1. **Agent reference format** (`plugin:agent` vs `plugin::agent`)
2. **Skill reference format** (single colon)
3. **File existence** for all referenced agents/skills
4. **plugin.json structure** and required fields
5. **SKILL.md frontmatter** validation

**See validation rules**: [Plugin Validation Rules](../docs/lint-plugins/plugin-validation-rules.md)

---

### Phase 2: Auto-Fix (if --fix flag)

Automatically correct common syntax errors:

```bash
# Auto-fix fixable issues
python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py \
  --fix \
  --verbose
```

**Auto-fixable issues**:
- ✅ Double colons (`::` → `:`)
- ✅ Whitespace in references
- ⚠️ Missing namespaces (partial - requires mapping)
- ❌ Non-existent agents (manual fix required)
- ❌ Invalid plugin.json (manual fix required)

**After auto-fix**:
1. Review changes carefully
2. Test affected commands
3. Commit fixes separately

---

### Phase 3: Dependency Analysis (Enterprise Mode)

Analyze cross-plugin dependencies and architecture:

#### 3.1 Build Dependency Graph

```bash
# Analyze dependencies
python validate_plugin_syntax.py \
  --plugins-dir plugins \
  --analyze-deps \
  --verbose
```

**Identifies**:
- Cross-plugin agent references
- Most-used agents across plugins
- Plugin coupling metrics (efferent/afferent)
- Dependency patterns

#### 3.2 Detect Circular Dependencies

```bash
# Check for circular dependencies
python validate_plugin_syntax.py \
  --check-circular \
  --verbose
```

**Example output**:
```
❌ Circular dependency detected:
   plugin-a → plugin-b → plugin-c → plugin-a

💡 Resolution: Break cycle by extracting shared agents
```

#### 3.3 Find Unused Agents

```bash
# Identify unused agents
python validate_plugin_syntax.py \
  --find-unused \
  --verbose
```

**Example output**:
```
⚠️  Unused agents:
  - backend-development:legacy-adapter (never referenced)
  - data-engineering:deprecated-transformer (never referenced)

💡 Action: Remove or archive unused agents
```

#### 3.4 Generate Dependency Graph

```bash
# Generate DOT file
python validate_plugin_syntax.py \
  --analyze-deps \
  --output-graph deps.dot

# Convert to image
dot -Tpng deps.dot -o deps.png
```

**See comprehensive guide**: [Dependency Analysis Guide](../docs/lint-plugins/dependency-analysis-guide.md)

---

## Validation Rules Reference

### Rule 1: Single Colon Format (SYNTAX_001)

**❌ Invalid**: `comprehensive-review::code-reviewer`
**✅ Valid**: `comprehensive-review:code-reviewer`
**Auto-fixable**: ✅ Yes

### Rule 2: Namespace Required (SYNTAX_002)

**❌ Invalid**: `code-reviewer` (bare name)
**✅ Valid**: `comprehensive-review:code-reviewer`
**Auto-fixable**: ✅ Partial (requires mapping)

### Rule 3: Agent File Exists (REFERENCE_001)

**❌ Invalid**: Reference to non-existent agent
**✅ Valid**: Agent file exists at expected path
**Auto-fixable**: ❌ No (create file or fix reference)

### Rule 4: plugin.json Structure (METADATA_001)

**Required fields**: name, version, description
**Optional fields**: agents, commands, skills, keywords
**Auto-fixable**: ❌ No (manual JSON editing)

**See detailed rules**: [Plugin Validation Rules](../docs/lint-plugins/plugin-validation-rules.md)

---

## Usage Examples

### Example 1: Basic Validation

```bash
/lint-plugins

# Output: Summary with error count
# Exit code: 0 (success) or 1 (errors found)
```

### Example 2: Validate Specific Plugin

```bash
/lint-plugins --plugin=backend-development

# Output: Validation results for single plugin only
```

### Example 3: Auto-Fix Errors

```bash
/lint-plugins --fix

# Output:
# ✅ Fixed 3 issue(s) automatically
# ⚠️  2 issue(s) require manual fixes
```

### Example 4: Generate Report

```bash
/lint-plugins --report

# Output: Detailed report with statistics and all errors
```

### Example 5: Dependency Analysis

```bash
/lint-plugins --analyze-deps

# Output:
# - Cross-plugin dependency graph
# - Most-used agents
# - Coupling metrics
# - Circular dependencies (if any)
# - Unused agents
```

---

## Integration with Development Workflow

### Pre-Commit Hook Setup

```bash
# Install pre-commit framework
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: lint-plugins
        name: Validate Plugin Syntax
        entry: python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py
        language: system
        pass_filenames: false
        files: '^plugins/.*\.(md|json)$'
EOF

# Install hooks
pre-commit install
```

**Behavior**: Validation runs automatically on `git commit`

### CI/CD Integration (GitHub Actions)

```yaml
# .github/workflows/lint-plugins.yml
name: Lint Plugins
on:
  push:
    paths: ['plugins/**/*.md', 'plugins/**/plugin.json']
  pull_request:
    paths: ['plugins/**/*.md', 'plugins/**/plugin.json']

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Validate plugins
        run: |
          python plugins/custom-commands/skills/plugin-syntax-validator/scripts/validate_plugin_syntax.py
```

**See workflow guide**: [Plugin Development Workflow](../docs/lint-plugins/plugin-development-workflow.md)

---

## Output Format

### Summary Mode (Default)

```
================================================================================
PLUGIN SYNTAX VALIDATION REPORT
================================================================================

📊 Statistics:
  Plugins scanned:      17
  Files scanned:        154
  Agent refs checked:   247
  Skill refs checked:   89

📈 Results:
  🔴 Errors:   3
  🟡 Warnings: 2
  🟢 Info:     5

────────────────────────────────────────────────────────────────────────────
🔴 ERRORS (Must Fix)
────────────────────────────────────────────────────────────────────────────

  [SYNTAX_001] backend-development/commands/feature-development.md:29
  Double colon in agent reference: 'comprehensive-review::code-reviewer'
  💡 Auto-fix: Change to 'comprehensive-review:code-reviewer'

⚠️  Found 3 error(s) - Run '/lint-plugins --fix' to auto-correct
```

### Detailed Mode (--report)

Includes:
- Per-plugin validation results
- Agent usage statistics
- Cross-reference analysis
- Suggestions for optimization

---

## Troubleshooting

### Issue: Auto-fix changes wrong references

**Cause**: Ambiguous agent names
**Solution**: Review changes before committing

### Issue: False positives for custom plugins

**Cause**: Plugin not in standard location
**Solution**: Ensure plugin follows structure:
```
plugins/my-plugin/
├── plugin.json
├── agents/
│   └── agent-name.md
└── commands/
    └── command-name.md
```

### Issue: Validation too slow

**Cause**: Scanning many large files
**Solution**: Use `--plugin=name` for specific plugin validation

---

## External Documentation

- [Plugin Validation Rules](../docs/lint-plugins/plugin-validation-rules.md) - All validation rules with before/after examples
- [Plugin Development Workflow](../docs/lint-plugins/plugin-development-workflow.md) - Pre-commit hooks, CI/CD integration, release workflow
- [Dependency Analysis Guide](../docs/lint-plugins/dependency-analysis-guide.md) - Cross-plugin dependencies, circular detection, dependency graphs

---

## Success Criteria

**Quick Mode**:
- ✅ Single plugin validated
- ✅ Syntax errors identified
- ✅ Exit code indicates pass/fail

**Standard Mode**:
- ✅ All plugins validated
- ✅ All validation rules checked
- ✅ File existence verified
- ✅ plugin.json structure valid
- ✅ Auto-fix suggestions provided

**Enterprise Mode**:
- ✅ All Standard criteria met
- ✅ Cross-plugin dependencies mapped
- ✅ No circular dependencies
- ✅ Unused agents identified
- ✅ Dependency graph generated
- ✅ Architecture review complete

---

Execute plugin validation for selected mode, provide detailed error locations and auto-fix recommendations.
