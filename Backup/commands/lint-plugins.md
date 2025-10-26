---
description: Validate claude-code-workflows plugin syntax for agent references
argument-hint: [--fix] [--plugin=name] [--report]
allowed-tools: Read, Write, Bash, Glob, Grep, TodoWrite
color: cyan
---

Validate all plugin command files for correct agent namespace syntax and identify potential issues before they cause runtime failures.

**Purpose**: Prevent agent loading errors by catching syntax issues in plugin files:
- Double colons (::) instead of single colons (:)
- Missing namespace prefixes (bare agent names)
- Invalid agent references
- Unreachable agent files

## Workflow

1. **Scan plugins**: Search all command files for `subagent_type=` references
2. **Validate syntax**: Check for correct `plugin:agent` format
3. **Verify agents exist**: Confirm referenced agents have corresponding files
4. **Report findings**: Generate detailed report with severity levels
5. **Auto-fix (optional)**: Apply corrections when `--fix` flag provided

## Validation Rules

### Rule 1: Single Colon Format
```bash
✅ VALID:   subagent_type="comprehensive-review:code-reviewer"
❌ INVALID: subagent_type="comprehensive-review::code-reviewer"  # Double colon
```

### Rule 2: Namespace Required
```bash
✅ VALID:   subagent_type="unit-testing:test-automator"
❌ INVALID: subagent_type="test-automator"  # Missing namespace
```

### Rule 3: Agent File Exists
```bash
✅ VALID:   "backend-development:backend-architect"
            → File exists: backend-development/agents/backend-architect.md
❌ INVALID: "backend-development:nonexistent-agent"
            → File missing
```

## Argument Handling

```bash
# Basic validation (read-only)
/lint-plugins

# Auto-fix all issues
/lint-plugins --fix

# Check specific plugin only
/lint-plugins --plugin=git-pr-workflows

# Generate detailed report file
/lint-plugins --report
```

## Implementation

### Step 1: Setup and Planning
Use TodoWrite to track progress:
1. Scan plugin directory structure
2. Extract all subagent_type references
3. Validate each reference
4. Generate report
5. Apply fixes if requested

### Step 2: Detection Logic

```bash
# Find all plugin command files
PLUGIN_DIR="$HOME/.claude/plugins/marketplaces/claude-code-workflows/plugins"

# Extract agent references
grep -rn "subagent_type=" "$PLUGIN_DIR/*/commands/" | while read line; do
  # Parse file:line:content
  FILE=$(echo "$line" | cut -d: -f1)
  LINE_NUM=$(echo "$line" | cut -d: -f2)
  CONTENT=$(echo "$line" | cut -d: -f3-)

  # Extract agent reference (between quotes)
  AGENT_REF=$(echo "$CONTENT" | sed -n 's/.*subagent_type="\([^"]*\)".*/\1/p')

  # Validate format
  if echo "$AGENT_REF" | grep -q "::"; then
    echo "ERROR: Double colon in $FILE:$LINE_NUM → $AGENT_REF"
  elif ! echo "$AGENT_REF" | grep -q ":"; then
    echo "WARNING: Missing namespace in $FILE:$LINE_NUM → $AGENT_REF"
  else
    # Check if agent file exists
    PLUGIN_NAME=$(echo "$AGENT_REF" | cut -d: -f1)
    AGENT_NAME=$(echo "$AGENT_REF" | cut -d: -f2)
    AGENT_FILE="$PLUGIN_DIR/$PLUGIN_NAME/agents/$AGENT_NAME.md"

    if [ ! -f "$AGENT_FILE" ]; then
      echo "ERROR: Agent file not found for $FILE:$LINE_NUM → $AGENT_REF"
      echo "  Expected: $AGENT_FILE"
    fi
  fi
done
```

### Step 3: Auto-Fix Logic

```bash
# Fix double colons (:: → :)
find "$PLUGIN_DIR" -name "*.md" -path "*/commands/*" -type f \
  -exec sed -i 's/subagent_type="\([^:]*\)::\([^"]*\)"/subagent_type="\1:\2"/g' {} \;

# Fix common bare agent names (requires mapping table)
declare -A AGENT_MAPPING=(
  ["performance-engineer"]="application-performance:performance-engineer"
  ["code-reviewer"]="comprehensive-review:code-reviewer"
  ["test-automator"]="unit-testing:test-automator"
  ["security-auditor"]="comprehensive-review:security-auditor"
  ["backend-architect"]="backend-development:backend-architect"
  ["deployment-engineer"]="full-stack-orchestration:deployment-engineer"
  ["observability-engineer"]="observability-monitoring:observability-engineer"
  ["frontend-developer"]="frontend-mobile-development:frontend-developer"
  ["architect-review"]="comprehensive-review:architect-review"
  ["legacy-modernizer"]="framework-migration:legacy-modernizer"
  ["debugger"]="debugging-toolkit:debugger"
)

for bare_name in "${!AGENT_MAPPING[@]}"; do
  namespaced="${AGENT_MAPPING[$bare_name]}"
  find "$PLUGIN_DIR" -name "*.md" -path "*/commands/*" -type f \
    -exec sed -i "s/subagent_type=\"$bare_name\"/subagent_type=\"$namespaced\"/g" {} \;
done
```

### Step 4: Report Generation

```markdown
# Plugin Lint Report
**Generated**: $(date +%Y-%m-%d)
**Plugins Scanned**: $(find "$PLUGIN_DIR" -name "*.md" -path "*/commands/*" | wc -l)
**Total Agent References**: $(grep -r "subagent_type=" "$PLUGIN_DIR/*/commands/" | wc -l)

## Summary
- ❌ Errors: 0 (double colons, missing agent files)
- ⚠️  Warnings: 0 (missing namespaces)
- ✅ Passed: 154 (all references valid)

## Issues Found
(List of all violations with file:line references)

## Recommendations
- Enable pre-commit hook for automatic validation
- Update plugin development documentation
- Add CI check to prevent regressions
```

## Examples

### Basic Validation
```bash
/lint-plugins

# Output:
# Scanning 17 plugins with 154 agent references...
# ✅ All references valid (100%)
# 0 errors, 0 warnings
```

### Validation with Issues
```bash
/lint-plugins

# Output:
# Scanning 17 plugins with 154 agent references...
# ❌ Found 3 errors:
#   - git-pr-workflows/commands/git-workflow.md:23 → Double colon in "code-reviewer::main"
#   - comprehensive-review/commands/full-review.md:20 → Missing namespace: "code-reviewer"
#   - incident-response/commands/smart-fix.md:92 → Agent file not found: custom-agent:debugger
# ⚠️  Found 2 warnings:
#   - application-performance/commands/performance-optimization.md:8 → Bare name: "performance-engineer"
```

### Auto-Fix Mode
```bash
/lint-plugins --fix

# Output:
# Scanning and fixing 17 plugins...
# Fixed 3 errors:
#   ✅ Replaced :: with : (1 instance)
#   ✅ Added namespaces (2 instances)
# Fixed 2 warnings:
#   ✅ Mapped bare names to namespaces (2 instances)
#
# Re-scanning to verify...
# ✅ All 154 references now valid (100%)
```

### Plugin-Specific Check
```bash
/lint-plugins --plugin=git-pr-workflows

# Output:
# Scanning git-pr-workflows plugin...
# ✅ 10/10 agent references valid
```

### Generate Report File
```bash
/lint-plugins --report

# Output:
# Scanning 17 plugins...
# ✅ Validation complete
# 📄 Report saved: /tmp/plugin-lint-report-2025-10-23.md
```

## Integration with CI/CD

Add to `.github/workflows/lint-plugins.yml`:

```yaml
name: Lint Plugins

on:
  push:
    paths:
      - '.claude/plugins/**/*.md'
  pull_request:
    paths:
      - '.claude/plugins/**/*.md'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate plugin syntax
        run: |
          claude-code /lint-plugins
          if [ $? -ne 0 ]; then
            echo "❌ Plugin validation failed"
            echo "Run '/lint-plugins --fix' to auto-correct issues"
            exit 1
          fi
```

## Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Validate plugins before commit

if git diff --cached --name-only | grep -q "plugins/.*/commands/.*\.md"; then
  echo "Validating plugin syntax..."

  # Run lint check
  if ! /lint-plugins; then
    echo "❌ Plugin validation failed"
    echo "Fix issues or run '/lint-plugins --fix' to auto-correct"
    exit 1
  fi

  echo "✅ Plugin validation passed"
fi
```

## Known Agent Namespace Mappings

Common bare agent names and their correct namespaces:

| Bare Name | Correct Namespace |
|-----------|-------------------|
| `code-reviewer` | `comprehensive-review:code-reviewer` |
| `performance-engineer` | `application-performance:performance-engineer` |
| `test-automator` | `unit-testing:test-automator` |
| `security-auditor` | `comprehensive-review:security-auditor` |
| `backend-architect` | `backend-development:backend-architect` |
| `deployment-engineer` | `full-stack-orchestration:deployment-engineer` |
| `debugger` | `debugging-toolkit:debugger` |
| `frontend-developer` | `frontend-mobile-development:frontend-developer` |
| `observability-engineer` | `observability-monitoring:observability-engineer` |
| `architect-review` | `comprehensive-review:architect-review` |
| `legacy-modernizer` | `framework-migration:legacy-modernizer` |
| `incident-responder` | `incident-response:incident-responder` |
| `ui-ux-designer` | `multi-platform-apps:ui-ux-designer` |
| `ios-developer` | `multi-platform-apps:ios-developer` |
| `mobile-developer` | `multi-platform-apps:mobile-developer` |
| `tdd-orchestrator` | `tdd-workflows:tdd-orchestrator` |

## Error Severity Levels

- **ERROR** (Exit code 1):
  - Double colons in agent references
  - Agent file does not exist
  - Malformed subagent_type syntax

- **WARNING** (Exit code 0):
  - Missing namespace (bare agent name)
  - Deprecated agent reference
  - Duplicate agent definitions

- **INFO** (Exit code 0):
  - Total references scanned
  - Validation statistics
  - Suggestions for optimization

## Output Format

### Summary Mode (Default)
```
✅ Validation complete
├─ Plugins scanned: 17
├─ References checked: 154
├─ Errors: 0
├─ Warnings: 0
└─ Pass rate: 100%
```

### Detailed Mode (--report)
```
📊 Plugin Lint Report

Scan Date: 2025-10-23 14:30:00
Duration: 1.2s

Plugins: 17 scanned
└─ application-performance          ✅ 13/13 valid
└─ backend-development              ✅ 10/10 valid
└─ comprehensive-review             ✅ 8/8 valid
└─ data-engineering                 ✅ 17/17 valid
└─ framework-migration              ✅ 13/13 valid
└─ full-stack-orchestration         ✅ 11/11 valid
└─ git-pr-workflows                 ✅ 10/10 valid
└─ incident-response                ✅ 27/27 valid
└─ (9 more...)                      ✅ 45/45 valid

Agent References: 154 total
└─ comprehensive-review:code-reviewer        (7 uses)
└─ unit-testing:test-automator               (6 uses)
└─ application-performance:performance-engineer (13 uses)
└─ (20 more distinct agents)

Validation Results: ✅ PASSED
└─ All agent references use correct plugin:agent format
└─ All agent files exist and are accessible
└─ No syntax violations detected
```

## Troubleshooting

### Issue: Command not found
**Solution**: Ensure command file is in `~/.claude/commands/` or `.claude/commands/`

### Issue: Permission denied accessing plugins
**Solution**: Check file permissions on plugin directory:
```bash
chmod -R u+r ~/.claude/plugins/marketplaces/claude-code-workflows/
```

### Issue: Auto-fix creates incorrect namespaces
**Solution**: Review agent mapping table and update for project-specific agents

### Issue: False positives for custom agents
**Solution**: Add custom agents to whitelist or use `--plugin` flag to skip

## Best Practices

1. **Run before commits**: Integrate into pre-commit hooks
2. **Fix issues immediately**: Use `--fix` flag for batch corrections
3. **Review changes**: Always verify auto-fixes before committing
4. **Update mappings**: Keep agent namespace mappings current
5. **CI integration**: Add to GitHub Actions for PR validation
6. **Documentation**: Update when adding new plugins or agents

## See Also

- `/command-creator` - Create new custom commands
- Plugin development documentation
- Agent namespace reference guide

---

**Created**: 2025-10-23
**Version**: 1.0.0
**Maintainer**: Claude Code Workflows Team
