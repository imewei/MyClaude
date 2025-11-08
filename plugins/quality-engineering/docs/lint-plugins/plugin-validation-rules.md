# Plugin Validation Rules

Comprehensive guide to Claude Code plugin validation rules with auto-fix capabilities and detailed examples.

## Overview

The plugin linter validates Claude Code plugins against strict syntax and structure rules to prevent runtime agent loading errors. This guide covers all validation rules with before/after examples.

---

## Rule Categories

1. **Syntax Rules** - Agent reference format validation
2. **Structure Rules** - Plugin directory and file structure
3. **Metadata Rules** - plugin.json validation
4. **Reference Rules** - Cross-plugin dependency validation
5. **Skill Rules** - SKILL.md frontmatter validation

---

## Rule 1: Single Colon Format

**Rule ID**: `SYNTAX_001`
**Severity**: ERROR
**Auto-fixable**: ✅ Yes

### Description

Agent and skill references must use single colon (`:`) format, not double colons (`::`) or other separators.

### Why This Matters

The Claude Code agent orchestration system expects the `plugin:agent` format. Double colons cause agent lookup failures at runtime.

### Examples

#### Example 1: Agent Reference in Command

**❌ INVALID**:
```markdown
<!-- commands/feature-development.md -->
Use Task tool with subagent_type="comprehensive-review::code-reviewer"
```

**Error message**:
```
[SYNTAX_001] commands/feature-development.md:29
Double colon (::) in agent reference: 'comprehensive-review::code-reviewer'
💡 Auto-fix available: comprehensive-review:code-reviewer
```

**✅ VALID (after fix)**:
```markdown
Use Task tool with subagent_type="comprehensive-review:code-reviewer"
```

#### Example 2: Skill Reference

**❌ INVALID**:
```markdown
Invoke Skill: "backend-development::api-design-principles"
```

**✅ VALID**:
```markdown
Invoke Skill: "backend-development:api-design-principles"
```

#### Example 3: Multiple References

**❌ INVALID**:
```markdown
agents:
  - comprehensive-review::code-reviewer
  - unit-testing::test-automator
  - backend-development::backend-architect
```

**✅ VALID**:
```markdown
agents:
  - comprehensive-review:code-reviewer
  - unit-testing:test-automator
  - backend-development:backend-architect
```

### Auto-Fix Behavior

The linter automatically replaces `::` with `:` in all recognized contexts:
- YAML frontmatter `agents:` lists
- Markdown inline code references
- Task tool `subagent_type` parameters
- Skill invocation strings

### Manual Review Required

After auto-fix, verify:
- [ ] Agent names are correct
- [ ] Plugin namespaces match actual plugin names
- [ ] References work in context

---

## Rule 2: Namespace Required

**Rule ID**: `SYNTAX_002`
**Severity**: WARNING
**Auto-fixable**: ✅ Partial (requires namespace mapping)

### Description

All agent and skill references must include plugin namespace. Bare agent names without namespace are ambiguous and error-prone.

### Why This Matters

Multiple plugins may have agents with similar names. Without namespace:
- Ambiguous references lead to wrong agent invocation
- Plugin reorganization breaks references
- Cross-plugin dependencies are unclear

### Examples

#### Example 1: Bare Agent Name

**❌ INVALID**:
```markdown
<!-- commands/smart-fix.md -->
Use Task tool with subagent_type="debugger"
```

**Error message**:
```
[SYNTAX_002] commands/smart-fix.md:42
Missing namespace in agent reference: 'debugger'
💡 Did you mean: debugging-toolkit:debugger?
```

**✅ VALID**:
```markdown
Use Task tool with subagent_type="debugging-toolkit:debugger"
```

#### Example 2: Common Agent Names

**❌ INVALID**:
```markdown
agents:
  - code-reviewer
  - backend-architect
  - test-automator
```

**Warning message**:
```
[SYNTAX_002] Multiple bare agent names detected
  - code-reviewer → comprehensive-review:code-reviewer
  - backend-architect → backend-development:backend-architect
  - test-automator → unit-testing:test-automator
```

**✅ VALID**:
```markdown
agents:
  - comprehensive-review:code-reviewer
  - backend-development:backend-architect
  - unit-testing:test-automator
```

### Namespace Mapping Table

| Bare Name | Correct Reference | Plugin |
|-----------|-------------------|--------|
| `code-reviewer` | `comprehensive-review:code-reviewer` | comprehensive-review |
| `architect-review` | `comprehensive-review:architect-review` | comprehensive-review |
| `backend-architect` | `backend-development:backend-architect` | backend-development |
| `api-specialist` | `backend-development:api-specialist` | backend-development |
| `database-expert` | `backend-development:database-expert` | backend-development |
| `frontend-developer` | `frontend-mobile-development:frontend-developer` | frontend-mobile-development |
| `mobile-developer` | `frontend-mobile-development:mobile-developer` | frontend-mobile-development |
| `performance-engineer` | `full-stack-orchestration:performance-engineer` | full-stack-orchestration |
| `security-auditor` | `full-stack-orchestration:security-auditor` | full-stack-orchestration |
| `test-automator` | `unit-testing:test-automator` | unit-testing |
| `debugger` | `debugging-toolkit:debugger` | debugging-toolkit |
| `dx-optimizer` | `debugging-toolkit:dx-optimizer` | debugging-toolkit |

### Auto-Fix Behavior

**Automatic** (if unambiguous):
```markdown
# Before
"debugger"

# After (only one agent named "debugger" exists)
"debugging-toolkit:debugger"
```

**Manual** (if ambiguous):
```
[SYNTAX_002] Ambiguous agent name: 'architect'
Multiple matches found:
  - comprehensive-review:architect-review
  - backend-development:backend-architect
❌ Cannot auto-fix: Manual selection required
```

---

## Rule 3: Agent File Existence

**Rule ID**: `REFERENCE_001`
**Severity**: ERROR
**Auto-fixable**: ❌ No

### Description

All referenced agents must have corresponding files in the plugin's `agents/` directory.

### Why This Matters

Missing agent files cause runtime errors when the orchestration system attempts to load agent context.

### Examples

#### Example 1: Non-Existent Agent

**❌ INVALID**:
```markdown
<!-- commands/code-migrate.md -->
agents:
  - backend-development:backend-architect
  - backend-development:migration-specialist  # Does not exist!
```

**Error message**:
```
[REFERENCE_001] commands/code-migrate.md:12
Agent file not found: 'backend-development:migration-specialist'
Expected: plugins/backend-development/agents/migration-specialist.md
💡 Create the agent file or remove the reference
```

**✅ VALID (Option 1: Create file)**:
```bash
# Create the missing agent
touch plugins/backend-development/agents/migration-specialist.md
```

**✅ VALID (Option 2: Remove reference)**:
```markdown
agents:
  - backend-development:backend-architect
  # migration-specialist removed
```

#### Example 2: Typo in Agent Name

**❌ INVALID**:
```markdown
agents:
  - unit-testing:test-auomator  # Typo: "auomator" instead of "automator"
```

**Error message**:
```
[REFERENCE_001] Agent file not found: 'unit-testing:test-auomator'
Expected: plugins/unit-testing/agents/test-auomator.md
💡 Did you mean: unit-testing:test-automator?
Available agents in unit-testing:
  - test-automator
  - test-generator
```

**✅ VALID**:
```markdown
agents:
  - unit-testing:test-automator
```

### Validation Process

The linter checks:
1. Plugin directory exists: `plugins/{plugin-name}/`
2. Agent directory exists: `plugins/{plugin-name}/agents/`
3. Agent file exists: `plugins/{plugin-name}/agents/{agent-name}.md`
4. File is readable and not empty

### Troubleshooting

**Issue**: False positive for custom plugins

**Solution**: Ensure plugin follows structure:
```
plugins/my-custom-plugin/
├── plugin.json
├── agents/
│   ├── my-agent.md
│   └── another-agent.md
└── commands/
    └── my-command.md
```

---

## Rule 4: plugin.json Structure

**Rule ID**: `METADATA_001`
**Severity**: ERROR
**Auto-fixable**: ❌ No (requires manual correction)

### Description

Every plugin must have a valid `plugin.json` file with required fields and correct structure.

### Required Fields

```json
{
  "name": "string (required)",
  "version": "string (required, semver)",
  "description": "string (required)",
  "agents": "array (optional)",
  "commands": "array (optional)",
  "skills": "array (optional)",
  "keywords": "array (optional)"
}
```

### Examples

#### Example 1: Missing Required Fields

**❌ INVALID**:
```json
{
  "name": "my-plugin"
}
```

**Error message**:
```
[METADATA_001] plugin.json validation failed
Missing required fields:
  - version
  - description
```

**✅ VALID**:
```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "My custom plugin for X, Y, Z workflows"
}
```

#### Example 2: Invalid Version Format

**❌ INVALID**:
```json
{
  "name": "my-plugin",
  "version": "1.0",  // Not semver
  "description": "My plugin"
}
```

**Error message**:
```
[METADATA_001] Invalid version format: '1.0'
Expected semantic versioning (e.g., 1.0.0, 1.2.3-beta.1)
```

**✅ VALID**:
```json
{
  "version": "1.0.0"
}
```

#### Example 3: Agent Metadata Mismatch

**❌ INVALID**:
```json
{
  "name": "backend-development",
  "version": "1.0.0",
  "description": "Backend workflows",
  "agents": [
    {
      "name": "backend-architect",
      "description": "System architecture expert"
    },
    {
      "name": "migration-specialist",  // File missing!
      "description": "Migration expert"
    }
  ]
}
```

**Error message**:
```
[METADATA_001] Agent listed in plugin.json but file missing
Agent: migration-specialist
Expected: agents/migration-specialist.md
```

**✅ VALID**:
```json
{
  "agents": [
    {
      "name": "backend-architect",
      "description": "System architecture expert",
      "status": "active"
    }
  ]
}
```

#### Example 4: Duplicate Agent Names

**❌ INVALID**:
```json
{
  "agents": [
    {"name": "backend-architect", "description": "Architect"},
    {"name": "backend-architect", "description": "Another architect"}
  ]
}
```

**Error message**:
```
[METADATA_001] Duplicate agent name: 'backend-architect'
Each agent name must be unique within a plugin
```

### Complete Valid Example

```json
{
  "name": "backend-development",
  "version": "1.0.3",
  "description": "Backend development workflows with API design, database optimization, and testing",
  "agents": [
    {
      "name": "backend-architect",
      "description": "Expert in system architecture, API design, and scalability",
      "status": "active",
      "tags": ["architecture", "api", "design"]
    },
    {
      "name": "database-expert",
      "description": "Database optimization, query tuning, and schema design",
      "status": "active",
      "tags": ["database", "sql", "optimization"]
    }
  ],
  "commands": [
    {
      "name": "/api-design",
      "description": "Design RESTful APIs with best practices"
    }
  ],
  "keywords": [
    "backend",
    "api",
    "database",
    "architecture",
    "rest",
    "graphql"
  ],
  "dependencies": {
    "comprehensive-review": ">=1.0.0",
    "unit-testing": ">=1.0.0"
  }
}
```

---

## Rule 5: SKILL.md Frontmatter

**Rule ID**: `SKILL_001`
**Severity**: WARNING
**Auto-fixable**: ❌ No

### Description

Skill files must have valid YAML frontmatter with required metadata.

### Required Frontmatter

```yaml
---
name: skill-name
description: Brief description
version: 1.0.0
type: skill  # or "agent-skill", "command-skill"
tags:
  - tag1
  - tag2
---
```

### Examples

#### Example 1: Missing Frontmatter

**❌ INVALID**:
```markdown
# My Skill

This is a skill without frontmatter.
```

**Warning message**:
```
[SKILL_001] skills/my-skill/SKILL.md
Missing YAML frontmatter
Expected format:
---
name: my-skill
description: Description here
version: 1.0.0
---
```

**✅ VALID**:
```markdown
---
name: my-skill
description: Comprehensive skill for X
version: 1.0.0
type: skill
tags:
  - validation
  - automation
---

# My Skill

Skill content here.
```

#### Example 2: Invalid YAML Syntax

**❌ INVALID**:
```markdown
---
name: my-skill
description: Missing closing quote
version: 1.0.0"
---
```

**Error message**:
```
[SKILL_001] Invalid YAML syntax in frontmatter
Line 4: Unexpected character '"' at position 15
```

---

## Rule 6: Cross-Plugin References

**Rule ID**: `REFERENCE_002`
**Severity**: WARNING
**Auto-fixable**: ❌ No

### Description

Cross-plugin agent references must point to active, existing agents in other plugins.

### Examples

#### Example 1: Valid Cross-Plugin Reference

**✅ VALID**:
```markdown
<!-- plugins/backend-development/commands/api-design.md -->
agents:
  - backend-development:backend-architect  # Same plugin
  - comprehensive-review:code-reviewer     # Cross-plugin (valid)
  - unit-testing:test-automator           # Cross-plugin (valid)
```

#### Example 2: Invalid Cross-Plugin Reference

**❌ INVALID**:
```markdown
<!-- plugins/backend-development/commands/api-design.md -->
agents:
  - backend-development:backend-architect
  - code-quality:reviewer  # Plugin 'code-quality' doesn't exist
```

**Error message**:
```
[REFERENCE_002] Cross-plugin reference validation failed
Agent: code-quality:reviewer
Plugin 'code-quality' not found in plugins/
💡 Available plugins: backend-development, comprehensive-review, unit-testing, ...
```

---

## Validation Severity Levels

| Level | Description | Action |
|-------|-------------|--------|
| **ERROR** | Breaks functionality | Must fix before commit |
| **WARNING** | Potential issues | Should fix soon |
| **INFO** | Suggestions | Optional improvements |

### Exit Codes

```bash
# Success (no errors)
/lint-plugins
# Exit code: 0

# Errors found
/lint-plugins
# Exit code: 1

# Warnings only
/lint-plugins
# Exit code: 0 (warnings don't fail)
```

---

## Auto-Fix Summary

| Rule ID | Rule Name | Auto-Fixable | Method |
|---------|-----------|--------------|--------|
| SYNTAX_001 | Single colon format | ✅ Yes | String replacement `::` → `:` |
| SYNTAX_002 | Namespace required | ✅ Partial | Lookup in namespace mapping |
| REFERENCE_001 | Agent file exists | ❌ No | Manual file creation |
| METADATA_001 | plugin.json structure | ❌ No | Manual JSON editing |
| SKILL_001 | SKILL.md frontmatter | ❌ No | Manual YAML editing |
| REFERENCE_002 | Cross-plugin refs | ❌ No | Manual reference update |

---

## Validation Report Format

### Summary Report

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
  Double colon (::) in agent reference: 'comprehensive-review::code-reviewer'
  💡 Auto-fix: Change to 'comprehensive-review:code-reviewer'

  [REFERENCE_001] custom-commands/commands/smart-fix.md:92
  Agent file not found: 'debugging-toolkit:debuger'
  💡 Did you mean: debugging-toolkit:debugger?

  [METADATA_001] custom-plugin/plugin.json
  Missing required field: 'version'

────────────────────────────────────────────────────────────────────────────
🟡 WARNINGS (Should Fix)
────────────────────────────────────────────────────────────────────────────

  [SYNTAX_002] backend-development/commands/optimize.md:45
  Bare agent name without namespace: 'performance-engineer'
  💡 Suggestion: full-stack-orchestration:performance-engineer

────────────────────────────────────────────────────────────────────────────
💡 SUGGESTIONS
────────────────────────────────────────────────────────────────────────────

  Run '/lint-plugins --fix' to automatically correct 1 fixable issue(s)

  Manual fixes required for 2 error(s)

⚠️  Found 3 error(s) that must be fixed before deployment.
```

---

## Best Practices

1. **Run linter before commits**: Catch errors early
2. **Use --fix for syntax errors**: Auto-correct when possible
3. **Keep namespace mapping updated**: Document custom agents
4. **Validate in CI/CD**: Prevent broken references in PRs
5. **Test after auto-fix**: Verify changes are correct
6. **Review cross-plugin deps**: Understand coupling
7. **Version plugin.json**: Track metadata changes

---

## Troubleshooting

### Issue: Auto-fix changes wrong references

**Cause**: Ambiguous agent names
**Solution**: Review changes, manually correct if needed

### Issue: False positives for valid references

**Cause**: Plugin not in standard location
**Solution**: Update plugin path in linter config

### Issue: Linter too slow

**Cause**: Scanning many large files
**Solution**: Use `--plugin=name` to validate specific plugin

---

This guide covers all validation rules enforced by `/lint-plugins` with examples and auto-fix capabilities.
