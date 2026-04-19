---
name: plugin-syntax-validator
description: Validates plugin structure, manifest correctness, and component syntax against official standards. Checks for required files (plugin.json, README.md, LICENSE), validates YAML frontmatter in agents/commands/skills, and ensures directory compliance. Use when creating, modifying, or reviewing Claude Code plugins, or validating plugin.json manifests before publishing.
---

# Plugin Syntax Validator

## Expert Agent

For plugin structure validation, manifest correctness, and component syntax checking, delegate to:

- **`quality-specialist`**: Ensures compliance with architectural standards and enforces coding guidelines.
  - *Location*: `plugins/dev-suite/agents/quality-specialist.md`

Expert validation skill for Claude Code plugins. Ensures compliance with architectural standards.

## Capabilities

1.  **Structure Validation**: Checks for required directories (`agents/`, `commands/`, `skills/`) and files.
2.  **Manifest Verification**: Validates `plugin.json` schema and required fields.
3.  **Frontmatter Analysis**: Parses and validates YAML frontmatter in markdown components.
4.  **Best Practices**: Checks for recommended patterns (colors, descriptions, examples).

## Usage

```bash
# Validate current plugin
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/validate_plugin_syntax.py .
```

## plugin.json Schema Requirements

```json
{
  "name": "suite-name",
  "version": "3.0.0",
  "description": "Suite description",
  "agents": [
    { "$ref": "agents/my-agent.md" }
  ],
  "commands": [
    { "$ref": "commands/my-command.md" }
  ],
  "skills": [
    { "$ref": "skills/my-skill/SKILL.md" }
  ]
}
```

| Field | Required | Validation Rule |
|-------|----------|-----------------|
| `name` | Yes | Match directory name |
| `version` | Yes | Semver format, consistent across all suites |
| `description` | Yes | Non-empty string |
| `agents` | Yes | Array of `$ref` paths pointing to existing `.md` files |
| `commands` | No | Array of `$ref` paths pointing to existing `.md` files |
| `skills` | No | Array of `$ref` paths pointing to existing `SKILL.md` files |

## Frontmatter Validation Rules

**Agents** must include:
```yaml
---
name: agent-name           # kebab-case, matches filename
description: "..."         # Non-empty
model: opus|sonnet|haiku   # Valid model tier
effort: low|medium|high    # Processing effort
---
```

**Commands** must include:
```yaml
---
name: command-name         # kebab-case, matches filename
description: "..."         # Non-empty
---
```

**Skills** must include:
```yaml
---
name: skill-name           # kebab-case, matches directory name
description: "..."         # Non-empty, under 300 chars
---
```

## Cross-Reference Validation

- Every `$ref` in plugin.json must resolve to an existing file.
- Every agent/command/skill file must be referenced by exactly one plugin.json.
- Skill names in frontmatter must match their directory name.
- Agent `model` field must use a valid tier (opus, sonnet, haiku).

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing $ref target` | File path typo in plugin.json | Correct the path to match actual file location |
| `Name mismatch` | Frontmatter `name` differs from filename | Align `name` field with filename (without `.md`) |
| `Invalid model tier` | Typo in model field | Use exactly `opus`, `sonnet`, or `haiku` |
| `Orphan component` | File exists but not in manifest | Add `$ref` entry to plugin.json |
| `Duplicate name` | Two components share a name | Rename one to be unique within the suite |

## Validation Checklist

- [ ] `plugin.json` exists, is valid JSON, and follows schema above
- [ ] `README.md` and `LICENSE` present in suite root
- [ ] All `$ref` paths in plugin.json resolve to existing files
- [ ] Agent frontmatter includes `name`, `description`, `model`, `effort`
- [ ] Command frontmatter includes `name` and `description`
- [ ] Skills follow `skills/<name>/SKILL.md` directory pattern
- [ ] Frontmatter `name` fields match filenames/directory names
- [ ] Version string is consistent across all plugin.json manifests
- [ ] No orphan components (every file referenced in manifest)
