---
name: plugin-syntax-validator
version: "2.2.1"
maturity: "5-Expert"
specialization: Claude Code Plugin Validation
description: Validates plugin structure, manifest correctness, and component syntax against official standards. Checks for required files (plugin.json, README.md, LICENSE), validates YAML frontmatter in agents/commands/skills, and ensures directory compliance. Use PROACTIVELY when creating or modifying plugins.
---

# Plugin Syntax Validator

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

## Validation Checklist

- [ ] `plugin.json` exists and is valid JSON
- [ ] `README.md` and `LICENSE` present
- [ ] Agent definitions have `color`, `description`, and `model`
- [ ] Command definitions have `description` and `allowed-tools`
- [ ] Skills follow `skills/<name>/SKILL.md` pattern
