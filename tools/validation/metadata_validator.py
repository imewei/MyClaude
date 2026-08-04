#!/usr/bin/env python3
"""
Plugin Metadata Validator

Validates plugin.json metadata against schema requirements:
- JSON schema compliance
- Required fields presence
- Semantic versioning format
- Tag and category assignments
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

# Allow `python tools/validation/metadata_validator.py ...` CLI invocation by
# adding the repo root to sys.path before resolving the `tools` package. The
# Makefile already sets `PYTHONPATH=.`; this ensures ad-hoc CLI runs work too.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.common.models import ValidationResult


class MetadataValidator:
    """Plugin metadata validator"""

    # Schema definition
    SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r"^[a-z0-9]+(-[a-z0-9]+)*$",
                "description": "Plugin name in kebab-case",
            },
            "version": {
                "type": "string",
                "pattern": r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$",
                "description": "Semantic version (e.g., 1.0.0)",
            },
            "description": {
                "type": "string",
                "min_length": 20,
                "max_length": 500,
                "description": "Brief plugin description",
            },
            "author": {
                "type": ["string", "object"],
                "description": "Author name or object with name/url",
            },
            "license": {
                "type": "string",
                "enum": ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "ISC"],
                "description": "Open source license identifier",
            },
        },
        "recommended": {
            "agents": {
                "type": "array",
                "min_items": 1,
                "description": "List of agent definitions",
            },
            "commands": {"type": "array", "description": "List of command definitions"},
            "skills": {"type": "array", "description": "List of skill definitions"},
            "keywords": {
                "type": "array",
                "min_items": 3,
                "description": "List of searchable keywords",
            },
            "category": {
                "type": "string",
                "enum": ["core", "dev", "science", "research"],
                "description": "Plugin category",
            },
        },
        "optional": {
            "homepage": {
                "type": "string",
                "pattern": r"^https?://.+",
                "description": "Plugin homepage URL",
            },
            "repository": {
                "type": ["string", "object"],
                "description": "Repository URL or object",
            },
            "bugs": {
                "type": ["string", "object"],
                "description": "Bug tracker URL or object",
            },
            "dependencies": {"type": "object", "description": "Plugin dependencies"},
            "engines": {"type": "object", "description": "Required engine versions"},
            "hooks": {
                "type": ["string", "array", "object"],
                "description": "Hook config path(s) or inline configuration (v2.1.9+)",
            },
            "lspServers": {
                "type": ["string", "array", "object"],
                "description": "Language Server Protocol configurations",
            },
            "outputStyles": {
                "type": ["string", "array"],
                "description": "Output style files or directories",
            },
        },
    }

    # Agent schema — aligned with Claude Code v2.1.88 subagent frontmatter spec
    AGENT_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r"^[a-z0-9]+(-[a-z0-9]+)*$",
                "description": "Agent name in kebab-case",
            },
            "description": {
                "type": "string",
                "min_length": 20,
                "description": "Agent description for delegation routing",
            },
        },
        "optional": {
            "tools": {
                "type": "string",
                "description": "Comma-separated list of allowed tools",
            },
            "disallowedTools": {
                "type": "string",
                "description": "Comma-separated list of denied tools",
            },
            "model": {
                "type": "string",
                "enum": ["sonnet", "opus", "haiku", "inherit"],
                "description": "Model to use (default: inherit)",
            },
            "permissionMode": {
                "type": "string",
                "enum": [
                    "default",
                    "acceptEdits",
                    "delegate",
                    "dontAsk",
                    "bypassPermissions",
                    "plan",
                    "auto",
                ],
                "description": "Permission handling mode",
            },
            "maxTurns": {
                "type": "integer",
                "min": 1,
                "description": "Maximum agentic turns before stopping",
            },
            "effort": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Model reasoning depth (v2.1.88+)",
            },
            "background": {
                "type": "boolean",
                "description": "Run agent in background (v2.1.88+)",
            },
            "isolation": {
                "type": "string",
                "enum": ["worktree"],
                "description": "Run agent in isolated git worktree (v2.1.88+)",
            },
            "skills": {
                "type": "array",
                "description": "Skills to preload into agent context",
            },
            "mcpServers": {
                "type": ["string", "object", "array"],
                "description": "MCP server configurations",
            },
            "hooks": {
                "type": ["string", "object"],
                "description": "Lifecycle hooks scoped to this agent",
            },
            "memory": {
                "type": "string",
                "enum": ["user", "project", "local"],
                "description": "Persistent memory scope (v2.1.40+)",
            },
        },
    }

    # Command schema
    COMMAND_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r"^/?[a-z0-9]+(-[a-z0-9]+)*$",
                "description": "Command name (with or without leading /)",
            },
            "description": {
                "type": "string",
                "min_length": 10,
                "description": "Command description",
            },
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "beta", "deprecated"],
                "description": "Command status",
            },
        },
        "optional": {
            "priority": {
                "type": "integer",
                "min": 1,
                "max": 10,
                "description": "Command priority (1=highest)",
            },
            "parameters": {"type": "array", "description": "Command parameters"},
        },
    }

    # Skill schema
    SKILL_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r"^[a-z0-9]+(-[a-z0-9]+)*$",
                "description": "Skill name in kebab-case",
            },
            "description": {
                "type": "string",
                "min_length": 10,
                "description": "Skill description",
            },
        },
        "optional": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "beta", "deprecated"],
                "description": "Skill status",
            },
            "tags": {"type": "array", "description": "Skill tags"},
        },
    }

    def __init__(self):
        """Initialize the validator"""

    def validate_plugin_json(self, plugin_path: Path) -> ValidationResult:
        """Validate plugin.json file"""
        plugin_name = plugin_path.name
        result = ValidationResult(plugin_name=plugin_name, is_valid=True)

        plugin_json_path = plugin_path / ".claude-plugin" / "plugin.json"

        # Check file exists
        if not plugin_json_path.exists():
            result.add_error("file", f"plugin.json not found at {plugin_json_path}")
            return result

        # Read and parse JSON
        try:
            with open(plugin_json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            result.add_error("json", f"Invalid JSON syntax: {e}")
            return result
        except OSError as e:
            result.add_error("file", f"Failed to read plugin.json: {e}")
            return result

        # Validate required fields
        self._validate_fields(metadata, self.SCHEMA["required"], result, required=True)

        # Validate recommended fields
        self._validate_fields(
            metadata, self.SCHEMA["recommended"], result, required=False
        )

        # Validate optional fields (if present)
        for field_name, field_schema in self.SCHEMA["optional"].items():
            if field_name in metadata:
                self._validate_field(
                    field_name, metadata[field_name], field_schema, result
                )

        # Validate nested structures
        if "agents" in metadata and isinstance(metadata["agents"], list):
            self._validate_agents(metadata["agents"], result)

        if "commands" in metadata and isinstance(metadata["commands"], list):
            self._validate_commands(metadata["commands"], result)

        if "skills" in metadata and isinstance(metadata["skills"], list):
            self._validate_skills(metadata["skills"], result, plugin_path)

        return result

    def _validate_fields(
        self,
        metadata: dict[str, Any],
        schema: dict[str, Any],
        result: ValidationResult,
        required: bool,
    ):
        """Validate a set of fields"""
        for field_name, field_schema in schema.items():
            if field_name not in metadata:
                if required:
                    result.add_error(
                        field_name,
                        f"Missing required field: {field_name}",
                        f"Add '{field_name}': {field_schema['description']}",
                    )
                else:
                    result.add_warning(
                        field_name,
                        f"Missing recommended field: {field_name}",
                        f"Consider adding: {field_schema['description']}",
                    )
            else:
                self._validate_field(
                    field_name, metadata[field_name], field_schema, result
                )

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        schema: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate a single field"""
        # Validation chain
        if not self._validate_type_constraint(field_name, value, schema, result):
            return

        self._validate_pattern_constraint(field_name, value, schema, result)
        self._validate_string_length_constraint(field_name, value, schema, result)
        self._validate_enum_constraint(field_name, value, schema, result)
        self._validate_array_constraint(field_name, value, schema, result)
        self._validate_numeric_constraint(field_name, value, schema, result)

    def _validate_type_constraint(
        self,
        field_name: str,
        value: Any,
        schema: dict[str, Any],
        result: ValidationResult,
    ) -> bool:
        """Validate type constraint. Returns False if type check fails."""
        expected_type = schema.get("type")
        if not expected_type:
            return True

        if isinstance(expected_type, list):
            # Multiple allowed types
            type_names = list(expected_type)
            valid_type = any(self._check_type(value, t) for t in expected_type)

            if not valid_type:
                result.add_error(
                    field_name,
                    f"Invalid type. Expected one of: {', '.join(type_names)}",
                    f"Current type: {type(value).__name__}",
                )
                return False
        else:
            # Single type
            if not self._check_type(value, expected_type):
                result.add_error(
                    field_name,
                    f"Invalid type. Expected: {expected_type}",
                    f"Current type: {type(value).__name__}",
                )
                return False
        return True

    def _validate_pattern_constraint(
        self,
        field_name: str,
        value: Any,
        schema: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate regex pattern constraint for strings"""
        if isinstance(value, str) and "pattern" in schema:
            pattern = schema["pattern"]
            if not re.match(pattern, value):
                result.add_error(
                    field_name,
                    f"Invalid format: '{value}'",
                    f"Expected pattern: {schema.get('description', pattern)}",
                )

    def _validate_string_length_constraint(
        self,
        field_name: str,
        value: Any,
        schema: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate string length constraints"""
        if isinstance(value, str):
            if "min_length" in schema and len(value) < schema["min_length"]:
                result.add_error(
                    field_name,
                    f"Too short (min {schema['min_length']} chars)",
                    f"Current length: {len(value)}",
                )
            if "max_length" in schema and len(value) > schema["max_length"]:
                result.add_warning(
                    field_name,
                    f"Too long (max {schema['max_length']} chars recommended)",
                    f"Current length: {len(value)}",
                )

    def _validate_enum_constraint(
        self,
        field_name: str,
        value: Any,
        schema: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate enum constraints"""
        if "enum" in schema and value not in schema["enum"]:
            result.add_error(
                field_name,
                f"Invalid value: '{value}'",
                f"Allowed values: {', '.join(map(str, schema['enum']))}",
            )

    def _validate_array_constraint(
        self,
        field_name: str,
        value: Any,
        schema: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate array constraints"""
        if (
            isinstance(value, list)
            and "min_items" in schema
            and len(value) < schema["min_items"]
        ):
            result.add_warning(
                field_name,
                f"Should have at least {schema['min_items']} items",
                f"Current count: {len(value)}",
            )

    def _validate_numeric_constraint(
        self,
        field_name: str,
        value: Any,
        schema: dict[str, Any],
        result: ValidationResult,
    ):
        """Validate numeric min/max constraints"""
        if isinstance(value, int):
            if "min" in schema and value < schema["min"]:
                result.add_error(
                    field_name,
                    f"Value too small (min: {schema['min']})",
                    f"Current value: {value}",
                )
            if "max" in schema and value > schema["max"]:
                result.add_error(
                    field_name,
                    f"Value too large (max: {schema['max']})",
                    f"Current value: {value}",
                )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_mapping: dict[str, Any] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected = type_mapping.get(expected_type)
        if expected is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected)

    def _validate_agents(self, agents: list[dict[str, Any]], result: ValidationResult):
        """Validate agents array (supports both file paths and inline objects)"""
        if not agents:
            result.add_warning("agents", "Agents array is empty")
            return

        for idx, agent in enumerate(agents):
            if isinstance(agent, str):
                # File path reference format: "./agents/orchestrator.md"
                if not agent.endswith(".md"):
                    result.add_warning(
                        f"agents[{idx}]",
                        f"Agent file path should end with .md: {agent}",
                    )
                continue

            if not isinstance(agent, dict):
                result.add_error(
                    f"agents[{idx}]", "Agent must be a string path or object"
                )
                continue

            # Validate required fields for inline object format
            for field_name, field_schema in self.AGENT_SCHEMA["required"].items():
                if field_name not in agent:
                    result.add_error(
                        f"agents[{idx}].{field_name}",
                        f"Missing required field in agent {idx}",
                    )
                else:
                    self._validate_field(
                        f"agents[{idx}].{field_name}",
                        agent[field_name],
                        field_schema,
                        result,
                    )

    def _validate_commands(
        self, commands: list[dict[str, Any]], result: ValidationResult
    ):
        """Validate commands array (supports both file paths and inline objects)"""
        for idx, command in enumerate(commands):
            if isinstance(command, str):
                # File path reference format: "./commands/commit.md"
                if not command.endswith(".md"):
                    result.add_warning(
                        f"commands[{idx}]",
                        f"Command file path should end with .md: {command}",
                    )
                continue

            if not isinstance(command, dict):
                result.add_error(
                    f"commands[{idx}]", "Command must be a string path or object"
                )
                continue

            # Validate required fields for inline object format
            for field_name, field_schema in self.COMMAND_SCHEMA["required"].items():
                if field_name not in command:
                    result.add_error(
                        f"commands[{idx}].{field_name}",
                        f"Missing required field in command {idx}",
                    )
                else:
                    self._validate_field(
                        f"commands[{idx}].{field_name}",
                        command[field_name],
                        field_schema,
                        result,
                    )

            # Validate optional fields if present
            for field_name, field_schema in self.COMMAND_SCHEMA["optional"].items():
                if field_name in command:
                    self._validate_field(
                        f"commands[{idx}].{field_name}",
                        command[field_name],
                        field_schema,
                        result,
                    )

    # Skills that lack a routing tree but are explicitly approved as user-facing standalones.
    # These may be added to plugin.json without triggering the tier-compliance warning.
    _TIER2_STANDALONE_WHITELIST: ClassVar[set[str]] = {
        "three-brain",      # dev-suite: multi-model routing (one-shot review or persistent team)
        "scientific-review",  # research-suite: peer-review deliverable
        "research-spark",     # research-suite: 5-stage core + optional extension orchestrator
        "spark-articulator",  # research-suite: stage 1
        "landscape-scanner",  # research-suite: stage 2
        "falsifiable-claim",  # research-suite: stage 3
        "theory-scaffold",    # research-suite: stages 4-5
        "numerical-prototype",  # research-suite: stage 6
        "experiment-designer",  # research-suite: stage 7
        "premortem-critique",   # research-suite: stage 8
    }

    def _validate_skills(
        self,
        skills: list[dict[str, Any]],
        result: ValidationResult,
        plugin_path: Path | None = None,
    ):
        """Validate skills array (supports both directory paths and inline objects).

        Tier-compliance: registered skills should be routing hubs (have a
        '## Routing Decision Tree' section) or appear in the standalone whitelist.
        Sub-skills that are only reachable via hub routing trees must NOT be
        registered here — they inflate the session context budget.
        """
        for idx, skill in enumerate(skills):
            if isinstance(skill, str):
                # Directory path reference format: "./skills/advanced-reasoning"
                # Skills reference directories, not .md files
                self._check_skill_tier_compliance(skill, idx, result, plugin_path)
                continue

            if not isinstance(skill, dict):
                result.add_error(
                    f"skills[{idx}]", "Skill must be a string path or object"
                )
                continue

            # Validate required fields for inline object format
            for field_name, field_schema in self.SKILL_SCHEMA["required"].items():
                if field_name not in skill:
                    result.add_error(
                        f"skills[{idx}].{field_name}",
                        f"Missing required field in skill {idx}",
                    )
                else:
                    self._validate_field(
                        f"skills[{idx}].{field_name}",
                        skill[field_name],
                        field_schema,
                        result,
                    )

    def _check_skill_tier_compliance(
        self,
        skill_path: str,
        idx: int,
        result: ValidationResult,
        plugin_path: Path | None,
    ) -> None:
        """Warn when a Tier-3 sub-skill (no routing tree) is registered in plugin.json.

        A registered skill must either:
        - Have a '## Routing Decision Tree' header in its SKILL.md, OR
        - Be in the _TIER2_STANDALONE_WHITELIST
        """
        if plugin_path is None:
            return

        # Derive skill name from path like "./skills/jax-mastery"
        skill_name = skill_path.rstrip("/").split("/")[-1]
        if skill_name in self._TIER2_STANDALONE_WHITELIST:
            return

        skill_md = plugin_path / skill_path.lstrip("./") / "SKILL.md"
        if not skill_md.exists():
            return  # Missing file handled elsewhere

        content = skill_md.read_text(encoding="utf-8")
        if "## Routing Decision Tree" not in content:
            result.add_warning(
                f"skills[{idx}]",
                f"'{skill_name}' appears to be a Tier-3 sub-skill (no routing decision "
                f"tree). Registering sub-skills inflates the session context budget. "
                f"Remove it from plugin.json — hubs will route to it via the Skill tool.",
                "Only register hub skills (meta-orchestrators with routing trees) "
                "and approved standalones in plugin.json.",
            )

    def generate_report(self, result: ValidationResult) -> str:
        """Generate validation report"""
        lines = []
        lines.append(f"# Metadata Validation Report: {result.plugin_name}\n")

        errors = result.errors
        warnings = result.warnings

        # Status
        if result.is_valid and not warnings:
            lines.append("**Status:** ✅ VALID - No issues found\n")
        elif result.is_valid:
            lines.append(f"**Status:** ✅ VALID - {len(warnings)} warnings\n")
        else:
            lines.append(f"**Status:** ❌ INVALID - {len(errors)} errors\n")

        # Summary
        lines.append("## Summary\n")
        lines.append(f"- **Errors:** {len(errors)}")
        lines.append(f"- **Warnings:** {len(warnings)}\n")

        # Errors
        if errors:
            lines.append("## Errors\n")
            for error in errors:
                lines.append(f"❌ **{error.field}**: {error.message}")
                if error.suggestion:
                    lines.append(f"   → {error.suggestion}")
                lines.append("")

        # Warnings
        if warnings:
            lines.append("## Warnings\n")
            for warning in warnings:
                lines.append(f"⚠️  **{warning.field}**: {warning.message}")
                if warning.suggestion:
                    lines.append(f"   → {warning.suggestion}")
                lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python metadata_validator.py <plugin-path>")
        print("\nExample:")
        print("  python metadata_validator.py plugins/julia-development")
        sys.exit(1)

    plugin_path = Path(sys.argv[1])

    if not plugin_path.exists():
        print(f"Error: Plugin path not found: {plugin_path}")
        sys.exit(1)

    validator = MetadataValidator()
    result = validator.validate_plugin_json(plugin_path)

    # Generate and print report
    report = validator.generate_report(result)
    print(report)

    # Exit with appropriate code
    if not result.is_valid:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
