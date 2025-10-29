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
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import re
from dataclasses import dataclass, field


@dataclass
class ValidationError:
    """Represents a validation error"""
    field: str
    severity: str  # error, warning
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Results of metadata validation"""
    plugin_name: str
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str, suggestion: Optional[str] = None):
        """Add a validation error"""
        self.errors.append(ValidationError(field, "error", message, suggestion))
        self.is_valid = False

    def add_warning(self, field: str, message: str, suggestion: Optional[str] = None):
        """Add a validation warning"""
        self.warnings.append(ValidationError(field, "warning", message, suggestion))


class MetadataValidator:
    """Plugin metadata validator"""

    # Schema definition
    SCHEMA = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r'^[a-z0-9]+(-[a-z0-9]+)*$',
                "description": "Plugin name in kebab-case"
            },
            "version": {
                "type": "string",
                "pattern": r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$',
                "description": "Semantic version (e.g., 1.0.0)"
            },
            "description": {
                "type": "string",
                "min_length": 20,
                "max_length": 500,
                "description": "Brief plugin description"
            },
            "author": {
                "type": ["string", "object"],
                "description": "Author name or object with name/url"
            },
            "license": {
                "type": "string",
                "enum": ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "ISC"],
                "description": "Open source license identifier"
            }
        },
        "recommended": {
            "agents": {
                "type": "array",
                "min_items": 1,
                "description": "List of agent definitions"
            },
            "commands": {
                "type": "array",
                "description": "List of command definitions"
            },
            "skills": {
                "type": "array",
                "description": "List of skill definitions"
            },
            "keywords": {
                "type": "array",
                "min_items": 3,
                "description": "List of searchable keywords"
            },
            "category": {
                "type": "string",
                "enum": [
                    "scientific-computing",
                    "development",
                    "devops",
                    "quality-engineering",
                    "infrastructure",
                    "tools"
                ],
                "description": "Plugin category"
            }
        },
        "optional": {
            "homepage": {
                "type": "string",
                "pattern": r'^https?://.+',
                "description": "Plugin homepage URL"
            },
            "repository": {
                "type": ["string", "object"],
                "description": "Repository URL or object"
            },
            "bugs": {
                "type": ["string", "object"],
                "description": "Bug tracker URL or object"
            },
            "dependencies": {
                "type": "object",
                "description": "Plugin dependencies"
            },
            "engines": {
                "type": "object",
                "description": "Required engine versions"
            }
        }
    }

    # Agent schema
    AGENT_SCHEMA = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r'^[a-z0-9]+(-[a-z0-9]+)*$',
                "description": "Agent name in kebab-case"
            },
            "description": {
                "type": "string",
                "min_length": 20,
                "description": "Agent description"
            },
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "beta", "deprecated"],
                "description": "Agent status"
            }
        },
        "optional": {
            "expertise": {
                "type": "array",
                "description": "List of expertise areas"
            },
            "triggers": {
                "type": "object",
                "description": "Activation triggers"
            }
        }
    }

    # Command schema
    COMMAND_SCHEMA = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r'^/?[a-z0-9]+(-[a-z0-9]+)*$',
                "description": "Command name (with or without leading /)"
            },
            "description": {
                "type": "string",
                "min_length": 10,
                "description": "Command description"
            },
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "beta", "deprecated"],
                "description": "Command status"
            }
        },
        "optional": {
            "priority": {
                "type": "integer",
                "min": 1,
                "max": 10,
                "description": "Command priority (1=highest)"
            },
            "parameters": {
                "type": "array",
                "description": "Command parameters"
            }
        }
    }

    # Skill schema
    SKILL_SCHEMA = {
        "required": {
            "name": {
                "type": "string",
                "pattern": r'^[a-z0-9]+(-[a-z0-9]+)*$',
                "description": "Skill name in kebab-case"
            },
            "description": {
                "type": "string",
                "min_length": 10,
                "description": "Skill description"
            }
        },
        "optional": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "beta", "deprecated"],
                "description": "Skill status"
            },
            "tags": {
                "type": "array",
                "description": "Skill tags"
            }
        }
    }

    def __init__(self):
        """Initialize the validator"""
        pass

    def validate_plugin_json(self, plugin_path: Path) -> ValidationResult:
        """Validate plugin.json file"""
        plugin_name = plugin_path.name
        result = ValidationResult(plugin_name=plugin_name, is_valid=True)

        plugin_json_path = plugin_path / "plugin.json"

        # Check file exists
        if not plugin_json_path.exists():
            result.add_error("file", f"plugin.json not found at {plugin_json_path}")
            return result

        # Read and parse JSON
        try:
            with open(plugin_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            result.add_error("json", f"Invalid JSON syntax: {e}")
            return result
        except Exception as e:
            result.add_error("file", f"Failed to read plugin.json: {e}")
            return result

        # Validate required fields
        self._validate_fields(metadata, self.SCHEMA["required"], result, required=True)

        # Validate recommended fields
        self._validate_fields(metadata, self.SCHEMA["recommended"], result, required=False)

        # Validate optional fields (if present)
        for field, field_schema in self.SCHEMA["optional"].items():
            if field in metadata:
                self._validate_field(field, metadata[field], field_schema, result)

        # Validate nested structures
        if "agents" in metadata and isinstance(metadata["agents"], list):
            self._validate_agents(metadata["agents"], result)

        if "commands" in metadata and isinstance(metadata["commands"], list):
            self._validate_commands(metadata["commands"], result)

        if "skills" in metadata and isinstance(metadata["skills"], list):
            self._validate_skills(metadata["skills"], result)

        return result

    def _validate_fields(self, metadata: Dict[str, Any], schema: Dict[str, Any],
                        result: ValidationResult, required: bool):
        """Validate a set of fields"""
        for field, field_schema in schema.items():
            if field not in metadata:
                if required:
                    result.add_error(
                        field,
                        f"Missing required field: {field}",
                        f"Add '{field}': {field_schema['description']}"
                    )
                else:
                    result.add_warning(
                        field,
                        f"Missing recommended field: {field}",
                        f"Consider adding: {field_schema['description']}"
                    )
            else:
                self._validate_field(field, metadata[field], field_schema, result)

    def _validate_field(self, field_name: str, value: Any, schema: Dict[str, Any],
                       result: ValidationResult):
        """Validate a single field"""
        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            if isinstance(expected_type, list):
                # Multiple allowed types
                type_names = []
                valid_type = False
                for t in expected_type:
                    type_names.append(t)
                    if self._check_type(value, t):
                        valid_type = True
                        break

                if not valid_type:
                    result.add_error(
                        field_name,
                        f"Invalid type. Expected one of: {', '.join(type_names)}",
                        f"Current type: {type(value).__name__}"
                    )
                    return
            else:
                # Single type
                if not self._check_type(value, expected_type):
                    result.add_error(
                        field_name,
                        f"Invalid type. Expected: {expected_type}",
                        f"Current type: {type(value).__name__}"
                    )
                    return

        # Pattern validation (for strings)
        if isinstance(value, str) and "pattern" in schema:
            pattern = schema["pattern"]
            if not re.match(pattern, value):
                result.add_error(
                    field_name,
                    f"Invalid format: '{value}'",
                    f"Expected pattern: {schema.get('description', pattern)}"
                )

        # Length validation (for strings)
        if isinstance(value, str):
            if "min_length" in schema and len(value) < schema["min_length"]:
                result.add_error(
                    field_name,
                    f"Too short (min {schema['min_length']} chars)",
                    f"Current length: {len(value)}"
                )
            if "max_length" in schema and len(value) > schema["max_length"]:
                result.add_warning(
                    field_name,
                    f"Too long (max {schema['max_length']} chars recommended)",
                    f"Current length: {len(value)}"
                )

        # Enum validation
        if "enum" in schema:
            if value not in schema["enum"]:
                result.add_error(
                    field_name,
                    f"Invalid value: '{value}'",
                    f"Allowed values: {', '.join(map(str, schema['enum']))}"
                )

        # Array validation
        if isinstance(value, list):
            if "min_items" in schema and len(value) < schema["min_items"]:
                result.add_warning(
                    field_name,
                    f"Should have at least {schema['min_items']} items",
                    f"Current count: {len(value)}"
                )

        # Integer validation
        if isinstance(value, int):
            if "min" in schema and value < schema["min"]:
                result.add_error(
                    field_name,
                    f"Value too small (min: {schema['min']})",
                    f"Current value: {value}"
                )
            if "max" in schema and value > schema["max"]:
                result.add_error(
                    field_name,
                    f"Value too large (max: {schema['max']})",
                    f"Current value: {value}"
                )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }

        expected = type_mapping.get(expected_type)
        if expected is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected)

    def _validate_agents(self, agents: List[Dict[str, Any]], result: ValidationResult):
        """Validate agents array"""
        if not agents:
            result.add_warning("agents", "Agents array is empty")
            return

        for idx, agent in enumerate(agents):
            if not isinstance(agent, dict):
                result.add_error(f"agents[{idx}]", "Agent must be an object")
                continue

            # Validate required fields
            for field, field_schema in self.AGENT_SCHEMA["required"].items():
                if field not in agent:
                    result.add_error(
                        f"agents[{idx}].{field}",
                        f"Missing required field in agent {idx}"
                    )
                else:
                    self._validate_field(
                        f"agents[{idx}].{field}",
                        agent[field],
                        field_schema,
                        result
                    )

    def _validate_commands(self, commands: List[Dict[str, Any]], result: ValidationResult):
        """Validate commands array"""
        for idx, command in enumerate(commands):
            if not isinstance(command, dict):
                result.add_error(f"commands[{idx}]", "Command must be an object")
                continue

            # Validate required fields
            for field, field_schema in self.COMMAND_SCHEMA["required"].items():
                if field not in command:
                    result.add_error(
                        f"commands[{idx}].{field}",
                        f"Missing required field in command {idx}"
                    )
                else:
                    self._validate_field(
                        f"commands[{idx}].{field}",
                        command[field],
                        field_schema,
                        result
                    )

            # Validate optional fields if present
            for field, field_schema in self.COMMAND_SCHEMA["optional"].items():
                if field in command:
                    self._validate_field(
                        f"commands[{idx}].{field}",
                        command[field],
                        field_schema,
                        result
                    )

    def _validate_skills(self, skills: List[Dict[str, Any]], result: ValidationResult):
        """Validate skills array"""
        for idx, skill in enumerate(skills):
            if not isinstance(skill, dict):
                result.add_error(f"skills[{idx}]", "Skill must be an object")
                continue

            # Validate required fields
            for field, field_schema in self.SKILL_SCHEMA["required"].items():
                if field not in skill:
                    result.add_error(
                        f"skills[{idx}].{field}",
                        f"Missing required field in skill {idx}"
                    )
                else:
                    self._validate_field(
                        f"skills[{idx}].{field}",
                        skill[field],
                        field_schema,
                        result
                    )

    def generate_report(self, result: ValidationResult) -> str:
        """Generate validation report"""
        lines = []
        lines.append(f"# Metadata Validation Report: {result.plugin_name}\n")

        # Status
        if result.is_valid and not result.warnings:
            lines.append("**Status:** ✅ VALID - No issues found\n")
        elif result.is_valid:
            lines.append(f"**Status:** ✅ VALID - {len(result.warnings)} warnings\n")
        else:
            lines.append(f"**Status:** ❌ INVALID - {len(result.errors)} errors\n")

        # Summary
        lines.append("## Summary\n")
        lines.append(f"- **Errors:** {len(result.errors)}")
        lines.append(f"- **Warnings:** {len(result.warnings)}\n")

        # Errors
        if result.errors:
            lines.append("## Errors\n")
            for error in result.errors:
                lines.append(f"❌ **{error.field}**: {error.message}")
                if error.suggestion:
                    lines.append(f"   → {error.suggestion}")
                lines.append("")

        # Warnings
        if result.warnings:
            lines.append("## Warnings\n")
            for warning in result.warnings:
                lines.append(f"⚠️  **{warning.field}**: {warning.message}")
                if warning.suggestion:
                    lines.append(f"   → {warning.suggestion}")
                lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python metadata-validator.py <plugin-path>")
        print("\nExample:")
        print("  python metadata-validator.py plugins/julia-development")
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
