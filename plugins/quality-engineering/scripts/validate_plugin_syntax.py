import os
import sys
import yaml
import json
import re
from pathlib import Path

def validate_yaml_frontmatter(content):
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return False, "Missing YAML frontmatter"
    try:
        yaml.safe_load(match.group(1))
        return True, "Valid YAML"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML: {e}"

def validate_plugin_structure(plugin_path):
    plugin_path = Path(plugin_path)
    issues = []

    # Check for plugin.json (root or .claude-plugin/)
    manifest_path = plugin_path / "plugin.json"
    if not manifest_path.exists():
        manifest_path = plugin_path / ".claude-plugin" / "plugin.json"

    if not manifest_path.exists():
        issues.append("Missing plugin.json (checked root and .claude-plugin/)")

    # Check README.md
    if not (plugin_path / "README.md").exists():
        issues.append("Missing README.md")

    # Check agent definitions
    agents_dir = plugin_path / "agents"
    if agents_dir.exists():
        for agent_file in agents_dir.glob("*.md"):
            with open(agent_file, 'r') as f:
                content = f.read()
                valid, msg = validate_yaml_frontmatter(content)
                if not valid:
                    issues.append(f"Agent {agent_file.name}: {msg}")

    # Check command definitions
    commands_dir = plugin_path / "commands"
    if commands_dir.exists():
        for cmd_file in commands_dir.glob("*.md"):
            with open(cmd_file, 'r') as f:
                content = f.read()
                valid, msg = validate_yaml_frontmatter(content)
                if not valid:
                    issues.append(f"Command {cmd_file.name}: {msg}")

    return issues

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_plugin_syntax.py <plugin_path>")
        sys.exit(1)

    path = sys.argv[1]
    issues = validate_plugin_structure(path)

    if issues:
        print("Validation FAILED:")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)
    else:
        print("Validation PASSED")
        sys.exit(0)
