import os
import yaml
import re

def validate_plugin(plugin_path):
    issues = []
    plugin_name = os.path.basename(plugin_path)

    manifest_path = os.path.join(plugin_path, "plugin.json")
    if not os.path.exists(manifest_path):
        issues.append(f"CRITICAL: {plugin_name} is missing plugin.json")
        return issues

    # 1. Agent Validation
    agents_dir = os.path.join(plugin_path, "agents")
    if os.path.exists(agents_dir):
        for agent_file in os.listdir(agents_dir):
            if agent_file.endswith(".md"):
                file_path = os.path.join(agents_dir, agent_file)
                with open(file_path, 'r') as f:
                    content = f.read()

                # Check frontmatter
                fm_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
                if not fm_match:
                    issues.append(f"MAJOR: {agent_file} missing frontmatter")
                    continue

                try:
                    fm = yaml.safe_load(fm_match.group(1))
                    for field in ['name', 'description', 'model', 'color']:
                        if field not in fm:
                            issues.append(f"MAJOR: {agent_file} frontmatter missing '{field}'")

                    if 'maturity' in fm:
                        issues.append(f"MAJOR: {agent_file} frontmatter contains forbidden 'maturity'")
                    if 'specialization' in fm:
                        issues.append(f"MAJOR: {agent_file} frontmatter contains forbidden 'specialization'")
                except Exception as e:
                    issues.append(f"MAJOR: {agent_file} frontmatter YAML error: {e}")

                # Check for examples
                if '<example>' not in content:
                    issues.append(f"MAJOR: {agent_file} missing <example> tags")

    # 2. Command Validation
    commands_dir = os.path.join(plugin_path, "commands")
    if os.path.exists(commands_dir):
        for cmd_file in os.listdir(commands_dir):
            if cmd_file.endswith(".md"):
                file_path = os.path.join(commands_dir, cmd_file)
                with open(file_path, 'r') as f:
                    content = f.read()

                fm_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
                if fm_match:
                    try:
                        # Use a custom loader or just check keys manually to avoid issues with complex YAML
                        lines = fm_match.group(1).split('\n')
                        for line in lines:
                            if ':' in line and not line.strip().startswith('-'):
                                key = line.split(':')[0].strip()
                                if '_' in key or any(c.isupper() for c in key):
                                    issues.append(f"MINOR: {cmd_file} frontmatter key '{key}' is not kebab-case")
                    except Exception as e:
                        issues.append(f"MINOR: {cmd_file} frontmatter check error: {e}")

    # 3. Skills Validation (Flattened)
    skills_dir = os.path.join(plugin_path, "skills")
    if os.path.exists(skills_dir):
        for root, dirs, files in os.walk(skills_dir):
            if "SKILL.md" in files:
                rel_path = os.path.relpath(root, skills_dir)
                depth = len(rel_path.split(os.sep))
                if depth > 1:
                    issues.append(f"MAJOR: Skill at {rel_path} is nested too deep (flatten to skills/name/SKILL.md)")

    return issues

plugins_root = "/Users/b80985/Projects/MyClaude/plugins"
all_issues = {}

for plugin in ["agent-core", "engineering-suite", "infrastructure-suite", "quality-suite", "science-suite"]:
    plugin_path = os.path.join(plugins_root, plugin)
    issues = validate_plugin(plugin_path)
    all_issues[plugin] = issues

import json
print(json.dumps(all_issues, indent=2))
