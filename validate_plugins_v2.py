import os
import yaml
import re
import json

def get_frontmatter(content):
    if not content.startswith('---'):
        return None
    parts = content.split('---')
    if len(parts) < 3:
        return None
    return parts[1]

def is_kebab_case(s):
    return re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', s) is not None

def validate_plugin(plugin_path):
    issues = []
    plugin_name = os.path.basename(plugin_path)

    manifest_path = os.path.join(plugin_path, "plugin.json")
    if not os.path.exists(manifest_path):
        issues.append(f"CRITICAL: {plugin_name} is missing plugin.json")
        return issues

    with open(manifest_path, 'r') as f:
        try:
            manifest = json.load(f)
            desc = manifest.get('description', '')
            if any(x in desc.lower() for x in ['placeholder', 'add description', 'tbd', 'tbc', 'todo']):
                issues.append(f"MAJOR: plugin.json has placeholder description: '{desc}'")
        except Exception as e:
            issues.append(f"CRITICAL: Failed to parse plugin.json: {e}")

    # 1. Agent Validation
    agents_dir = os.path.join(plugin_path, "agents")
    if os.path.exists(agents_dir):
        for agent_file in os.listdir(agents_dir):
            if agent_file.endswith(".md"):
                file_path = os.path.join(agents_dir, agent_file)
                with open(file_path, 'r') as f:
                    content = f.read()

                fm_text = get_frontmatter(content)
                if not fm_text:
                    issues.append(f"MAJOR: {agent_file} missing frontmatter")
                    continue

                try:
                    fm = yaml.safe_load(fm_text)
                    if not fm:
                        issues.append(f"MAJOR: {agent_file} has empty frontmatter")
                        continue

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

                fm_text = get_frontmatter(content)
                if fm_text:
                    try:
                        fm = yaml.safe_load(fm_text)
                        if fm:
                            for key in fm.keys():
                                if not is_kebab_case(key):
                                    issues.append(f"MINOR: {cmd_file} frontmatter key '{key}' is not kebab-case")
                    except Exception as e:
                        issues.append(f"MINOR: {cmd_file} frontmatter check error: {e}")

    # 3. Skills Validation (Flattened)
    skills_dir = os.path.join(plugin_path, "skills")
    if os.path.exists(skills_dir):
        for skill_name in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_name)
            if os.path.isdir(skill_path):
                # Check for SKILL.md directly under skill_path
                if not os.path.exists(os.path.join(skill_path, "SKILL.md")):
                    issues.append(f"MAJOR: Skill '{skill_name}' is missing SKILL.md")

                # Check for deep nesting
                for root, dirs, files in os.walk(skill_path):
                    if root == skill_path: continue
                    if "SKILL.md" in files:
                        rel = os.path.relpath(root, skills_dir)
                        issues.append(f"MAJOR: Nested skill found at {rel}/SKILL.md (should be flattened)")

    return issues

plugins_root = "/Users/b80985/Projects/MyClaude/plugins"
all_results = {}

plugins = ["agent-core", "engineering-suite", "infrastructure-suite", "quality-suite", "science-suite"]
for plugin in plugins:
    p_path = os.path.join(plugins_root, plugin)
    all_results[plugin] = validate_plugin(p_path)

print(json.dumps(all_results, indent=2))
