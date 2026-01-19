import os
import json
import yaml
import re
from pathlib import Path

def parse_frontmatter(content):
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return {}
    return {}

def analyze_ecosystem(base_path):
    plugins_dir = Path(base_path) / "plugins"
    capabilities = {
        "plugins": [],
        "capability_matrix": {}
    }

    if not plugins_dir.exists():
        print(f"Directory {plugins_dir} not found.")
        return capabilities

    # Functional domain mapping rules (keywords -> domain)
    domain_map = {
        "testing": "Quality Engineering",
        "validation": "Quality Engineering",
        "qa": "Quality Engineering",
        "optimization": "Performance Optimization",
        "performance": "Performance Optimization",
        "security": "Security",
        "vulnerability": "Security",
        "jax": "Scientific Computing",
        "numpy": "Scientific Computing",
        "scientific": "Scientific Computing",
        "hpc": "High Performance Computing",
        "parallel": "High Performance Computing",
        "monitoring": "Observability",
        "observability": "Observability",
        "deployment": "DevOps",
        "ci/cd": "DevOps",
        "docker": "DevOps"
    }

    for plugin_path in plugins_dir.iterdir():
        if not plugin_path.is_dir():
            continue

        plugin_info = {
            "name": plugin_path.name,
            "path": str(plugin_path),
            "agents": [],
            "skills": [],
            "domains": set()
        }

        # 1. Parse plugin.json
        manifest_path = plugin_path / "plugin.json"
        if not manifest_path.exists():
            manifest_path = plugin_path / ".claude-plugin" / "plugin.json"

        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    keywords = manifest.get("keywords", [])
                    categories = manifest.get("categories", [])

                    for k in keywords + categories:
                        k_lower = k.lower()
                        for key, domain in domain_map.items():
                            if key in k_lower:
                                plugin_info["domains"].add(domain)
            except Exception as e:
                print(f"Error parsing {manifest_path}: {e}")

        # 2. Parse agents
        agents_dir = plugin_path / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.md"):
                try:
                    with open(agent_file, 'r') as f:
                        content = f.read()
                        frontmatter = parse_frontmatter(content)
                        if frontmatter:
                            agent_data = {
                                "name": agent_file.stem,
                                "specialization": frontmatter.get("specialization", ""),
                                "description": frontmatter.get("description", "")
                            }
                            plugin_info["agents"].append(agent_data)

                            spec = agent_data["specialization"].lower()
                            desc = agent_data["description"].lower()
                            for key, domain in domain_map.items():
                                if key in spec or key in desc:
                                    plugin_info["domains"].add(domain)
                except Exception as e:
                    print(f"Error parsing agent {agent_file}: {e}")

        # 3. Parse skills
        skills_dir = plugin_path / "skills"
        if skills_dir.exists():
            for skill_path in skills_dir.iterdir():
                if not skill_path.is_dir():
                    continue
                skill_file = skill_path / "SKILL.md"
                if skill_file.exists():
                    try:
                        with open(skill_file, 'r') as f:
                            content = f.read()
                            frontmatter = parse_frontmatter(content)
                            if frontmatter:
                                skill_data = {
                                    "name": skill_path.name,
                                    "specialization": frontmatter.get("specialization", ""),
                                    "description": frontmatter.get("description", "")
                                }
                                plugin_info["skills"].append(skill_data)

                                spec = skill_data["specialization"].lower()
                                desc = skill_data["description"].lower()
                                for key, domain in domain_map.items():
                                    if key in spec or key in desc:
                                        plugin_info["domains"].add(domain)
                    except Exception as e:
                        print(f"Error parsing skill {skill_file}: {e}")

        plugin_info["domains"] = list(plugin_info["domains"])
        capabilities["plugins"].append(plugin_info)

        # Update capability matrix
        for domain in plugin_info["domains"]:
            if domain not in capabilities["capability_matrix"]:
                capabilities["capability_matrix"][domain] = []
            capabilities["capability_matrix"][domain].append(plugin_path.name)

    return capabilities

if __name__ == "__main__":
    repo_root = os.getcwd()
    report = analyze_ecosystem(repo_root)

    output_path = Path(repo_root) / "ecosystem_capabilities.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Generated {output_path}")
