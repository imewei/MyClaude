import os
import json
import yaml
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Union

# Functional domain mapping rules (keywords -> domain)
DOMAIN_MAP: Dict[str, str] = {
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

def parse_frontmatter(content: str) -> Dict[str, Any]:
    """Parses YAML frontmatter from a markdown string."""
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if match:
        try:
            data = yaml.safe_load(match.group(1))
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}
    return {}

def extract_domains(texts: List[str], domain_map: Dict[str, str]) -> Set[str]:
    """Extracts domains from a list of texts based on a mapping of keywords."""
    found_domains: Set[str] = set()
    for text in texts:
        if not text:
            continue
        text_lower = text.lower()
        for keyword, domain in domain_map.items():
            if re.search(rf'\b{re.escape(keyword.lower())}\b', text_lower):
                found_domains.add(domain)
    return found_domains

def analyze_ecosystem(base_path: Union[str, Path]) -> Dict[str, Any]:
    """Analyzes the plugin ecosystem and extracts capabilities."""
    base_path = Path(base_path)
    plugins_dir = base_path / "plugins"
    capabilities: Dict[str, Any] = {
        "plugins": [],
        "capability_matrix": {}
    }

    if not plugins_dir.exists():
        print(f"Directory {plugins_dir} not found.")
        return capabilities

    for plugin_path in plugins_dir.iterdir():
        if not plugin_path.is_dir() or plugin_path.name.startswith('.'):
            continue

        plugin_info: Dict[str, Any] = {
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
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    keywords = manifest.get("keywords", [])
                    categories = manifest.get("categories", [])
                    plugin_info["domains"].update(extract_domains(keywords + categories, DOMAIN_MAP))
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error parsing {manifest_path}: {e}")

        # 2. Parse agents
        agents_dir = plugin_path / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.md"):
                try:
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        frontmatter = parse_frontmatter(content)
                        if frontmatter:
                            agent_data = {
                                "name": agent_file.stem,
                                "specialization": frontmatter.get("specialization", ""),
                                "description": frontmatter.get("description", "")
                            }
                            plugin_info["agents"].append(agent_data)
                            plugin_info["domains"].update(
                                extract_domains([agent_data["specialization"], agent_data["description"]], DOMAIN_MAP)
                            )
                except (yaml.YAMLError, OSError) as e:
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
                        with open(skill_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            frontmatter = parse_frontmatter(content)
                            if frontmatter:
                                skill_data = {
                                    "name": skill_path.name,
                                    "specialization": frontmatter.get("specialization", ""),
                                    "description": frontmatter.get("description", "")
                                }
                                plugin_info["skills"].append(skill_data)
                                plugin_info["domains"].update(
                                    extract_domains([skill_data["specialization"], skill_data["description"]], DOMAIN_MAP)
                                )
                    except (yaml.YAMLError, OSError) as e:
                        print(f"Error parsing skill {skill_file}: {e}")

        plugin_info["domains"] = sorted(list(plugin_info["domains"]))
        capabilities["plugins"].append(plugin_info)

        # Update capability matrix
        for domain in plugin_info["domains"]:
            if domain not in capabilities["capability_matrix"]:
                capabilities["capability_matrix"][domain] = []
            capabilities["capability_matrix"][domain].append(plugin_path.name)

    return capabilities

if __name__ == "__main__":
    import sys

    # Determine repo root relative to this script
    script_dir = Path(__file__).parent.absolute()
    default_repo_root = script_dir.parent

    # Allow overriding repo root via argument
    repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_repo_root

    report = analyze_ecosystem(repo_root)

    output_path = repo_root / "ecosystem_capabilities.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"Generated {output_path}")
    except OSError as e:
        print(f"Error writing report to {output_path}: {e}")
        sys.exit(1)
