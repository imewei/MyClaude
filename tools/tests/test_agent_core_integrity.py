import json
import yaml  # type: ignore
import pytest
import re
from pathlib import Path

# Paths
# tools/tests/test_agent_core_integrity.py -> .../tools/tests -> .../tools -> .../ (root) -> .../plugins/agent-core
PLUGIN_ROOT = Path(__file__).parent.parent.parent / "plugins" / "agent-core"
PLUGIN_JSON = PLUGIN_ROOT / "plugin.json"
AGENTS_DIR = PLUGIN_ROOT / "agents"
SKILLS_DIR = PLUGIN_ROOT / "skills"

def load_frontmatter(file_path):
    """Extract and parse YAML frontmatter from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
    return None

def test_plugin_json_exists():
    assert PLUGIN_JSON.exists(), "plugin.json missing"

def test_plugin_json_structure():
    with open(PLUGIN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_fields = ["name", "version", "description", "agents"]
    for field in required_fields:
        assert field in data, f"Missing required field in plugin.json: {field}"
    
    assert isinstance(data["agents"], list), "'agents' must be a list"
    assert len(data["agents"]) > 0, "'agents' list should not be empty"

def test_agents_exist():
    with open(PLUGIN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for agent_info in data.get("agents", []):
        agent_name = agent_info["name"] if isinstance(agent_info, dict) else agent_info
        agent_file = AGENTS_DIR / f"{agent_name}.md"
        assert agent_file.exists(), f"Agent file missing: {agent_file}"

def test_agent_frontmatter_validity():
    if not AGENTS_DIR.exists():
        pytest.skip("Agents directory not found")
        
    for agent_file in AGENTS_DIR.glob("*.md"):
        frontmatter = load_frontmatter(agent_file)
        assert frontmatter is not None, f"Invalid or missing YAML frontmatter in {agent_file.name}"
        assert "description" in frontmatter, f"Missing 'description' in frontmatter of {agent_file.name}"

def test_skill_frontmatter_validity():
    if not SKILLS_DIR.exists():
        return # Skills are optional
        
    for skill_file in SKILLS_DIR.rglob("*.md"):
        # Skip READMEs or non-skill markdowns if convention dictates, 
        # but typically SKILL.md is the standard
        if skill_file.name == "SKILL.md":
            frontmatter = load_frontmatter(skill_file)
            assert frontmatter is not None, f"Invalid or missing YAML frontmatter in {skill_file}"
            assert "name" in frontmatter or "description" in frontmatter, \
                f"Missing required fields in frontmatter of {skill_file}"
