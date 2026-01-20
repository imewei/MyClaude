import json
import pytest
import re
import yaml  # type: ignore
from pathlib import Path

# Paths adjustment: .../tools/tests -> .../tools -> .../ (root) -> .../plugins/quality-suite
PLUGIN_ROOT = Path(__file__).parent.parent.parent / "plugins" / "quality-suite"
PLUGIN_JSON = PLUGIN_ROOT / "plugin.json"
AGENTS_DIR = PLUGIN_ROOT / "agents"

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

def test_plugin_exists():
    assert PLUGIN_ROOT.exists(), f"Plugin directory not found: {PLUGIN_ROOT}"

def test_plugin_json_structure():
    assert PLUGIN_JSON.exists(), "plugin.json missing"
    with open(PLUGIN_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_fields = ["name", "version", "description", "agents"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    assert data["name"] == "quality-suite"

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
        assert frontmatter is not None, f"Invalid YAML frontmatter in {agent_file.name}"
