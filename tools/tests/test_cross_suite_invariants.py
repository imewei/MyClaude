"""Cross-suite invariant tests.

Validates constraints that span all plugin suites:
- Version sync across manifests
- Agent model tier validity
- Hub skill routing tree structure
- Sub-skill orphan reachability
- Registered command count
- Hook script syntax validity
"""

import ast
import json
import re
from pathlib import Path

import pytest
import yaml  # type: ignore

REPO_ROOT = Path(__file__).parent.parent.parent
PLUGINS_ROOT = REPO_ROOT / "plugins"
SUITES = ["agent-core", "dev-suite", "science-suite"]

VALID_MODEL_TIERS = {"opus", "sonnet", "haiku", "inherit"}

# Expected registered command counts per CLAUDE.md
EXPECTED_REGISTERED_COMMANDS = {
    "agent-core": 2,
    "dev-suite": 12,
    "science-suite": 0,
}


def load_frontmatter(file_path: Path) -> dict | None:
    """Extract and parse YAML frontmatter from a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
    return None


def load_plugin_json(suite: str) -> dict:
    """Load a suite's plugin.json."""
    path = PLUGINS_ROOT / suite / ".claude-plugin" / "plugin.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Version Sync ---


class TestVersionSync:
    """All plugin.json files must declare the same version string."""

    def test_all_plugin_json_versions_match(self):
        versions = {}
        for suite in SUITES:
            data = load_plugin_json(suite)
            versions[suite] = data.get("version", "MISSING")

        unique_versions = set(versions.values())
        assert len(unique_versions) == 1, (
            f"Version mismatch across suites: {versions}"
        )
        assert "MISSING" not in unique_versions, (
            f"Suite missing version field: {versions}"
        )


# --- Agent Model Tier ---


class TestAgentModelTier:
    """Agent model field must be a valid tier value."""

    @pytest.mark.parametrize("suite", SUITES)
    def test_agent_model_field_is_valid_tier(self, suite: str):
        agents_dir = PLUGINS_ROOT / suite / "agents"
        if not agents_dir.exists():
            pytest.skip(f"{suite} has no agents directory")

        broken_yaml = []
        invalid = []
        for agent_file in agents_dir.glob("*.md"):
            fm = load_frontmatter(agent_file)
            if fm is None:
                broken_yaml.append(agent_file.name)
                continue
            model = fm.get("model")
            if model is not None and model not in VALID_MODEL_TIERS:
                invalid.append(f"{agent_file.name}: model='{model}'")

        assert not broken_yaml, (
            f"{suite} agents with broken YAML frontmatter: {broken_yaml}"
        )
        assert not invalid, (
            f"{suite} agents with invalid model tier: {invalid}"
        )


# --- Hub Skill Routing Tree ---


class TestHubRoutingTree:
    """Hub skills must contain a Routing Decision Tree code block and Checklist."""

    @pytest.mark.parametrize("suite", SUITES)
    def test_hub_skills_have_routing_decision_tree(self, suite: str):
        data = load_plugin_json(suite)
        skills = data.get("skills", [])
        if not skills:
            pytest.skip(f"{suite} has no registered skills")

        missing_tree = []
        for skill_ref in skills:
            # skill_ref is a path like "./skills/backend-patterns"
            skill_path = skill_ref if isinstance(skill_ref, str) else skill_ref.get("name", "")
            skill_md = PLUGINS_ROOT / suite / skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            content = skill_md.read_text(encoding="utf-8")
            has_routing = "## Routing Decision Tree" in content or "## Routing" in content
            has_code_block = False
            if has_routing:
                parts = content.split("## Routing", 1)
                if len(parts) > 1:
                    has_code_block = "```" in parts[1]

            if not (has_routing and has_code_block):
                missing_tree.append(skill_path)

        assert not missing_tree, (
            f"{suite} hub skills missing Routing Decision Tree code block: {missing_tree}"
        )

    @pytest.mark.parametrize("suite", SUITES)
    def test_hub_skills_have_checklist(self, suite: str):
        data = load_plugin_json(suite)
        skills = data.get("skills", [])
        if not skills:
            pytest.skip(f"{suite} has no registered skills")

        missing_checklist = []
        for skill_ref in skills:
            skill_path = skill_ref if isinstance(skill_ref, str) else skill_ref.get("name", "")
            skill_md = PLUGINS_ROOT / suite / skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            content = skill_md.read_text(encoding="utf-8")
            if "## Checklist" not in content:
                missing_checklist.append(skill_path)

        assert not missing_checklist, (
            f"{suite} hub skills missing Checklist section: {missing_checklist}"
        )


# --- Sub-skill Orphan Reachability ---


class TestSubSkillReachability:
    """Every sub-skill directory must be referenced by at least one hub."""

    @pytest.mark.parametrize("suite", SUITES)
    def test_no_orphan_sub_skills(self, suite: str):
        skills_dir = PLUGINS_ROOT / suite / "skills"
        if not skills_dir.exists():
            pytest.skip(f"{suite} has no skills directory")

        data = load_plugin_json(suite)
        hub_refs = data.get("skills", [])
        hub_names = set()
        for ref in hub_refs:
            name = ref if isinstance(ref, str) else ref.get("name", "")
            # Extract just the directory name from path like "./skills/foo"
            hub_names.add(Path(name).name)

        # Collect all references from hub SKILL.md files
        referenced_skills = set()
        for hub_name in hub_names:
            hub_md = skills_dir / hub_name / "SKILL.md"
            if not hub_md.exists():
                continue
            content = hub_md.read_text(encoding="utf-8")
            # Find relative references like ../sub-skill-name/SKILL.md
            refs = re.findall(r"\.\./([^/]+)/SKILL\.md", content)
            referenced_skills.update(refs)

        # All skill dirs on disk
        all_skill_dirs = {
            d.name
            for d in skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        }

        # Hub skills reference themselves implicitly
        referenced_skills.update(hub_names)

        orphans = all_skill_dirs - referenced_skills
        assert not orphans, (
            f"{suite} has orphan sub-skills not referenced by any hub: {sorted(orphans)}"
        )


# --- Registered Command Count ---


class TestCommandRegistration:
    """Registered command counts must match documented expectations."""

    @pytest.mark.parametrize("suite", SUITES)
    def test_registered_command_count(self, suite: str):
        data = load_plugin_json(suite)
        commands = data.get("commands", [])
        expected = EXPECTED_REGISTERED_COMMANDS[suite]
        assert len(commands) == expected, (
            f"{suite}: expected {expected} registered commands, got {len(commands)}. "
            f"Check if a skill-invoked command was accidentally registered."
        )


# --- Hook Script Syntax ---


class TestHookScriptSyntax:
    """All hook Python scripts must parse without syntax errors."""

    @pytest.mark.parametrize("suite", SUITES)
    def test_hook_scripts_parse(self, suite: str):
        hooks_dir = PLUGINS_ROOT / suite / "hooks"
        if not hooks_dir.exists():
            pytest.skip(f"{suite} has no hooks directory")

        syntax_errors = []
        for script in hooks_dir.glob("*.py"):
            try:
                source = script.read_text(encoding="utf-8")
                ast.parse(source, filename=str(script))
            except SyntaxError as e:
                syntax_errors.append(f"{script.name}: {e}")

        assert not syntax_errors, (
            f"{suite} hook scripts with syntax errors: {syntax_errors}"
        )
