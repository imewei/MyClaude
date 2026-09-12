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
SUITES = ["dev-suite", "science-suite"]

VALID_MODEL_TIERS = {"opus", "sonnet", "haiku", "inherit"}

# Expected registered command counts per CLAUDE.md
EXPECTED_REGISTERED_COMMANDS = {
    "dev-suite": 10,
    "science-suite": 2,
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


def _is_hub_skill(skill_md: Path) -> bool:
    """A hub skill routes to sub-skills via ``../`` relative links."""
    if not skill_md.exists():
        return False
    content = skill_md.read_text(encoding="utf-8")
    return "../" in content


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
            if not _is_hub_skill(skill_md):
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
            if not _is_hub_skill(skill_md):
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

        # Transitively collect all references reachable from registered hubs.
        # Handles multi-level chains: meta-router → hub → sub-skill.
        referenced_skills: set = set(hub_names)
        frontier = set(hub_names)
        while frontier:
            next_frontier: set = set()
            for skill_name in frontier:
                skill_md = skills_dir / skill_name / "SKILL.md"
                if not skill_md.exists():
                    continue
                content = skill_md.read_text(encoding="utf-8")
                refs = re.findall(r"\.\./([^/]+)/SKILL\.md", content)
                for r in refs:
                    if r not in referenced_skills:
                        referenced_skills.add(r)
                        next_frontier.add(r)
            frontier = next_frontier

        # All skill dirs on disk
        all_skill_dirs = {
            d.name
            for d in skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        }

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


ALL_SUITES = ["dev-suite", "research-suite", "science-suite"]


def _all_skill_names() -> set[str]:
    return {
        p.parent.name
        for suite in ALL_SUITES
        for p in (PLUGINS_ROOT / suite / "skills").rglob("SKILL.md")
    }


def _registered_skill_names() -> set[str]:
    names: set[str] = set()
    for suite in ALL_SUITES:
        data = json.loads(
            (PLUGINS_ROOT / suite / ".claude-plugin" / "plugin.json").read_text()
        )
        names |= {Path(ref).name for ref in data.get("skills", [])}
    return names


def _agent_names() -> set[str]:
    return {
        a.stem for suite in ALL_SUITES for a in (PLUGINS_ROOT / suite / "agents").glob("*.md")
    }


class TestDispatchEdgesAreDeclared:
    """Every command and agent must name where work goes next.

    Routing here is carried by prose, not by a manifest field: a command says which
    agent or hub handles it, and an agent names the skills holding the worked detail.
    Nothing else in the toolchain checks those edges, so a command that names no
    target reads as a dead end and an agent that names no skill invites rewriting from
    memory what a skill already maintains.
    """

    @pytest.mark.parametrize("suite", ALL_SUITES)
    def test_every_command_names_an_agent_or_hub(self, suite: str):
        agents = _agent_names()
        hubs = _registered_skill_names()
        orphans = []
        for command in sorted((PLUGINS_ROOT / suite / "commands").glob("*.md")):
            text = command.read_text(encoding="utf-8")
            named = any(re.search(rf"\b{re.escape(n)}\b", text) for n in agents | hubs)
            if not named:
                orphans.append(command.stem)
        assert not orphans, (
            f"{suite} commands naming no agent and no registered hub: {orphans}. "
            "Add a 'Routes to `<agent>` via `<suite>:<hub>`' line."
        )

    @pytest.mark.parametrize("suite", ALL_SUITES)
    def test_every_agent_names_at_least_one_skill(self, suite: str):
        skills = _all_skill_names()
        orphans = []
        for agent in sorted((PLUGINS_ROOT / suite / "agents").glob("*.md")):
            text = agent.read_text(encoding="utf-8")
            if not any(re.search(rf"`{re.escape(s)}`", text) for s in skills):
                orphans.append(agent.stem)
        assert not orphans, (
            f"{suite} agents naming no skill: {orphans}. Add a 'Related Skills' "
            "section pointing at the skills that carry the worked detail."
        )

    @pytest.mark.parametrize("suite", ALL_SUITES)
    def test_agent_skill_references_resolve(self, suite: str):
        """A pointer to a skill that does not exist is worse than no pointer."""
        skills = _all_skill_names()
        agents = _agent_names()
        broken = []
        for agent in sorted((PLUGINS_ROOT / suite / "agents").glob("*.md")):
            section = re.search(
                r"## Related Skills(.*?)(?=\n## |\Z)", agent.read_text(), re.DOTALL
            )
            if not section:
                continue
            for name in re.findall(r"`([a-z0-9][a-z0-9-]{3,})`", section.group(1)):
                # Backticks in these sections also wrap suite names and library
                # names (`remake`, `tick`, `freud`). Skill and agent names are
                # always hyphenated, so require a hyphen and skip the suites.
                if "-" not in name or name in ALL_SUITES:
                    continue
                if name not in skills and name not in agents:
                    broken.append(f"{agent.stem} -> {name}")
        assert not broken, f"{suite} agents pointing at non-existent skills: {broken}"
