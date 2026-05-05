# tools/tests/test_scicomp_redesign.py
"""Tests for the scientific computing plugin redesign (v3.5.0).

All tests are written before implementation. Run with:
  uv run pytest tools/tests/test_scicomp_redesign.py -v
"""

import json
import pytest
from pathlib import Path

import yaml

REPO = Path(__file__).parent.parent.parent
PLUGINS = REPO / "plugins"
SCIENCE = PLUGINS / "science-suite"
RESEARCH = PLUGINS / "research-suite"
AGENT_CORE = PLUGINS / "agent-core"
DEV_SUITE = PLUGINS / "dev-suite"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter delimited by --- from a markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    return yaml.safe_load(parts[1]) if len(parts) >= 3 else {}


def _plugin_json(suite_dir: Path) -> dict:
    return json.loads((suite_dir / ".claude-plugin/plugin.json").read_text())


# ---------------------------------------------------------------------------
# Model Tier Changes
# ---------------------------------------------------------------------------

class TestModelTiers:
    def test_jax_pro_is_opus(self):
        fm = _frontmatter(SCIENCE / "agents/jax-pro.md")
        assert fm["model"] == "opus", "jax-pro must be upgraded to opus"

    def test_julia_pro_is_opus(self):
        fm = _frontmatter(SCIENCE / "agents/julia-pro.md")
        assert fm["model"] == "opus", "julia-pro must be upgraded to opus"

    def test_ml_expert_is_haiku(self):
        fm = _frontmatter(SCIENCE / "agents/ml-expert.md")
        assert fm["model"] == "haiku", "ml-expert must be demoted to haiku"


# ---------------------------------------------------------------------------
# Agent Repurposing
# ---------------------------------------------------------------------------

class TestAgentRepurposing:
    def test_pinn_engineer_exists(self):
        assert (SCIENCE / "agents/pinn-engineer.md").exists()

    def test_sci_workflow_engineer_exists(self):
        assert (SCIENCE / "agents/sci-workflow-engineer.md").exists()

    def test_ai_engineer_deleted(self):
        assert not (SCIENCE / "agents/ai-engineer.md").exists(), \
            "ai-engineer.md must be deleted (replaced by pinn-engineer.md)"

    def test_prompt_engineer_deleted(self):
        assert not (SCIENCE / "agents/prompt-engineer.md").exists(), \
            "prompt-engineer.md must be deleted (replaced by sci-workflow-engineer.md)"

    def test_pinn_engineer_name_field(self):
        fm = _frontmatter(SCIENCE / "agents/pinn-engineer.md")
        assert fm.get("name") == "pinn-engineer"

    def test_sci_workflow_engineer_name_field(self):
        fm = _frontmatter(SCIENCE / "agents/sci-workflow-engineer.md")
        assert fm.get("name") == "sci-workflow-engineer"

    def test_pinn_engineer_model(self):
        fm = _frontmatter(SCIENCE / "agents/pinn-engineer.md")
        assert fm.get("model") in ("sonnet", "opus")

    def test_sci_workflow_engineer_model(self):
        fm = _frontmatter(SCIENCE / "agents/sci-workflow-engineer.md")
        assert fm.get("model") in ("sonnet", "opus")


# ---------------------------------------------------------------------------
# Description Trimming (<=180 chars for all science-suite agents)
# ---------------------------------------------------------------------------

SCIENCE_AGENTS = [
    "jax-pro", "julia-pro", "julia-ml-hpc", "ml-expert",
    "neural-network-master", "nonlinear-dynamics-expert",
    "python-pro", "simulation-expert", "statistical-physicist",
    "pinn-engineer", "sci-workflow-engineer",
]


class TestDescriptionTrimming:
    @pytest.mark.parametrize("agent", SCIENCE_AGENTS)
    def test_description_at_most_180_chars(self, agent):
        path = SCIENCE / f"agents/{agent}.md"
        fm = _frontmatter(path)
        desc = fm.get("description", "")
        assert len(desc) <= 180, (
            f"{agent} description is {len(desc)} chars (max 180): {desc!r}"
        )

    @pytest.mark.parametrize("agent", SCIENCE_AGENTS)
    def test_description_not_empty(self, agent):
        path = SCIENCE / f"agents/{agent}.md"
        fm = _frontmatter(path)
        assert fm.get("description"), f"{agent} description must not be empty"


# ---------------------------------------------------------------------------
# New Commands - science-suite
# ---------------------------------------------------------------------------

class TestScienceSuiteCommands:
    def test_md_sim_exists(self):
        assert (SCIENCE / "commands/md-sim.md").exists()

    def test_benchmark_exists(self):
        assert (SCIENCE / "commands/benchmark.md").exists()

    @pytest.mark.parametrize("cmd", ["md-sim", "benchmark"])
    def test_command_has_name(self, cmd):
        fm = _frontmatter(SCIENCE / f"commands/{cmd}.md")
        assert fm.get("name") == cmd

    @pytest.mark.parametrize("cmd", ["md-sim", "benchmark"])
    def test_command_has_description(self, cmd):
        fm = _frontmatter(SCIENCE / f"commands/{cmd}.md")
        assert fm.get("description"), f"{cmd} must have a description"

    @pytest.mark.parametrize("cmd", ["md-sim", "benchmark"])
    def test_command_has_argument_hint(self, cmd):
        fm = _frontmatter(SCIENCE / f"commands/{cmd}.md")
        assert fm.get("argument-hint"), f"{cmd} must have an argument-hint"


# ---------------------------------------------------------------------------
# New Commands - research-suite
# ---------------------------------------------------------------------------

class TestResearchSuiteCommands:
    def test_paper_implement_exists(self):
        assert (RESEARCH / "commands/paper-implement.md").exists()

    def test_lit_review_exists(self):
        assert (RESEARCH / "commands/lit-review.md").exists()

    def test_replicate_exists(self):
        assert (RESEARCH / "commands/replicate.md").exists()

    @pytest.mark.parametrize("cmd", ["paper-implement", "lit-review", "replicate"])
    def test_command_has_name(self, cmd):
        fm = _frontmatter(RESEARCH / f"commands/{cmd}.md")
        assert fm.get("name") == cmd

    @pytest.mark.parametrize("cmd", ["paper-implement", "lit-review", "replicate"])
    def test_command_has_description(self, cmd):
        fm = _frontmatter(RESEARCH / f"commands/{cmd}.md")
        assert fm.get("description"), f"{cmd} must have a description"


# ---------------------------------------------------------------------------
# Plugin Manifest (plugin.json) Changes
# ---------------------------------------------------------------------------

class TestManifests:
    @pytest.mark.parametrize("suite_dir", [
        AGENT_CORE, DEV_SUITE, RESEARCH, SCIENCE
    ], ids=["agent-core", "dev-suite", "research-suite", "science-suite"])
    def test_version_is_350(self, suite_dir):
        plugin = _plugin_json(suite_dir)
        assert plugin["version"] == "3.5.0", \
            f"{suite_dir.name} version must be 3.5.0, got {plugin['version']}"

    def test_science_suite_has_md_sim_command(self):
        plugin = _plugin_json(SCIENCE)
        assert "./commands/md-sim.md" in plugin.get("commands", [])

    def test_science_suite_has_benchmark_command(self):
        plugin = _plugin_json(SCIENCE)
        assert "./commands/benchmark.md" in plugin.get("commands", [])

    def test_research_suite_has_paper_implement(self):
        plugin = _plugin_json(RESEARCH)
        assert "./commands/paper-implement.md" in plugin.get("commands", [])

    def test_research_suite_has_lit_review(self):
        plugin = _plugin_json(RESEARCH)
        assert "./commands/lit-review.md" in plugin.get("commands", [])

    def test_research_suite_has_replicate(self):
        plugin = _plugin_json(RESEARCH)
        assert "./commands/replicate.md" in plugin.get("commands", [])

    def test_science_suite_agent_pinn_engineer_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/pinn-engineer.md" in plugin.get("agents", [])

    def test_science_suite_agent_sci_workflow_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/sci-workflow-engineer.md" in plugin.get("agents", [])

    def test_science_suite_ai_engineer_not_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/ai-engineer.md" not in plugin.get("agents", [])

    def test_science_suite_prompt_engineer_not_registered(self):
        plugin = _plugin_json(SCIENCE)
        assert "./agents/prompt-engineer.md" not in plugin.get("agents", [])


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

class TestInfrastructure:
    def test_claudeignore_exists(self):
        assert (REPO / ".claudeignore").exists(), \
            ".claudeignore must exist at repo root"

    def test_claudeignore_has_graphify_out(self):
        content = (REPO / ".claudeignore").read_text()
        assert "graphify-out/" in content

    def test_claudeignore_has_pycache(self):
        content = (REPO / ".claudeignore").read_text()
        assert "__pycache__" in content

    def test_pre_compact_has_priority_skills(self):
        script = (AGENT_CORE / "hooks/pre_compact.py").read_text()
        assert "PRIORITY_SKILLS" in script, \
            "pre_compact.py must define PRIORITY_SKILLS list"
        assert "jax-computing" in script
        assert "simulation-and-hpc" in script
