#!/usr/bin/env python3
"""Unit tests for tools.validation.xref_validator._build_plugin_index.

Covers the merge of (a) manifest-declared hubs and (b) disk-discovered
sub-skills, agents, and commands.
"""

import sys
import tempfile
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from tools.validation.xref_validator import CrossReferenceValidator  # noqa: E402


def _make_plugin(
    root: Path,
    name: str,
    *,
    agents: list[str] | None = None,
    commands: list[str] | None = None,
    skills: list[str] | None = None,
    sub_skills: list[str] | None = None,
    extra_agents_on_disk: list[str] | None = None,
) -> None:
    """Create a minimal plugin tree for index-builder testing."""
    plugin = root / name
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / "agents").mkdir()
    (plugin / "commands").mkdir()
    (plugin / "skills").mkdir()

    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": f"test {name}",
        "category": "test",
        "agents": [f"./agents/{a}.md" for a in (agents or [])],
        "commands": [f"./commands/{c}.md" for c in (commands or [])],
        "skills": [f"./skills/{s}/SKILL.md" for s in (skills or [])],
    }
    import json

    (plugin / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest))

    # Manifest-declared component files on disk
    for a in agents or []:
        (plugin / "agents" / f"{a}.md").write_text(f"---\nname: {a}\n---\n")
    for c in commands or []:
        (plugin / "commands" / f"{c}.md").write_text(f"---\nname: {c}\n---\n")
    for s in skills or []:
        (plugin / "skills" / s).mkdir(exist_ok=True)
        (plugin / "skills" / s / "SKILL.md").write_text(
            f"---\nname: {s}\ndescription: hub {s}\n---\n"
        )

    # Disk-only sub-skills (intentionally not in manifest, per hub-skill arch)
    for s in sub_skills or []:
        (plugin / "skills" / s).mkdir(exist_ok=True)
        (plugin / "skills" / s / "SKILL.md").write_text(
            f"---\nname: {s}\ndescription: sub {s}\n---\n"
        )

    # Disk-only agents (e.g., introduced after a manifest update)
    for a in extra_agents_on_disk or []:
        (plugin / "agents" / f"{a}.md").write_text(f"---\nname: {a}\n---\n")


class TestBuildPluginIndex(unittest.TestCase):
    def test_manifest_declared_components_indexed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_plugin(
                root,
                "p1",
                agents=["arch", "review"],
                commands=["build"],
                skills=["hub-a"],
            )
            v = CrossReferenceValidator(root)
            entry = v.result.plugin_index["p1"]
            self.assertEqual(set(entry["agents"].keys()), {"arch", "review"})
            self.assertEqual(set(entry["commands"].keys()), {"build"})
            self.assertIn("hub-a", entry["skills"])

    def test_disk_subskills_merged_with_manifest_hubs(self):
        # The hub-skill architecture: manifest declares hubs only, sub-skills
        # are discovered from disk.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_plugin(
                root,
                "p2",
                skills=["hub-a"],
                sub_skills=["sub-1", "sub-2", "sub-3"],
            )
            v = CrossReferenceValidator(root)
            entry = v.result.plugin_index["p2"]
            self.assertIn("hub-a", entry["skills"])  # manifest
            self.assertIn("sub-1", entry["skills"])  # disk-discovered
            self.assertIn("sub-2", entry["skills"])
            self.assertIn("sub-3", entry["skills"])

    def test_disk_agents_merged_with_manifest_agents(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_plugin(
                root,
                "p3",
                agents=["registered"],
                extra_agents_on_disk=["disk-only"],
            )
            v = CrossReferenceValidator(root)
            entry = v.result.plugin_index["p3"]
            self.assertIn("registered", entry["agents"])
            self.assertIn("disk-only", entry["agents"])

    def test_empty_plugin_yields_empty_index(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_plugin(root, "empty")
            v = CrossReferenceValidator(root)
            entry = v.result.plugin_index["empty"]
            self.assertEqual(entry["agents"], {})
            self.assertEqual(entry["commands"], {})
            self.assertEqual(entry["skills"], {})

    def test_multiple_plugins_indexed_independently(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_plugin(root, "p-a", agents=["x"])
            _make_plugin(root, "p-b", agents=["y"])
            v = CrossReferenceValidator(root)
            self.assertIn("p-a", v.result.plugin_index)
            self.assertIn("p-b", v.result.plugin_index)
            self.assertEqual(set(v.result.plugin_index["p-a"]["agents"]), {"x"})
            self.assertEqual(set(v.result.plugin_index["p-b"]["agents"]), {"y"})


class TestImport(unittest.TestCase):
    def test_import(self):
        from tools.validation.xref_validator import main  # noqa: F401


if __name__ == "__main__":
    unittest.main()
