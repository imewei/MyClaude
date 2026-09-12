#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestRoutingTreeDetection(unittest.TestCase):
    """The tier check must tell a real routing tree from a decorative heading.

    Matching only the '## Routing Decision Tree' heading would let an empty code
    fence pass. Requiring one specific arrow style is the opposite failure: hubs
    write targets as '-->', '\u2192', '../<skill>/SKILL.md', or '<suite>:<skill>',
    and demanding ASCII arrows alone flags seven working science-suite hubs as
    sub-skills. Accept any target style; reject only a fence with no target at all.
    """

    def _validator(self):
        from tools.validation.metadata_validator import MetadataValidator

        return MetadataValidator

    def test_empty_fence_is_not_a_routing_tree(self):
        content = "# Skill\n\n## Routing Decision Tree\n\n```\n```\n\n## Checklist\n"
        self.assertFalse(self._validator()._has_routing_targets(content))

    def test_heading_without_fence_is_not_a_routing_tree(self):
        content = "# Skill\n\n## Routing Decision Tree\n\nSee the table below.\n"
        self.assertFalse(self._validator()._has_routing_targets(content))

    def test_ascii_arrow_target_counts(self):
        content = (
            "## Routing Decision Tree\n\n```\n"
            "+-- Setting up MD?\n|   --> science-suite:md-simulation-setup\n```\n"
        )
        self.assertTrue(self._validator()._has_routing_targets(content))

    def test_unicode_arrow_target_counts(self):
        """science-suite hubs use \u2192, not '-->'; both are valid."""
        content = (
            "## Routing Decision Tree\n\n```\n"
            "Need posterior uncertainty on a PINN solution?\n"
            "  \u2192 science-suite:bayesian-pinn (../bayesian-pinn/SKILL.md)\n```\n"
        )
        self.assertTrue(self._validator()._has_routing_targets(content))

    def test_relative_link_target_counts(self):
        content = (
            "## Routing Decision Tree\n\n```\n"
            "Trajectory observables? ../trajectory-analysis/SKILL.md\n```\n"
        )
        self.assertTrue(self._validator()._has_routing_targets(content))

    def test_every_registered_skill_passes_the_tier_check(self):
        """All 50 registered skills are either real routers or whitelisted standalones."""
        import json

        validator = self._validator()
        plugins = project_root / "plugins"
        offenders = []
        for suite in ("dev-suite", "research-suite", "science-suite"):
            manifest = json.loads(
                (plugins / suite / ".claude-plugin" / "plugin.json").read_text()
            )
            for ref in manifest.get("skills", []):
                name = Path(ref).name
                if name in validator._TIER2_STANDALONE_WHITELIST:
                    continue
                skill_md = plugins / suite / "skills" / name / "SKILL.md"
                if not validator._has_routing_targets(skill_md.read_text()):
                    offenders.append(f"{suite}/{name}")
        self.assertEqual(offenders, [])


class TestRoutingRedundancy(unittest.TestCase):
    """A registered hub should be able to route somewhere its parents cannot.

    Eight skills were registered while every target they named was already reachable
    from a hub that routes to them, so the entry bought an always-loaded description
    for a decision a parent already offered.
    """

    def _validator(self):
        from tools.validation.metadata_validator import MetadataValidator

        return MetadataValidator

    def test_targets_parsed_from_tree_and_core_skills(self):
        content = (
            "## Core Skills\n\n### [Alpha](../alpha/SKILL.md)\n\n"
            "## Routing Decision Tree\n\n```\nBeta? \u2192 science-suite:beta\n```\n"
        )
        self.assertEqual(self._validator()._routing_targets(content), {"alpha", "beta"})

    def test_redundant_hub_is_flagged(self):
        """Every target already reachable from the parent -> warn."""
        from tools.validation.metadata_validator import (
            MetadataValidator,
            ValidationResult,
        )

        validator = MetadataValidator()
        child = "## Routing Decision Tree\n\n```\n--> science-suite:leaf\n```\n"
        MetadataValidator._REGISTRY_CACHE[str(Path("/fake").resolve())] = {
            "parent": (
                "## Routing Decision Tree\n\n```\n"
                "--> science-suite:child\n--> science-suite:leaf\n```\n"
            ),
            "child": child,
        }
        result = ValidationResult(plugin_name="test")
        validator._check_routing_redundancy(
            "child", child, 0, result, Path("/fake/suite")
        )
        self.assertTrue(any("adds no routing reach" in w.message for w in result.warnings))

    def test_hub_with_unique_reach_is_not_flagged(self):
        """One target the parent cannot reach is enough to earn registration."""
        from tools.validation.metadata_validator import (
            MetadataValidator,
            ValidationResult,
        )

        validator = MetadataValidator()
        child = (
            "## Routing Decision Tree\n\n```\n"
            "--> science-suite:leaf\n--> science-suite:only-here\n```\n"
        )
        MetadataValidator._REGISTRY_CACHE[str(Path("/fake2").resolve())] = {
            "parent": (
                "## Routing Decision Tree\n\n```\n"
                "--> science-suite:child\n--> science-suite:leaf\n```\n"
            ),
            "child": child,
        }
        result = ValidationResult(plugin_name="test")
        validator._check_routing_redundancy(
            "child", child, 0, result, Path("/fake2/suite")
        )
        self.assertEqual(result.warnings, [])

    def test_root_hub_without_parents_is_not_flagged(self):
        """science-hub and friends are the way in; nothing routes to them."""
        from tools.validation.metadata_validator import (
            MetadataValidator,
            ValidationResult,
        )

        validator = MetadataValidator()
        root = "## Routing Decision Tree\n\n```\n--> science-suite:leaf\n```\n"
        MetadataValidator._REGISTRY_CACHE[str(Path("/fake3").resolve())] = {"root": root}
        result = ValidationResult(plugin_name="test")
        validator._check_routing_redundancy("root", root, 0, result, Path("/fake3/suite"))
        self.assertEqual(result.warnings, [])

    def test_every_registered_skill_adds_reach(self):
        """The live tree must stay clean, so a redundant registration is caught."""
        import json

        from tools.validation.metadata_validator import (
            MetadataValidator,
            ValidationResult,
        )

        validator = MetadataValidator()
        plugins = project_root / "plugins"
        offenders = []
        for suite in ("dev-suite", "research-suite", "science-suite"):
            manifest = json.loads(
                (plugins / suite / ".claude-plugin" / "plugin.json").read_text()
            )
            for ref in manifest.get("skills", []):
                name = Path(ref).name
                skill_md = plugins / suite / "skills" / name / "SKILL.md"
                if not skill_md.exists():
                    continue
                result = ValidationResult(plugin_name=suite)
                validator._check_routing_redundancy(
                    name, skill_md.read_text(), 0, result, plugins / suite
                )
                if result.warnings:
                    offenders.append(f"{suite}/{name}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
