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


if __name__ == "__main__":
    unittest.main()
