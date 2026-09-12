#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

# Add tools directory to path
current_dir = Path(__file__).parent
tools_root = current_dir.parent
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestSkillValidator(unittest.TestCase):
    def test_import(self):
        try:
            from tools.validation.skill_validator import main  # noqa: F401
        except ImportError:
            self.fail("Could not import skill_validator")

    def test_reports_registered_skill_inventory(self):
        from tools.validation.skill_validator import SkillInventoryValidator

        validator = SkillInventoryValidator(str(project_root / "plugins"))
        validator.load_skills()
        report = validator.generate_report()

        self.assertGreater(len(validator.skills), 0)
        self.assertIn("Skill Inventory Report", report)
        for plugin in ("dev-suite", "research-suite", "science-suite"):
            self.assertIn(plugin, report)

    def test_flags_missing_description(self):
        from tools.validation.skill_validator import SkillInventoryValidator

        validator = SkillInventoryValidator(str(project_root / "plugins"))
        validator._add_skill({"name": "bare-skill"}, "test-plugin")

        self.assertTrue(
            any("missing `description`" in issue for issue in validator.issues)
        )
        self.assertIn("❌", validator.generate_report())

    def test_no_triggering_metrics_are_reported(self):
        """Regression: the accuracy/precision table derived its ground truth
        from the score it was testing, so it reported 100% for any input.
        It must not come back."""
        from tools.validation.skill_validator import SkillInventoryValidator

        validator = SkillInventoryValidator(str(project_root / "plugins"))
        validator.load_skills()
        report = validator.generate_report()

        for banned in (
            "Overall Accuracy",
            "Over-Trigger Rate",
            "Under-Trigger Rate",
            "Precision",
            "EXCELLENT",
        ):
            self.assertNotIn(banned, report)


if __name__ == "__main__":
    unittest.main()
