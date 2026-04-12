#!/usr/bin/env python3
import unittest
import sys
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

    def test_no_corpus_reports_no_data(self):
        """Regression: validator must NOT report EXCELLENT when no tests ran."""
        from tools.validation.skill_validator import SkillApplicationValidator

        validator = SkillApplicationValidator(
            str(project_root / "plugins"), corpus_dir=None
        )
        validator.load_skills()
        validator.test_skill_application()
        report = validator.generate_report()
        self.assertIn("NO DATA", report)
        self.assertNotIn("EXCELLENT", report)

    def test_zero_tests_metrics_accuracy(self):
        """Verify accuracy is 0.0% with no tests, not a false positive."""
        from tools.validation.skill_validator import SkillValidationMetrics

        metrics = SkillValidationMetrics()
        self.assertEqual(metrics.total_tests, 0)
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.over_trigger_rate, 0.0)
        self.assertEqual(metrics.under_trigger_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
