#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path

# Add tools directory to path
current_dir = Path(__file__).parent
tools_root = current_dir.parent
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

class TestPluginReview(unittest.TestCase):
    def test_import(self):
        try:
            from tools.validation.plugin_review_script import PluginReviewer  # noqa: F401
        except ImportError:
            self.fail("Could not import PluginReviewer")

if __name__ == "__main__":
    unittest.main()

