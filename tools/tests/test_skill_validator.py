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
            from tools.validation.skill_validator import main
        except ImportError:
            self.fail("Could not import skill_validator")

if __name__ == "__main__":
    unittest.main()

