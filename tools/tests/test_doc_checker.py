#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path

# Add tools directory to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

class TestDocChecker(unittest.TestCase):
    def test_import(self):
        try:
            from tools.validation.doc_checker import DocumentationChecker  # noqa: F401
        except ImportError as e:
            self.fail(f"Could not import DocumentationChecker: {e}")

if __name__ == "__main__":
    unittest.main()
