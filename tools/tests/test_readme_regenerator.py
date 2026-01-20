
import unittest
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.generation.readme_regenerator import ReadmeRegenerator
except ImportError:
    pass

class TestReadmeRegenerator(unittest.TestCase):
    def setUp(self):
        if 'ReadmeRegenerator' not in globals():
            self.skipTest("ReadmeRegenerator class not found")

        self.regenerator = ReadmeRegenerator()
        self.plugin_path = Path("/tmp/test-plugin")
        self.plugin_data = {
            "name": "test-plugin",
            "description": "A test plugin",
            "version": "1.0.0",
            "category": "development",
            "license": "MIT",
            "agents": [
                {"name": "test-agent", "description": "A test agent", "status": "active"}
            ],
            "commands": [
                {"name": "/test", "description": "A test command", "status": "beta"}
            ],
            "skills": [
                {"name": "test-skill", "description": "A test skill"}
            ]
        }

    def test_generate_readme(self):
        with patch.object(ReadmeRegenerator, 'parse_plugin_json', return_value=self.plugin_data):
            content = self.regenerator.generate_readme(self.plugin_path)

            self.assertIn("# Test Plugin", content)
            self.assertIn("A test plugin", content)
            self.assertIn("**Version:** 1.0.0", content)
            self.assertIn("## Agents (1)", content)
            self.assertIn("### test-agent", content)
            self.assertIn("## Commands (1)", content)
            self.assertIn("### `/test`", content)
            self.assertIn("## Skills (1)", content)
            self.assertIn("### test-skill", content)

    def test_format_title(self):
        self.assertEqual(self.regenerator._format_title("test-plugin"), "Test Plugin")
        self.assertEqual(self.regenerator._format_title("my-cool-plugin"), "My Cool Plugin")

if __name__ == '__main__':
    unittest.main()
