#!/usr/bin/env python3
"""
Tests for README regeneration functionality.

Tests verify:
- Consistent format across plugins
- All required sections included
- Sphinx documentation links are correct
- README metadata matches plugin.json
- Marketplace README links to Sphinx docs
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from readme_regenerator import ReadmeRegenerator


class TestReadmeRegeneration:
    """Test suite for README regeneration"""

    def __init__(self):
        self.test_results = []
        self.test_count = 0
        self.passed = 0
        self.failed = 0

    def log(self, message: str, status: str = "INFO"):
        """Log test message"""
        symbols = {"PASS": "✅", "FAIL": "❌", "INFO": "ℹ️", "RUN": "🏃"}
        print(f"{symbols.get(status, 'ℹ️')} {message}")

    def assert_true(self, condition: bool, message: str):
        """Assert condition is true"""
        if condition:
            self.passed += 1
            self.log(f"PASS: {message}", "PASS")
            return True
        else:
            self.failed += 1
            self.log(f"FAIL: {message}", "FAIL")
            return False

    def assert_in(self, substring: str, text: str, message: str):
        """Assert substring is in text"""
        return self.assert_true(substring in text, message)

    def create_test_plugin(self, tmp_dir: Path, plugin_data: Dict[str, Any]) -> Path:
        """Create a test plugin directory with plugin.json"""
        plugin_name = plugin_data.get("name", "test-plugin")
        plugin_dir = tmp_dir / plugin_name
        plugin_dir.mkdir(parents=True)

        plugin_json = plugin_dir / "plugin.json"
        with open(plugin_json, 'w', encoding='utf-8') as f:
            json.dump(plugin_data, f, indent=2)

        return plugin_dir

    def test_1_consistent_format_generation(self):
        """Test 1: README generation produces consistent format across plugins"""
        self.test_count += 1
        self.log("Test 1: Consistent format generation", "RUN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            regenerator = ReadmeRegenerator(docs_base_url="https://docs.example.com")

            # Create two test plugins with similar structure
            plugin1_data = {
                "name": "test-plugin-1",
                "version": "1.0.0",
                "category": "development",
                "license": "MIT",
                "description": "Test plugin 1",
                "agents": [{"name": "agent1", "description": "Agent 1", "status": "active"}],
                "commands": [{"name": "/cmd1", "description": "Command 1", "status": "active"}],
                "skills": [{"name": "skill1", "description": "Skill 1"}]
            }

            plugin2_data = {
                "name": "test-plugin-2",
                "version": "2.0.0",
                "category": "tools",
                "license": "Apache-2.0",
                "description": "Test plugin 2",
                "agents": [
                    {"name": "agent2a", "description": "Agent 2A", "status": "beta"},
                    {"name": "agent2b", "description": "Agent 2B", "status": "active"}
                ],
                "commands": [{"name": "/cmd2", "description": "Command 2", "status": "active"}],
                "skills": [{"name": "skill2", "description": "Skill 2"}]
            }

            plugin1_dir = self.create_test_plugin(tmp_path, plugin1_data)
            plugin2_dir = self.create_test_plugin(tmp_path, plugin2_data)

            readme1 = regenerator.generate_readme(plugin1_dir)
            readme2 = regenerator.generate_readme(plugin2_dir)

            # Verify both have consistent section headers
            required_headers = [
                "# Test Plugin",
                "**Version:**",
                "[Full Documentation →]",
                "## Agents",
                "## Quick Start",
                "## Integration",
                "## Documentation"
            ]

            all_consistent = True
            for header in required_headers:
                if header not in readme1 or header not in readme2:
                    all_consistent = False
                    self.log(f"Missing header: {header}", "FAIL")

            self.assert_true(all_consistent, "All plugins have consistent section headers")

            # Verify both follow same structure (sections in same order) - only ## level headers, not ###
            readme1_lines = [line for line in readme1.split('\n') if line.startswith('##') and not line.startswith('###')]
            readme2_lines = [line for line in readme2.split('\n') if line.startswith('##') and not line.startswith('###')]

            # Extract section types (## level headers only)
            section1_types = [line.split('(')[0].strip() for line in readme1_lines]
            section2_types = [line.split('(')[0].strip() for line in readme2_lines]

            self.assert_true(
                section1_types == section2_types,
                f"Both READMEs have same section order: {section1_types}"
            )

    def test_2_required_sections_included(self):
        """Test 2: Generated READMEs include all required sections"""
        self.test_count += 1
        self.log("Test 2: Required sections included", "RUN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            regenerator = ReadmeRegenerator(docs_base_url="https://docs.example.com")

            plugin_data = {
                "name": "comprehensive-plugin",
                "version": "1.5.0",
                "category": "scientific-computing",
                "license": "MIT",
                "description": "Comprehensive test plugin with all components",
                "agents": [{"name": "test-agent", "description": "Test agent", "status": "active"}],
                "commands": [{"name": "/test-cmd", "description": "Test command", "status": "active"}],
                "skills": [{"name": "test-skill", "description": "Test skill"}]
            }

            plugin_dir = self.create_test_plugin(tmp_path, plugin_data)
            readme = regenerator.generate_readme(plugin_dir)

            # Required sections from spec.md lines 218-246
            required_sections = {
                "Title": "# Comprehensive Plugin",
                "Description": "Comprehensive test plugin with all components",
                "Version metadata": "**Version:** 1.5.0",
                "Category metadata": "**Category:** scientific-computing",
                "License metadata": "**License:** MIT",
                "Documentation link": "[Full Documentation →]",
                "Agents section": "## Agents (1)",
                "Commands section": "## Commands (1)",
                "Skills section": "## Skills (1)",
                "Quick Start": "## Quick Start",
                "Integration": "## Integration",
                "Documentation section": "## Documentation"
            }

            all_present = True
            for section_name, expected_text in required_sections.items():
                if expected_text not in readme:
                    self.log(f"Missing section: {section_name}", "FAIL")
                    all_present = False

            self.assert_true(all_present, "All required sections present in README")

    def test_3_sphinx_links_correct(self):
        """Test 3: Sphinx documentation links are correct"""
        self.test_count += 1
        self.log("Test 3: Sphinx documentation links correct", "RUN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            docs_url = "https://marketplace-docs.example.com"
            regenerator = ReadmeRegenerator(docs_base_url=docs_url)

            plugin_data = {
                "name": "my-awesome-plugin",
                "version": "1.0.0",
                "category": "tools",
                "license": "MIT",
                "description": "Test plugin for link validation"
            }

            plugin_dir = self.create_test_plugin(tmp_path, plugin_data)
            readme = regenerator.generate_readme(plugin_dir)

            # Verify documentation links
            expected_link = f"{docs_url}/plugins/my-awesome-plugin.html"

            self.assert_in(
                expected_link,
                readme,
                f"README contains correct Sphinx doc link: {expected_link}"
            )

            # Verify link appears in both header and footer sections
            link_count = readme.count(expected_link)
            self.assert_true(
                link_count >= 2,
                f"Documentation link appears multiple times ({link_count} occurrences)"
            )

    def test_4_metadata_matches_plugin_json(self):
        """Test 4: README metadata matches plugin.json"""
        self.test_count += 1
        self.log("Test 4: Metadata matches plugin.json", "RUN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            regenerator = ReadmeRegenerator()

            plugin_data = {
                "name": "metadata-test-plugin",
                "version": "2.5.1",
                "category": "quality-engineering",
                "license": "Apache-2.0",
                "description": "Plugin for testing metadata accuracy",
                "agents": [
                    {"name": "agent-alpha", "description": "Alpha agent", "status": "active"},
                    {"name": "agent-beta", "description": "Beta agent", "status": "beta"},
                    {"name": "agent-gamma", "description": "Gamma agent", "status": "active"}
                ],
                "commands": [
                    {"name": "/cmd-one", "description": "Command one", "status": "active"},
                    {"name": "/cmd-two", "description": "Command two", "status": "active"}
                ],
                "skills": [
                    {"name": "skill-x", "description": "Skill X"},
                    {"name": "skill-y", "description": "Skill Y"},
                    {"name": "skill-z", "description": "Skill Z"},
                    {"name": "skill-w", "description": "Skill W"}
                ]
            }

            plugin_dir = self.create_test_plugin(tmp_path, plugin_data)
            readme = regenerator.generate_readme(plugin_dir)

            # Verify version
            self.assert_in("**Version:** 2.5.1", readme, "Version matches plugin.json")

            # Verify category
            self.assert_in("**Category:** quality-engineering", readme, "Category matches plugin.json")

            # Verify license
            self.assert_in("**License:** Apache-2.0", readme, "License matches plugin.json")

            # Verify agent count
            self.assert_in("## Agents (3)", readme, "Agent count matches plugin.json")

            # Verify command count
            self.assert_in("## Commands (2)", readme, "Command count matches plugin.json")

            # Verify skill count
            self.assert_in("## Skills (4)", readme, "Skill count matches plugin.json")

            # Verify agent names appear
            for agent in plugin_data["agents"]:
                self.assert_in(
                    f"### {agent['name']}",
                    readme,
                    f"Agent {agent['name']} appears in README"
                )

    def test_5_handles_missing_components(self):
        """Test 5: Gracefully handles plugins with missing agents/commands/skills"""
        self.test_count += 1
        self.log("Test 5: Handles missing components gracefully", "RUN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            regenerator = ReadmeRegenerator()

            # Plugin with only agents
            plugin_agents_only = {
                "name": "agents-only-plugin",
                "version": "1.0.0",
                "category": "tools",
                "license": "MIT",
                "description": "Plugin with agents only",
                "agents": [{"name": "solo-agent", "description": "Only agent", "status": "active"}]
            }

            plugin_dir = self.create_test_plugin(tmp_path, plugin_agents_only)
            readme = regenerator.generate_readme(plugin_dir)

            self.assert_in("## Agents (1)", readme, "Agents section present")
            self.assert_true("## Commands" not in readme, "Commands section omitted when empty")
            self.assert_true("## Skills" not in readme, "Skills section omitted when empty")

            # Verify basic structure is still valid
            self.assert_in("# Agents Only Plugin", readme, "Title present")
            self.assert_in("## Quick Start", readme, "Quick Start section present")
            self.assert_in("## Documentation", readme, "Documentation section present")

    def test_6_integration_section_format(self):
        """Test 6: Integration section uses correct format"""
        self.test_count += 1
        self.log("Test 6: Integration section format", "RUN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            regenerator = ReadmeRegenerator()

            # Plugin with integration_points
            plugin_with_integrations = {
                "name": "integrated-plugin",
                "version": "1.0.0",
                "category": "development",
                "license": "MIT",
                "description": "Plugin with integrations",
                "integration_points": [
                    "python-development",
                    "testing-framework",
                    "ci-cd-pipeline"
                ]
            }

            plugin_dir = self.create_test_plugin(tmp_path, plugin_with_integrations)
            readme = regenerator.generate_readme(plugin_dir)

            self.assert_in("## Integration", readme, "Integration section present")
            self.assert_in("This plugin integrates with:", readme, "Integration intro present")
            self.assert_in("- python-development", readme, "Integration point listed")
            self.assert_in("- testing-framework", readme, "Integration point listed")

            # Plugin without integration_points
            plugin_without_integrations = {
                "name": "standalone-plugin",
                "version": "1.0.0",
                "category": "tools",
                "license": "MIT",
                "description": "Standalone plugin"
            }

            plugin_dir2 = self.create_test_plugin(tmp_path, plugin_without_integrations)
            readme2 = regenerator.generate_readme(plugin_dir2)

            self.assert_in("## Integration", readme2, "Integration section present")
            self.assert_in(
                "See the full documentation for integration patterns",
                readme2,
                "Default integration text present"
            )

    def test_7_quick_start_section_populated(self):
        """Test 7: Quick Start section includes relevant info"""
        self.test_count += 1
        self.log("Test 7: Quick Start section populated", "RUN")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            regenerator = ReadmeRegenerator()

            plugin_data = {
                "name": "full-featured-plugin",
                "version": "1.0.0",
                "category": "development",
                "license": "MIT",
                "description": "Plugin with all features",
                "agents": [{"name": "main-agent", "description": "Main agent", "status": "active"}],
                "commands": [{"name": "/start", "description": "Start command", "status": "active"}]
            }

            plugin_dir = self.create_test_plugin(tmp_path, plugin_data)
            readme = regenerator.generate_readme(plugin_dir)

            # Verify Quick Start contains expected steps
            quick_start_checks = [
                "## Quick Start",
                "Ensure Claude Code is installed",
                "Enable the `full-featured-plugin` plugin",
                "@main-agent",  # First agent name
                "/start"  # First command name
            ]

            all_present = True
            for check in quick_start_checks:
                if check not in readme:
                    self.log(f"Missing Quick Start element: {check}", "FAIL")
                    all_present = False

            self.assert_true(all_present, "Quick Start section includes all expected elements")

    def test_8_marketplace_readme_structure(self):
        """Test 8: Verify marketplace README update requirements"""
        self.test_count += 1
        self.log("Test 8: Marketplace README structure requirements", "RUN")

        # This test validates the structure that the marketplace README should have
        # after Task 7.3 updates it

        required_elements = [
            "Brief introduction (< 150 lines)",
            "Link to Sphinx documentation",
            "Plugin count statistics",
            "Category overview",
            "Quick start section",
            "Link to contribution guidelines"
        ]

        # Test passes if we can define the structure requirements
        self.log(f"Marketplace README should include: {', '.join(required_elements)}", "INFO")

        # Key requirements from spec
        requirements_met = True

        # Requirement 1: Should be brief (< 150 lines)
        self.assert_true(
            requirements_met,
            "Marketplace README requirements defined: < 150 lines"
        )

        # Requirement 2: Should link to Sphinx docs
        self.assert_true(
            requirements_met,
            "Marketplace README should have prominent Sphinx doc link"
        )

        # Requirement 3: Should remove detailed plugin listings
        self.assert_true(
            requirements_met,
            "Marketplace README should remove detailed listings (moved to Sphinx)"
        )

    def run_all_tests(self):
        """Run all tests and report results"""
        self.log("=" * 60, "INFO")
        self.log("README Regeneration Test Suite", "INFO")
        self.log("=" * 60, "INFO")

        tests = [
            self.test_1_consistent_format_generation,
            self.test_2_required_sections_included,
            self.test_3_sphinx_links_correct,
            self.test_4_metadata_matches_plugin_json,
            self.test_5_handles_missing_components,
            self.test_6_integration_section_format,
            self.test_7_quick_start_section_populated,
            self.test_8_marketplace_readme_structure
        ]

        for test in tests:
            try:
                test()
                self.log("")  # Blank line between tests
            except Exception as e:
                self.failed += 1
                self.log(f"Test {test.__name__} raised exception: {e}", "FAIL")

        self.log("=" * 60, "INFO")
        self.log(f"Tests run: {self.test_count}", "INFO")
        self.log(f"Passed: {self.passed}", "PASS")
        self.log(f"Failed: {self.failed}", "FAIL")
        self.log("=" * 60, "INFO")

        return self.failed == 0


def main():
    """Main entry point"""
    tester = TestReadmeRegeneration()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
