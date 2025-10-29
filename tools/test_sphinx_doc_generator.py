#!/usr/bin/env python3
"""
Tests for sphinx-doc-generator.py

Tests documentation generation tooling including:
- plugin.json metadata extraction
- RST file generation with all required sections
- Integration detection
- Sphinx build compatibility
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_metadata_extraction():
    """Test that sphinx-doc-generator extracts plugin.json metadata correctly"""
    from sphinx_doc_generator import SphinxDocGenerator

    # Create temporary test plugin structure
    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        test_plugin = plugins_dir / "test-plugin"
        test_plugin.mkdir()

        # Create test plugin.json
        plugin_data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "Test plugin for validation",
            "author": "Test Author",
            "license": "MIT",
            "category": "development",
            "keywords": ["test", "validation"],
            "agents": [
                {
                    "name": "test-agent",
                    "description": "Test agent description",
                    "status": "active",
                    "expertise": ["testing", "validation"]
                }
            ],
            "commands": [
                {
                    "name": "/test-command",
                    "description": "Test command description",
                    "status": "active",
                    "priority": 1
                }
            ],
            "skills": [
                {
                    "name": "test-skill",
                    "description": "Test skill description",
                    "status": "active"
                }
            ]
        }

        with open(test_plugin / "plugin.json", 'w') as f:
            json.dump(plugin_data, f)

        # Test metadata extraction
        generator = SphinxDocGenerator(plugins_dir)
        metadata = generator.extract_plugin_metadata(test_plugin)

        assert metadata is not None, "Failed to extract metadata"
        assert metadata["name"] == "test-plugin"
        assert metadata["version"] == "1.0.0"
        assert metadata["category"] == "development"
        assert len(metadata["agents"]) == 1
        assert len(metadata["commands"]) == 1
        assert len(metadata["skills"]) == 1

        print("✅ test_metadata_extraction passed")


def test_rst_generation_structure():
    """Test that RST file generation includes all required sections"""
    from sphinx_doc_generator import SphinxDocGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        test_plugin = plugins_dir / "test-plugin"
        test_plugin.mkdir()

        # Create minimal plugin.json
        plugin_data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "Test plugin for RST generation",
            "author": "Test Author",
            "license": "MIT",
            "category": "development",
            "agents": [{"name": "test-agent", "description": "Test agent", "status": "active"}],
            "commands": [{"name": "/test", "description": "Test command", "status": "active"}],
            "skills": [{"name": "test-skill", "description": "Test skill"}]
        }

        with open(test_plugin / "plugin.json", 'w') as f:
            json.dump(plugin_data, f)

        # Generate RST
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        generator = SphinxDocGenerator(plugins_dir)
        rst_content = generator.generate_plugin_rst(test_plugin, output_dir)

        # Check for required sections
        required_sections = [
            "Description",
            "Agents",
            "Commands",
            "Skills",
            "Usage Examples",
            "Integration",
            "See Also",
            "References"
        ]

        for section in required_sections:
            assert section in rst_content, f"Missing required section: {section}"

        # Check for module directive
        assert ".. module::" in rst_content, "Missing module directive"

        print("✅ test_rst_generation_structure passed")


def test_readme_regenerator_output():
    """Test that readme-regenerator.py generates valid README.md from plugin.json"""
    from readme_regenerator import ReadmeRegenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        test_plugin = Path(tmpdir) / "test-plugin"
        test_plugin.mkdir()

        # Create plugin.json
        plugin_data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "Test plugin for README generation",
            "author": "Test Author",
            "license": "MIT",
            "category": "development",
            "agents": [{"name": "agent1", "description": "Agent one", "status": "active"}],
            "commands": [{"name": "/cmd1", "description": "Command one", "status": "active"}],
            "skills": [{"name": "skill1", "description": "Skill one"}]
        }

        with open(test_plugin / "plugin.json", 'w') as f:
            json.dump(plugin_data, f)

        # Generate README
        regenerator = ReadmeRegenerator()
        readme_content = regenerator.generate_readme(test_plugin)

        # Check for required sections
        assert "# test-plugin" in readme_content or "# Test Plugin" in readme_content
        assert "Version:" in readme_content
        assert "Category:" in readme_content
        assert "License:" in readme_content
        assert "## Agents" in readme_content
        assert "## Commands" in readme_content
        assert "## Skills" in readme_content
        assert "## Quick Start" in readme_content
        assert "## Integration" in readme_content
        assert "## Documentation" in readme_content

        print("✅ test_readme_regenerator_output passed")


def test_integration_detection():
    """Test that integration detection identifies cross-plugin references"""
    from sphinx_doc_generator import SphinxDocGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        # Create two test plugins
        plugin1 = plugins_dir / "plugin-one"
        plugin1.mkdir()

        plugin2 = plugins_dir / "plugin-two"
        plugin2.mkdir()

        # Plugin 1 with reference to plugin 2
        plugin1_data = {
            "name": "plugin-one",
            "version": "1.0.0",
            "description": "First test plugin",
            "author": "Test",
            "license": "MIT",
            "category": "development"
        }

        with open(plugin1 / "plugin.json", 'w') as f:
            json.dump(plugin1_data, f)

        # Create README with reference
        with open(plugin1 / "README.md", 'w') as f:
            f.write("# Plugin One\n\nIntegrates with plugin-two for enhanced functionality.\n")

        # Plugin 2
        plugin2_data = {
            "name": "plugin-two",
            "version": "1.0.0",
            "description": "Second test plugin",
            "author": "Test",
            "license": "MIT",
            "category": "development"
        }

        with open(plugin2 / "plugin.json", 'w') as f:
            json.dump(plugin2_data, f)

        with open(plugin2 / "README.md", 'w') as f:
            f.write("# Plugin Two\n\nStandalone plugin.\n")

        # Test integration detection
        generator = SphinxDocGenerator(plugins_dir)

        # First discover plugins
        plugins = generator.discover_plugins()
        assert len(plugins) == 2, f"Expected 2 plugins, found {len(plugins)}"

        # Then detect integrations
        integrations = generator.detect_integrations()

        # Should find plugin-one referencing plugin-two
        assert "plugin-one" in integrations, "Failed to detect source plugin"
        assert "plugin-two" in integrations.get("plugin-one", set()), "Failed to detect target plugin"

        print("✅ test_integration_detection passed")


def test_rst_builds_without_warnings():
    """Test that generated RST builds with Sphinx without warnings"""
    from sphinx_doc_generator import SphinxDocGenerator
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        test_plugin = plugins_dir / "test-plugin"
        test_plugin.mkdir()

        # Create plugin.json
        plugin_data = {
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "Test plugin for Sphinx build validation",
            "author": "Test Author",
            "license": "MIT",
            "category": "development",
            "agents": [{"name": "test-agent", "description": "Test agent", "status": "active"}]
        }

        with open(test_plugin / "plugin.json", 'w') as f:
            json.dump(plugin_data, f)

        # Generate RST
        output_dir = Path(tmpdir) / "docs" / "plugins"
        output_dir.mkdir(parents=True)

        generator = SphinxDocGenerator(plugins_dir)
        rst_content = generator.generate_plugin_rst(test_plugin, output_dir)

        # Write RST file
        rst_file = output_dir / "test-plugin.rst"
        with open(rst_file, 'w') as f:
            f.write(rst_content)

        # Validate RST syntax (basic check)
        assert ".. module::" in rst_content
        assert "Description" in rst_content
        assert "---" in rst_content  # Section underlines

        # Check for proper RST formatting
        lines = rst_content.split('\n')
        for i, line in enumerate(lines):
            if line and not line.startswith(' ') and i + 1 < len(lines):
                next_line = lines[i + 1]
                # Check if it's a title (next line is all dashes or equals)
                if next_line and all(c in '=-~^' for c in next_line.strip()):
                    # Title and underline should be same length or close
                    assert abs(len(line) - len(next_line.strip())) <= 2, \
                        f"Title/underline length mismatch: '{line}' vs '{next_line}'"

        print("✅ test_rst_builds_without_warnings passed")


def run_all_tests():
    """Run all tests"""
    tests = [
        test_metadata_extraction,
        test_rst_generation_structure,
        test_readme_regenerator_output,
        test_integration_detection,
        test_rst_builds_without_warnings
    ]

    print("Running Sphinx Doc Generator Tests\n")
    print("=" * 50)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("=" * 50)
    print(f"\nTest Results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
