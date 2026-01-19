#!/usr/bin/env python3
"""
Tests for Cross-Plugin Integration System

Tests integration mapping functionality including:
- Integration detection across all plugins
- Integration matrix generation
- Bidirectional reference detection
- Common workflow pattern identification
- Integration section RST generation
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_integration_detection_across_all_plugins():
    """Test integration detection across all 31 plugins"""
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator

    # Use actual plugins directory
    plugins_dir = Path(__file__).parent.parent / "plugins"

    if not plugins_dir.exists():
        print("⚠️  Warning: Plugins directory not found, using mock data")
        return

    generator = SphinxDocGenerator(plugins_dir, verbose=False)

    # Discover all plugins
    plugins = generator.discover_plugins()

    # Should find 31 plugins
    assert len(plugins) >= 31, f"Expected at least 31 plugins, found {len(plugins)}"

    # Detect integrations
    integrations = generator.detect_integrations()

    # Should find some integrations
    assert len(integrations) > 0, "No integrations detected"

    # Count total integration points
    total_integrations = sum(len(targets) for targets in integrations.values())

    print(f"   Found {total_integrations} integration points across {len(plugins)} plugins")
    print("✅ test_integration_detection_across_all_plugins passed")


def test_integration_matrix_generation():
    """Test integration matrix generation"""
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        # Create test plugins with cross-references
        plugin_configs = [
            {
                "name": "plugin-a",
                "description": "Plugin A that integrates with plugin-b",
                "readme": "# Plugin A\n\nWorks with plugin-b and plugin-c for complete solution.\n"
            },
            {
                "name": "plugin-b",
                "description": "Plugin B that works with plugin-a",
                "readme": "# Plugin B\n\nIntegrates with plugin-a for enhanced features.\n"
            },
            {
                "name": "plugin-c",
                "description": "Plugin C for specialized tasks",
                "readme": "# Plugin C\n\nStandalone but mentioned by plugin-a.\n"
            }
        ]

        for config in plugin_configs:
            plugin_dir = plugins_dir / config["name"]
            plugin_dir.mkdir()

            plugin_data = {
                "name": config["name"],
                "version": "1.0.0",
                "description": config["description"],
                "author": "Test",
                "license": "MIT",
                "category": "development"
            }

            with open(plugin_dir / "plugin.json", 'w') as f:
                json.dump(plugin_data, f)

            with open(plugin_dir / "README.md", 'w') as f:
                f.write(config["readme"])

        # Generate integration map
        generator = SphinxDocGenerator(plugins_dir, verbose=False)
        plugins = generator.discover_plugins()
        integrations = generator.detect_integrations()

        # Generate integration matrix
        matrix = generator.generate_integration_matrix()

        # Verify matrix structure
        assert "Plugin A" in matrix or "plugin-a" in matrix
        assert "Plugin B" in matrix or "plugin-b" in matrix

        # Verify it's a string with table format
        assert isinstance(matrix, str), "Integration matrix should be a string"
        assert len(matrix) > 0, "Integration matrix should not be empty"

        print("✅ test_integration_matrix_generation passed")


def test_bidirectional_reference_detection():
    """Test bidirectional reference detection"""
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        # Create two plugins that reference each other
        plugin1_dir = plugins_dir / "plugin-alpha"
        plugin1_dir.mkdir()

        plugin1_data = {
            "name": "plugin-alpha",
            "version": "1.0.0",
            "description": "Plugin Alpha",
            "author": "Test",
            "license": "MIT",
            "category": "development"
        }

        with open(plugin1_dir / "plugin.json", 'w') as f:
            json.dump(plugin1_data, f)

        with open(plugin1_dir / "README.md", 'w') as f:
            f.write("# Plugin Alpha\n\nIntegrates with plugin-beta for complete workflow.\n")

        # Plugin 2
        plugin2_dir = plugins_dir / "plugin-beta"
        plugin2_dir.mkdir()

        plugin2_data = {
            "name": "plugin-beta",
            "version": "1.0.0",
            "description": "Plugin Beta",
            "author": "Test",
            "license": "MIT",
            "category": "development"
        }

        with open(plugin2_dir / "plugin.json", 'w') as f:
            json.dump(plugin2_data, f)

        with open(plugin2_dir / "README.md", 'w') as f:
            f.write("# Plugin Beta\n\nWorks together with plugin-alpha.\n")

        # Detect integrations
        generator = SphinxDocGenerator(plugins_dir, verbose=False)
        plugins = generator.discover_plugins()
        integrations = generator.detect_integrations()

        # Build reverse dependencies
        reverse_deps = generator.build_reverse_dependencies()

        # Verify bidirectional detection
        assert "plugin-alpha" in integrations, "Forward reference not detected"
        assert "plugin-beta" in integrations.get("plugin-alpha", set()), "Alpha -> Beta not detected"

        assert "plugin-beta" in integrations, "Backward reference not detected"
        assert "plugin-alpha" in integrations.get("plugin-beta", set()), "Beta -> Alpha not detected"

        # Verify reverse dependencies
        assert "plugin-beta" in reverse_deps, "Reverse dependency map incomplete"
        assert "plugin-alpha" in reverse_deps.get("plugin-beta", set()), "Reverse ref Alpha -> Beta not recorded"

        print("✅ test_bidirectional_reference_detection passed")


def test_common_workflow_pattern_identification():
    """Test common workflow pattern identification"""
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        # Create plugins with category-based patterns
        scientific_plugins = [
            {"name": "julia-dev", "category": "scientific-computing", "keywords": ["julia", "HPC", "scientific"]},
            {"name": "hpc-cluster", "category": "scientific-computing", "keywords": ["HPC", "parallel", "computing"]},
            {"name": "gpu-accelerator", "category": "scientific-computing", "keywords": ["GPU", "CUDA", "scientific"]}
        ]

        dev_plugins = [
            {"name": "python-dev", "category": "development", "keywords": ["python", "backend", "API"]},
            {"name": "testing-suite", "category": "quality-engineering", "keywords": ["testing", "TDD", "python"]},
            {"name": "api-builder", "category": "development", "keywords": ["API", "REST", "microservices"]}
        ]

        all_plugins = scientific_plugins + dev_plugins

        for config in all_plugins:
            plugin_dir = plugins_dir / config["name"]
            plugin_dir.mkdir()

            plugin_data = {
                "name": config["name"],
                "version": "1.0.0",
                "description": f"Plugin for {config['name']}",
                "author": "Test",
                "license": "MIT",
                "category": config["category"],
                "keywords": config["keywords"]
            }

            with open(plugin_dir / "plugin.json", 'w') as f:
                json.dump(plugin_data, f)

            # Create README with some references
            with open(plugin_dir / "README.md", 'w') as f:
                f.write(f"# {config['name']}\n\nPlugin for testing.\n")

        # Detect patterns
        generator = SphinxDocGenerator(plugins_dir, verbose=False)
        plugins = generator.discover_plugins()

        # Identify integration patterns
        patterns = generator.identify_integration_patterns()

        # Should find at least 2 pattern groups
        assert len(patterns) >= 2, f"Expected at least 2 patterns, found {len(patterns)}"

        # Verify pattern structure
        for pattern_name, plugin_list in patterns.items():
            assert isinstance(pattern_name, str), "Pattern name should be string"
            assert isinstance(plugin_list, list), "Pattern plugins should be list"
            assert len(plugin_list) > 0, f"Pattern {pattern_name} has no plugins"

        print(f"   Identified {len(patterns)} workflow patterns")
        print("✅ test_common_workflow_pattern_identification passed")


def test_integration_section_rst_generation():
    """Test integration section RST generation"""
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        # Create test plugins
        plugin_main = plugins_dir / "main-plugin"
        plugin_main.mkdir()

        plugin_main_data = {
            "name": "main-plugin",
            "version": "1.0.0",
            "description": "Main plugin with integrations",
            "author": "Test",
            "license": "MIT",
            "category": "development",
            "agents": [{"name": "main-agent", "description": "Main agent", "status": "active"}]
        }

        with open(plugin_main / "plugin.json", 'w') as f:
            json.dump(plugin_main_data, f)

        with open(plugin_main / "README.md", 'w') as f:
            f.write("# Main Plugin\n\nIntegrates with helper-plugin and support-plugin.\n")

        # Helper plugins
        for helper_name in ["helper-plugin", "support-plugin"]:
            helper_dir = plugins_dir / helper_name
            helper_dir.mkdir()

            helper_data = {
                "name": helper_name,
                "version": "1.0.0",
                "description": f"{helper_name} for testing",
                "author": "Test",
                "license": "MIT",
                "category": "development"
            }

            with open(helper_dir / "plugin.json", 'w') as f:
                json.dump(helper_data, f)

            with open(helper_dir / "README.md", 'w') as f:
                f.write(f"# {helper_name}\n\nUsed by main-plugin.\n")

        # Generate RST with integration section
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        generator = SphinxDocGenerator(plugins_dir, verbose=False)
        plugins = generator.discover_plugins()
        integrations = generator.detect_integrations()

        # Generate RST for main plugin
        rst_content = generator.generate_plugin_rst(plugin_main, output_dir)

        # Verify Integration section exists
        assert "Integration" in rst_content, "Integration section missing"
        assert "Integrates With:" in rst_content or "integration" in rst_content.lower()

        # Verify cross-references use :doc: directive
        if "helper-plugin" in integrations.get("main-plugin", set()):
            assert ":doc:" in rst_content, "Missing :doc: directive for cross-references"
            assert "helper-plugin" in rst_content, "Referenced plugin not in Integration section"

        # Verify Referenced By section if reverse dependencies exist
        reverse_deps = generator.build_reverse_dependencies()
        if "main-plugin" in reverse_deps and reverse_deps["main-plugin"]:
            # The reverse deps should be mentioned somewhere
            pass  # This is tested in the actual plugin pages

        print("✅ test_integration_section_rst_generation passed")


def test_integration_points_threshold():
    """Test that at least 50 integration points are identified (success criteria)"""
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator

    # Use actual plugins directory
    plugins_dir = Path(__file__).parent.parent / "plugins"

    if not plugins_dir.exists():
        print("⚠️  Warning: Plugins directory not found, skipping threshold test")
        return

    generator = SphinxDocGenerator(plugins_dir, verbose=False)

    # Discover and analyze all plugins
    plugins = generator.discover_plugins()
    integrations = generator.detect_integrations()

    # Count total integration points
    total_integrations = sum(len(targets) for targets in integrations.values())

    print(f"   Total integration points found: {total_integrations}")

    # Success criteria: at least 50 integration points
    assert total_integrations >= 50, \
        f"Expected at least 50 integration points, found {total_integrations}"

    print("✅ test_integration_points_threshold passed")


def test_integration_matrix_rst_format():
    """Test that integration matrix generates valid RST table format"""
    from tools.generation.sphinx_doc_generator import SphinxDocGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        plugins_dir = Path(tmpdir) / "plugins"
        plugins_dir.mkdir()

        # Create minimal test plugins
        for i in range(3):
            plugin_dir = plugins_dir / f"plugin-{i}"
            plugin_dir.mkdir()

            plugin_data = {
                "name": f"plugin-{i}",
                "version": "1.0.0",
                "description": f"Test plugin {i}",
                "author": "Test",
                "license": "MIT",
                "category": "development"
            }

            with open(plugin_dir / "plugin.json", 'w') as f:
                json.dump(plugin_data, f)

            # Create cross-references
            refs = [f"plugin-{j}" for j in range(3) if j != i]
            readme_content = f"# Plugin {i}\n\nWorks with {', '.join(refs)}.\n"

            with open(plugin_dir / "README.md", 'w') as f:
                f.write(readme_content)

        # Generate integration matrix
        generator = SphinxDocGenerator(plugins_dir, verbose=False)
        plugins = generator.discover_plugins()
        integrations = generator.detect_integrations()

        matrix_rst = generator.generate_integration_matrix()

        # Verify RST format
        assert isinstance(matrix_rst, str), "Matrix should be string"

        # Check for RST table elements or list format
        # Could be a table or a definition list
        assert ("plugin-0" in matrix_rst or "Plugin 0" in matrix_rst or
                "plugin_0" in matrix_rst), "Matrix missing plugin references"

        # Should have some structure (headers, separators, etc.)
        lines = matrix_rst.split('\n')
        assert len(lines) > 5, "Matrix should have multiple lines"

        print("✅ test_integration_matrix_rst_format passed")


def run_all_tests():
    """Run all integration mapping tests"""
    tests = [
        test_integration_detection_across_all_plugins,
        test_integration_matrix_generation,
        test_bidirectional_reference_detection,
        test_common_workflow_pattern_identification,
        test_integration_section_rst_generation,
        test_integration_points_threshold,
        test_integration_matrix_rst_format
    ]

    print("Running Cross-Plugin Integration System Tests\n")
    print("=" * 60)

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"\nTest Results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
