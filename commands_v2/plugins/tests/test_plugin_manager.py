#!/usr/bin/env python3
"""
Plugin Manager Tests
====================

Test suite for plugin management system.
"""

import unittest
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.plugin_base import (
    BasePlugin, CommandPlugin, PluginMetadata, PluginType,
    PluginContext, PluginResult, PluginStatus
)
from core.plugin_manager import PluginManager, PluginDiscovery, PluginLoader
from core.plugin_hooks import HookRegistry, HookManager, HookType


class TestPluginBase(unittest.TestCase):
    """Test base plugin functionality"""

    def setUp(self):
        """Setup test plugin"""
        self.metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            plugin_type=PluginType.COMMAND,
            description="Test plugin",
            author="Test Author"
        )

    def test_plugin_metadata(self):
        """Test plugin metadata"""
        self.assertEqual(self.metadata.name, "test-plugin")
        self.assertEqual(self.metadata.version, "1.0.0")
        self.assertEqual(self.metadata.plugin_type, PluginType.COMMAND)

    def test_plugin_validation(self):
        """Test plugin validation"""
        # Create mock plugin
        class MockPlugin(CommandPlugin):
            def load(self):
                return True

            def execute(self, context):
                return PluginResult(success=True, plugin_name=self.metadata.name)

            def get_command_info(self):
                return {"name": "test"}

        plugin = MockPlugin(self.metadata)

        # Validate
        valid, errors = plugin.validate()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_plugin_config(self):
        """Test plugin configuration"""
        class MockPlugin(CommandPlugin):
            def load(self):
                return True

            def execute(self, context):
                return PluginResult(success=True, plugin_name=self.metadata.name)

            def get_command_info(self):
                return {"name": "test"}

        config = {"key": "value"}
        plugin = MockPlugin(self.metadata, config)

        self.assertEqual(plugin.get_config("key"), "value")
        self.assertEqual(plugin.get_config("missing", "default"), "default")

        plugin.set_config("new_key", "new_value")
        self.assertEqual(plugin.get_config("new_key"), "new_value")

    def test_plugin_state(self):
        """Test plugin state management"""
        class MockPlugin(CommandPlugin):
            def load(self):
                return True

            def execute(self, context):
                return PluginResult(success=True, plugin_name=self.metadata.name)

            def get_command_info(self):
                return {"name": "test"}

        plugin = MockPlugin(self.metadata)

        # Test state
        plugin.set_state("counter", 0)
        self.assertEqual(plugin.get_state("counter"), 0)

        plugin.set_state("counter", 1)
        self.assertEqual(plugin.get_state("counter"), 1)


class TestPluginDiscovery(unittest.TestCase):
    """Test plugin discovery"""

    def test_discover_from_directory(self):
        """Test discovery from directory"""
        # Create temporary plugin directory
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "test-plugin"
            plugin_dir.mkdir()

            # Create plugin manifest
            manifest = {
                "name": "test-plugin",
                "version": "1.0.0",
                "type": "command",
                "description": "Test",
                "author": "Test"
            }

            (plugin_dir / "plugin.json").write_text(json.dumps(manifest))

            # Discover
            discovery = PluginDiscovery([Path(tmpdir)])
            sources = discovery.discover_all()

            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].type, "directory")
            self.assertEqual(sources[0].metadata["name"], "test-plugin")


class TestHookRegistry(unittest.TestCase):
    """Test hook registry"""

    def test_register_hook(self):
        """Test hook registration"""
        registry = HookRegistry()

        def test_hook(context, data):
            data["test"] = True
            return data

        registry.register_hook(
            "test-plugin",
            HookType.PRE_EXECUTION,
            test_hook,
            priority=5
        )

        # Check registration
        hooks = registry.get_hooks(HookType.PRE_EXECUTION)
        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0].plugin_name, "test-plugin")

    def test_execute_hooks(self):
        """Test hook execution"""
        registry = HookRegistry()

        def test_hook(context, data):
            data["executed"] = True
            return data

        registry.register_hook(
            "test-plugin",
            HookType.PRE_EXECUTION,
            test_hook
        )

        # Execute
        context = PluginContext(
            plugin_name="test",
            command_name="test",
            work_dir=Path.cwd(),
            config={},
            framework_version="2.0.0"
        )

        result = registry.execute_hooks(HookType.PRE_EXECUTION, context, {})

        self.assertTrue(result.get("executed"))

    def test_hook_priority(self):
        """Test hook priority ordering"""
        registry = HookRegistry()

        execution_order = []

        def hook1(context, data):
            execution_order.append(1)
            return data

        def hook2(context, data):
            execution_order.append(2)
            return data

        # Register with different priorities
        registry.register_hook("plugin1", HookType.PRE_EXECUTION, hook1, priority=5)
        registry.register_hook("plugin2", HookType.PRE_EXECUTION, hook2, priority=8)

        # Execute
        context = PluginContext(
            plugin_name="test",
            command_name="test",
            work_dir=Path.cwd(),
            config={},
            framework_version="2.0.0"
        )

        registry.execute_hooks(HookType.PRE_EXECUTION, context, {})

        # Hook2 should execute first (higher priority)
        self.assertEqual(execution_order, [2, 1])


class TestPluginResult(unittest.TestCase):
    """Test plugin result"""

    def test_success_result(self):
        """Test success result"""
        result = PluginResult(
            success=True,
            plugin_name="test",
            data={"key": "value"}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.plugin_name, "test")
        self.assertEqual(result.data["key"], "value")

    def test_error_result(self):
        """Test error result"""
        result = PluginResult(
            success=False,
            plugin_name="test",
            errors=["Error message"]
        )

        self.assertFalse(result.success)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0], "Error message")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPluginBase))
    suite.addTests(loader.loadTestsFromTestCase(TestPluginDiscovery))
    suite.addTests(loader.loadTestsFromTestCase(TestHookRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestPluginResult))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())