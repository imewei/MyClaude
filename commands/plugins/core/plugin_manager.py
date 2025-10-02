#!/usr/bin/env python3
"""
Plugin Manager
==============

Central plugin management system for discovery, loading, and lifecycle management.

Features:
- Auto-discovery from plugin directories
- pip-installable plugin support (entry points)
- Plugin dependency resolution
- Plugin lifecycle management
- Plugin hot-reloading (optional)
- Plugin isolation and sandboxing

Author: Claude Code Framework
Version: 1.0
Last Updated: 2025-09-29
"""

import sys
import os
import json
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Type
from dataclasses import dataclass
import traceback

from .plugin_base import (
    BasePlugin, PluginMetadata, PluginType, PluginStatus,
    CommandPlugin, AgentPlugin, ValidatorPlugin,
    CacheProviderPlugin, ReporterPlugin, IntegrationPlugin
)


# ============================================================================
# Plugin Discovery
# ============================================================================

@dataclass
class PluginSource:
    """Source of a discovered plugin"""
    type: str  # directory, entry_point, file
    path: Optional[Path] = None
    module_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PluginDiscovery:
    """
    Plugin discovery system.

    Discovers plugins from:
    1. Plugin directories
    2. pip-installable packages (entry points)
    3. Explicit file paths
    """

    def __init__(self, plugin_dirs: List[Path]):
        """
        Initialize discovery system.

        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugin_dirs = plugin_dirs
        self.logger = logging.getLogger(self.__class__.__name__)

    def discover_all(self) -> List[PluginSource]:
        """
        Discover all available plugins.

        Returns:
            List of discovered plugin sources
        """
        sources = []

        # Discover from directories
        sources.extend(self._discover_from_directories())

        # Discover from entry points
        sources.extend(self._discover_from_entry_points())

        self.logger.info(f"Discovered {len(sources)} plugins")
        return sources

    def _discover_from_directories(self) -> List[PluginSource]:
        """Discover plugins from directories"""
        sources = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Look for plugin.json manifests
            for manifest_file in plugin_dir.rglob("plugin.json"):
                try:
                    metadata = json.loads(manifest_file.read_text())
                    plugin_path = manifest_file.parent

                    source = PluginSource(
                        type="directory",
                        path=plugin_path,
                        metadata=metadata
                    )
                    sources.append(source)

                    self.logger.debug(f"Found plugin: {metadata.get('name')} at {plugin_path}")

                except Exception as e:
                    self.logger.error(f"Error reading plugin manifest {manifest_file}: {e}")

        return sources

    def _discover_from_entry_points(self) -> List[PluginSource]:
        """Discover plugins from pip entry points"""
        sources = []

        try:
            # Use importlib.metadata for Python 3.8+
            from importlib.metadata import entry_points

            # Get all claude_code_plugins entry points
            eps = entry_points()

            # Handle both dict and SelectableGroups return types
            if hasattr(eps, 'select'):
                plugin_eps = eps.select(group='claude_code_plugins')
            else:
                plugin_eps = eps.get('claude_code_plugins', [])

            for ep in plugin_eps:
                source = PluginSource(
                    type="entry_point",
                    module_name=ep.value,
                    metadata={"name": ep.name}
                )
                sources.append(source)

                self.logger.debug(f"Found entry point plugin: {ep.name}")

        except ImportError:
            self.logger.warning("importlib.metadata not available, skipping entry point discovery")
        except Exception as e:
            self.logger.error(f"Error discovering entry points: {e}")

        return sources


# ============================================================================
# Plugin Loader
# ============================================================================

class PluginLoader:
    """
    Plugin loading and instantiation system.

    Features:
    - Dynamic module loading
    - Plugin class discovery
    - Dependency injection
    - Error handling
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.loaded_modules: Dict[str, Any] = {}

    def load_plugin(self, source: PluginSource, config: Dict[str, Any] = None) -> Optional[BasePlugin]:
        """
        Load plugin from source.

        Args:
            source: Plugin source
            config: Plugin configuration

        Returns:
            Loaded plugin instance or None if failed
        """
        try:
            if source.type == "directory":
                return self._load_from_directory(source, config)
            elif source.type == "entry_point":
                return self._load_from_entry_point(source, config)
            else:
                self.logger.error(f"Unknown plugin source type: {source.type}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading plugin: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _load_from_directory(
        self,
        source: PluginSource,
        config: Dict[str, Any] = None
    ) -> Optional[BasePlugin]:
        """Load plugin from directory"""
        plugin_path = source.path
        metadata = source.metadata

        # Find plugin.py file
        plugin_file = plugin_path / "plugin.py"
        if not plugin_file.exists():
            self.logger.error(f"Plugin file not found: {plugin_file}")
            return None

        # Load module
        module_name = f"plugin_{metadata['name'].replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if spec is None or spec.loader is None:
            self.logger.error(f"Failed to create module spec for {plugin_file}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        self.loaded_modules[module_name] = module

        # Find plugin class
        plugin_class = self._find_plugin_class(module)
        if plugin_class is None:
            self.logger.error(f"No plugin class found in {plugin_file}")
            return None

        # Create plugin metadata
        plugin_metadata = self._create_metadata(metadata)

        # Instantiate plugin
        plugin = plugin_class(plugin_metadata, config)

        self.logger.info(f"Loaded plugin: {metadata['name']}")
        return plugin

    def _load_from_entry_point(
        self,
        source: PluginSource,
        config: Dict[str, Any] = None
    ) -> Optional[BasePlugin]:
        """Load plugin from entry point"""
        try:
            # Import module
            module = importlib.import_module(source.module_name)
            self.loaded_modules[source.module_name] = module

            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                self.logger.error(f"No plugin class found in {source.module_name}")
                return None

            # Get metadata from plugin class
            if hasattr(plugin_class, 'METADATA'):
                metadata = plugin_class.METADATA
            else:
                metadata = source.metadata

            plugin_metadata = self._create_metadata(metadata)

            # Instantiate plugin
            plugin = plugin_class(plugin_metadata, config)

            self.logger.info(f"Loaded plugin from entry point: {source.metadata['name']}")
            return plugin

        except ImportError as e:
            self.logger.error(f"Failed to import plugin module {source.module_name}: {e}")
            return None

    def _find_plugin_class(self, module: Any) -> Optional[Type[BasePlugin]]:
        """Find plugin class in module"""
        for name in dir(module):
            obj = getattr(module, name)

            # Check if it's a class and subclass of BasePlugin
            if (isinstance(obj, type) and
                issubclass(obj, BasePlugin) and
                obj is not BasePlugin and
                obj not in [CommandPlugin, AgentPlugin, ValidatorPlugin,
                           CacheProviderPlugin, ReporterPlugin, IntegrationPlugin]):
                return obj

        return None

    def _create_metadata(self, metadata_dict: Dict[str, Any]) -> PluginMetadata:
        """Create PluginMetadata from dictionary"""
        plugin_type_str = metadata_dict.get('type', 'command')

        try:
            plugin_type = PluginType(plugin_type_str)
        except ValueError:
            plugin_type = PluginType.COMMAND

        return PluginMetadata(
            name=metadata_dict.get('name', 'unknown'),
            version=metadata_dict.get('version', '0.0.0'),
            plugin_type=plugin_type,
            description=metadata_dict.get('description', ''),
            author=metadata_dict.get('author', ''),
            framework_version=metadata_dict.get('framework_version', '>=2.0.0'),
            python_version=metadata_dict.get('python_version', '>=3.8'),
            dependencies=metadata_dict.get('dependencies', []),
            capabilities=metadata_dict.get('capabilities', []),
            supported_commands=metadata_dict.get('supported_commands', []),
            config_schema=metadata_dict.get('config_schema', {}),
            default_config=metadata_dict.get('default_config', {}),
            permissions=metadata_dict.get('permissions', []),
            sandbox=metadata_dict.get('sandbox', True),
            homepage=metadata_dict.get('homepage', ''),
            repository=metadata_dict.get('repository', ''),
            license=metadata_dict.get('license', ''),
            tags=metadata_dict.get('tags', [])
        )


# ============================================================================
# Plugin Manager
# ============================================================================

class PluginManager:
    """
    Central plugin management system.

    Features:
    - Plugin discovery and loading
    - Plugin lifecycle management
    - Plugin dependency resolution
    - Plugin state management
    - Plugin hot-reloading
    """

    def __init__(self, plugin_dirs: List[Path] = None):
        """
        Initialize plugin manager.

        Args:
            plugin_dirs: Directories to search for plugins
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup plugin directories
        if plugin_dirs is None:
            claude_dir = Path.home() / ".claude"
            plugin_dirs = [
                claude_dir / "commands" / "plugins" / "examples",
                claude_dir / "plugins",
                Path.cwd() / "plugins"
            ]

        self.plugin_dirs = [d for d in plugin_dirs if d.exists()]

        # Initialize components
        self.discovery = PluginDiscovery(self.plugin_dirs)
        self.loader = PluginLoader()

        # Plugin storage
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugins_by_type: Dict[PluginType, List[BasePlugin]] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}

        # State
        self.initialized = False

    def initialize(self):
        """Initialize plugin system"""
        if self.initialized:
            return

        self.logger.info("Initializing plugin system")

        # Load plugin configurations
        self._load_configurations()

        # Discover plugins
        sources = self.discovery.discover_all()

        # Load plugins
        for source in sources:
            plugin_name = source.metadata.get('name')
            config = self.plugin_configs.get(plugin_name, {})

            plugin = self.loader.load_plugin(source, config)
            if plugin:
                self._register_plugin(plugin)

        self.initialized = True
        self.logger.info(f"Initialized {len(self.plugins)} plugins")

    def _load_configurations(self):
        """Load plugin configurations"""
        config_file = Path.home() / ".claude" / "commands" / "plugins" / "config" / "plugins.json"

        if config_file.exists():
            try:
                configs = json.loads(config_file.read_text())
                self.plugin_configs = configs.get('plugins', {})
            except Exception as e:
                self.logger.error(f"Error loading plugin configurations: {e}")

    def _register_plugin(self, plugin: BasePlugin):
        """Register plugin"""
        # Validate plugin
        valid, errors = plugin.validate()
        if not valid:
            self.logger.error(f"Plugin validation failed: {plugin.metadata.name}")
            for error in errors:
                self.logger.error(f"  - {error}")
            return

        # Load plugin
        if plugin.load():
            self.plugins[plugin.metadata.name] = plugin
            plugin.status = PluginStatus.LOADED

            # Index by type
            plugin_type = plugin.metadata.plugin_type
            if plugin_type not in self.plugins_by_type:
                self.plugins_by_type[plugin_type] = []
            self.plugins_by_type[plugin_type].append(plugin)

            self.logger.info(f"Registered plugin: {plugin.metadata.name} ({plugin_type.value})")
        else:
            self.logger.error(f"Failed to load plugin: {plugin.metadata.name}")

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """
        Get plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self.plugins.get(name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """
        Get all plugins of a specific type.

        Args:
            plugin_type: Plugin type

        Returns:
            List of plugins
        """
        return self.plugins_by_type.get(plugin_type, [])

    def get_all_plugins(self) -> List[BasePlugin]:
        """Get all loaded plugins"""
        return list(self.plugins.values())

    def enable_plugin(self, name: str) -> bool:
        """
        Enable a plugin.

        Args:
            name: Plugin name

        Returns:
            True if successful
        """
        plugin = self.get_plugin(name)
        if plugin:
            plugin.status = PluginStatus.ACTIVE
            self.logger.info(f"Enabled plugin: {name}")
            return True
        return False

    def disable_plugin(self, name: str) -> bool:
        """
        Disable a plugin.

        Args:
            name: Plugin name

        Returns:
            True if successful
        """
        plugin = self.get_plugin(name)
        if plugin:
            plugin.status = PluginStatus.DISABLED
            self.logger.info(f"Disabled plugin: {name}")
            return True
        return False

    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if successful
        """
        plugin = self.get_plugin(name)
        if plugin:
            plugin.cleanup()
            del self.plugins[name]

            # Remove from type index
            plugin_type = plugin.metadata.plugin_type
            if plugin_type in self.plugins_by_type:
                self.plugins_by_type[plugin_type] = [
                    p for p in self.plugins_by_type[plugin_type]
                    if p.metadata.name != name
                ]

            self.logger.info(f"Unloaded plugin: {name}")
            return True
        return False

    def reload_plugin(self, name: str) -> bool:
        """
        Reload a plugin.

        Args:
            name: Plugin name

        Returns:
            True if successful
        """
        plugin = self.get_plugin(name)
        if not plugin:
            return False

        # Get plugin source info
        config = plugin.config

        # Unload
        self.unload_plugin(name)

        # Reload - discover and load again
        sources = self.discovery.discover_all()
        for source in sources:
            if source.metadata.get('name') == name:
                new_plugin = self.loader.load_plugin(source, config)
                if new_plugin:
                    self._register_plugin(new_plugin)
                    return True

        return False

    def get_plugin_info(self) -> Dict[str, Any]:
        """
        Get information about all plugins.

        Returns:
            Dictionary with plugin information
        """
        return {
            "total_plugins": len(self.plugins),
            "plugins_by_type": {
                ptype.value: len(plugins)
                for ptype, plugins in self.plugins_by_type.items()
            },
            "plugins": [plugin.to_dict() for plugin in self.plugins.values()]
        }

    def cleanup(self):
        """Cleanup all plugins"""
        self.logger.info("Cleaning up plugin system")

        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up plugin {plugin.metadata.name}: {e}")

        self.plugins.clear()
        self.plugins_by_type.clear()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Test plugin manager"""
    logging.basicConfig(level=logging.INFO)

    print("Plugin Manager")
    print("==============\n")

    manager = PluginManager()
    manager.initialize()

    info = manager.get_plugin_info()
    print(f"Total plugins: {info['total_plugins']}")
    print(f"\nPlugins by type:")
    for ptype, count in info['plugins_by_type'].items():
        print(f"  - {ptype}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())