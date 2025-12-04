"""
Unified plugin.json loader with caching.

Consolidates duplicate plugin loading logic from:
- metadata-validator.py
- plugin-review-script.py
- validate_plugins.py
- xref-validator.py
- dependency-mapper.py
- load-profiler.py
- activation-profiler.py
"""

import json
from pathlib import Path
from typing import Optional

from tools.common.models import PluginMetadata, ValidationIssue


class PluginLoader:
    """Loads and caches plugin metadata from plugin.json files.

    Usage:
        loader = PluginLoader(Path("plugins"))

        # Load single plugin
        metadata = loader.load_plugin("julia-development")

        # Load all plugins
        all_plugins = loader.load_all_plugins()

        # Get cached plugin
        cached = loader.get_plugin("julia-development")
    """

    def __init__(self, plugins_dir: Path) -> None:
        self.plugins_dir = plugins_dir
        self._cache: dict[str, PluginMetadata] = {}
        self._load_errors: dict[str, list[ValidationIssue]] = {}

    def load_plugin(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Load a single plugin's metadata.

        Returns None if plugin doesn't exist or has invalid JSON.
        Errors are stored in _load_errors[plugin_name].
        """
        if plugin_name in self._cache:
            return self._cache[plugin_name]

        plugin_path = self.plugins_dir / plugin_name
        plugin_json_path = plugin_path / "plugin.json"

        if not plugin_path.exists():
            self._add_error(plugin_name, "path", f"Plugin directory not found: {plugin_path}")
            return None

        if not plugin_json_path.exists():
            self._add_error(plugin_name, "plugin.json", f"plugin.json not found: {plugin_json_path}")
            return None

        try:
            with open(plugin_json_path, encoding="utf-8") as f:
                data = json.load(f)

            metadata = PluginMetadata(
                name=data.get("name", plugin_name),
                version=data.get("version", "unknown"),
                description=data.get("description", ""),
                category=data.get("category", "uncategorized"),
                path=plugin_path,
                agents=data.get("agents", []),
                commands=data.get("commands", []),
                skills=data.get("skills", []),
                keywords=data.get("keywords", []),
            )

            self._cache[plugin_name] = metadata
            return metadata

        except json.JSONDecodeError as e:
            self._add_error(
                plugin_name,
                "plugin.json",
                f"Invalid JSON: {e.msg} at line {e.lineno}",
            )
            return None
        except OSError as e:
            self._add_error(plugin_name, "plugin.json", f"Error reading file: {e}")
            return None

    def load_all_plugins(self) -> dict[str, PluginMetadata]:
        """Load all plugins from the plugins directory.

        Returns dict mapping plugin_name -> PluginMetadata.
        Skips hidden directories and non-directories.
        """
        if not self.plugins_dir.exists():
            return {}

        for entry in sorted(self.plugins_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                self.load_plugin(entry.name)

        return dict(self._cache)

    def get_plugin(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin from cache, or load it if not cached."""
        if plugin_name in self._cache:
            return self._cache[plugin_name]
        return self.load_plugin(plugin_name)

    def get_all_cached(self) -> dict[str, PluginMetadata]:
        """Get all currently cached plugins."""
        return dict(self._cache)

    def get_errors(self, plugin_name: Optional[str] = None) -> list[ValidationIssue]:
        """Get load errors for a plugin, or all errors if no name specified."""
        if plugin_name:
            return self._load_errors.get(plugin_name, [])
        all_errors: list[ValidationIssue] = []
        for errors in self._load_errors.values():
            all_errors.extend(errors)
        return all_errors

    def clear_cache(self) -> None:
        """Clear the plugin cache."""
        self._cache.clear()
        self._load_errors.clear()

    def _add_error(self, plugin_name: str, field: str, message: str) -> None:
        """Add a load error for a plugin."""
        if plugin_name not in self._load_errors:
            self._load_errors[plugin_name] = []
        self._load_errors[plugin_name].append(
            ValidationIssue(field=field, severity="error", message=message)
        )

    # Convenience methods for common queries

    def get_plugin_names(self) -> list[str]:
        """Get list of all loaded plugin names."""
        return list(self._cache.keys())

    def get_plugins_by_category(self) -> dict[str, list[PluginMetadata]]:
        """Group plugins by category."""
        by_category: dict[str, list[PluginMetadata]] = {}
        for metadata in self._cache.values():
            category = metadata.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(metadata)
        return by_category

    def find_agent(self, agent_name: str) -> Optional[tuple[str, dict]]:
        """Find which plugin contains an agent by name.

        Returns (plugin_name, agent_dict) or None if not found.
        """
        for plugin_name, metadata in self._cache.items():
            for agent in metadata.agents:
                if agent.get("name") == agent_name:
                    return (plugin_name, agent)
        return None

    def find_command(self, command_name: str) -> Optional[tuple[str, dict]]:
        """Find which plugin contains a command by name."""
        for plugin_name, metadata in self._cache.items():
            for command in metadata.commands:
                if command.get("name") == command_name:
                    return (plugin_name, command)
        return None

    def find_skill(self, skill_name: str) -> Optional[tuple[str, dict]]:
        """Find which plugin contains a skill by name."""
        for plugin_name, metadata in self._cache.items():
            for skill in metadata.skills:
                if skill.get("name") == skill_name:
                    return (plugin_name, skill)
        return None

    def get_total_counts(self) -> dict[str, int]:
        """Get total counts across all loaded plugins."""
        return {
            "plugins": len(self._cache),
            "agents": sum(m.agent_count for m in self._cache.values()),
            "commands": sum(m.command_count for m in self._cache.values()),
            "skills": sum(m.skill_count for m in self._cache.values()),
        }
