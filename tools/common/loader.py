"""
Unified plugin.json loader with caching.

Consolidates duplicate plugin loading logic from:
- metadata_validator.py
- plugin_review_script.py
- validate_plugins.py
- xref_validator.py
- dependency_mapper.py
- load_profiler.py
- activation_profiler.py
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from tools.common.models import PluginMetadata, ValidationIssue

# YAML frontmatter delimiter pattern
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _read_frontmatter(file_path: Path) -> dict[str, Any]:
    """Extract YAML frontmatter from a markdown file.

    Returns an empty dict if the file is missing, has no frontmatter,
    or the frontmatter cannot be parsed.
    """
    if not file_path.exists():
        return {}
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_component_entry(
    entry: Any, plugin_path: Path, kind: str
) -> dict[str, Any]:
    """Normalize an agents/commands/skills entry to a dict.

    plugin.json entries can be either:
      - A file/directory path string (e.g. "./agents/foo.md", "./skills/foo")
      - An inline dict with name/description fields

    For path strings we read the target's YAML frontmatter and merge in
    {name, description, path}. For inline dicts we pass them through and
    just attach a path field if missing.
    """
    if isinstance(entry, dict):
        # Already an inline dict; pass through.
        return entry

    if not isinstance(entry, str):
        # Unknown shape — return a placeholder so callers don't crash.
        return {"name": "", "description": "", "path": ""}

    # Resolve relative path against the plugin directory.
    rel_path = entry.lstrip("./")
    target = (plugin_path / rel_path).resolve()

    # Skills point to directories containing SKILL.md; agents/commands
    # point directly to a markdown file.
    if kind == "skills" and target.is_dir():
        frontmatter_path = target / "SKILL.md"
    else:
        frontmatter_path = target

    fm = _read_frontmatter(frontmatter_path)

    name = fm.get("name") or target.stem
    description = fm.get("description", "")

    return {
        "name": name,
        "description": description,
        "path": str(target),
    }


def _normalize_component_list(
    entries: list[Any], plugin_path: Path, kind: str
) -> list[dict[str, Any]]:
    """Normalize a list of agents/commands/skills entries to dicts."""
    return [_normalize_component_entry(e, plugin_path, kind) for e in entries]


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
        plugin_json_path = plugin_path / ".claude-plugin" / "plugin.json"

        if not plugin_path.exists():
            self._add_error(
                plugin_name, "path", f"Plugin directory not found: {plugin_path}"
            )
            return None

        if not plugin_json_path.exists():
            self._add_error(
                plugin_name, "plugin.json", f"plugin.json not found: {plugin_json_path}"
            )
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
                agents=_normalize_component_list(
                    data.get("agents", []), plugin_path, "agents"
                ),
                commands=_normalize_component_list(
                    data.get("commands", []), plugin_path, "commands"
                ),
                skills=_normalize_component_list(
                    data.get("skills", []), plugin_path, "skills"
                ),
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
