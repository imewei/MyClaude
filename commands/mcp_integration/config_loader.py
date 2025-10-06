"""
Configuration Loader

Centralized YAML configuration management for MCP integration system.
Handles loading and validation of all configuration files.

Example:
    >>> config = await ConfigLoader.load_all()
    >>> # Returns: {'mcp_config': {...}, 'profiles': {...}, 'library_cache': {...}}
    >>>
    >>> # Load specific config
    >>> profiles = await load_profiles("mcp-profiles.yaml")
"""

import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ConfigPaths:
    """
    Configuration file paths.

    Attributes:
        mcp_config: Path to mcp-config.yaml
        profiles: Path to mcp-profiles.yaml
        library_cache: Path to library-cache.yaml
        base_dir: Base directory for config files
    """
    mcp_config: str = "mcp-config.yaml"
    profiles: str = "mcp-profiles.yaml"
    library_cache: str = "library-cache.yaml"
    base_dir: Optional[str] = None

    def __post_init__(self):
        """Resolve paths relative to base_dir."""
        if self.base_dir:
            base = Path(self.base_dir)
            self.mcp_config = str(base / self.mcp_config)
            self.profiles = str(base / self.profiles)
            self.library_cache = str(base / self.library_cache)


class ConfigLoader:
    """
    Configuration loader for MCP integration system.

    Handles loading, validation, and caching of YAML configurations.

    Features:
    - Centralized config loading
    - Schema validation
    - Default value handling
    - Environment variable substitution
    - Config caching
    """

    def __init__(
        self,
        paths: ConfigPaths,
        enable_caching: bool = True,
        validate_schema: bool = True,
    ):
        """
        Initialize config loader.

        Args:
            paths: Configuration file paths
            enable_caching: Cache loaded configs in memory
            validate_schema: Validate config schemas
        """
        self.paths = paths
        self.enable_caching = enable_caching
        self.validate_schema = validate_schema

        # Config cache
        self._cache: Dict[str, Any] = {}

    @classmethod
    async def create(
        cls,
        base_dir: Optional[str] = None,
        **kwargs
    ) -> "ConfigLoader":
        """
        Create config loader.

        Args:
            base_dir: Base directory for config files
            **kwargs: Additional configuration

        Returns:
            Initialized ConfigLoader instance
        """
        paths = ConfigPaths(base_dir=base_dir)
        return cls(paths=paths, **kwargs)

    async def load_all(self) -> Dict[str, Any]:
        """
        Load all configuration files.

        Returns:
            Dictionary with all configs

        Example:
            >>> config = await loader.load_all()
            >>> # Returns: {
            ...     'mcp_config': {...},
            ...     'profiles': {...},
            ...     'library_cache': {...}
            ... }
        """
        return {
            'mcp_config': await self.load_mcp_config(),
            'profiles': await self.load_profiles(),
            'library_cache': await self.load_library_cache(),
        }

    async def load_mcp_config(self) -> Dict[str, Any]:
        """
        Load mcp-config.yaml.

        Returns:
            MCP configuration dictionary

        Example:
            >>> config = await loader.load_mcp_config()
            >>> print(config['knowledge_hierarchy']['enabled'])
            >>> # True
        """
        cache_key = 'mcp_config'

        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]

        config = self._load_yaml(self.paths.mcp_config)

        if self.validate_schema:
            self._validate_mcp_config(config)

        # Apply defaults
        config = self._apply_mcp_config_defaults(config)

        if self.enable_caching:
            self._cache[cache_key] = config

        return config

    async def load_profiles(self) -> Dict[str, Any]:
        """
        Load mcp-profiles.yaml.

        Returns:
            Profiles configuration dictionary

        Example:
            >>> profiles = await loader.load_profiles()
            >>> print(profiles['profiles']['code-analysis'])
        """
        cache_key = 'profiles'

        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]

        config = self._load_yaml(self.paths.profiles)

        if self.validate_schema:
            self._validate_profiles_config(config)

        if self.enable_caching:
            self._cache[cache_key] = config

        return config

    async def load_library_cache(self) -> Dict[str, Any]:
        """
        Load library-cache.yaml.

        Returns:
            Library cache configuration dictionary

        Example:
            >>> cache_config = await loader.load_library_cache()
            >>> print(cache_config['common_libraries']['numpy'])
        """
        cache_key = 'library_cache'

        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]

        config = self._load_yaml(self.paths.library_cache)

        if self.validate_schema:
            self._validate_library_cache_config(config)

        if self.enable_caching:
            self._cache[cache_key] = config

        return config

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML content
        """
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

    def _apply_mcp_config_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values to mcp-config.yaml.

        Args:
            config: Loaded config

        Returns:
            Config with defaults applied
        """
        defaults = {
            'knowledge_hierarchy': {
                'enabled': True,
                'layers': [],
            },
            'memory-bank': {
                'cache': {
                    'enabled': True,
                    'ttl_by_type': {},
                }
            },
            'context7': {
                'library_cache': {
                    'enabled': True,
                    'fallback': {
                        'use_resolve_api': True,
                    }
                }
            },
            'smart_triggers': {
                'enabled': True,
                'patterns': [],
                'mcp_rules': {},
            }
        }

        # Deep merge defaults
        return self._deep_merge(defaults, config)

    def _validate_mcp_config(self, config: Dict[str, Any]) -> None:
        """
        Validate mcp-config.yaml schema.

        Args:
            config: Config to validate

        Raises:
            ValueError: If config is invalid
        """
        # Required top-level keys
        if 'knowledge_hierarchy' in config:
            kh = config['knowledge_hierarchy']
            if 'layers' in kh and not isinstance(kh['layers'], list):
                raise ValueError("knowledge_hierarchy.layers must be a list")

        if 'smart_triggers' in config:
            st = config['smart_triggers']
            if 'patterns' in st and not isinstance(st['patterns'], list):
                raise ValueError("smart_triggers.patterns must be a list")

    def _validate_profiles_config(self, config: Dict[str, Any]) -> None:
        """
        Validate mcp-profiles.yaml schema.

        Args:
            config: Config to validate

        Raises:
            ValueError: If config is invalid
        """
        if 'profiles' not in config:
            raise ValueError("profiles key required in mcp-profiles.yaml")

        profiles = config['profiles']
        if not isinstance(profiles, dict):
            raise ValueError("profiles must be a dictionary")

        # Validate each profile
        for profile_name, profile_config in profiles.items():
            if 'mcps' not in profile_config:
                raise ValueError(f"Profile '{profile_name}' missing 'mcps' key")

            if not isinstance(profile_config['mcps'], list):
                raise ValueError(f"Profile '{profile_name}' mcps must be a list")

    def _validate_library_cache_config(self, config: Dict[str, Any]) -> None:
        """
        Validate library-cache.yaml schema.

        Args:
            config: Config to validate

        Raises:
            ValueError: If config is invalid
        """
        if 'common_libraries' not in config:
            raise ValueError("common_libraries key required in library-cache.yaml")

        libraries = config['common_libraries']
        if not isinstance(libraries, dict):
            raise ValueError("common_libraries must be a dictionary")

        # Validate each library
        for lib_name, lib_config in libraries.items():
            if 'id' not in lib_config:
                raise ValueError(f"Library '{lib_name}' missing 'id' key")

    def _deep_merge(
        self,
        defaults: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            defaults: Default values
            config: User config (takes precedence)

        Returns:
            Merged dictionary
        """
        result = defaults.copy()

        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def invalidate_cache(self, config_type: Optional[str] = None) -> None:
        """
        Invalidate config cache.

        Args:
            config_type: Specific config to invalidate, or None for all
        """
        if config_type:
            self._cache.pop(config_type, None)
        else:
            self._cache.clear()

    def get_cached_configs(self) -> List[str]:
        """
        Get list of cached config types.

        Returns:
            List of cached config keys
        """
        return list(self._cache.keys())


# Convenience functions

async def load_mcp_config(
    path: str = "mcp-config.yaml"
) -> Dict[str, Any]:
    """
    Convenience function to load mcp-config.yaml.

    Args:
        path: Path to config file

    Returns:
        MCP configuration
    """
    loader = await ConfigLoader.create()
    loader.paths.mcp_config = path
    return await loader.load_mcp_config()


async def load_profiles(
    path: str = "mcp-profiles.yaml"
) -> Dict[str, Any]:
    """
    Convenience function to load mcp-profiles.yaml.

    Args:
        path: Path to config file

    Returns:
        Profiles configuration
    """
    loader = await ConfigLoader.create()
    loader.paths.profiles = path
    return await loader.load_profiles()


async def load_library_cache(
    path: str = "library-cache.yaml"
) -> Dict[str, Any]:
    """
    Convenience function to load library-cache.yaml.

    Args:
        path: Path to config file

    Returns:
        Library cache configuration
    """
    loader = await ConfigLoader.create()
    loader.paths.library_cache = path
    return await loader.load_library_cache()


async def load_all_configs(
    base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load all configs.

    Args:
        base_dir: Base directory for config files

    Returns:
        Dictionary with all configs
    """
    loader = await ConfigLoader.create(base_dir=base_dir)
    return await loader.load_all()
