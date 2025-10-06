"""
MCP Profile Manager

Manages MCP server initialization and lifecycle based on profiles.
Supports parallel loading, priority-based initialization, and profile-to-command mapping.

Example:
    >>> manager = await MCPProfileManager.create("mcp-profiles.yaml")
    >>>
    >>> # Load profile for a command
    >>> profile = await manager.activate_profile("code-analysis")
    >>> # Returns: MCPProfile with serena (critical) + memory-bank (medium)
    >>>
    >>> # Get initialized MCPs
    >>> mcps = manager.get_active_mcps()
    >>> # Returns: {'serena': <MCP>, 'memory-bank': <MCP>}
"""

import asyncio
import yaml
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class MCPPriority(Enum):
    """MCP priority levels."""
    CRITICAL = 5  # Must load, block if fails
    HIGH = 4      # Important, warn if fails
    MEDIUM = 3    # Optional, silent fail ok
    LOW = 2       # Best effort
    OPTIONAL = 1  # Load only if available


@dataclass
class MCPConfig:
    """
    Configuration for a single MCP server.

    Attributes:
        name: MCP server name (e.g., 'serena', 'context7')
        priority: Loading priority
        preload: Load immediately when profile activates
        operations: Allowed operations (read, write, etc.)
        config: MCP-specific configuration
        timeout_ms: Initialization timeout
    """
    name: str
    priority: MCPPriority
    preload: bool = False
    operations: List[str] = field(default_factory=lambda: ["read"])
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 5000

    def __post_init__(self):
        """Ensure priority is an MCPPriority enum."""
        if isinstance(self.priority, str):
            priority_map = {
                "critical": MCPPriority.CRITICAL,
                "high": MCPPriority.HIGH,
                "medium": MCPPriority.MEDIUM,
                "medium-high": MCPPriority.HIGH,
                "medium-low": MCPPriority.MEDIUM,
                "low": MCPPriority.LOW,
                "optional": MCPPriority.OPTIONAL,
            }
            self.priority = priority_map.get(self.priority.lower(), MCPPriority.MEDIUM)


@dataclass
class MCPProfile:
    """
    MCP profile for a command or workflow.

    Attributes:
        name: Profile name (e.g., 'code-analysis', 'meta-reasoning')
        mcps: List of MCP configurations
        commands: Commands that use this profile
        parallel_init: Initialize MCPs in parallel
        orchestrated: Use orchestrated multi-agent flow
        description: Profile description
    """
    name: str
    mcps: List[MCPConfig]
    commands: List[str] = field(default_factory=list)
    parallel_init: bool = True
    orchestrated: bool = False
    description: str = ""

    def get_critical_mcps(self) -> List[MCPConfig]:
        """Get MCPs with critical priority."""
        return [mcp for mcp in self.mcps if mcp.priority == MCPPriority.CRITICAL]

    def get_preload_mcps(self) -> List[MCPConfig]:
        """Get MCPs marked for preloading."""
        return [mcp for mcp in self.mcps if mcp.preload]

    def sorted_by_priority(self) -> List[MCPConfig]:
        """Get MCPs sorted by priority (highest first)."""
        return sorted(self.mcps, key=lambda m: m.priority.value, reverse=True)


class MCPProfileManager:
    """
    Manages MCP profiles and server lifecycle.

    Features:
    - Profile-based MCP initialization
    - Parallel loading for performance
    - Priority-based failure handling
    - Command-to-profile mapping
    - MCP sharing across commands
    - Statistics tracking
    """

    def __init__(
        self,
        profiles: Dict[str, MCPProfile],
        mcp_factory: Optional[Callable] = None,
        enable_parallel: bool = True,
    ):
        """
        Initialize profile manager.

        Args:
            profiles: Dictionary of profile name → MCPProfile
            mcp_factory: Factory function to create MCP instances
            enable_parallel: Enable parallel MCP loading
        """
        self.profiles = profiles
        self.mcp_factory = mcp_factory
        self.enable_parallel = enable_parallel

        # Active MCPs (shared across profiles)
        self._active_mcps: Dict[str, Any] = {}

        # Current profile
        self._current_profile: Optional[MCPProfile] = None

        # Command to profile mapping
        self._command_map: Dict[str, str] = {}
        for profile_name, profile in profiles.items():
            for command in profile.commands:
                self._command_map[command] = profile_name

        # Statistics
        self.stats = {
            "profiles_activated": 0,
            "mcps_loaded": 0,
            "load_failures": 0,
            "parallel_loads": 0,
            "total_load_time_ms": 0,
        }

    @classmethod
    async def create(
        cls,
        config_path: str,
        mcp_factory: Optional[Callable] = None,
        **kwargs
    ) -> "MCPProfileManager":
        """
        Create profile manager from YAML configuration.

        Args:
            config_path: Path to mcp-profiles.yaml
            mcp_factory: Factory function for MCP instances
            **kwargs: Additional configuration

        Returns:
            Initialized MCPProfileManager instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load profiles
        profiles = {}
        for profile_name, profile_config in config.get('profiles', {}).items():
            # Parse MCPs
            mcps = []
            for mcp_config in profile_config.get('mcps', []):
                # Handle both dict and string formats
                if isinstance(mcp_config, str):
                    # Format: "name (priority)"
                    parts = mcp_config.split('(')
                    name = parts[0].strip()
                    priority = parts[1].rstrip(')').strip() if len(parts) > 1 else "medium"
                    mcps.append(MCPConfig(name=name, priority=priority))
                else:
                    # Dict format
                    mcps.append(MCPConfig(
                        name=mcp_config['name'],
                        priority=mcp_config.get('priority', 'medium'),
                        preload=mcp_config.get('preload', False),
                        operations=mcp_config.get('operations', ['read']),
                        config=mcp_config.get('config', {}),
                        timeout_ms=mcp_config.get('timeout_ms', 5000),
                    ))

            profiles[profile_name] = MCPProfile(
                name=profile_name,
                mcps=mcps,
                commands=profile_config.get('commands', []),
                parallel_init=profile_config.get('parallel_init', True),
                orchestrated=profile_config.get('orchestrated', False),
                description=profile_config.get('description', ''),
            )

        return cls(
            profiles=profiles,
            mcp_factory=mcp_factory,
            **kwargs
        )

    async def activate_profile(
        self,
        profile_name: str,
        force_reload: bool = False
    ) -> Optional[MCPProfile]:
        """
        Activate a profile and load its MCPs.

        Args:
            profile_name: Name of profile to activate
            force_reload: Force reload even if MCPs already active

        Returns:
            Activated MCPProfile or None if not found

        Example:
            >>> profile = await manager.activate_profile("code-analysis")
            >>> # Loads: serena (critical), memory-bank (medium)
        """
        if profile_name not in self.profiles:
            return None

        profile = self.profiles[profile_name]
        self._current_profile = profile
        self.stats["profiles_activated"] += 1

        # Load MCPs
        if profile.parallel_init and self.enable_parallel:
            await self._load_mcps_parallel(profile, force_reload)
        else:
            await self._load_mcps_sequential(profile, force_reload)

        return profile

    async def activate_for_command(
        self,
        command: str,
        force_reload: bool = False
    ) -> Optional[MCPProfile]:
        """
        Activate profile for a specific command.

        Args:
            command: Command name (e.g., 'fix', 'quality')
            force_reload: Force reload MCPs

        Returns:
            Activated profile or None

        Example:
            >>> profile = await manager.activate_for_command("fix")
            >>> # Activates: code-analysis profile
        """
        profile_name = self._command_map.get(command)
        if not profile_name:
            return None

        return await self.activate_profile(profile_name, force_reload)

    async def _load_mcps_parallel(
        self,
        profile: MCPProfile,
        force_reload: bool
    ) -> None:
        """
        Load MCPs in parallel.

        Args:
            profile: Profile to load
            force_reload: Force reload
        """
        import time
        start_time = time.time()

        # Get MCPs to load
        mcps_to_load = [
            mcp for mcp in profile.mcps
            if force_reload or mcp.name not in self._active_mcps
        ]

        if not mcps_to_load:
            return

        # Load in parallel
        tasks = [
            self._load_single_mcp(mcp_config)
            for mcp_config in mcps_to_load
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update statistics
        load_time = int((time.time() - start_time) * 1000)
        self.stats["parallel_loads"] += 1
        self.stats["total_load_time_ms"] += load_time

        # Handle results
        for mcp_config, result in zip(mcps_to_load, results):
            if isinstance(result, Exception):
                self._handle_load_failure(mcp_config, result)
            elif result is not None:
                self._active_mcps[mcp_config.name] = result
                self.stats["mcps_loaded"] += 1

    async def _load_mcps_sequential(
        self,
        profile: MCPProfile,
        force_reload: bool
    ) -> None:
        """
        Load MCPs sequentially (priority order).

        Args:
            profile: Profile to load
            force_reload: Force reload
        """
        import time
        start_time = time.time()

        # Load by priority (highest first)
        for mcp_config in profile.sorted_by_priority():
            if not force_reload and mcp_config.name in self._active_mcps:
                continue

            try:
                mcp_instance = await self._load_single_mcp(mcp_config)
                if mcp_instance:
                    self._active_mcps[mcp_config.name] = mcp_instance
                    self.stats["mcps_loaded"] += 1
            except Exception as e:
                self._handle_load_failure(mcp_config, e)

        load_time = int((time.time() - start_time) * 1000)
        self.stats["total_load_time_ms"] += load_time

    async def _load_single_mcp(
        self,
        mcp_config: MCPConfig
    ) -> Optional[Any]:
        """
        Load a single MCP server.

        Args:
            mcp_config: MCP configuration

        Returns:
            MCP instance or None
        """
        if self.mcp_factory is None:
            # No factory provided, return mock
            return {"name": mcp_config.name, "config": mcp_config.config}

        try:
            # Use factory to create MCP instance
            mcp_instance = await asyncio.wait_for(
                self.mcp_factory(mcp_config),
                timeout=mcp_config.timeout_ms / 1000.0
            )
            return mcp_instance
        except asyncio.TimeoutError:
            raise Exception(f"MCP '{mcp_config.name}' load timeout after {mcp_config.timeout_ms}ms")
        except Exception as e:
            raise Exception(f"MCP '{mcp_config.name}' load failed: {e}")

    def _handle_load_failure(
        self,
        mcp_config: MCPConfig,
        error: Exception
    ) -> None:
        """
        Handle MCP load failure based on priority.

        Args:
            mcp_config: Failed MCP configuration
            error: Error that occurred
        """
        self.stats["load_failures"] += 1

        if mcp_config.priority == MCPPriority.CRITICAL:
            # Critical MCP - raise error
            raise Exception(f"Critical MCP '{mcp_config.name}' failed to load: {error}")
        elif mcp_config.priority == MCPPriority.HIGH:
            # High priority - warn
            print(f"Warning: High-priority MCP '{mcp_config.name}' failed to load: {error}")
        else:
            # Medium/Low/Optional - silent fail
            pass

    def get_active_mcps(self) -> Dict[str, Any]:
        """
        Get currently active MCP instances.

        Returns:
            Dictionary of MCP name → instance
        """
        return self._active_mcps.copy()

    def get_mcp(self, name: str) -> Optional[Any]:
        """
        Get specific MCP instance.

        Args:
            name: MCP name

        Returns:
            MCP instance or None
        """
        return self._active_mcps.get(name)

    def get_current_profile(self) -> Optional[MCPProfile]:
        """Get currently active profile."""
        return self._current_profile

    def get_profile_for_command(self, command: str) -> Optional[str]:
        """
        Get profile name for a command.

        Args:
            command: Command name

        Returns:
            Profile name or None
        """
        return self._command_map.get(command)

    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self.profiles.keys())

    async def preload_mcps(
        self,
        profile_name: str
    ) -> int:
        """
        Preload MCPs marked for preloading in a profile.

        Args:
            profile_name: Profile name

        Returns:
            Number of MCPs preloaded

        Example:
            >>> count = await manager.preload_mcps("code-analysis")
            >>> # Preloads: serena (if marked preload: true)
        """
        if profile_name not in self.profiles:
            return 0

        profile = self.profiles[profile_name]
        preload_mcps = profile.get_preload_mcps()

        if not preload_mcps:
            return 0

        # Load preload MCPs
        tasks = [
            self._load_single_mcp(mcp_config)
            for mcp_config in preload_mcps
            if mcp_config.name not in self._active_mcps
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        loaded_count = 0
        for mcp_config, result in zip(preload_mcps, results):
            if not isinstance(result, Exception) and result is not None:
                self._active_mcps[mcp_config.name] = result
                loaded_count += 1
                self.stats["mcps_loaded"] += 1

        return loaded_count

    async def unload_mcp(self, name: str) -> bool:
        """
        Unload a specific MCP.

        Args:
            name: MCP name

        Returns:
            True if unloaded successfully
        """
        if name not in self._active_mcps:
            return False

        # In production, call cleanup/shutdown on MCP
        # if hasattr(self._active_mcps[name], 'shutdown'):
        #     await self._active_mcps[name].shutdown()

        del self._active_mcps[name]
        return True

    async def unload_all_mcps(self) -> int:
        """
        Unload all active MCPs.

        Returns:
            Number of MCPs unloaded
        """
        count = len(self._active_mcps)

        # Unload all
        mcp_names = list(self._active_mcps.keys())
        for name in mcp_names:
            await self.unload_mcp(name)

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get manager statistics.

        Returns:
            Statistics dictionary

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Avg load time: {stats['avg_load_time_ms']}ms")
        """
        profiles_activated = self.stats["profiles_activated"]
        avg_load_time = (
            self.stats["total_load_time_ms"] / profiles_activated
            if profiles_activated > 0 else 0
        )

        return {
            **self.stats,
            "active_mcps": len(self._active_mcps),
            "total_profiles": len(self.profiles),
            "avg_load_time_ms": avg_load_time,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "profiles_activated": 0,
            "mcps_loaded": 0,
            "load_failures": 0,
            "parallel_loads": 0,
            "total_load_time_ms": 0,
        }


# Convenience function
async def load_profile(
    profile_name: str,
    config_path: str = "mcp-profiles.yaml",
    mcp_factory: Optional[Callable] = None
) -> Optional[MCPProfile]:
    """
    Convenience function to load a profile.

    Args:
        profile_name: Profile to load
        config_path: Path to profiles config
        mcp_factory: MCP factory function

    Returns:
        Loaded profile or None
    """
    manager = await MCPProfileManager.create(config_path, mcp_factory)
    return await manager.activate_profile(profile_name)
