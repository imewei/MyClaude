"""
Shared MCP runtime for Option A deployment.
Uses lazy initialization - starts on first use.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from mcp_integration import (
    KnowledgeHierarchy,
    LibraryCache,
    MCPProfileManager,
    SmartTrigger,
    create_mcp_adapters,
)
from mcp_integration.learning_system import LearningSystem
from mcp_integration.predictive_preloader import PredictivePreloader
from mcp_integration.monitoring import Monitor


class SharedMCPRuntime:
    """
    Shared MCP runtime with lazy initialization.

    Usage:
        from mcp_shared_runtime import get_mcp_runtime

        runtime = await get_mcp_runtime(mcp_servers)
        result = await runtime.hierarchy.fetch(...)
    """

    _instance: Optional["SharedMCPRuntime"] = None
    _initialized: bool = False

    def __init__(self):
        self.hierarchy = None
        self.lib_cache = None
        self.profile_manager = None
        self.smart_trigger = None
        self.learner = None
        self.preloader = None
        self.monitor = None

    @classmethod
    async def get_instance(cls, mcp_servers: Dict[str, Any]) -> "SharedMCPRuntime":
        """
        Get or create the shared runtime instance.

        Args:
            mcp_servers: Dictionary of MCP server instances
                {
                    'memory_bank': memory_bank_mcp,
                    'serena': serena_mcp,
                    'context7': context7_mcp,
                    'github': github_mcp (optional),
                }

        Returns:
            Initialized SharedMCPRuntime instance
        """
        if cls._instance is None:
            cls._instance = cls()

        if not cls._initialized:
            await cls._instance._initialize(mcp_servers)
            cls._initialized = True

        return cls._instance

    async def _initialize(self, mcp_servers: Dict[str, Any]):
        """Initialize all MCP integration components."""
        print("[MCP Runtime] Initializing (lazy init on first use)...")

        config_dir = Path.home() / ".claude" / "commands"

        # Create adapters
        adapters = await create_mcp_adapters(
            memory_bank_mcp=mcp_servers.get("memory_bank"),
            serena_mcp=mcp_servers.get("serena"),
            context7_mcp=mcp_servers.get("context7"),
            github_mcp=mcp_servers.get("github"),
            playwright_mcp=mcp_servers.get("playwright"),
            sequential_thinking_mcp=mcp_servers.get("sequential_thinking"),
        )

        # Initialize components
        self.hierarchy = await KnowledgeHierarchy.create(
            memory_bank=adapters.get("memory-bank"),
            serena=adapters.get("serena"),
            context7=adapters.get("context7"),
            github=adapters.get("github"),
        )

        self.lib_cache = await LibraryCache.create(
            str(config_dir / "library-cache.yaml"),
            context7_mcp=adapters.get("context7"),
        )

        self.profile_manager = await MCPProfileManager.create(
            str(config_dir / "mcp-profiles.yaml")
        )

        self.smart_trigger = await SmartTrigger.create(
            str(config_dir / "mcp-config.yaml")
        )

        self.learner = await LearningSystem.create(
            memory_bank=adapters.get("memory-bank")
        )

        self.preloader = await PredictivePreloader.create(
            profile_manager=self.profile_manager, learning_system=self.learner
        )

        self.monitor = await Monitor.create()

        # Start background tasks
        await self.preloader.start_background_preloading()

        print("[MCP Runtime] ✓ Initialized")


# Convenience function
async def get_mcp_runtime(mcp_servers: Dict[str, Any]) -> SharedMCPRuntime:
    """
    Get the shared MCP runtime (initializes on first call).

    Usage:
        runtime = await get_mcp_runtime({
            'memory_bank': memory_bank_mcp,
            'serena': serena_mcp,
            'context7': context7_mcp,
        })

        result = await runtime.hierarchy.fetch(...)
    """
    return await SharedMCPRuntime.get_instance(mcp_servers)
