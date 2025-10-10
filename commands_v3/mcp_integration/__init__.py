"""
Claude Code MCP Integration System

A comprehensive system for optimizing MCP (Model Context Protocol) server usage
through intelligent caching, hierarchical knowledge retrieval, and profile-based
configuration.

Components:
    - KnowledgeHierarchy: Three-tier knowledge retrieval system
    - MCPProfileManager: Profile-based MCP configuration
    - LibraryCache: Pre-cached library IDs
    - SmartTrigger: Pattern-based MCP activation
    - ConfigLoader: YAML configuration management

Example:
    >>> from mcp_integration import KnowledgeHierarchy, LibraryCache
    >>>
    >>> # Initialize components
    >>> knowledge = await KnowledgeHierarchy.create(
    ...     memory_bank=memory_bank_mcp,
    ...     serena=serena_mcp,
    ...     context7=context7_mcp
    ... )
    >>>
    >>> # Fetch knowledge with hierarchy
    >>> result = await knowledge.fetch("numpy.array", context_type="library_api")
    >>> print(f"Source: {result.source}, Latency: {result.latency_ms}ms")
"""

__version__ = "1.0.0"
__author__ = "Claude Code Team"

from .knowledge_hierarchy import (
    KnowledgeHierarchy,
    Knowledge,
    KnowledgeSource,
    AuthorityRule,
)
from .library_cache import (
    LibraryCache,
    LibraryInfo,
    DetectionPattern,
)
from .profile_manager import (
    MCPProfileManager,
    MCPProfile,
    MCPConfig,
)
from .smart_trigger import (
    SmartTrigger,
    TriggerResult,
)
from .config_loader import (
    ConfigLoader,
    load_mcp_config,
    load_profiles,
    load_library_cache,
)
from .cache_backend import (
    CacheBackend,
    MemoryCacheBackend,
    FileCacheBackend,
)
from .mcp_adapter import (
    MCPAdapter,
    create_mcp_adapters,
)

__all__ = [
    # Knowledge Hierarchy
    "KnowledgeHierarchy",
    "Knowledge",
    "KnowledgeSource",
    "AuthorityRule",

    # Library Cache
    "LibraryCache",
    "LibraryInfo",
    "DetectionPattern",

    # Profile Manager
    "MCPProfileManager",
    "MCPProfile",
    "MCPConfig",

    # Smart Trigger
    "SmartTrigger",
    "TriggerResult",

    # Config Loader
    "ConfigLoader",
    "load_mcp_config",
    "load_profiles",
    "load_library_cache",

    # Cache Backend
    "CacheBackend",
    "MemoryCacheBackend",
    "FileCacheBackend",

    # MCP Adapter
    "MCPAdapter",
    "create_mcp_adapters",
]
