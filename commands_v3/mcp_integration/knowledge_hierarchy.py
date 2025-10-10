"""
Knowledge Hierarchy System

Implements three-tier knowledge retrieval with authority rules and caching:
1. memory-bank (fastest, cached)
2. serena (local codebase)
3. context7 (external libraries)

Example:
    >>> hierarchy = await KnowledgeHierarchy.create(
    ...     memory_bank=memory_bank_mcp,
    ...     serena=serena_mcp,
    ...     context7=context7_mcp
    ... )
    >>>
    >>> result = await hierarchy.fetch("numpy.array", context_type="library_api")
    >>> print(f"Found in {result.source} ({result.latency_ms}ms)")
"""

import time
import hashlib
from typing import Optional, Dict, Any, List, Protocol
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class KnowledgeSource(Enum):
    """Sources of knowledge in the hierarchy."""
    MEMORY_BANK = "memory-bank"
    SERENA = "serena"
    CONTEXT7 = "context7"
    GITHUB = "github"
    NONE = "none"


class AuthorityRule(Enum):
    """Authority rules for knowledge source ordering."""
    LIBRARY_API = "library_api"  # context7 > memory-bank > serena
    PROJECT_CODE = "project_code"  # serena > memory-bank > context7
    PATTERNS = "patterns"  # memory-bank > serena > context7
    AUTO = "auto"  # Determine based on context_type


@dataclass
class Knowledge:
    """
    Knowledge retrieved from the hierarchy.

    Attributes:
        content: The actual knowledge data
        source: Which MCP provided this knowledge
        latency_ms: Time taken to retrieve (milliseconds)
        confidence: Confidence score (0.0-1.0)
        cache_key: Key used for caching
        cached: Whether this was retrieved from cache
        error: Error message if retrieval failed
    """
    content: Optional[Dict[str, Any]]
    source: KnowledgeSource
    latency_ms: int
    confidence: float = 1.0
    cache_key: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if knowledge retrieval was successful."""
        return self.content is not None and self.error is None


class MCPInterface(Protocol):
    """
    Protocol for MCP server interface.

    Claude Code runtime should implement this interface for each MCP server.
    """

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch knowledge from the MCP server.

        Args:
            query: Search query or symbol name
            context_type: Type of knowledge (api, code, pattern, etc.)
            **kwargs: Additional MCP-specific parameters

        Returns:
            Knowledge data or None if not found
        """
        ...

    async def store(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Store knowledge in the MCP (if applicable, e.g., memory-bank).

        Args:
            key: Storage key
            value: Value to store
            ttl: Time-to-live in seconds
            **kwargs: Additional parameters

        Returns:
            True if stored successfully
        """
        ...


class KnowledgeHierarchy:
    """
    Three-tier knowledge retrieval system with authority rules.

    The hierarchy optimizes knowledge retrieval by checking faster/cheaper
    sources first and falling back to slower/expensive sources.

    Layer 1: memory-bank (50-100ms, cached knowledge)
    Layer 2: serena (100-200ms, local codebase)
    Layer 3: context7 (300-500ms, external libraries)

    Authority rules determine the search order based on knowledge type:
    - Library APIs: context7 is authoritative (latest docs)
    - Project code: serena is authoritative (current state)
    - Patterns: memory-bank is authoritative (learned patterns)
    """

    def __init__(
        self,
        memory_bank: Optional[MCPInterface] = None,
        serena: Optional[MCPInterface] = None,
        context7: Optional[MCPInterface] = None,
        github: Optional[MCPInterface] = None,
        enable_caching: bool = True,
        cache_ttl: int = 1800,  # 30 minutes
    ):
        """
        Initialize the knowledge hierarchy.

        Args:
            memory_bank: Memory-bank MCP interface
            serena: Serena MCP interface
            context7: Context7 MCP interface
            github: GitHub MCP interface
            enable_caching: Enable intermediate caching
            cache_ttl: Default cache TTL in seconds
        """
        self.memory_bank = memory_bank
        self.serena = serena
        self.context7 = context7
        self.github = github
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "source_hits": {source.value: 0 for source in KnowledgeSource},
            "total_latency_ms": 0,
        }

    @classmethod
    async def create(
        cls,
        memory_bank: Optional[MCPInterface] = None,
        serena: Optional[MCPInterface] = None,
        context7: Optional[MCPInterface] = None,
        github: Optional[MCPInterface] = None,
        **kwargs
    ) -> "KnowledgeHierarchy":
        """
        Factory method to create and initialize hierarchy.

        Args:
            memory_bank: Memory-bank MCP interface
            serena: Serena MCP interface
            context7: Context7 MCP interface
            github: GitHub MCP interface
            **kwargs: Additional configuration

        Returns:
            Initialized KnowledgeHierarchy instance
        """
        hierarchy = cls(
            memory_bank=memory_bank,
            serena=serena,
            context7=context7,
            github=github,
            **kwargs
        )
        return hierarchy

    def _generate_cache_key(
        self,
        query: str,
        context_type: str,
        **kwargs
    ) -> str:
        """Generate cache key from query parameters."""
        key_parts = [context_type, query]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_search_order(
        self,
        authority_rule: AuthorityRule,
        context_type: str
    ) -> List[tuple[KnowledgeSource, Optional[MCPInterface]]]:
        """
        Determine MCP search order based on authority rule.

        Args:
            authority_rule: Authority rule to apply
            context_type: Type of knowledge being fetched

        Returns:
            List of (source, mcp) tuples in search order
        """
        # Determine rule if AUTO
        if authority_rule == AuthorityRule.AUTO:
            if context_type in ["api", "library"]:
                authority_rule = AuthorityRule.LIBRARY_API
            elif context_type in ["code", "project"]:
                authority_rule = AuthorityRule.PROJECT_CODE
            else:
                authority_rule = AuthorityRule.PATTERNS

        # Define search orders
        orders = {
            AuthorityRule.LIBRARY_API: [
                (KnowledgeSource.CONTEXT7, self.context7),
                (KnowledgeSource.MEMORY_BANK, self.memory_bank),
                (KnowledgeSource.SERENA, self.serena),
            ],
            AuthorityRule.PROJECT_CODE: [
                (KnowledgeSource.SERENA, self.serena),
                (KnowledgeSource.MEMORY_BANK, self.memory_bank),
                (KnowledgeSource.CONTEXT7, self.context7),
            ],
            AuthorityRule.PATTERNS: [
                (KnowledgeSource.MEMORY_BANK, self.memory_bank),
                (KnowledgeSource.SERENA, self.serena),
                (KnowledgeSource.CONTEXT7, self.context7),
            ],
        }

        # Filter out None MCPs
        order = [
            (source, mcp)
            for source, mcp in orders[authority_rule]
            if mcp is not None
        ]

        return order

    async def fetch(
        self,
        query: str,
        context_type: str = "general",
        authority_rule: AuthorityRule = AuthorityRule.AUTO,
        **kwargs
    ) -> Knowledge:
        """
        Fetch knowledge using hierarchical retrieval.

        Args:
            query: Search query or symbol name
            context_type: Type of knowledge (api, code, pattern, general)
            authority_rule: Authority rule for source ordering
            **kwargs: Additional parameters passed to MCPs

        Returns:
            Knowledge object with content and metadata

        Example:
            >>> result = await hierarchy.fetch(
            ...     "numpy.array",
            ...     context_type="library_api"
            ... )
            >>> if result.success:
            ...     print(f"Found in {result.source}")
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        # Generate cache key
        cache_key = self._generate_cache_key(query, context_type, **kwargs)

        # Get search order
        search_order = self._get_search_order(authority_rule, context_type)

        if not search_order:
            return Knowledge(
                content=None,
                source=KnowledgeSource.NONE,
                latency_ms=0,
                error="No MCP servers available"
            )

        # Try each source in order
        for source, mcp in search_order:
            try:
                fetch_start = time.time()
                content = await mcp.fetch(query, context_type, **kwargs)
                fetch_latency = int((time.time() - fetch_start) * 1000)

                if content:
                    # Update statistics
                    total_latency = int((time.time() - start_time) * 1000)
                    self.stats["source_hits"][source.value] += 1
                    self.stats["total_latency_ms"] += total_latency

                    # Check if this came from cache
                    cached = (
                        source == KnowledgeSource.MEMORY_BANK and
                        fetch_latency < 100
                    )
                    if cached:
                        self.stats["cache_hits"] += 1

                    # Cache in memory-bank if enabled and not already cached
                    if (
                        self.enable_caching and
                        not cached and
                        source != KnowledgeSource.MEMORY_BANK and
                        self.memory_bank is not None
                    ):
                        await self._cache_knowledge(
                            cache_key,
                            content,
                            source,
                            context_type
                        )

                    return Knowledge(
                        content=content,
                        source=source,
                        latency_ms=total_latency,
                        cache_key=cache_key,
                        cached=cached,
                        metadata={
                            "fetch_latency_ms": fetch_latency,
                            "authority_rule": authority_rule.value,
                        }
                    )

            except Exception as e:
                # Log error and continue to next source
                # In production, use proper logging
                print(f"Error fetching from {source.value}: {e}")
                continue

        # No source found the knowledge
        total_latency = int((time.time() - start_time) * 1000)
        return Knowledge(
            content=None,
            source=KnowledgeSource.NONE,
            latency_ms=total_latency,
            error="Knowledge not found in any source"
        )

    async def _cache_knowledge(
        self,
        cache_key: str,
        content: Any,
        source: KnowledgeSource,
        context_type: str
    ) -> None:
        """
        Cache knowledge in memory-bank.

        Args:
            cache_key: Cache key
            content: Content to cache
            source: Original source
            context_type: Type of knowledge
        """
        if self.memory_bank is None:
            return

        # Determine TTL based on context type
        ttl_map = {
            "library_api": 7 * 24 * 3600,  # 7 days
            "code": 30 * 24 * 3600,  # 30 days
            "pattern": 30 * 24 * 3600,  # 30 days
            "error": 90 * 24 * 3600,  # 90 days
        }
        ttl = ttl_map.get(context_type, self.cache_ttl)

        try:
            await self.memory_bank.store(
                key=cache_key,
                value={
                    "content": content,
                    "source": source.value,
                    "context_type": context_type,
                    "cached_at": time.time(),
                },
                ttl=ttl,
                tags=[context_type, source.value]
            )
        except Exception as e:
            # Log but don't fail
            print(f"Failed to cache knowledge: {e}")

    async def invalidate_cache(
        self,
        pattern: Optional[str] = None,
        context_type: Optional[str] = None
    ) -> int:
        """
        Invalidate cached knowledge.

        Args:
            pattern: Key pattern to invalidate (e.g., "numpy*")
            context_type: Invalidate all keys of this type

        Returns:
            Number of keys invalidated
        """
        if self.memory_bank is None:
            return 0

        # Implementation depends on memory-bank capabilities
        # This is a placeholder
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hierarchy statistics.

        Returns:
            Dictionary with usage statistics

        Example:
            >>> stats = hierarchy.get_stats()
            >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        """
        total_queries = self.stats["total_queries"]
        if total_queries == 0:
            return {**self.stats, "cache_hit_rate": 0.0, "avg_latency_ms": 0}

        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / total_queries,
            "avg_latency_ms": self.stats["total_latency_ms"] / total_queries,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "source_hits": {source.value: 0 for source in KnowledgeSource},
            "total_latency_ms": 0,
        }


# Convenience function for simple usage
async def fetch_knowledge(
    query: str,
    memory_bank: Optional[MCPInterface] = None,
    serena: Optional[MCPInterface] = None,
    context7: Optional[MCPInterface] = None,
    context_type: str = "general",
) -> Knowledge:
    """
    Convenience function for one-off knowledge retrieval.

    Args:
        query: Search query
        memory_bank: Memory-bank MCP
        serena: Serena MCP
        context7: Context7 MCP
        context_type: Type of knowledge

    Returns:
        Knowledge object
    """
    hierarchy = await KnowledgeHierarchy.create(
        memory_bank=memory_bank,
        serena=serena,
        context7=context7
    )
    return await hierarchy.fetch(query, context_type=context_type)
