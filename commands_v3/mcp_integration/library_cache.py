"""
Library Cache System

Pre-caches library IDs to eliminate the resolve-library-id API call overhead.
Supports auto-detection of library usage from code patterns.

Example:
    >>> cache = await LibraryCache.create("library-cache.yaml")
    >>>
    >>> # Get library ID (no API call needed)
    >>> lib_id = await cache.get_library_id("numpy")
    >>> # Returns: "/numpy/numpy"
    >>>
    >>> # Auto-detect libraries in code
    >>> code = "import numpy as np\\nimport torch"
    >>> libs = cache.detect_libraries(code)
    >>> # Returns: [LibraryInfo(name='numpy', id='/numpy/numpy'), ...]
"""

import re
import yaml
import hashlib
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class DetectionType(Enum):
    """Types of library detection patterns."""
    IMPORT = "import"
    DECORATOR = "decorator"
    FUNCTION = "function"


@dataclass
class DetectionPattern:
    """
    Pattern for detecting library usage in code.

    Attributes:
        pattern: Regex pattern to match
        library: Library name this pattern indicates
        detection_type: Type of pattern (import, decorator, function)
        confidence: Confidence score (0.0-1.0)
    """
    pattern: str
    library: str
    detection_type: DetectionType
    confidence: float = 1.0

    @property
    def compiled_pattern(self) -> re.Pattern:
        """Get compiled regex pattern."""
        if not hasattr(self, '_compiled'):
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        return self._compiled


@dataclass
class LibraryInfo:
    """
    Information about a cached library.

    Attributes:
        name: Library name (e.g., 'numpy')
        id: Context7 library ID (e.g., '/numpy/numpy')
        aliases: Alternative names (e.g., ['np'] for numpy)
        category: Library category (scientific, ml, testing, etc.)
        description: Brief description
    """
    name: str
    id: str
    aliases: List[str] = field(default_factory=list)
    category: str = "general"
    description: str = ""

    def matches(self, query: str) -> bool:
        """Check if query matches this library."""
        query_lower = query.lower()
        return (
            query_lower == self.name.lower() or
            query_lower in [alias.lower() for alias in self.aliases]
        )


class LibraryCache:
    """
    Library ID cache with auto-detection capabilities.

    Eliminates the need for context7.resolve_library_id() calls by
    maintaining a pre-populated cache of common libraries.

    Features:
    - Pre-cached IDs for 40+ common libraries
    - Auto-detection from code patterns
    - Category-based grouping
    - Alias support (e.g., 'np' → 'numpy')
    - Pattern matching for imports, decorators, functions
    """

    def __init__(
        self,
        libraries: Dict[str, LibraryInfo],
        detection_patterns: List[DetectionPattern],
        enable_fallback: bool = True
    ):
        """
        Initialize library cache.

        Args:
            libraries: Dictionary of library name → LibraryInfo
            detection_patterns: List of detection patterns
            enable_fallback: Enable fallback to resolve API if not cached
        """
        self.libraries = libraries
        self.detection_patterns = detection_patterns
        self.enable_fallback = enable_fallback

        # Build alias lookup for fast matching
        self._alias_map: Dict[str, str] = {}
        for lib_name, lib_info in libraries.items():
            self._alias_map[lib_name.lower()] = lib_name
            for alias in lib_info.aliases:
                self._alias_map[alias.lower()] = lib_name

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "detections": 0,
            "fallback_calls": 0,
        }

    @classmethod
    async def create(
        cls,
        config_path: str,
        context7_mcp: Optional[Any] = None
    ) -> "LibraryCache":
        """
        Create library cache from YAML configuration.

        Args:
            config_path: Path to library-cache.yaml
            context7_mcp: Optional context7 MCP for fallback

        Returns:
            Initialized LibraryCache instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load libraries
        libraries = {}
        for lib_name, lib_config in config.get('common_libraries', {}).items():
            libraries[lib_name] = LibraryInfo(
                name=lib_name,
                id=lib_config['id'],
                aliases=lib_config.get('aliases', []),
                category=lib_config.get('category', 'general'),
                description=lib_config.get('description', '')
            )

        # Load detection patterns
        patterns = []
        for lang, lang_patterns in config.get('detection_patterns', {}).items():
            for pattern_config in lang_patterns:
                # Get detection_type from config or default to IMPORT
                detection_type_str = pattern_config.get('detection_type', 'import')
                patterns.append(DetectionPattern(
                    pattern=pattern_config['pattern'],
                    library=pattern_config['library'],
                    detection_type=DetectionType(detection_type_str)
                ))

        cache = cls(
            libraries=libraries,
            detection_patterns=patterns,
            enable_fallback=config.get('cache_strategy', {}).get('fallback', {}).get('use_resolve_api', True)
        )

        cache.context7_mcp = context7_mcp

        return cache

    async def get_library_id(
        self,
        library_name: str,
        use_fallback: bool = True
    ) -> Optional[str]:
        """
        Get library ID with three-tier lookup.

        Tier 1: Pre-populated cache (immediate)
        Tier 2: Check aliases
        Tier 3: Fallback to context7 resolve API

        Args:
            library_name: Library name (e.g., 'numpy', 'np')
            use_fallback: Use fallback to resolve API if not cached

        Returns:
            Library ID (e.g., '/numpy/numpy') or None

        Example:
            >>> lib_id = await cache.get_library_id("numpy")
            >>> # Returns: "/numpy/numpy" (cache hit, ~1ms)
            >>>
            >>> lib_id = await cache.get_library_id("unknown-lib")
            >>> # Returns: "/owner/unknown-lib" (fallback, ~300ms)
        """
        # Tier 1 & 2: Check cache and aliases
        canonical_name = self._alias_map.get(library_name.lower())

        if canonical_name and canonical_name in self.libraries:
            self.stats["cache_hits"] += 1
            return self.libraries[canonical_name].id

        # Cache miss
        self.stats["cache_misses"] += 1

        # Tier 3: Fallback to context7 resolve API
        if use_fallback and self.enable_fallback and hasattr(self, 'context7_mcp'):
            try:
                self.stats["fallback_calls"] += 1
                resolved_id = await self.context7_mcp.resolve_library_id(library_name)

                # Cache the resolved ID for future use
                if resolved_id:
                    await self._cache_resolved_library(library_name, resolved_id)

                return resolved_id
            except Exception as e:
                print(f"Fallback resolution failed: {e}")
                return None

        return None

    def detect_libraries(
        self,
        code: str,
        language: Optional[str] = None
    ) -> List[LibraryInfo]:
        """
        Detect libraries used in code based on patterns.

        Args:
            code: Source code to analyze
            language: Programming language (auto-detect if None)

        Returns:
            List of detected libraries

        Example:
            >>> code = '''
            ... import numpy as np
            ... import torch
            ... @jax.jit
            ... def foo(): pass
            ... '''
            >>> libs = cache.detect_libraries(code)
            >>> # Returns: [LibraryInfo(name='numpy', ...),
            ...              LibraryInfo(name='pytorch', ...),
            ...              LibraryInfo(name='jax', ...)]
        """
        detected: Set[str] = set()

        for pattern in self.detection_patterns:
            if pattern.compiled_pattern.search(code):
                detected.add(pattern.library)
                self.stats["detections"] += 1

        # Convert to LibraryInfo objects
        result = []
        for lib_name in detected:
            if lib_name in self.libraries:
                result.append(self.libraries[lib_name])

        return result

    def get_by_category(self, category: str) -> List[LibraryInfo]:
        """
        Get all libraries in a category.

        Args:
            category: Category name (scientific, ml, testing, etc.)

        Returns:
            List of libraries in that category
        """
        return [
            lib for lib in self.libraries.values()
            if lib.category == category
        ]

    def get_common_libraries(self, limit: int = 10) -> List[LibraryInfo]:
        """
        Get most commonly used libraries.

        Args:
            limit: Maximum number to return

        Returns:
            List of common libraries
        """
        # In a production system, this would be based on usage stats
        # For now, return by category priority
        priority_categories = ['scientific', 'ml', 'testing', 'frontend']

        result = []
        for category in priority_categories:
            result.extend(self.get_by_category(category))
            if len(result) >= limit:
                break

        return result[:limit]

    async def _cache_resolved_library(
        self,
        library_name: str,
        library_id: str
    ) -> None:
        """
        Cache a newly resolved library ID.

        Args:
            library_name: Library name
            library_id: Resolved library ID
        """
        # Add to in-memory cache
        self.libraries[library_name] = LibraryInfo(
            name=library_name,
            id=library_id,
            category="resolved"
        )
        self._alias_map[library_name.lower()] = library_name

        # In production, persist to memory-bank MCP
        # if hasattr(self, 'memory_bank_mcp'):
        #     await self.memory_bank_mcp.store(
        #         key=f"lib_id:{library_name}",
        #         value=library_id,
        #         ttl=30 * 24 * 3600  # 30 days
        #     )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
        """
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = self.stats["cache_hits"] / total if total > 0 else 0

        return {
            **self.stats,
            "total_queries": total,
            "hit_rate": hit_rate,
            "total_libraries": len(self.libraries),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "detections": 0,
            "fallback_calls": 0,
        }


# Convenience functions

async def get_library_id(
    library_name: str,
    cache_path: str = "library-cache.yaml",
    context7_mcp: Optional[Any] = None
) -> Optional[str]:
    """
    Convenience function to get library ID.

    Args:
        library_name: Library name
        cache_path: Path to cache config
        context7_mcp: Optional context7 MCP for fallback

    Returns:
        Library ID or None
    """
    cache = await LibraryCache.create(cache_path, context7_mcp)
    return await cache.get_library_id(library_name)


def detect_libraries_in_code(
    code: str,
    cache_path: str = "library-cache.yaml"
) -> List[str]:
    """
    Convenience function to detect libraries in code.

    Args:
        code: Source code
        cache_path: Path to cache config

    Returns:
        List of library names
    """
    import asyncio

    async def _detect():
        cache = await LibraryCache.create(cache_path)
        libs = cache.detect_libraries(code)
        return [lib.name for lib in libs]

    return asyncio.run(_detect())
