"""
Cache Backend System

Pluggable cache backends for knowledge hierarchy and library cache.
Supports in-memory, file-based, and memory-bank MCP caching.

Example:
    >>> # In-memory cache (fastest, volatile)
    >>> cache = MemoryCacheBackend(max_size=1000)
    >>> await cache.set("key", {"data": "value"}, ttl=3600)
    >>> value = await cache.get("key")
    >>>
    >>> # File-based cache (persistent)
    >>> cache = FileCacheBackend(cache_dir=".cache")
    >>> await cache.set("key", {"data": "value"}, ttl=3600)
"""

import json
import time
import hashlib
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class CacheEntry:
    """
    Cache entry with metadata.

    Attributes:
        key: Cache key
        value: Cached value
        ttl: Time-to-live in seconds
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        tags: Tags for categorization
        size_bytes: Approximate size in bytes
    """
    key: str
    value: Any
    ttl: Optional[int]
    created_at: float
    accessed_at: float
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    All cache backends must implement:
    - get(key) -> value or None
    - set(key, value, ttl) -> bool
    - delete(key) -> bool
    - clear() -> int
    - exists(key) -> bool
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend with LRU eviction.

    Features:
    - Fast access (O(1) lookups)
    - LRU eviction when max_size reached
    - TTL support
    - Tag-based categorization
    - Statistics tracking

    Best for:
    - Short-lived data
    - High-frequency access
    - Development/testing
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: Optional[float] = None,
    ):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Tags index
        self._tags: Dict[str, Set[str]] = {}

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "total_size_bytes": 0,
        }

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            self.stats["misses"] += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired():
            await self.delete(key)
            self.stats["misses"] += 1
            return None

        # Update access time and move to end (LRU)
        entry.accessed_at = time.time()
        self._cache.move_to_end(key)

        self.stats["hits"] += 1
        return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        # Calculate approximate size
        size_bytes = self._estimate_size(value)

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            created_at=time.time(),
            accessed_at=time.time(),
            tags=set(tags) if tags else set(),
            size_bytes=size_bytes,
        )

        # Check if we need to evict
        while len(self._cache) >= self.max_size:
            await self._evict_lru()

        # Check memory limit
        if self.max_memory_mb:
            memory_mb = self.stats["total_size_bytes"] / (1024 * 1024)
            while memory_mb > self.max_memory_mb:
                await self._evict_lru()
                memory_mb = self.stats["total_size_bytes"] / (1024 * 1024)

        # Remove old entry if exists
        if key in self._cache:
            old_entry = self._cache[key]
            self.stats["total_size_bytes"] -= old_entry.size_bytes
            self._remove_from_tags(key, old_entry.tags)

        # Add new entry
        self._cache[key] = entry
        self.stats["total_size_bytes"] += size_bytes
        self.stats["sets"] += 1

        # Update tags index
        for tag in entry.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(key)

        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key not in self._cache:
            return False

        entry = self._cache.pop(key)
        self.stats["total_size_bytes"] -= entry.size_bytes
        self.stats["deletes"] += 1

        # Remove from tags
        self._remove_from_tags(key, entry.tags)

        return True

    async def clear(self) -> int:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._tags.clear()
        self.stats["total_size_bytes"] = 0
        return count

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if key not in self._cache:
            return False

        entry = self._cache[key]
        if entry.is_expired():
            await self.delete(key)
            return False

        return True

    async def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """
        Get all entries with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            Dictionary of key -> value
        """
        if tag not in self._tags:
            return {}

        result = {}
        for key in self._tags[tag].copy():  # Copy to avoid modification during iteration
            value = await self.get(key)
            if value is not None:
                result[key] = value

        return result

    async def delete_by_tag(self, tag: str) -> int:
        """
        Delete all entries with a specific tag.

        Args:
            tag: Tag to delete

        Returns:
            Number of entries deleted
        """
        if tag not in self._tags:
            return 0

        keys = list(self._tags[tag])
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1

        return count

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Remove first item (oldest)
        key, entry = self._cache.popitem(last=False)
        self.stats["total_size_bytes"] -= entry.size_bytes
        self.stats["evictions"] += 1

        # Remove from tags
        self._remove_from_tags(key, entry.tags)

    def _remove_from_tags(self, key: str, tags: Set[str]) -> None:
        """Remove key from tags index."""
        for tag in tags:
            if tag in self._tags:
                self._tags[tag].discard(key)
                if not self._tags[tag]:
                    del self._tags[tag]

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            # Try pickle for accurate size
            return len(pickle.dumps(value))
        except Exception:
            # Fallback to rough estimate
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            else:
                return 100  # Default estimate

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        memory_mb = self.stats["total_size_bytes"] / (1024 * 1024)

        return {
            **self.stats,
            "total_entries": len(self._cache),
            "total_tags": len(self._tags),
            "hit_rate": hit_rate,
            "memory_mb": memory_mb,
        }


class FileCacheBackend(CacheBackend):
    """
    File-based cache backend with JSON/pickle storage.

    Features:
    - Persistent storage
    - TTL support
    - Metadata tracking
    - Directory-based organization

    Best for:
    - Long-lived data
    - Cross-session persistence
    - Production use
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        format: str = "json",  # 'json' or 'pickle'
    ):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory for cache files
            format: Storage format ('json' or 'pickle')
        """
        self.cache_dir = Path(cache_dir)
        self.format = format

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.cache_dir / "_metadata.json"

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            self.stats["misses"] += 1
            return None

        # Load entry
        entry = self._load_entry(cache_file)

        if entry is None:
            self.stats["misses"] += 1
            return None

        # Check expiration
        if entry.is_expired():
            await self.delete(key)
            self.stats["misses"] += 1
            return None

        # Update access time
        entry.accessed_at = time.time()
        self._save_entry(cache_file, entry)

        self.stats["hits"] += 1
        return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache."""
        cache_file = self._get_cache_file(key)

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            created_at=time.time(),
            accessed_at=time.time(),
            tags=set(tags) if tags else set(),
        )

        # Save entry
        try:
            self._save_entry(cache_file, entry)
            self.stats["sets"] += 1
            return True
        except Exception as e:
            print(f"Failed to save cache entry: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return False

        try:
            cache_file.unlink()
            self.stats["deletes"] += 1
            return True
        except Exception:
            return False

    async def clear(self) -> int:
        """Clear all cache entries."""
        count = 0
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.name.startswith("_"):
                continue
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass

        return count

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        cache_file = self._get_cache_file(key)
        if not cache_file.exists():
            return False

        entry = self._load_entry(cache_file)
        if entry is None or entry.is_expired():
            await self.delete(key)
            return False

        return True

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash key for safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        extension = ".json" if self.format == "json" else ".pkl"
        return self.cache_dir / f"{key_hash}{extension}"

    def _load_entry(self, cache_file: Path) -> Optional[CacheEntry]:
        """Load cache entry from file."""
        try:
            if self.format == "json":
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return CacheEntry(
                        key=data['key'],
                        value=data['value'],
                        ttl=data['ttl'],
                        created_at=data['created_at'],
                        accessed_at=data['accessed_at'],
                        tags=set(data.get('tags', [])),
                    )
            else:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            return None

    def _save_entry(self, cache_file: Path, entry: CacheEntry) -> None:
        """Save cache entry to file."""
        if self.format == "json":
            with open(cache_file, 'w') as f:
                json.dump({
                    'key': entry.key,
                    'value': entry.value,
                    'ttl': entry.ttl,
                    'created_at': entry.created_at,
                    'accessed_at': entry.accessed_at,
                    'tags': list(entry.tags),
                }, f)
        else:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        # Count files
        total_entries = len(list(self.cache_dir.glob("*"))) - 1  # Exclude metadata

        return {
            **self.stats,
            "total_entries": total_entries,
            "hit_rate": hit_rate,
            "cache_dir": str(self.cache_dir),
        }


# Factory function
def create_cache_backend(
    backend_type: str = "memory",
    **kwargs
) -> CacheBackend:
    """
    Factory function to create cache backend.

    Args:
        backend_type: Type of backend ('memory', 'file')
        **kwargs: Backend-specific arguments

    Returns:
        Cache backend instance

    Example:
        >>> cache = create_cache_backend("memory", max_size=1000)
        >>> cache = create_cache_backend("file", cache_dir=".cache")
    """
    if backend_type == "memory":
        return MemoryCacheBackend(**kwargs)
    elif backend_type == "file":
        return FileCacheBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
