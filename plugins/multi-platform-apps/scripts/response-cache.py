#!/usr/bin/env python3
"""
Response Caching Layer for Python Development Plugin

Implements intelligent caching with:
- Pattern-based caching
- TTL management
- Cache invalidation
- Hit rate tracking
"""

import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pickle


@dataclass
class CacheEntry:
    """Single cache entry"""
    key: str
    query: str
    response: str
    agent: str
    model: str
    timestamp: float
    ttl_seconds: int
    hit_count: int = 0
    last_accessed: float = 0.0

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() - self.timestamp > self.ttl_seconds

    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if entry is stale (old but not expired)"""
        age_hours = (time.time() - self.timestamp) / 3600
        return age_hours > max_age_hours

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


class ResponseCache:
    """Response caching with intelligent invalidation"""

    def __init__(self, cache_dir: Optional[Path] = None, default_ttl: int = 3600):
        """
        Initialize cache

        Args:
            cache_dir: Directory for cache files (default: plugin/.cache)
            default_ttl: Default TTL in seconds (default: 1 hour)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / ".cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl

        self.cache_file = self.cache_dir / "response_cache.pkl"
        self.stats_file = self.cache_dir / "cache_stats.json"

        # Load existing cache
        self.cache: Dict[str, CacheEntry] = self._load_cache()
        self.stats = self._load_stats()

    def _generate_key(self, query: str, agent: str, context: Optional[Dict] = None) -> str:
        """Generate cache key from query and context"""
        # Normalize query
        normalized = query.lower().strip()

        # Include agent in key
        key_components = [normalized, agent]

        # Include relevant context
        if context:
            if 'model' in context:
                key_components.append(context['model'])

        key_str = "|".join(key_components)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        query: str,
        agent: str,
        context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get cached response

        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(query, agent, context)

        if key not in self.cache:
            self.stats['misses'] = self.stats.get('misses', 0) + 1
            return None

        entry = self.cache[key]

        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self.stats['expirations'] = self.stats.get('expirations', 0) + 1
            self.stats['misses'] = self.stats.get('misses', 0) + 1
            self._save_cache()
            return None

        # Update hit stats
        entry.hit_count += 1
        entry.last_accessed = time.time()
        self.stats['hits'] = self.stats.get('hits', 0) + 1

        self._save_cache()
        return entry.response

    def set(
        self,
        query: str,
        response: str,
        agent: str,
        model: str,
        context: Optional[Dict] = None,
        ttl: Optional[int] = None
    ):
        """Cache a response"""
        key = self._generate_key(query, agent, context)

        entry = CacheEntry(
            key=key,
            query=query,
            response=response,
            agent=agent,
            model=model,
            timestamp=time.time(),
            ttl_seconds=ttl or self.default_ttl,
            last_accessed=time.time()
        )

        self.cache[key] = entry
        self.stats['stores'] = self.stats.get('stores', 0) + 1

        self._save_cache()

    def invalidate_agent(self, agent: str):
        """Invalidate all cache entries for an agent"""
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if entry.agent == agent
        ]

        for key in keys_to_remove:
            del self.cache[key]

        self.stats['invalidations'] = self.stats.get('invalidations', 0) + len(keys_to_remove)
        self._save_cache()

    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern in query"""
        pattern_lower = pattern.lower()
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if pattern_lower in entry.query.lower()
        ]

        for key in keys_to_remove:
            del self.cache[key]

        self.stats['invalidations'] = self.stats.get('invalidations', 0) + len(keys_to_remove)
        self._save_cache()

    def clean_expired(self):
        """Remove expired entries"""
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]

        for key in keys_to_remove:
            del self.cache[key]

        return len(keys_to_remove)

    def clean_stale(self, max_age_hours: int = 24):
        """Remove stale entries"""
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if entry.is_stale(max_age_hours)
        ]

        for key in keys_to_remove:
            del self.cache[key]

        return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats.get('hits', 0) + self.stats.get('misses', 0)
        hit_rate = self.stats.get('hits', 0) / total_requests if total_requests > 0 else 0

        # Calculate cache size
        cache_size_bytes = 0
        if self.cache_file.exists():
            cache_size_bytes = self.cache_file.stat().st_size

        # Top hit entries
        top_entries = sorted(
            self.cache.values(),
            key=lambda e: e.hit_count,
            reverse=True
        )[:10]

        return {
            'entries': len(self.cache),
            'hits': self.stats.get('hits', 0),
            'misses': self.stats.get('misses', 0),
            'hit_rate': hit_rate,
            'stores': self.stats.get('stores', 0),
            'invalidations': self.stats.get('invalidations', 0),
            'expirations': self.stats.get('expirations', 0),
            'cache_size_bytes': cache_size_bytes,
            'cache_size_mb': cache_size_bytes / (1024 * 1024),
            'top_queries': [
                {
                    'query': e.query[:60],
                    'hits': e.hit_count,
                    'agent': e.agent,
                    'model': e.model
                }
                for e in top_entries[:5]
            ]
        }

    def clear_all(self):
        """Clear entire cache"""
        count = len(self.cache)
        self.cache.clear()
        self.stats['clears'] = self.stats.get('clears', 0) + 1
        self._save_cache()
        return count

    def _load_cache(self) -> Dict[str, CacheEntry]:
        """Load cache from disk"""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)

            self._save_stats()
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _load_stats(self) -> Dict[str, int]:
        """Load stats from disk"""
        if not self.stats_file.exists():
            return {}

        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_stats(self):
        """Save stats to disk"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save stats: {e}")


def demo():
    """Demonstration of caching functionality"""
    print("=" * 80)
    print("Response Cache Demo")
    print("=" * 80)

    cache = ResponseCache()

    # Clear cache for demo
    cache.clear_all()
    print("\n✓ Cache cleared for demo")

    # Simulate caching some responses
    test_scenarios = [
        ("How do I create a FastAPI endpoint?", "fastapi-pro", "haiku", "Use @app.get..."),
        ("Show me async patterns", "python-pro", "sonnet", "Here are async patterns..."),
        ("How do I create a FastAPI endpoint?", "fastapi-pro", "haiku", "Use @app.get..."),  # Duplicate
        ("Install Django", "django-pro", "haiku", "uv pip install django..."),
    ]

    print("\n--- Simulating Requests ---")
    for query, agent, model, response in test_scenarios:
        # Try to get from cache first
        cached = cache.get(query, agent, {'model': model})

        if cached:
            print(f"✓ CACHE HIT: {query[:40]}... [{agent}]")
        else:
            print(f"✗ CACHE MISS: {query[:40]}... [{agent}]")
            # Store in cache
            cache.set(query, response, agent, model, {'model': model})

    # Show statistics
    print("\n--- Cache Statistics ---")
    stats = cache.get_stats()
    print(f"Entries: {stats['entries']}")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    print(f"Cache Size: {stats['cache_size_mb']:.2f} MB")

    print("\n--- Top Queries ---")
    for i, query_stat in enumerate(stats['top_queries'], 1):
        print(f"{i}. {query_stat['query']}... (hits: {query_stat['hits']}, model: {query_stat['model']})")

    # Cleanup demo
    print("\n--- Cleanup ---")
    expired_count = cache.clean_expired()
    print(f"Removed {expired_count} expired entries")

    print("\n✓ Demo complete")


if __name__ == "__main__":
    demo()
