"""
Basic Usage Examples

Demonstrates core functionality of the MCP integration system.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_integration import (
    KnowledgeHierarchy,
    LibraryCache,
    MCPProfileManager,
    SmartTrigger,
    ConfigLoader,
    MemoryCacheBackend,
)


async def example_1_library_cache():
    """Example 1: Using LibraryCache to resolve library IDs."""
    print("\n=== Example 1: Library Cache ===\n")

    # Create library cache
    cache = await LibraryCache.create("../library-cache.yaml")

    # Get library IDs (cache hits)
    print("Getting library IDs...")
    numpy_id = await cache.get_library_id("numpy")
    print(f"numpy → {numpy_id}")

    pytorch_id = await cache.get_library_id("torch")
    print(f"torch (alias) → {pytorch_id}")

    # Auto-detect libraries in code
    print("\nDetecting libraries in code...")
    code = """
import numpy as np
import torch
from sklearn import datasets
    """
    detected = cache.detect_libraries(code)
    print(f"Detected: {[lib.name for lib in detected]}")

    # Show statistics
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Total libraries: {stats['total_libraries']}")


async def example_2_knowledge_hierarchy():
    """Example 2: Using KnowledgeHierarchy for retrieval."""
    print("\n=== Example 2: Knowledge Hierarchy ===\n")

    # Mock MCP interfaces
    class MockMemoryBank:
        async def fetch(self, query, context_type, **kwargs):
            # Simulate cache miss
            return None

        async def store(self, key, value, ttl, **kwargs):
            print(f"  [memory-bank] Cached: {key[:50]}...")
            return True

    class MockSerena:
        async def fetch(self, query, context_type, **kwargs):
            if "main.py" in query:
                return {"content": "def main(): pass", "file": "main.py"}
            return None

    class MockContext7:
        async def fetch(self, query, context_type, **kwargs):
            if "numpy" in query.lower():
                return {"docs": "numpy.array creates an array...", "examples": [...]}
            return None

    # Create hierarchy
    hierarchy = await KnowledgeHierarchy.create(
        memory_bank=MockMemoryBank(),
        serena=MockSerena(),
        context7=MockContext7()
    )

    # Fetch library API (context7 is authoritative)
    print("Fetching library API documentation...")
    result = await hierarchy.fetch(
        "numpy.array",
        context_type="library_api"
    )
    print(f"  Source: {result.source.value}")
    print(f"  Latency: {result.latency_ms}ms")
    print(f"  Cached: {result.cached}")

    # Fetch project code (serena is authoritative)
    print("\nFetching project code...")
    result = await hierarchy.fetch(
        "main.py function",
        context_type="project_code"
    )
    print(f"  Source: {result.source.value}")
    print(f"  Latency: {result.latency_ms}ms")

    # Show statistics
    stats = hierarchy.get_stats()
    print(f"\nHierarchy Stats:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.0f}ms")


async def example_3_profile_manager():
    """Example 3: Using MCPProfileManager for profiles."""
    print("\n=== Example 3: Profile Manager ===\n")

    # Mock MCP factory
    async def mock_factory(mcp_config):
        print(f"  Loading {mcp_config.name} (priority: {mcp_config.priority.value})")
        await asyncio.sleep(0.1)  # Simulate load time
        return {"name": mcp_config.name, "status": "loaded"}

    # Create manager
    manager = await MCPProfileManager.create(
        "../mcp-profiles.yaml",
        mcp_factory=mock_factory
    )

    # List available profiles
    print("Available profiles:")
    for profile_name in manager.list_profiles():
        print(f"  - {profile_name}")

    # Activate profile
    print("\nActivating 'code-analysis' profile...")
    profile = await manager.activate_profile("code-analysis")
    print(f"  MCPs loaded: {list(manager.get_active_mcps().keys())}")

    # Activate for command
    print("\nActivating profile for 'fix' command...")
    profile = await manager.activate_for_command("fix")
    if profile:
        print(f"  Profile: {profile.name}")
        print(f"  Orchestrated: {profile.orchestrated}")

    # Show statistics
    stats = manager.get_stats()
    print(f"\nManager Stats:")
    print(f"  Profiles activated: {stats['profiles_activated']}")
    print(f"  MCPs loaded: {stats['mcps_loaded']}")
    print(f"  Avg load time: {stats['avg_load_time_ms']:.0f}ms")


async def example_4_smart_trigger():
    """Example 4: Using SmartTrigger for conditional activation."""
    print("\n=== Example 4: Smart Trigger ===\n")

    # Create trigger
    trigger = await SmartTrigger.create("../mcp-config.yaml")

    # Analyze different query types
    queries = [
        ("How do I use numpy.array?", "ultra-think"),
        ("Fix the error in main.py", "fix"),
        ("List open pull requests", None),
        ("Analyze this code", "code-review"),
    ]

    for query, command in queries:
        result = trigger.analyze(query, command)
        print(f"\nQuery: \"{query}\"")
        print(f"  Type: {result.query_type.value}")
        print(f"  Recommended MCPs: {result.recommended_mcps}")
        print(f"  Confidence: {result.confidence:.2f}")

    # Check specific MCP activation
    print("\n--- Specific MCP Checks ---")
    should_use = trigger.should_activate_mcp(
        mcp_name="context7",
        query="How do I use numpy.array?",
        threshold=0.6
    )
    print(f"Use context7 for numpy query? {should_use}")

    should_use = trigger.should_activate_mcp(
        mcp_name="github",
        query="Implement new feature",
        threshold=0.6
    )
    print(f"Use github for feature request? {should_use}")

    # Show statistics
    stats = trigger.get_stats()
    print(f"\nTrigger Stats:")
    print(f"  Queries analyzed: {stats['queries_analyzed']}")
    print(f"  Trigger rate: {stats['trigger_rate']:.1%}")


async def example_5_cache_backend():
    """Example 5: Using CacheBackend for storage."""
    print("\n=== Example 5: Cache Backend ===\n")

    # Create memory cache
    cache = MemoryCacheBackend(max_size=100)

    # Store values
    print("Storing values...")
    await cache.set("user:1", {"name": "Alice", "role": "admin"}, ttl=3600)
    await cache.set("user:2", {"name": "Bob", "role": "user"}, ttl=3600)
    await cache.set("config:app", {"debug": True}, ttl=7200, tags=["config"])

    # Retrieve values
    print("\nRetrieving values...")
    user1 = await cache.get("user:1")
    print(f"  user:1 → {user1}")

    # Check existence
    exists = await cache.exists("user:2")
    print(f"  user:2 exists? {exists}")

    # Get by tag
    print("\nGetting by tag...")
    configs = await cache.get_by_tag("config")
    print(f"  Tag 'config': {list(configs.keys())}")

    # Show statistics
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")


async def example_6_integrated_workflow():
    """Example 6: Integrated workflow combining all components."""
    print("\n=== Example 6: Integrated Workflow ===\n")

    # Scenario: User asks "How do I use numpy.array?"

    # Step 1: Smart trigger analyzes query
    print("Step 1: Analyzing query with SmartTrigger...")
    trigger = await SmartTrigger.create("../mcp-config.yaml")
    result = trigger.analyze("How do I use numpy.array?", "ultra-think")
    print(f"  Recommended MCPs: {result.recommended_mcps}")

    # Step 2: Library cache resolves numpy ID
    print("\nStep 2: Resolving library ID...")
    lib_cache = await LibraryCache.create("../library-cache.yaml")
    numpy_id = await lib_cache.get_library_id("numpy")
    print(f"  numpy ID: {numpy_id}")

    # Step 3: Knowledge hierarchy fetches docs
    print("\nStep 3: Fetching documentation via hierarchy...")
    # (Would use actual MCPs in production)
    print("  [context7] Fetching numpy.array docs...")
    print("  [memory-bank] Caching for future use...")

    # Step 4: Profile manager ensures MCPs loaded
    print("\nStep 4: Profile manager ensures MCPs ready...")
    # (Would activate meta-reasoning profile)
    print("  Profile 'meta-reasoning' activated")
    print("  MCPs loaded: sequential-thinking, memory-bank, context7")

    print("\n✓ Workflow complete!")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("MCP Integration System - Usage Examples")
    print("=" * 60)

    await example_1_library_cache()
    await example_2_knowledge_hierarchy()
    await example_3_profile_manager()
    await example_4_smart_trigger()
    await example_5_cache_backend()
    await example_6_integrated_workflow()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
