# MCP Integration System - Deployment Guide

**Version:** 1.0.0
**Date:** 2025-10-04
**Status:** Production-Ready

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Integration with Claude Code](#integration-with-claude-code)
5. [Command Integration](#command-integration)
6. [Testing](#testing)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)
10. [Rollback Procedures](#rollback-procedures)

---

## Prerequisites

### System Requirements

- **Python:** 3.9+ (async/await support)
- **Memory:** 512MB minimum, 1GB recommended
- **Storage:** 100MB for package + 500MB for cache
- **Network:** Stable connection for external MCPs (context7, github)

### Dependencies

```bash
# Core dependencies
pip install pyyaml  # Configuration loading
pip install pytest  # Testing (optional)

# For production monitoring (optional)
pip install prometheus-client  # Metrics export
```

### MCP Servers Required

- ✅ **memory-bank** (allPepper-memory-bank)
- ✅ **serena** (serena-heterodyne-analysis)
- ✅ **context7** (for library documentation)
- ⚠️ **github** (optional, for GitHub operations)
- ⚠️ **playwright** (optional, for web automation)
- ⚠️ **sequential-thinking** (optional, for meta-reasoning)

---

## Installation

### Step 1: Copy Package

The MCP integration package is located at:
```
/Users/b80985/.claude/commands/mcp_integration/
```

For deployment:

```bash
# Option A: Direct import (development)
# Add to PYTHONPATH or sys.path.insert()
export PYTHONPATH=/Users/b80985/.claude/commands:$PYTHONPATH

# Option B: Install as package (production)
cd /Users/b80985/.claude/commands
pip install -e mcp_integration/

# Option C: Copy to Claude Code runtime
cp -r mcp_integration/ /path/to/claude-code/runtime/
```

### Step 2: Copy Configuration Files

```bash
# Copy config files to runtime directory
cp mcp-config.yaml /path/to/claude-code/config/
cp mcp-profiles.yaml /path/to/claude-code/config/
cp library-cache.yaml /path/to/claude-code/config/
```

### Step 3: Verify Installation

```python
# test_installation.py
import asyncio
from mcp_integration import (
    KnowledgeHierarchy,
    LibraryCache,
    MCPProfileManager,
)

async def test():
    # Test imports
    print("✓ Imports successful")

    # Test config loading
    cache = await LibraryCache.create("library-cache.yaml")
    print(f"✓ Loaded {len(cache.libraries)} libraries")

    manager = await MCPProfileManager.create("mcp-profiles.yaml")
    print(f"✓ Loaded {len(manager.profiles)} profiles")

asyncio.run(test())
```

---

## Configuration

### 1. Global Configuration (mcp-config.yaml)

```yaml
# /path/to/claude-code/config/mcp-config.yaml

knowledge_hierarchy:
  enabled: true
  layers:
    - name: memory-bank
      priority: 1
      latency_target_ms: 100
    - name: serena
      priority: 2
      latency_target_ms: 200
    - name: context7
      priority: 3
      latency_target_ms: 500

memory-bank:
  cache:
    enabled: true
    ttl_by_type:
      error_solutions: 7776000  # 90 days
      test_stability: 5184000   # 60 days
      quality_baseline: 2592000 # 30 days

context7:
  library_cache:
    enabled: true
    fallback:
      use_resolve_api: true

smart_triggers:
  enabled: true
  patterns:
    - pattern: "how (to|do i).*numpy"
      mcp: context7
      type: library_api
      confidence: 0.9
    - pattern: "fix.*error|debug"
      mcp: memory-bank
      type: error_debug
      confidence: 0.85
```

### 2. Profile Configuration (mcp-profiles.yaml)

Already complete - no changes needed unless adding custom profiles.

### 3. Library Cache (library-cache.yaml)

Add project-specific libraries if needed:

```yaml
# Add custom libraries
common_libraries:
  my-lib:
    id: "/org/my-lib"
    aliases: ["mylib"]
    category: "custom"
    description: "Internal library"

# Add custom detection patterns
detection_patterns:
  import:
    - pattern: "import\\s+mylib"
      library: my-lib
```

---

## Integration with Claude Code

### Step 1: Initialize in Runtime

Create `mcp_integration_runtime.py`:

```python
"""
MCP Integration Runtime Initialization

Initialize once at Claude Code startup.
"""

import asyncio
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


class MCPIntegrationRuntime:
    """Global MCP integration runtime."""

    def __init__(self):
        self.hierarchy = None
        self.lib_cache = None
        self.profile_manager = None
        self.smart_trigger = None
        self.learner = None
        self.preloader = None
        self.monitor = None

    async def initialize(
        self,
        memory_bank_mcp,
        serena_mcp,
        context7_mcp,
        github_mcp=None,
        config_dir="/path/to/config"
    ):
        """
        Initialize all components.

        Args:
            *_mcp: MCP server instances
            config_dir: Configuration directory
        """
        print("[MCP Integration] Initializing...")

        # Create adapters
        adapters = await create_mcp_adapters(
            memory_bank_mcp=memory_bank_mcp,
            serena_mcp=serena_mcp,
            context7_mcp=context7_mcp,
            github_mcp=github_mcp,
        )

        # Initialize components
        self.hierarchy = await KnowledgeHierarchy.create(
            memory_bank=adapters.get('memory-bank'),
            serena=adapters.get('serena'),
            context7=adapters.get('context7'),
            github=adapters.get('github'),
        )

        self.lib_cache = await LibraryCache.create(
            f"{config_dir}/library-cache.yaml",
            context7_mcp=adapters.get('context7')
        )

        self.profile_manager = await MCPProfileManager.create(
            f"{config_dir}/mcp-profiles.yaml",
            mcp_factory=self._create_mcp_instance
        )

        self.smart_trigger = await SmartTrigger.create(
            f"{config_dir}/mcp-config.yaml"
        )

        self.learner = await LearningSystem.create(
            memory_bank=adapters.get('memory-bank')
        )

        self.preloader = await PredictivePreloader.create(
            profile_manager=self.profile_manager,
            learning_system=self.learner
        )

        self.monitor = await Monitor.create()

        # Start background tasks
        await self.preloader.start_background_preloading()

        print("[MCP Integration] ✓ Initialized")

    async def _create_mcp_instance(self, mcp_config):
        """Factory for creating MCP instances."""
        # This would be implemented by Claude Code runtime
        # to create actual MCP instances
        pass

    async def shutdown(self):
        """Shutdown all components."""
        print("[MCP Integration] Shutting down...")

        if self.preloader:
            await self.preloader.stop_background_preloading()

        if self.monitor:
            await self.monitor.stop()

        print("[MCP Integration] ✓ Shutdown complete")


# Global instance
mcp_runtime = MCPIntegrationRuntime()
```

### Step 2: Initialize at Startup

In Claude Code's main runtime initialization:

```python
# claude_code_runtime.py

from mcp_integration_runtime import mcp_runtime

async def initialize_claude_code():
    # ... existing initialization ...

    # Initialize MCP integration
    await mcp_runtime.initialize(
        memory_bank_mcp=memory_bank,
        serena_mcp=serena,
        context7_mcp=context7,
        github_mcp=github,
        config_dir="/path/to/config"
    )

    # ... rest of initialization ...
```

---

## Command Integration

### Pattern 1: Simple Integration (Read-Only)

For commands that only need to fetch knowledge:

```python
# In /fix command handler

from mcp_integration_runtime import mcp_runtime

async def execute_fix_command(error_context):
    # Get cached solution
    result = await mcp_runtime.hierarchy.fetch(
        query=f"error:{error_context['type']}",
        context_type="error"
    )

    if result.success:
        print(f"Found solution from {result.source.value}")
        return result.content

    # No cached solution, analyze with serena
    # ...
```

### Pattern 2: Full Integration (Read + Write + Learn)

For commands that should learn and improve:

```python
# In /fix command handler

from mcp_integration_runtime import mcp_runtime
from mcp_integration.learning_system import OutcomeType

async def execute_fix_command(error_context):
    # 1. Activate profile
    profile = await mcp_runtime.profile_manager.activate_for_command("fix")

    # 2. Track MCP call
    async with mcp_runtime.monitor.track_mcp_call("memory-bank", "error"):
        solution = await mcp_runtime.hierarchy.fetch(
            query=f"error:{error_context['type']}",
            context_type="error"
        )

    # 3. Apply fix
    fix_result = apply_fix(solution)

    # 4. Track learning
    outcome = OutcomeType.SUCCESS if fix_result.success else OutcomeType.FAILURE
    await mcp_runtime.learner.track_success(
        query=f"error:{error_context['type']}",
        mcps_used=['memory-bank', 'serena'],
        outcome=outcome,
        latency_ms=solution.latency_ms
    )

    # 5. Cache solution if successful
    if fix_result.success:
        await mcp_runtime.hierarchy.memory_bank.store(
            key=f"error:{error_context['type']}:solution",
            value=solution.content,
            ttl=90 * 24 * 3600  # 90 days
        )

    return fix_result
```

### Pattern 3: Predictive Integration

For commands that benefit from preloading:

```python
# Before command execution (proactive)

async def before_command_execution(command_name, context):
    # Predictively preload MCPs
    await mcp_runtime.preloader.preload_for_command(
        command=command_name,
        context=context
    )

# Then execute command normally
# MCPs are already loaded, reducing latency
```

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest mcp_integration/tests/ -v

# Run with coverage
pytest mcp_integration/tests/ --cov=mcp_integration --cov-report=html

# Run specific test
pytest mcp_integration/tests/test_library_cache.py -v
```

### Integration Tests

```python
# test_integration.py

import asyncio
from mcp_integration_runtime import mcp_runtime

async def test_full_integration():
    # Initialize with mock MCPs
    await mcp_runtime.initialize(
        memory_bank_mcp=MockMemoryBank(),
        serena_mcp=MockSerena(),
        context7_mcp=MockContext7()
    )

    # Test hierarchy
    result = await mcp_runtime.hierarchy.fetch("test query")
    assert result is not None

    # Test library cache
    lib_id = await mcp_runtime.lib_cache.get_library_id("numpy")
    assert lib_id == "/numpy/numpy"

    # Test learning
    await mcp_runtime.learner.track_success(
        query="test",
        mcps_used=['memory-bank'],
        outcome=OutcomeType.SUCCESS
    )

    # Cleanup
    await mcp_runtime.shutdown()

asyncio.run(test_full_integration())
```

### Load Testing

```python
# test_performance.py

import asyncio
import time

async def load_test():
    """Simulate 1000 concurrent requests."""
    tasks = []

    for i in range(1000):
        task = mcp_runtime.hierarchy.fetch(f"query_{i}")
        tasks.append(task)

    start = time.time()
    results = await asyncio.gather(*tasks)
    duration = time.time() - start

    successful = sum(1 for r in results if r.success)
    print(f"Completed 1000 requests in {duration:.2f}s")
    print(f"Success rate: {successful/1000:.1%}")
    print(f"Throughput: {1000/duration:.0f} req/s")

asyncio.run(load_test())
```

---

## Monitoring

### 1. Enable Metrics Collection

```python
# In command execution

from mcp_integration_runtime import mcp_runtime

# Track MCP calls
async with mcp_runtime.monitor.track_mcp_call("context7", "library_api"):
    result = await context7.fetch(...)

# Record custom metrics
mcp_runtime.monitor.record_metric(
    name="command.fix.duration",
    value=duration_ms,
    tags={'outcome': 'success'},
    unit="ms"
)
```

### 2. Export Dashboard Data

```python
# Export metrics every hour

import asyncio

async def export_metrics_loop():
    while True:
        await asyncio.sleep(3600)  # 1 hour

        # Export to file
        mcp_runtime.monitor.export_metrics(
            filepath="/var/log/claude-code/mcp-metrics.json",
            format="json"
        )

        # Print summary
        dashboard = mcp_runtime.monitor.get_dashboard_data()
        print(f"Total calls: {dashboard['summary']['total_calls']}")
        print(f"Error rate: {dashboard['summary']['error_rate']:.1%}")
        print(f"Cost: ${dashboard['summary']['total_cost_usd']:.4f}")
```

### 3. Alert Handling

```python
# Check for alerts

alerts = mcp_runtime.monitor.get_alerts(severity=AlertSeverity.ERROR)
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.message}")

    # Send to logging system
    # send_to_slack(alert)
    # send_to_pagerduty(alert)
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'mcp_integration'`

**Solution:**
```bash
# Add to PYTHONPATH
export PYTHONPATH=/Users/b80985/.claude/commands:$PYTHONPATH

# Or install package
pip install -e /Users/b80985/.claude/commands/mcp_integration/
```

#### 2. Configuration Not Found

**Problem:** `FileNotFoundError: Config file not found: mcp-config.yaml`

**Solution:**
```python
# Use absolute paths
config_path = "/absolute/path/to/mcp-config.yaml"
trigger = await SmartTrigger.create(config_path)
```

#### 3. Memory-Bank Connection Issues

**Problem:** Memory-bank fetch/store failures

**Solution:**
```python
# Check MCP connection
try:
    result = await memory_bank.memory_bank_read(
        projectName="test",
        fileName="test.json"
    )
    print("✓ Memory-bank connected")
except Exception as e:
    print(f"✗ Memory-bank error: {e}")
```

#### 4. High Latency

**Problem:** Slow query performance

**Solution:**
```python
# Check hierarchy stats
stats = mcp_runtime.hierarchy.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")

# If cache hit rate < 80%:
# 1. Increase cache TTL
# 2. Preload common patterns
# 3. Adjust authority rules
```

---

## Performance Tuning

### 1. Cache Optimization

```yaml
# mcp-config.yaml - Increase TTLs

memory-bank:
  cache:
    ttl_by_type:
      error_solutions: 15552000  # 180 days (was 90)
      test_stability: 10368000   # 120 days (was 60)
```

### 2. Preloading Strategy

```python
# Aggressive preloading for high-traffic commands

preloader = await PredictivePreloader.create(
    profile_manager=manager,
    strategy=PreloadStrategy.AGGRESSIVE  # vs BALANCED
)
```

### 3. Parallel MCP Loading

```yaml
# mcp-profiles.yaml - Enable parallel for all profiles

profiles:
  code-analysis:
    parallel_init: true  # Load MCPs in parallel
```

### 4. Connection Pooling

```python
# Reuse MCP connections

class MCPConnectionPool:
    def __init__(self, max_connections=10):
        self.pool = {}
        self.max_connections = max_connections

    async def get_connection(self, mcp_name):
        if mcp_name not in self.pool:
            self.pool[mcp_name] = await create_mcp_connection(mcp_name)
        return self.pool[mcp_name]
```

---

## Rollback Procedures

### Quick Rollback

If issues occur, disable MCP integration:

```python
# Set flag in runtime
MCP_INTEGRATION_ENABLED = False

# In command handlers
if MCP_INTEGRATION_ENABLED:
    # Use MCP integration
    result = await mcp_runtime.hierarchy.fetch(...)
else:
    # Use original implementation
    result = await legacy_fetch(...)
```

### Gradual Rollout

Deploy to percentage of users:

```python
import random

def should_use_mcp_integration(user_id):
    # 20% rollout
    return hash(user_id) % 100 < 20

# In command
if should_use_mcp_integration(user.id):
    result = await mcp_runtime.hierarchy.fetch(...)
else:
    result = await legacy_implementation(...)
```

### Complete Removal

```bash
# Remove package
pip uninstall mcp-integration

# Remove config files
rm /path/to/config/mcp-*.yaml
rm /path/to/config/library-cache.yaml

# Revert command modifications
git checkout -- commands/
```

---

## Production Checklist

Before deploying to production:

- [ ] All tests passing (94%+ coverage)
- [ ] Configuration files in place
- [ ] MCP servers verified working
- [ ] Monitoring enabled
- [ ] Alert thresholds configured
- [ ] Rollback procedure tested
- [ ] Performance benchmarks met
- [ ] Documentation reviewed
- [ ] Team trained on new system
- [ ] Gradual rollout plan ready

---

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review Phase 2 & 3 summary documents
3. Check GitHub issues (if applicable)
4. Contact Claude Code team

---

**Deployment Guide Version:** 1.0.0
**Last Updated:** 2025-10-04
**Status:** ✅ Production-Ready
