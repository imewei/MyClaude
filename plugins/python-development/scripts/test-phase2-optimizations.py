#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 2 Optimizations

Tests:
1. Query complexity analyzer accuracy
2. Response caching effectiveness
3. Performance monitoring accuracy
4. Agent complexity hints parsing
5. Adaptive router integration
"""

import sys
import time
import importlib.util
from pathlib import Path
from typing import List

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import Phase 2 modules


def load_module(name: str, path: Path):
    """Dynamically load module"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
qca_module = load_module("qca", SCRIPTS_DIR / "query-complexity-analyzer.py")
cache_module = load_module("cache", SCRIPTS_DIR / "response-cache.py")
monitor_module = load_module("monitor", SCRIPTS_DIR / "performance-monitor.py")

QueryComplexityAnalyzer = qca_module.QueryComplexityAnalyzer
QueryComplexity = qca_module.QueryComplexity
ResponseCache = cache_module.ResponseCache
PerformanceMonitor = monitor_module.PerformanceMonitor


class TestResult:
    """Test result tracker"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures: List[str] = []

    def assert_true(self, condition: bool, message: str):
        """Assert condition is true"""
        if condition:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.failures.append(message)
            print(f"  ✗ {message}")

    def assert_equal(self, actual, expected, message: str):
        """Assert values are equal"""
        if actual == expected:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.failures.append(f"{message} (expected: {expected}, got: {actual})")
            print(f"  ✗ {message} (expected: {expected}, got: {actual})")

    def assert_in_range(self, value: float, min_val: float, max_val: float, message: str):
        """Assert value is in range"""
        if min_val <= value <= max_val:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.failures.append(f"{message} (value: {value}, range: [{min_val}, {max_val}])")
            print(f"  ✗ {message} (value: {value}, range: [{min_val}, {max_val}])")

    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if self.failures:
            print("\n--- Failures ---")
            for failure in self.failures:
                print(f"  ✗ {failure}")

        print("\n" + ("✓ ALL TESTS PASSED" if self.failed == 0 else "✗ SOME TESTS FAILED"))
        return self.failed == 0


def test_query_complexity_analyzer(results: TestResult):
    """Test query complexity analyzer"""
    print("\n=== Test: Query Complexity Analyzer ===")

    analyzer = QueryComplexityAnalyzer()

    # Test simple queries
    simple_queries = [
        "How do I create a FastAPI endpoint?",
        "Show me a hello world example",
        "What is the basic syntax?",
        "Install Django"
    ]

    for query in simple_queries:
        analysis = analyzer.analyze(query)
        results.assert_equal(
            analysis.complexity.value,
            "simple",
            f"Simple query classified correctly: '{query[:30]}...'"
        )
        results.assert_equal(
            analysis.recommended_model.value,
            "haiku",
            f"Simple query recommends haiku: '{query[:30]}...'"
        )

    # Test complex queries
    complex_queries = [
        "Design a scalable microservices architecture",
        "Debug this complex async race condition",
        "Compare authentication strategies for distributed systems",
        "Analyze performance bottlenecks"
    ]

    for query in complex_queries:
        analysis = analyzer.analyze(query)
        results.assert_equal(
            analysis.complexity.value,
            "complex",
            f"Complex query classified correctly: '{query[:30]}...'"
        )
        results.assert_equal(
            analysis.recommended_model.value,
            "sonnet",
            f"Complex query recommends sonnet: '{query[:30]}...'"
        )


def test_response_cache(results: TestResult):
    """Test response caching"""
    print("\n=== Test: Response Cache ===")

    cache = ResponseCache()
    cache.clear_all()  # Start fresh

    # Test cache miss
    result = cache.get("test query", "test-agent")
    results.assert_true(result is None, "Cache miss returns None")

    # Test cache set and hit
    cache.set("test query", "test response", "test-agent", "haiku")
    result = cache.get("test query", "test-agent")
    results.assert_equal(result, "test response", "Cache hit returns correct response")

    # Test cache stats
    stats = cache.get_stats()
    results.assert_equal(stats['hits'], 1, "Cache stats show 1 hit")
    results.assert_equal(stats['misses'], 1, "Cache stats show 1 miss")
    results.assert_equal(stats['entries'], 1, "Cache stats show 1 entry")
    results.assert_equal(stats['hit_rate'], 0.5, "Cache hit rate is 50%")

    # Test cache invalidation
    cache.invalidate_agent("test-agent")
    result = cache.get("test query", "test-agent")
    results.assert_true(result is None, "Invalidated cache entry returns None")


def test_performance_monitor(results: TestResult):
    """Test performance monitoring"""
    print("\n=== Test: Performance Monitor ===")

    monitor = PerformanceMonitor()

    # Log some metrics
    monitor.log_metric("test-agent", "haiku", "simple", 150.0, cache_hit=False)
    monitor.log_metric("test-agent", "haiku", "simple", 160.0, cache_hit=False)
    monitor.log_metric("test-agent", "sonnet", "complex", 800.0, cache_hit=False)
    monitor.log_metric("test-agent", "haiku", "simple", 5.0, cache_hit=True)

    # Analyze performance
    analysis = monitor.analyze_performance(hours=24)

    results.assert_equal(analysis['status'], 'OK', "Analysis returns OK status")
    results.assert_true(analysis['total_queries'] >= 4, "Analysis counts all queries")

    # Check model breakdown
    if 'by_model' in analysis and 'haiku' in analysis['by_model']:
        haiku_stats = analysis['by_model']['haiku']
        results.assert_true(haiku_stats['count'] >= 3, "Haiku query count correct")
        results.assert_in_range(
            haiku_stats['avg_ms'],
            0.0,
            200.0,
            "Haiku avg latency in expected range"
        )

    if 'by_model' in analysis and 'sonnet' in analysis['by_model']:
        sonnet_stats = analysis['by_model']['sonnet']
        results.assert_true(sonnet_stats['count'] >= 1, "Sonnet query count correct")

    # Check cache stats
    cache_stats = analysis['cache']
    results.assert_true(cache_stats['hit_rate'] > 0, "Cache hit rate is positive")


def test_agent_complexity_hints(results: TestResult):
    """Test agent complexity hints parsing"""
    print("\n=== Test: Agent Complexity Hints ===")

    plugin_dir = Path(__file__).parent.parent

    # Check agent files
    for agent_name in ['fastapi-pro', 'django-pro', 'python-pro']:
        agent_file = plugin_dir / "agents" / f"{agent_name}.md"
        results.assert_true(agent_file.exists(), f"{agent_name}.md exists")

        # Check for complexity_hints in frontmatter
        content = agent_file.read_text()
        results.assert_true(
            'complexity_hints:' in content,
            f"{agent_name} has complexity_hints"
        )
        results.assert_true(
            'simple_queries:' in content,
            f"{agent_name} has simple_queries hints"
        )
        results.assert_true(
            'complex_queries:' in content,
            f"{agent_name} has complex_queries hints"
        )


def test_end_to_end_optimization(results: TestResult):
    """Test end-to-end optimization pipeline"""
    print("\n=== Test: End-to-End Optimization ===")

    # Initialize components
    analyzer = QueryComplexityAnalyzer()
    cache = ResponseCache()
    monitor = PerformanceMonitor()

    cache.clear_all()

    # Simulate query pipeline
    query = "How do I create a FastAPI endpoint?"
    agent = "fastapi-pro"

    # Step 1: Analyze complexity
    analysis = analyzer.analyze(query)
    results.assert_equal(
        analysis.complexity.value,
        "simple",
        "End-to-end: Simple query classified"
    )

    # Step 2: Check cache (should miss)
    cached = cache.get(query, agent)
    results.assert_true(cached is None, "End-to-end: Cache miss on first request")

    # Step 3: Simulate response and cache
    response = "Create endpoint with @app.get decorator"
    start = time.perf_counter()
    time.sleep(analysis.recommended_model.value == "haiku" and 0.15 or 0.8)  # Simulate latency
    elapsed_ms = (time.perf_counter() - start) * 1000

    cache.set(query, response, agent, analysis.recommended_model.value)

    # Step 4: Log metrics
    monitor.log_metric(
        agent,
        analysis.recommended_model.value,
        analysis.complexity.value,
        elapsed_ms,
        cache_hit=False
    )

    # Step 5: Verify cache hit on repeat
    cached = cache.get(query, agent)
    results.assert_equal(
        cached,
        response,
        "End-to-end: Cache hit on repeat request"
    )

    # Verify full pipeline worked
    stats = cache.get_stats()
    results.assert_true(
        stats['hit_rate'] > 0,
        "End-to-end: Pipeline produced cache hits"
    )


def main():
    """Run all tests"""
    print("=" * 80)
    print("Phase 2 Optimization Test Suite")
    print("=" * 80)

    results = TestResult()

    # Run all test suites
    test_query_complexity_analyzer(results)
    test_response_cache(results)
    test_performance_monitor(results)
    test_agent_complexity_hints(results)
    test_end_to_end_optimization(results)

    # Print summary
    success = results.summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
