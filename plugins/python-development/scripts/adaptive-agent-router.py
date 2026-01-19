#!/usr/bin/env python3
"""
Adaptive Agent Router - Integration of Phase 2 Optimizations

Combines:
1. Query complexity analysis
2. Response caching
3. Performance monitoring
4. Agent complexity hints

To route queries to the optimal model with caching and monitoring.
"""

import json
import time
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import Phase 2 components


def load_module(name: str, path: Path):
    """Dynamically load module from path"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules dynamically
qca_module = load_module("query_complexity_analyzer", SCRIPTS_DIR / "query-complexity-analyzer.py")
cache_module = load_module("response_cache", SCRIPTS_DIR / "response-cache.py")
monitor_module = load_module("performance_monitor", SCRIPTS_DIR / "performance-monitor.py")

QueryComplexityAnalyzer = qca_module.QueryComplexityAnalyzer
QueryComplexity = qca_module.QueryComplexity
ModelRecommendation = qca_module.ModelRecommendation
ResponseCache = cache_module.ResponseCache
PerformanceMonitor = monitor_module.PerformanceMonitor


@dataclass
class RouteDecision:
    """Routing decision with reasoning"""
    agent: str
    model: str
    use_cache: bool
    complexity: str
    confidence: float
    reasoning: str
    estimated_latency_ms: float


class AdaptiveAgentRouter:
    """
    Intelligent router that combines:
    - Query complexity analysis
    - Agent complexity hints
    - Response caching
    - Performance monitoring
    """

    def __init__(self, plugin_dir: Optional[Path] = None):
        """Initialize adaptive router"""
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent.parent

        self.plugin_dir = plugin_dir

        # Initialize components
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.cache = ResponseCache()
        self.monitor = PerformanceMonitor()

        # Load agent configurations
        self.agents = self._load_agents()

    def _load_agents(self) -> Dict:
        """Load agent configurations with complexity hints"""
        with open(self.plugin_dir / "plugin.json") as f:
            plugin_config = json.load(f)

        agents = {}
        for agent_config in plugin_config.get('agents', []):
            agent_name = agent_config['name']

            # Load agent file for complexity hints
            agent_file = self.plugin_dir / "agents" / f"{agent_name}.md"
            if agent_file.exists():
                agents[agent_name] = self._parse_agent_file(agent_file)
            else:
                agents[agent_name] = agent_config

        return agents

    def _parse_agent_file(self, agent_file: Path) -> Dict:
        """Parse agent markdown file for complexity hints"""
        content = agent_file.read_text()

        # Extract YAML frontmatter
        if not content.startswith('---'):
            return {}

        end_marker = content.find('---', 3)
        if end_marker == -1:
            return {}

        frontmatter = content[3:end_marker]

        # Simple YAML parsing for our specific structure
        # (In production, use proper YAML parser)
        config = {}
        lines = frontmatter.split('\n')

        _ = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('name:'):
                config['name'] = line.split(':', 1)[1].strip()
            elif line.startswith('model:'):
                config['model'] = line.split(':', 1)[1].strip()
            elif line.startswith('complexity_hints:'):
                config['complexity_hints'] = {}
                _ = 'hints'

        return config

    def route_query(
        self,
        query: str,
        agent: str,
        context: Optional[Dict] = None
    ) -> RouteDecision:
        """
        Route query to optimal model with caching consideration

        Args:
            query: User query
            agent: Target agent name
            context: Optional context

        Returns:
            RouteDecision with routing details
        """
        # Check cache first
        cached_response = self.cache.get(query, agent, context)
        if cached_response:
            return RouteDecision(
                agent=agent,
                model="cached",
                use_cache=True,
                complexity="cached",
                confidence=1.0,
                reasoning="Response found in cache",
                estimated_latency_ms=5.0  # Cache hit is ~5ms
            )

        # Analyze query complexity
        analysis = self.complexity_analyzer.analyze(query, context)

        # Get agent configuration
        agent_config = self.agents.get(agent, {})
        _ = agent_config.get('model', 'sonnet')

        # Determine optimal model
        recommended_model = analysis.recommended_model.value

        # Check if agent has complexity hints that override
        complexity_hints = agent_config.get('complexity_hints', {})
        if complexity_hints:
            complexity_level = analysis.complexity.value
            hint = complexity_hints.get(f"{complexity_level}_queries", {})
            if hint and 'model' in hint:
                recommended_model = hint['model']
                estimated_latency = hint.get('latency_target_ms', 500)
            else:
                estimated_latency = self._estimate_latency(recommended_model)
        else:
            estimated_latency = self._estimate_latency(recommended_model)

        return RouteDecision(
            agent=agent,
            model=recommended_model,
            use_cache=False,
            complexity=analysis.complexity.value,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
            estimated_latency_ms=estimated_latency
        )

    def _estimate_latency(self, model: str) -> float:
        """Estimate latency based on model"""
        latency_map = {
            'haiku': 200.0,
            'sonnet': 800.0,
            'opus': 1500.0,
        }
        return latency_map.get(model, 500.0)

    def execute_query(
        self,
        query: str,
        agent: str,
        context: Optional[Dict] = None
    ) -> Tuple[str, RouteDecision]:
        """
        Execute query with full optimization pipeline

        Args:
            query: User query
            agent: Target agent
            context: Optional context

        Returns:
            (response, route_decision)
        """
        # Route the query
        decision = self.route_query(query, agent, context)

        # If cached, return immediately
        if decision.use_cache:
            cached_response = self.cache.get(query, agent, context)
            self.monitor.log_metric(
                agent=agent,
                model="cached",
                query_complexity=decision.complexity,
                response_time_ms=5.0,
                cache_hit=True
            )
            return cached_response, decision

        # Simulate query execution
        start_time = time.perf_counter()

        # In production, this would actually call the agent
        response = self._simulate_agent_call(query, decision)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Cache the response
        self.cache.set(
            query=query,
            response=response,
            agent=agent,
            model=decision.model,
            context=context
        )

        # Log performance metric
        self.monitor.log_metric(
            agent=agent,
            model=decision.model,
            query_complexity=decision.complexity,
            response_time_ms=elapsed_ms,
            cache_hit=False
        )

        return response, decision

    def _simulate_agent_call(self, query: str, decision: RouteDecision) -> str:
        """Simulate agent execution (replace with actual agent call)"""
        # Simulate latency
        time.sleep(decision.estimated_latency_ms / 1000.0)

        return f"Response for '{query[:50]}...' using {decision.model}"

    def get_performance_summary(self) -> Dict:
        """Get performance summary across all queries"""
        return self.monitor.analyze_performance(hours=24)

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()


def demo():
    """Comprehensive demo of adaptive routing"""
    print("=" * 80)
    print("Adaptive Agent Router Demo")
    print("=" * 80)

    router = AdaptiveAgentRouter()

    # Test scenarios
    scenarios = [
        ("How do I create a basic FastAPI endpoint?", "fastapi-pro"),
        ("Design a scalable microservices architecture", "fastapi-pro"),
        ("How do I create a basic FastAPI endpoint?", "fastapi-pro"),  # Should hit cache
        ("What are Python decorators?", "python-pro"),
        ("Optimize this async code for performance", "python-pro"),
        ("Create a simple Django model", "django-pro"),
        ("Build a multi-tenant Django architecture", "django-pro"),
    ]

    print("\n--- Executing Queries ---\n")
    for query, agent in scenarios:
        response, decision = router.execute_query(query, agent)

        print(f"Query: {query[:50]}...")
        print(f"  Agent: {decision.agent}")
        print(f"  Model: {decision.model}")
        print(f"  Complexity: {decision.complexity}")
        print(f"  Cache: {'HIT' if decision.use_cache else 'MISS'}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
        print(f"  Est. Latency: {decision.estimated_latency_ms:.0f}ms")
        print()

    # Performance summary
    print("=" * 80)
    print("Performance Summary")
    print("=" * 80)

    perf_summary = router.get_performance_summary()
    if perf_summary['status'] == 'OK':
        print(f"\nTotal Queries: {perf_summary['total_queries']}")
        print(f"Cache Hit Rate: {perf_summary['cache']['hit_rate']:.1%}")
        print(f"Cache Status: {perf_summary['cache']['status']}")

        print("\n--- By Model ---")
        for model, stats in perf_summary.get('by_model', {}).items():
            print(f"\n{model.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg: {stats['avg_ms']:.0f}ms")
            print(f"  P95: {stats['p95_ms']:.0f}ms")

        print("\n--- By Agent ---")
        for agent, count in perf_summary['by_agent'].items():
            print(f"  {agent}: {count}")

    # Cache statistics
    print("\n" + "=" * 80)
    print("Cache Statistics")
    print("=" * 80)

    cache_stats = router.get_cache_stats()
    print(f"\nEntries: {cache_stats['entries']}")
    print(f"Hits: {cache_stats['hits']}")
    print(f"Misses: {cache_stats['misses']}")
    print(f"Hit Rate: {cache_stats['hit_rate']:.1%}")

    # Savings calculation
    if cache_stats['hits'] > 0:
        avg_latency_saved = 500  # Assuming avg 500ms per query
        total_saved_ms = cache_stats['hits'] * avg_latency_saved
        print(f"\nEstimated Time Saved: {total_saved_ms:.0f}ms ({total_saved_ms/1000:.1f}s)")
        print(f"Cost Reduction: ~{cache_stats['hits'] * 0.015:.3f} USD (assuming $0.015/query)")

    print("\n✓ Demo complete")


if __name__ == "__main__":
    demo()
