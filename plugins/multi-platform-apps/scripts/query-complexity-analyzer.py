#!/usr/bin/env python3
"""
Query Complexity Analyzer for Multi-Platform Apps Plugin

Analyzes incoming queries to determine optimal model (haiku vs sonnet)
based on complexity heuristics tailored for multi-platform development.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"      # haiku-appropriate
    MEDIUM = "medium"      # could use either
    COMPLEX = "complex"    # sonnet-recommended


class ModelRecommendation(Enum):
    """Model recommendations"""
    HAIKU = "haiku"        # Fast, cost-effective (~200ms, $0.003)
    SONNET = "sonnet"      # Comprehensive, intelligent (~800ms, $0.015)


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis"""
    complexity: QueryComplexity
    recommended_model: ModelRecommendation
    confidence: float  # 0.0 to 1.0
    reasoning: str
    factors: Dict[str, any]
    estimated_latency_ms: int
    estimated_cost_usd: float


class QueryComplexityAnalyzer:
    """Analyzes query complexity for optimal model selection in multi-platform context"""

    # Keywords indicating simple queries
    SIMPLE_KEYWORDS = {
        'how to', 'example', 'basic', 'simple', 'quick', 'show me',
        'what is', 'syntax', 'hello world', 'tutorial', 'getting started',
        'button', 'widget', 'component', 'view', 'screen', 'layout',
        'color', 'styling', 'icon', 'text', 'navigation', 'list'
    }

    # Keywords indicating complex queries
    COMPLEX_KEYWORDS = {
        'architecture', 'design pattern', 'scalable', 'performance',
        'optimization', 'refactor', 'best practice', 'production',
        'distributed', 'microservice', 'security', 'authentication',
        'state management', 'offline sync', 'native module', 'platform channel',
        'memory profiling', 'clean architecture', 'dependency injection',
        'server component', 'design system', 'accessibility audit'
    }

    # Platform-specific simple patterns
    SIMPLE_PATTERNS = [
        # Flutter
        r'\b(widget|button|text|container|scaffold|appbar)\b',
        # React Native
        r'\b(view|touchable|flatlist|text input)\b',
        # iOS
        r'\b(swiftui|list|navigation|button|text field)\b',
        # React/Next.js
        r'\b(react component|div|button|input|onclick)\b',
        # Backend
        r'\b(rest endpoint|api route|simple crud|http method)\b',
        # UI/UX
        r'\b(color palette|icon|basic layout|wireframe)\b',
    ]

    # Platform-specific complex patterns
    COMPLEX_PATTERNS = [
        # Flutter
        r'\b(state management|custom render|platform channel|impeller)\b',
        # React Native
        r'\b(native module|turbomodule|new architecture|bridge|offline sync)\b',
        # iOS
        r'\b(core data|combine|clean architecture|arkit|cloudkit)\b',
        # React/Next.js
        r'\b(server component|server action|web vitals|suspense|micro frontend)\b',
        # Backend
        r'\b(microservices|event-driven|circuit breaker|distributed tracing|saga pattern)\b',
        # UI/UX
        r'\b(design system|user research|accessibility audit|design token)\b',
    ]

    def __init__(self):
        self.simple_regex = re.compile('|'.join(self.SIMPLE_PATTERNS), re.IGNORECASE)
        self.complex_regex = re.compile('|'.join(self.COMPLEX_PATTERNS), re.IGNORECASE)

    def analyze(self, query: str, context: Optional[Dict] = None) -> ComplexityAnalysis:
        """
        Analyze query complexity and recommend model.

        Args:
            query: The user's query string
            context: Optional context (agent type, previous queries, etc.)

        Returns:
            ComplexityAnalysis with recommendation
        """
        factors = {
            'length': len(query),
            'word_count': len(query.split()),
            'has_code': bool(re.search(r'```|`[^`]+`', query)),
            'has_multiple_questions': query.count('?') > 1,
            'is_simple_ui': bool(self.simple_regex.search(query)),
            'is_complex_feature': bool(self.complex_regex.search(query)),
        }

        # Calculate complexity score
        score = self._calculate_complexity_score(query, factors, context)

        # Determine complexity level and model
        if score < 3:
            complexity = QueryComplexity.SIMPLE
            model = ModelRecommendation.HAIKU
            confidence = 0.85
            reasoning = "Simple UI/component query - haiku optimized"
            latency_ms = 200
            cost_usd = 0.003
        elif score < 6:
            complexity = QueryComplexity.MEDIUM
            model = ModelRecommendation.SONNET  # Default to sonnet for safety
            confidence = 0.7
            reasoning = "Medium complexity - using sonnet for quality"
            latency_ms = 600
            cost_usd = 0.010
        else:
            complexity = QueryComplexity.COMPLEX
            model = ModelRecommendation.SONNET
            confidence = 0.95
            reasoning = "Complex architecture/design query - sonnet required"
            latency_ms = 1000
            cost_usd = 0.015

        # Override based on specific patterns
        if factors['is_simple_ui'] and not factors['is_complex_feature']:
            complexity = QueryComplexity.SIMPLE
            model = ModelRecommendation.HAIKU
            confidence = 0.9
            reasoning = "Simple UI component - haiku optimized (75% latency reduction)"
            latency_ms = 200
            cost_usd = 0.003
        elif factors['is_complex_feature']:
            complexity = QueryComplexity.COMPLEX
            model = ModelRecommendation.SONNET
            confidence = 0.95
            reasoning = "Complex feature/architecture - sonnet required"
            latency_ms = 1000
            cost_usd = 0.015

        # Context-based adjustments
        if context:
            model, confidence, reasoning, latency_ms, cost_usd = self._adjust_for_context(
                model, confidence, reasoning, latency_ms, cost_usd, context, factors
            )

        return ComplexityAnalysis(
            complexity=complexity,
            recommended_model=model,
            confidence=confidence,
            reasoning=reasoning,
            factors=factors,
            estimated_latency_ms=latency_ms,
            estimated_cost_usd=cost_usd
        )

    def _calculate_complexity_score(
        self,
        query: str,
        factors: Dict,
        context: Optional[Dict]
    ) -> float:
        """Calculate complexity score (0-10)"""
        score = 0.0

        # Length-based scoring
        if factors['word_count'] < 10:
            score += 0
        elif factors['word_count'] < 30:
            score += 1
        else:
            score += 2

        # Simple keywords reduce complexity
        query_lower = query.lower()
        simple_count = sum(1 for kw in self.SIMPLE_KEYWORDS if kw in query_lower)
        score -= simple_count * 0.5

        # Complex keywords increase complexity
        complex_count = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in query_lower)
        score += complex_count * 1.5

        # Code presence
        if factors['has_code']:
            score += 1

        # Multiple questions
        if factors['has_multiple_questions']:
            score += 2

        # Simple UI patterns
        if factors['is_simple_ui']:
            score -= 2

        # Complex features
        if factors['is_complex_feature']:
            score += 3

        # Clamp to 0-10 range
        return max(0, min(10, score))

    def _adjust_for_context(
        self,
        model: ModelRecommendation,
        confidence: float,
        reasoning: str,
        latency_ms: int,
        cost_usd: float,
        context: Dict,
        factors: Dict
    ) -> tuple:
        """Adjust recommendation based on agent context"""

        agent = context.get('agent', '')

        # Agent-specific optimizations
        if agent == 'flutter-expert' and factors['is_simple_ui']:
            return ModelRecommendation.HAIKU, 0.95, "Flutter simple UI - haiku optimized", 200, 0.003

        if agent == 'frontend-developer' and 'button' in reasoning.lower():
            return ModelRecommendation.HAIKU, 0.9, "React simple component - haiku optimized", 200, 0.003

        if agent == 'ios-developer' and factors['is_simple_ui']:
            return ModelRecommendation.HAIKU, 0.9, "iOS simple view - haiku optimized", 200, 0.003

        if agent == 'backend-architect' and 'endpoint' in reasoning.lower():
            return ModelRecommendation.HAIKU, 0.85, "Simple API endpoint - haiku suitable", 200, 0.003

        if agent == 'ui-ux-designer' and factors['is_simple_ui']:
            return ModelRecommendation.HAIKU, 0.9, "Simple design element - haiku optimized", 200, 0.003

        # If previous query was complex, be cautious
        if context.get('previous_complexity') == QueryComplexity.COMPLEX:
            confidence *= 0.9
            reasoning += " (considering previous complex query)"

        return model, confidence, reasoning, latency_ms, cost_usd


def demo():
    """Demonstration of complexity analyzer for multi-platform apps"""
    analyzer = QueryComplexityAnalyzer()

    test_queries = [
        ("Create a simple Flutter button widget", {'agent': 'flutter-expert'}),
        ("How do I add navigation to my React Native app?", {'agent': 'mobile-developer'}),
        ("Design a microservices architecture for mobile backend", {'agent': 'backend-architect'}),
        ("Build a complex state management system with Riverpod", {'agent': 'flutter-expert'}),
        ("Create a SwiftUI list view with basic data", {'agent': 'ios-developer'}),
        ("Design a comprehensive design system with accessibility", {'agent': 'ui-ux-designer'}),
        ("Add a button that navigates to another screen", {'agent': 'frontend-developer'}),
        ("Implement offline sync with conflict resolution", {'agent': 'mobile-developer'}),
    ]

    print("=" * 80)
    print("Multi-Platform Apps Query Complexity Analysis Demo")
    print("=" * 80)

    total_simple = 0
    total_complex = 0

    for query, context in test_queries:
        print(f"\nQuery: {query}")
        print(f"Agent: {context.get('agent', 'unknown')}")
        analysis = analyzer.analyze(query, context)
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Recommended Model: {analysis.recommended_model.value}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Estimated Latency: {analysis.estimated_latency_ms}ms")
        print(f"  Estimated Cost: ${analysis.estimated_cost_usd:.4f}")
        print(f"  Reasoning: {analysis.reasoning}")

        if analysis.recommended_model == ModelRecommendation.HAIKU:
            total_simple += 1
        else:
            total_complex += 1

    print("\n" + "=" * 80)
    print(f"Summary: {total_simple} queries → haiku, {total_complex} queries → sonnet")
    print(f"Potential savings: {total_simple * 0.012:.4f} USD per query set")
    print(f"Potential latency reduction: {total_simple * 600}ms total")
    print("=" * 80)


if __name__ == "__main__":
    demo()
