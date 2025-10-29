#!/usr/bin/env python3
"""
Query Complexity Analyzer for Adaptive Model Selection

Analyzes incoming queries to determine optimal model (haiku vs sonnet)
based on complexity heuristics.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"      # haiku-appropriate
    MEDIUM = "medium"      # could use either
    COMPLEX = "complex"    # sonnet-recommended


class ModelRecommendation(Enum):
    """Model recommendations"""
    HAIKU = "haiku"        # Fast, cost-effective
    SONNET = "sonnet"      # Comprehensive, intelligent
    OPUS = "opus"          # Maximum capability (rare)


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis"""
    complexity: QueryComplexity
    recommended_model: ModelRecommendation
    confidence: float  # 0.0 to 1.0
    reasoning: str
    factors: Dict[str, any]


class QueryComplexityAnalyzer:
    """Analyzes query complexity for optimal model selection"""

    # Keywords indicating simple queries
    SIMPLE_KEYWORDS = {
        'how to', 'example', 'basic', 'simple', 'quick', 'show me',
        'what is', 'syntax', 'hello world', 'tutorial', 'getting started',
        'install', 'setup', 'create', 'list', 'get', 'find'
    }

    # Keywords indicating complex queries
    COMPLEX_KEYWORDS = {
        'architecture', 'design pattern', 'scalable', 'performance',
        'optimization', 'refactor', 'best practice', 'production',
        'distributed', 'microservice', 'security', 'authentication',
        'algorithm', 'complex', 'advanced', 'enterprise', 'explain why',
        'compare', 'trade-off', 'analyze', 'debug', 'troubleshoot'
    }

    # Patterns indicating CRUD operations (simple)
    CRUD_PATTERNS = [
        r'\b(create|read|update|delete|crud)\b',
        r'\b(get|post|put|patch|delete)\s+(endpoint|route|api)',
        r'\b(select|insert|update|delete)\s+.*\b(database|table|record)',
    ]

    # Patterns indicating architectural queries (complex)
    ARCHITECTURE_PATTERNS = [
        r'\b(design|architect|structure)\s+.*\b(system|application)',
        r'\b(scalable|distributed|microservice)',
        r'\b(performance|optimization|efficiency)',
        r'\b(security|authentication|authorization)\s+.*\b(strategy|pattern)',
    ]

    def __init__(self):
        self.crud_regex = re.compile('|'.join(self.CRUD_PATTERNS), re.IGNORECASE)
        self.arch_regex = re.compile('|'.join(self.ARCHITECTURE_PATTERNS), re.IGNORECASE)

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
            'is_crud': bool(self.crud_regex.search(query)),
            'is_architectural': bool(self.arch_regex.search(query)),
        }

        # Calculate complexity score
        score = self._calculate_complexity_score(query, factors, context)

        # Determine complexity level
        if score < 3:
            complexity = QueryComplexity.SIMPLE
            model = ModelRecommendation.HAIKU
            confidence = 0.8
            reasoning = "Query is simple/straightforward"
        elif score < 6:
            complexity = QueryComplexity.MEDIUM
            model = ModelRecommendation.SONNET  # Default to sonnet for medium
            confidence = 0.6
            reasoning = "Query has medium complexity, using sonnet for safety"
        else:
            complexity = QueryComplexity.COMPLEX
            model = ModelRecommendation.SONNET
            confidence = 0.9
            reasoning = "Query requires comprehensive analysis"

        # Override based on specific patterns
        if factors['is_crud'] and not factors['is_architectural']:
            complexity = QueryComplexity.SIMPLE
            model = ModelRecommendation.HAIKU
            confidence = 0.85
            reasoning = "CRUD operation - suitable for haiku"
        elif factors['is_architectural']:
            complexity = QueryComplexity.COMPLEX
            model = ModelRecommendation.SONNET
            confidence = 0.9
            reasoning = "Architectural query requires deep analysis"

        # Context-based adjustments
        if context:
            model, confidence, reasoning = self._adjust_for_context(
                model, confidence, reasoning, context, factors
            )

        return ComplexityAnalysis(
            complexity=complexity,
            recommended_model=model,
            confidence=confidence,
            reasoning=reasoning,
            factors=factors
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

        # CRUD operations are simple
        if factors['is_crud']:
            score -= 2

        # Architectural queries are complex
        if factors['is_architectural']:
            score += 3

        # Clamp to 0-10 range
        return max(0, min(10, score))

    def _adjust_for_context(
        self,
        model: ModelRecommendation,
        confidence: float,
        reasoning: str,
        context: Dict,
        factors: Dict
    ) -> tuple:
        """Adjust recommendation based on context"""

        # If agent is FastAPI and it's a simple CRUD, definitely use haiku
        if context.get('agent') == 'fastapi-pro' and factors['is_crud']:
            return ModelRecommendation.HAIKU, 0.95, "FastAPI CRUD - optimized for haiku"

        # If previous query was complex, be cautious
        if context.get('previous_complexity') == QueryComplexity.COMPLEX:
            confidence *= 0.9
            reasoning += " (considering previous complex query)"

        # If user explicitly requests fast response
        if any(word in context.get('preferences', []) for word in ['fast', 'quick']):
            if model == ModelRecommendation.SONNET and confidence < 0.7:
                return ModelRecommendation.HAIKU, confidence, reasoning + " (user prefers speed)"

        return model, confidence, reasoning


def demo():
    """Demonstration of complexity analyzer"""
    analyzer = QueryComplexityAnalyzer()

    test_queries = [
        "How do I create a FastAPI endpoint?",
        "Show me a simple hello world example",
        "Design a scalable microservices architecture for an e-commerce platform",
        "What's the best authentication strategy for a distributed system?",
        "Create a GET endpoint that returns user data",
        "Explain the trade-offs between different caching strategies",
        "How to install FastAPI?",
        "Debug this complex async race condition in my application",
    ]

    print("=" * 80)
    print("Query Complexity Analysis Demo")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: {query[:60]}...")
        analysis = analyzer.analyze(query)
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Recommended Model: {analysis.recommended_model.value}")
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Reasoning: {analysis.reasoning}")
        print(f"  Factors: word_count={analysis.factors['word_count']}, "
              f"is_crud={analysis.factors['is_crud']}, "
              f"is_arch={analysis.factors['is_architectural']}")


if __name__ == "__main__":
    demo()
