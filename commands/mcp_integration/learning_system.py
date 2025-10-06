"""
MCP Learning System

Adaptive learning system that tracks patterns, solutions, and usage to improve
MCP integration performance over time.

Features:
- Pattern learning and recognition
- Solution effectiveness tracking
- Query-to-MCP mapping optimization
- Confidence scoring
- Automatic pattern refinement

Example:
    >>> learner = await LearningSystem.create(
    ...     memory_bank=memory_bank_mcp,
    ...     min_confidence=0.7
    ... )
    >>>
    >>> # Track successful pattern
    >>> await learner.track_success(
    ...     query="How to use numpy.array?",
    ...     mcps_used=['context7'],
    ...     outcome='success'
    ... )
    >>>
    >>> # Get recommendations
    >>> recommendations = await learner.recommend_mcps("numpy question")
    >>> # Returns: ['context7'] with high confidence
"""

import time
import hashlib
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class OutcomeType(Enum):
    """Types of interaction outcomes."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class Pattern:
    """
    Learned pattern.

    Attributes:
        pattern: Regex pattern or keyword
        mcps: MCPs associated with this pattern
        confidence: Confidence score (0.0-1.0)
        success_count: Number of successful uses
        total_count: Total number of uses
        last_used: Last usage timestamp
        context_type: Context where pattern applies
    """
    pattern: str
    mcps: List[str]
    confidence: float = 0.5
    success_count: int = 0
    total_count: int = 0
    last_used: float = 0.0
    context_type: str = "general"

    def update_confidence(self, outcome: OutcomeType) -> None:
        """Update confidence based on outcome."""
        self.total_count += 1
        self.last_used = time.time()

        if outcome == OutcomeType.SUCCESS:
            self.success_count += 1
            # Increase confidence
            self.confidence = min(
                self.confidence + 0.05,
                0.95
            )
        elif outcome == OutcomeType.FAILURE:
            # Decrease confidence
            self.confidence = max(
                self.confidence - 0.1,
                0.1
            )
        elif outcome == OutcomeType.PARTIAL:
            # Slight increase
            self.confidence = min(
                self.confidence + 0.02,
                0.9
            )

        # Also calculate from success rate
        success_rate = self.success_count / self.total_count
        # Weighted average: 70% from incremental updates, 30% from success rate
        self.confidence = 0.7 * self.confidence + 0.3 * success_rate

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0


@dataclass
class QueryHistory:
    """
    Query execution history.

    Attributes:
        query: Original query
        mcps_used: MCPs that were activated
        outcome: Execution outcome
        latency_ms: Total latency
        timestamp: Execution timestamp
        context_type: Query context type
    """
    query: str
    mcps_used: List[str]
    outcome: OutcomeType
    latency_ms: int
    timestamp: float
    context_type: str = "general"


class LearningSystem:
    """
    Adaptive learning system for MCP integration.

    Tracks patterns, learns from outcomes, and provides recommendations
    to optimize MCP usage over time.

    Features:
    - Pattern learning and recognition
    - Success/failure tracking
    - Confidence scoring
    - Query-to-MCP mapping
    - Automatic pattern refinement
    """

    def __init__(
        self,
        memory_bank: Optional[Any] = None,
        min_confidence: float = 0.7,
        pattern_decay_days: int = 30,
        enable_auto_learning: bool = True,
    ):
        """
        Initialize learning system.

        Args:
            memory_bank: Memory-bank MCP for persistence
            min_confidence: Minimum confidence for recommendations
            pattern_decay_days: Days before unused patterns decay
            enable_auto_learning: Enable automatic pattern learning
        """
        self.memory_bank = memory_bank
        self.min_confidence = min_confidence
        self.pattern_decay_days = pattern_decay_days
        self.enable_auto_learning = enable_auto_learning

        # Learned patterns
        self.patterns: Dict[str, Pattern] = {}

        # Query history (in-memory, persisted to memory-bank)
        self.history: List[QueryHistory] = []

        # MCP effectiveness tracking
        self.mcp_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'success_count': 0,
                'total_count': 0,
                'avg_latency_ms': 0,
                'total_latency_ms': 0,
            }
        )

        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'patterns_learned': 0,
            'recommendations_made': 0,
        }

    @classmethod
    async def create(
        cls,
        memory_bank: Optional[Any] = None,
        **kwargs
    ) -> "LearningSystem":
        """
        Create learning system and load existing patterns.

        Args:
            memory_bank: Memory-bank MCP for persistence
            **kwargs: Additional configuration

        Returns:
            Initialized LearningSystem instance
        """
        system = cls(memory_bank=memory_bank, **kwargs)

        # Load existing patterns from memory-bank
        if memory_bank:
            await system._load_patterns()

        return system

    async def track_success(
        self,
        query: str,
        mcps_used: List[str],
        outcome: OutcomeType,
        latency_ms: int = 0,
        context_type: str = "general"
    ) -> None:
        """
        Track a query execution.

        Args:
            query: User query
            mcps_used: MCPs that were activated
            outcome: Execution outcome
            latency_ms: Total latency
            context_type: Query context type

        Example:
            >>> await learner.track_success(
            ...     query="How to use numpy.array?",
            ...     mcps_used=['context7'],
            ...     outcome=OutcomeType.SUCCESS,
            ...     latency_ms=450
            ... )
        """
        # Record history
        history = QueryHistory(
            query=query,
            mcps_used=mcps_used,
            outcome=outcome,
            latency_ms=latency_ms,
            timestamp=time.time(),
            context_type=context_type
        )
        self.history.append(history)

        # Update statistics
        self.stats['total_queries'] += 1
        if outcome == OutcomeType.SUCCESS:
            self.stats['successful_queries'] += 1

        # Update MCP stats
        for mcp in mcps_used:
            self.mcp_stats[mcp]['total_count'] += 1
            self.mcp_stats[mcp]['total_latency_ms'] += latency_ms

            if outcome == OutcomeType.SUCCESS:
                self.mcp_stats[mcp]['success_count'] += 1

            # Update average latency
            total = self.mcp_stats[mcp]['total_count']
            self.mcp_stats[mcp]['avg_latency_ms'] = (
                self.mcp_stats[mcp]['total_latency_ms'] / total
            )

        # Learn patterns if enabled
        if self.enable_auto_learning:
            await self._learn_from_query(query, mcps_used, outcome, context_type)

        # Persist to memory-bank
        if self.memory_bank:
            await self._persist_learning()

    async def recommend_mcps(
        self,
        query: str,
        context_type: str = "general",
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Recommend MCPs for a query based on learned patterns.

        Args:
            query: User query
            context_type: Query context type
            top_k: Number of recommendations to return

        Returns:
            List of (mcp_name, confidence) tuples

        Example:
            >>> recommendations = await learner.recommend_mcps("numpy question")
            >>> # [('context7', 0.92), ('memory-bank', 0.75)]
        """
        self.stats['recommendations_made'] += 1

        # Find matching patterns
        matches: Dict[str, float] = {}

        for pattern_key, pattern in self.patterns.items():
            # Check if pattern matches query
            if self._pattern_matches(pattern.pattern, query):
                # Check context compatibility
                if pattern.context_type == context_type or pattern.context_type == "general":
                    # Decay confidence for old patterns
                    decayed_confidence = self._decay_confidence(pattern)

                    # Add all MCPs from pattern
                    for mcp in pattern.mcps:
                        if mcp in matches:
                            matches[mcp] = max(matches[mcp], decayed_confidence)
                        else:
                            matches[mcp] = decayed_confidence

        # Also consider MCP success rates
        for mcp, stats in self.mcp_stats.items():
            if stats['total_count'] > 0:
                success_rate = stats['success_count'] / stats['total_count']
                # Boost confidence based on historical success
                if mcp in matches:
                    matches[mcp] = min(matches[mcp] * (1 + success_rate * 0.2), 1.0)

        # Filter by minimum confidence
        filtered = [
            (mcp, conf) for mcp, conf in matches.items()
            if conf >= self.min_confidence
        ]

        # Sort by confidence
        sorted_recommendations = sorted(filtered, key=lambda x: x[1], reverse=True)

        return sorted_recommendations[:top_k]

    async def _learn_from_query(
        self,
        query: str,
        mcps_used: List[str],
        outcome: OutcomeType,
        context_type: str
    ) -> None:
        """
        Learn patterns from a query execution.

        Args:
            query: User query
            mcps_used: MCPs that were used
            outcome: Execution outcome
            context_type: Query context
        """
        # Extract keywords and patterns from query
        keywords = self._extract_keywords(query)

        for keyword in keywords:
            # Generate pattern key
            pattern_key = hashlib.md5(
                f"{keyword}:{context_type}".encode()
            ).hexdigest()

            # Update or create pattern
            if pattern_key in self.patterns:
                pattern = self.patterns[pattern_key]
                pattern.update_confidence(outcome)

                # Add new MCPs if successful
                if outcome == OutcomeType.SUCCESS:
                    for mcp in mcps_used:
                        if mcp not in pattern.mcps:
                            pattern.mcps.append(mcp)
            else:
                # Create new pattern
                if outcome == OutcomeType.SUCCESS:
                    self.patterns[pattern_key] = Pattern(
                        pattern=keyword,
                        mcps=mcps_used.copy(),
                        confidence=0.6,
                        success_count=1,
                        total_count=1,
                        last_used=time.time(),
                        context_type=context_type
                    )
                    self.stats['patterns_learned'] += 1

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query.

        Args:
            query: User query

        Returns:
            List of keywords/patterns
        """
        keywords = []

        # Remove common words
        stopwords = {'the', 'a', 'an', 'how', 'to', 'do', 'i', 'can', 'what', 'is', 'are'}

        # Tokenize
        words = re.findall(r'\w+', query.lower())

        # Filter and extract
        for word in words:
            if word not in stopwords and len(word) > 2:
                keywords.append(word)

        # Also look for library patterns
        library_pattern = r'\b(numpy|pandas|torch|react|vue|django|flask|pytest|jax)\b'
        libraries = re.findall(library_pattern, query.lower())
        keywords.extend(libraries)

        # Look for error patterns
        error_pattern = r'\b(error|exception|failure|bug|issue)\b'
        errors = re.findall(error_pattern, query.lower())
        keywords.extend(errors)

        return list(set(keywords))  # Unique keywords

    def _pattern_matches(self, pattern: str, query: str) -> bool:
        """
        Check if pattern matches query.

        Args:
            pattern: Pattern string (keyword or regex)
            query: Query to match against

        Returns:
            True if matches
        """
        # Try as regex first
        try:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        except re.error:
            pass

        # Fallback to simple keyword match
        return pattern.lower() in query.lower()

    def _decay_confidence(self, pattern: Pattern) -> float:
        """
        Decay confidence for patterns that haven't been used recently.

        Args:
            pattern: Pattern to decay

        Returns:
            Decayed confidence score
        """
        days_since_use = (time.time() - pattern.last_used) / (24 * 3600)

        if days_since_use > self.pattern_decay_days:
            # Decay by 50% after threshold
            decay_factor = 0.5
        elif days_since_use > self.pattern_decay_days / 2:
            # Linear decay
            decay_factor = 1.0 - (
                (days_since_use - self.pattern_decay_days / 2) /
                (self.pattern_decay_days / 2) * 0.5
            )
        else:
            # No decay
            decay_factor = 1.0

        return pattern.confidence * decay_factor

    async def _persist_learning(self) -> None:
        """Persist learned patterns to memory-bank."""
        if not self.memory_bank:
            return

        # Convert patterns to serializable format
        patterns_data = {
            pattern_key: {
                'pattern': pattern.pattern,
                'mcps': pattern.mcps,
                'confidence': pattern.confidence,
                'success_count': pattern.success_count,
                'total_count': pattern.total_count,
                'last_used': pattern.last_used,
                'context_type': pattern.context_type,
            }
            for pattern_key, pattern in self.patterns.items()
        }

        # Store in memory-bank
        await self.memory_bank.store(
            key='mcp_learning:patterns',
            value=patterns_data,
            ttl=90 * 24 * 3600,  # 90 days
            project_name='mcp_integration',
            context_type='learning'
        )

        # Store MCP stats
        await self.memory_bank.store(
            key='mcp_learning:stats',
            value=dict(self.mcp_stats),
            ttl=90 * 24 * 3600,
            project_name='mcp_integration',
            context_type='learning'
        )

    async def _load_patterns(self) -> None:
        """Load learned patterns from memory-bank."""
        if not self.memory_bank:
            return

        # Load patterns
        patterns_data = await self.memory_bank.fetch(
            query='mcp_learning:patterns',
            context_type='learning',
            project_name='mcp_integration'
        )

        if patterns_data:
            for pattern_key, pattern_dict in patterns_data.items():
                self.patterns[pattern_key] = Pattern(
                    pattern=pattern_dict['pattern'],
                    mcps=pattern_dict['mcps'],
                    confidence=pattern_dict['confidence'],
                    success_count=pattern_dict['success_count'],
                    total_count=pattern_dict['total_count'],
                    last_used=pattern_dict['last_used'],
                    context_type=pattern_dict.get('context_type', 'general'),
                )

        # Load MCP stats
        stats_data = await self.memory_bank.fetch(
            query='mcp_learning:stats',
            context_type='learning',
            project_name='mcp_integration'
        )

        if stats_data:
            self.mcp_stats = defaultdict(
                lambda: {
                    'success_count': 0,
                    'total_count': 0,
                    'avg_latency_ms': 0,
                    'total_latency_ms': 0,
                },
                stats_data
            )

    def get_patterns(
        self,
        min_confidence: Optional[float] = None,
        context_type: Optional[str] = None
    ) -> List[Pattern]:
        """
        Get learned patterns.

        Args:
            min_confidence: Minimum confidence threshold
            context_type: Filter by context type

        Returns:
            List of patterns
        """
        patterns = list(self.patterns.values())

        if min_confidence:
            patterns = [p for p in patterns if p.confidence >= min_confidence]

        if context_type:
            patterns = [p for p in patterns if p.context_type == context_type]

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def get_mcp_effectiveness(self, mcp_name: str) -> Dict[str, Any]:
        """
        Get effectiveness metrics for an MCP.

        Args:
            mcp_name: MCP name

        Returns:
            Effectiveness metrics
        """
        if mcp_name not in self.mcp_stats:
            return {
                'success_rate': 0.0,
                'avg_latency_ms': 0,
                'total_uses': 0,
            }

        stats = self.mcp_stats[mcp_name]
        success_rate = (
            stats['success_count'] / stats['total_count']
            if stats['total_count'] > 0 else 0.0
        )

        return {
            'success_rate': success_rate,
            'avg_latency_ms': stats['avg_latency_ms'],
            'total_uses': stats['total_count'],
            'successful_uses': stats['success_count'],
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get learning system statistics.

        Returns:
            Statistics dictionary
        """
        success_rate = (
            self.stats['successful_queries'] / self.stats['total_queries']
            if self.stats['total_queries'] > 0 else 0.0
        )

        return {
            **self.stats,
            'success_rate': success_rate,
            'total_patterns': len(self.patterns),
            'high_confidence_patterns': len([
                p for p in self.patterns.values()
                if p.confidence >= 0.8
            ]),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'patterns_learned': 0,
            'recommendations_made': 0,
        }
