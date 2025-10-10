"""
Smart MCP Trigger System

Pattern-based conditional MCP activation to minimize unnecessary MCP calls.
Analyzes user queries and command context to determine which MCPs are needed.

Example:
    >>> trigger = await SmartTrigger.create("mcp-config.yaml")
    >>>
    >>> # Analyze query
    >>> result = trigger.analyze("How do I use numpy.array?")
    >>> print(result.recommended_mcps)
    >>> # Returns: ['context7'] (detected library API query)
    >>>
    >>> # Check if MCP should activate
    >>> should_use = trigger.should_activate_mcp(
    ...     mcp_name="context7",
    ...     query="implement feature X",
    ...     command="fix"
    ... )
"""

import re
import yaml
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class QueryType(Enum):
    """Types of user queries."""
    LIBRARY_API = "library_api"        # "how to use numpy.array"
    PROJECT_CODE = "project_code"      # "where is the User class"
    ERROR_DEBUG = "error_debug"        # "fix TypeError in main.py"
    GENERAL = "general"                # "explain this code"
    GITHUB_ISSUE = "github_issue"      # "list open PRs"
    WEB_AUTOMATION = "web_automation"  # "click the submit button"
    META_REASONING = "meta_reasoning"  # "analyze trade-offs"


@dataclass
class TriggerPattern:
    """
    Pattern for MCP activation.

    Attributes:
        pattern: Regex pattern to match
        mcp: MCP name this pattern indicates
        query_type: Type of query this pattern matches
        confidence: Confidence score (0.0-1.0)
        priority: Priority boost when matched
    """
    pattern: str
    mcp: str
    query_type: QueryType
    confidence: float = 0.8
    priority: int = 0

    @property
    def compiled_pattern(self) -> re.Pattern:
        """Get compiled regex pattern."""
        if not hasattr(self, '_compiled'):
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        return self._compiled


@dataclass
class TriggerResult:
    """
    Result of trigger analysis.

    Attributes:
        query_type: Detected query type
        recommended_mcps: List of recommended MCPs
        confidence: Overall confidence (0.0-1.0)
        matched_patterns: Patterns that matched
        reasoning: Explanation of recommendations
    """
    query_type: QueryType
    recommended_mcps: List[str]
    confidence: float
    matched_patterns: List[TriggerPattern] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartTrigger:
    """
    Smart MCP trigger system with pattern-based activation.

    Reduces unnecessary MCP calls by intelligently determining which
    MCPs are needed based on query patterns and command context.

    Features:
    - Pattern-based MCP recommendation
    - Query type classification
    - Confidence scoring
    - Context-aware activation
    - Statistics tracking
    """

    def __init__(
        self,
        trigger_patterns: List[TriggerPattern],
        mcp_rules: Dict[str, Dict[str, Any]],
        enable_context_awareness: bool = True,
    ):
        """
        Initialize smart trigger system.

        Args:
            trigger_patterns: List of trigger patterns
            mcp_rules: MCP-specific activation rules
            enable_context_awareness: Use command context for decisions
        """
        self.trigger_patterns = trigger_patterns
        self.mcp_rules = mcp_rules
        self.enable_context_awareness = enable_context_awareness

        # Group patterns by MCP for faster lookup
        self._patterns_by_mcp: Dict[str, List[TriggerPattern]] = {}
        for pattern in trigger_patterns:
            if pattern.mcp not in self._patterns_by_mcp:
                self._patterns_by_mcp[pattern.mcp] = []
            self._patterns_by_mcp[pattern.mcp].append(pattern)

        # Statistics
        self.stats = {
            "queries_analyzed": 0,
            "mcps_triggered": 0,
            "mcps_skipped": 0,
            "total_confidence": 0.0,
        }

    @classmethod
    async def create(
        cls,
        config_path: str,
        **kwargs
    ) -> "SmartTrigger":
        """
        Create smart trigger from configuration.

        Args:
            config_path: Path to mcp-config.yaml
            **kwargs: Additional configuration

        Returns:
            Initialized SmartTrigger instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load trigger patterns
        patterns = []
        for pattern_config in config.get('smart_triggers', {}).get('patterns', []):
            patterns.append(TriggerPattern(
                pattern=pattern_config['pattern'],
                mcp=pattern_config['mcp'],
                query_type=QueryType(pattern_config.get('type', 'general')),
                confidence=pattern_config.get('confidence', 0.8),
                priority=pattern_config.get('priority', 0),
            ))

        # Load MCP rules
        mcp_rules = config.get('smart_triggers', {}).get('mcp_rules', {})

        return cls(
            trigger_patterns=patterns,
            mcp_rules=mcp_rules,
            **kwargs
        )

    def analyze(
        self,
        query: str,
        command: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerResult:
        """
        Analyze query and recommend MCPs.

        Args:
            query: User query or task description
            command: Command being executed (e.g., 'fix', 'quality')
            context: Additional context (file paths, error messages, etc.)

        Returns:
            TriggerResult with recommendations

        Example:
            >>> result = trigger.analyze(
            ...     query="How do I use numpy.array?",
            ...     command="ultra-think"
            ... )
            >>> print(result.recommended_mcps)
            >>> # ['context7', 'memory-bank']
        """
        self.stats["queries_analyzed"] += 1

        # Match patterns
        matched_patterns = []
        for pattern in self.trigger_patterns:
            if pattern.compiled_pattern.search(query):
                matched_patterns.append(pattern)

        # Classify query type
        query_type = self._classify_query_type(query, matched_patterns, context)

        # Get recommended MCPs
        recommended_mcps, confidence = self._recommend_mcps(
            query_type,
            matched_patterns,
            command,
            context
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            query_type,
            matched_patterns,
            recommended_mcps
        )

        self.stats["total_confidence"] += confidence

        return TriggerResult(
            query_type=query_type,
            recommended_mcps=recommended_mcps,
            confidence=confidence,
            matched_patterns=matched_patterns,
            reasoning=reasoning,
            metadata={
                "command": command,
                "pattern_count": len(matched_patterns),
            }
        )

    def _classify_query_type(
        self,
        query: str,
        matched_patterns: List[TriggerPattern],
        context: Optional[Dict[str, Any]]
    ) -> QueryType:
        """
        Classify the type of query.

        Args:
            query: User query
            matched_patterns: Matched trigger patterns
            context: Additional context

        Returns:
            Classified query type
        """
        # Check matched patterns
        if matched_patterns:
            # Use most confident pattern's type
            most_confident = max(matched_patterns, key=lambda p: p.confidence)
            return most_confident.query_type

        # Fallback heuristics
        query_lower = query.lower()

        # Library API patterns
        if any(word in query_lower for word in ['how to', 'how do i', 'api', 'docs', 'documentation']):
            return QueryType.LIBRARY_API

        # Error/debug patterns
        if any(word in query_lower for word in ['error', 'fix', 'bug', 'failure', 'exception']):
            return QueryType.ERROR_DEBUG

        # GitHub patterns
        if any(word in query_lower for word in ['pull request', 'pr', 'issue', 'commit', 'branch']):
            return QueryType.GITHUB_ISSUE

        # Meta-reasoning patterns
        if any(word in query_lower for word in ['analyze', 'evaluate', 'trade-off', 'approach', 'strategy']):
            return QueryType.META_REASONING

        # Check context
        if context:
            if context.get('file_paths'):
                return QueryType.PROJECT_CODE
            if context.get('error_message'):
                return QueryType.ERROR_DEBUG

        return QueryType.GENERAL

    def _recommend_mcps(
        self,
        query_type: QueryType,
        matched_patterns: List[TriggerPattern],
        command: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], float]:
        """
        Recommend MCPs based on analysis.

        Args:
            query_type: Classified query type
            matched_patterns: Matched patterns
            command: Command being executed
            context: Additional context

        Returns:
            Tuple of (recommended MCPs, confidence score)
        """
        recommendations: Dict[str, float] = {}

        # Add MCPs from matched patterns
        for pattern in matched_patterns:
            if pattern.mcp in recommendations:
                recommendations[pattern.mcp] = max(
                    recommendations[pattern.mcp],
                    pattern.confidence
                )
            else:
                recommendations[pattern.mcp] = pattern.confidence

        # Add MCPs based on query type
        type_to_mcps = {
            QueryType.LIBRARY_API: [('context7', 0.9), ('memory-bank', 0.6)],
            QueryType.PROJECT_CODE: [('serena', 0.9), ('memory-bank', 0.7)],
            QueryType.ERROR_DEBUG: [('memory-bank', 0.8), ('serena', 0.7)],
            QueryType.GITHUB_ISSUE: [('github', 0.95)],
            QueryType.WEB_AUTOMATION: [('playwright', 0.95)],
            QueryType.META_REASONING: [('sequential-thinking', 0.9), ('memory-bank', 0.6)],
            QueryType.GENERAL: [('memory-bank', 0.5)],
        }

        for mcp, confidence in type_to_mcps.get(query_type, []):
            if mcp in recommendations:
                recommendations[mcp] = max(recommendations[mcp], confidence)
            else:
                recommendations[mcp] = confidence

        # Apply command-specific rules
        if command and self.enable_context_awareness:
            command_mcps = self._get_command_mcps(command)
            for mcp in command_mcps:
                if mcp in recommendations:
                    recommendations[mcp] = min(recommendations[mcp] + 0.1, 1.0)
                else:
                    recommendations[mcp] = 0.7

        # Sort by confidence and return top recommendations
        sorted_mcps = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Filter by minimum confidence threshold
        min_confidence = 0.5
        recommended = [mcp for mcp, conf in sorted_mcps if conf >= min_confidence]

        # Calculate overall confidence
        overall_confidence = (
            sum(conf for _, conf in sorted_mcps) / len(sorted_mcps)
            if sorted_mcps else 0.0
        )

        return recommended, overall_confidence

    def _get_command_mcps(self, command: str) -> List[str]:
        """
        Get MCPs typically used by a command.

        Args:
            command: Command name

        Returns:
            List of MCP names
        """
        # Command to MCP mapping
        command_map = {
            'ultra-think': ['sequential-thinking', 'memory-bank'],
            'reflection': ['sequential-thinking', 'memory-bank'],
            'double-check': ['sequential-thinking'],
            'fix': ['serena', 'memory-bank'],
            'quality': ['serena', 'memory-bank'],
            'clean-codebase': ['serena', 'memory-bank'],
            'code-review': ['serena', 'memory-bank'],
            'generate-tests': ['serena', 'memory-bank'],
            'fix-commit-errors': ['github', 'memory-bank'],
            'commit': ['github'],
        }

        return command_map.get(command, [])

    def _generate_reasoning(
        self,
        query_type: QueryType,
        matched_patterns: List[TriggerPattern],
        recommended_mcps: List[str]
    ) -> str:
        """
        Generate human-readable reasoning.

        Args:
            query_type: Query type
            matched_patterns: Matched patterns
            recommended_mcps: Recommended MCPs

        Returns:
            Reasoning string
        """
        parts = [f"Query classified as: {query_type.value}"]

        if matched_patterns:
            pattern_mcps = list(set(p.mcp for p in matched_patterns))
            parts.append(f"Matched patterns for: {', '.join(pattern_mcps)}")

        if recommended_mcps:
            parts.append(f"Recommending: {', '.join(recommended_mcps)}")

        return ". ".join(parts)

    def should_activate_mcp(
        self,
        mcp_name: str,
        query: str,
        command: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5
    ) -> bool:
        """
        Determine if an MCP should be activated.

        Args:
            mcp_name: MCP to check
            query: User query
            command: Command being executed
            context: Additional context
            threshold: Minimum confidence threshold

        Returns:
            True if MCP should be activated

        Example:
            >>> should_use = trigger.should_activate_mcp(
            ...     mcp_name="context7",
            ...     query="implement feature X",
            ...     command="fix",
            ...     threshold=0.6
            ... )
        """
        result = self.analyze(query, command, context)

        # Check if MCP is recommended
        if mcp_name in result.recommended_mcps:
            # Check patterns specific to this MCP
            mcp_patterns = [
                p for p in result.matched_patterns
                if p.mcp == mcp_name
            ]

            if mcp_patterns:
                # Use highest confidence from matched patterns
                max_confidence = max(p.confidence for p in mcp_patterns)
                activated = max_confidence >= threshold
            else:
                # Use overall confidence
                activated = result.confidence >= threshold

            if activated:
                self.stats["mcps_triggered"] += 1
            else:
                self.stats["mcps_skipped"] += 1

            return activated

        self.stats["mcps_skipped"] += 1
        return False

    def get_activation_score(
        self,
        mcp_name: str,
        query: str,
        command: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Get activation score for an MCP.

        Args:
            mcp_name: MCP to score
            query: User query
            command: Command being executed
            context: Additional context

        Returns:
            Score (0.0-1.0), higher = more relevant

        Example:
            >>> score = trigger.get_activation_score("serena", "fix error in main.py")
            >>> # Returns: 0.85
        """
        result = self.analyze(query, command, context)

        if mcp_name not in result.recommended_mcps:
            return 0.0

        # Get MCP-specific patterns
        mcp_patterns = [
            p for p in result.matched_patterns
            if p.mcp == mcp_name
        ]

        if mcp_patterns:
            return max(p.confidence for p in mcp_patterns)

        # Return default confidence if recommended but no specific patterns
        return 0.5

    def add_pattern(
        self,
        pattern: str,
        mcp: str,
        query_type: QueryType,
        confidence: float = 0.8
    ) -> None:
        """
        Add a new trigger pattern.

        Args:
            pattern: Regex pattern
            mcp: MCP name
            query_type: Query type
            confidence: Confidence score
        """
        trigger_pattern = TriggerPattern(
            pattern=pattern,
            mcp=mcp,
            query_type=query_type,
            confidence=confidence
        )

        self.trigger_patterns.append(trigger_pattern)

        if mcp not in self._patterns_by_mcp:
            self._patterns_by_mcp[mcp] = []
        self._patterns_by_mcp[mcp].append(trigger_pattern)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get trigger statistics.

        Returns:
            Statistics dictionary

        Example:
            >>> stats = trigger.get_stats()
            >>> print(f"Trigger rate: {stats['trigger_rate']:.1%}")
        """
        total = self.stats["mcps_triggered"] + self.stats["mcps_skipped"]
        trigger_rate = (
            self.stats["mcps_triggered"] / total
            if total > 0 else 0.0
        )

        queries_analyzed = self.stats["queries_analyzed"]
        avg_confidence = (
            self.stats["total_confidence"] / queries_analyzed
            if queries_analyzed > 0 else 0.0
        )

        return {
            **self.stats,
            "trigger_rate": trigger_rate,
            "avg_confidence": avg_confidence,
            "total_patterns": len(self.trigger_patterns),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "queries_analyzed": 0,
            "mcps_triggered": 0,
            "mcps_skipped": 0,
            "total_confidence": 0.0,
        }


# Convenience function
async def should_use_mcp(
    mcp_name: str,
    query: str,
    config_path: str = "mcp-config.yaml",
    threshold: float = 0.5
) -> bool:
    """
    Convenience function to check if MCP should be used.

    Args:
        mcp_name: MCP to check
        query: User query
        config_path: Path to config
        threshold: Confidence threshold

    Returns:
        True if MCP should be activated
    """
    trigger = await SmartTrigger.create(config_path)
    return trigger.should_activate_mcp(mcp_name, query, threshold=threshold)
