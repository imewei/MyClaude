"""
Predictive Preloading System

Proactively loads MCPs and caches knowledge based on usage patterns,
command context, and learned behaviors.

Features:
- Command-based MCP prediction
- Usage pattern analysis
- Intelligent preloading
- Background caching
- Resource management

Example:
    >>> preloader = await PredictivePreloader.create(
    ...     profile_manager=manager,
    ...     learning_system=learner
    ... )
    >>>
    >>> # Predict and preload for command
    >>> await preloader.preload_for_command("fix", file_context="main.py")
    >>> # Preloads: serena (critical), memory-bank (high)
    >>>
    >>> # Background preloading
    >>> await preloader.start_background_preloading()
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter, deque
from enum import Enum


class PreloadStrategy(Enum):
    """Preloading strategies."""
    AGGRESSIVE = "aggressive"  # Preload everything predicted
    BALANCED = "balanced"      # Preload high-confidence only
    CONSERVATIVE = "conservative"  # Preload critical only
    DISABLED = "disabled"      # No preloading


@dataclass
class PreloadPrediction:
    """
    Preload prediction.

    Attributes:
        mcps: MCPs to preload
        confidence: Prediction confidence
        reason: Reasoning for prediction
        priority: Preload priority (1-5, 5=highest)
        estimated_load_time_ms: Estimated load time
    """
    mcps: List[str]
    confidence: float
    reason: str
    priority: int = 3
    estimated_load_time_ms: int = 500


@dataclass
class PreloadResult:
    """
    Result of preload operation.

    Attributes:
        mcps_loaded: MCPs that were loaded
        load_time_ms: Total load time
        cache_hit: Whether MCPs were already loaded
        strategy_used: Strategy used for preloading
    """
    mcps_loaded: List[str]
    load_time_ms: int
    cache_hit: bool
    strategy_used: PreloadStrategy


class PredictivePreloader:
    """
    Predictive preloading system for MCPs.

    Analyzes usage patterns and proactively loads MCPs to reduce
    latency when they're actually needed.

    Features:
    - Command-based prediction
    - Pattern-based preloading
    - Background caching
    - Resource-aware loading
    - Multi-strategy support
    """

    def __init__(
        self,
        profile_manager: Any,
        learning_system: Optional[Any] = None,
        strategy: PreloadStrategy = PreloadStrategy.BALANCED,
        max_concurrent_preloads: int = 3,
    ):
        """
        Initialize predictive preloader.

        Args:
            profile_manager: MCP profile manager
            learning_system: Learning system for patterns
            strategy: Preloading strategy
            max_concurrent_preloads: Max concurrent preload operations
        """
        self.profile_manager = profile_manager
        self.learning_system = learning_system
        self.strategy = strategy
        self.max_concurrent_preloads = max_concurrent_preloads

        # Preload queue
        self.preload_queue: deque = deque()

        # Command history for pattern analysis
        self.command_history: deque = deque(maxlen=100)

        # Preload cache (track what's already loaded)
        self.preload_cache: Set[str] = set()

        # Background task
        self.background_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            'predictions_made': 0,
            'preloads_attempted': 0,
            'preloads_successful': 0,
            'cache_hits': 0,
            'total_time_saved_ms': 0,
        }

    @classmethod
    async def create(
        cls,
        profile_manager: Any,
        learning_system: Optional[Any] = None,
        **kwargs
    ) -> "PredictivePreloader":
        """
        Create predictive preloader.

        Args:
            profile_manager: MCP profile manager
            learning_system: Learning system
            **kwargs: Additional configuration

        Returns:
            Initialized PredictivePreloader instance
        """
        return cls(
            profile_manager=profile_manager,
            learning_system=learning_system,
            **kwargs
        )

    async def predict_for_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PreloadPrediction:
        """
        Predict MCPs needed for a command.

        Args:
            command: Command name
            context: Additional context (file paths, error messages, etc.)

        Returns:
            PreloadPrediction with MCPs and confidence

        Example:
            >>> prediction = await preloader.predict_for_command(
            ...     "fix",
            ...     context={'file': 'main.py', 'error': 'TypeError'}
            ... )
            >>> # Returns: PreloadPrediction(
            ...     mcps=['serena', 'memory-bank'],
            ...     confidence=0.9,
            ...     reason='command profile + error pattern'
            ... )
        """
        self.stats['predictions_made'] += 1

        mcps_to_preload: Set[str] = set()
        reasons: List[str] = []
        confidence_scores: List[float] = []

        # 1. Profile-based prediction (high confidence)
        profile_name = self.profile_manager.get_profile_for_command(command)
        if profile_name and profile_name in self.profile_manager.profiles:
            profile = self.profile_manager.profiles[profile_name]
            profile_mcps = [mcp.name for mcp in profile.get_preload_mcps()]
            mcps_to_preload.update(profile_mcps)
            reasons.append(f"command profile: {profile_name}")
            confidence_scores.append(0.9)

        # 2. Learning system predictions (medium-high confidence)
        if self.learning_system and context:
            # Build query from context
            query_parts = [command]
            if 'file' in context:
                query_parts.append(context['file'])
            if 'error' in context:
                query_parts.append(context['error'])

            query = ' '.join(query_parts)

            recommendations = await self.learning_system.recommend_mcps(query)
            for mcp, conf in recommendations:
                mcps_to_preload.add(mcp)
                confidence_scores.append(conf)
            if recommendations:
                reasons.append('learned patterns')

        # 3. Sequential command patterns (medium confidence)
        if len(self.command_history) >= 2:
            # Look for patterns in command sequences
            recent_commands = list(self.command_history)[-5:]
            if command in self._get_command_sequences():
                # Predict based on what usually follows
                sequence_mcps = self._predict_from_sequence(recent_commands, command)
                mcps_to_preload.update(sequence_mcps)
                if sequence_mcps:
                    reasons.append('command sequence pattern')
                    confidence_scores.append(0.7)

        # 4. Context-based prediction (low-medium confidence)
        if context:
            context_mcps = self._predict_from_context(context)
            mcps_to_preload.update(context_mcps)
            if context_mcps:
                reasons.append('context analysis')
                confidence_scores.append(0.6)

        # Calculate overall confidence
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.5
        )

        # Estimate load time
        estimated_time = len(mcps_to_preload) * 150  # ~150ms per MCP

        # Determine priority
        priority = self._calculate_priority(command, overall_confidence)

        return PreloadPrediction(
            mcps=list(mcps_to_preload),
            confidence=overall_confidence,
            reason=' + '.join(reasons) if reasons else 'default prediction',
            priority=priority,
            estimated_load_time_ms=estimated_time
        )

    async def preload_for_command(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> PreloadResult:
        """
        Preload MCPs for a command.

        Args:
            command: Command name
            context: Additional context
            force: Force preload even if already loaded

        Returns:
            PreloadResult with loaded MCPs and timing

        Example:
            >>> result = await preloader.preload_for_command("fix")
            >>> print(f"Loaded {len(result.mcps_loaded)} MCPs in {result.load_time_ms}ms")
        """
        start_time = time.time()
        self.stats['preloads_attempted'] += 1

        # Get prediction
        prediction = await self.predict_for_command(command, context)

        # Check strategy
        if not self._should_preload(prediction):
            return PreloadResult(
                mcps_loaded=[],
                load_time_ms=0,
                cache_hit=True,
                strategy_used=self.strategy
            )

        # Filter MCPs to load
        mcps_to_load = [
            mcp for mcp in prediction.mcps
            if force or mcp not in self.preload_cache
        ]

        if not mcps_to_load:
            # All MCPs already loaded
            self.stats['cache_hits'] += 1
            return PreloadResult(
                mcps_loaded=[],
                load_time_ms=0,
                cache_hit=True,
                strategy_used=self.strategy
            )

        # Activate profile (will load MCPs)
        profile = await self.profile_manager.activate_for_command(command)

        # Update cache
        self.preload_cache.update(mcps_to_load)

        # Calculate time
        load_time = int((time.time() - start_time) * 1000)

        # Track command history
        self.command_history.append({
            'command': command,
            'mcps': mcps_to_load,
            'timestamp': time.time(),
        })

        self.stats['preloads_successful'] += 1
        self.stats['total_time_saved_ms'] += max(0, prediction.estimated_load_time_ms - load_time)

        return PreloadResult(
            mcps_loaded=mcps_to_load,
            load_time_ms=load_time,
            cache_hit=False,
            strategy_used=self.strategy
        )

    async def start_background_preloading(self) -> None:
        """
        Start background preloading task.

        Continuously analyzes patterns and preloads MCPs in the background.
        """
        if self.background_task and not self.background_task.done():
            return  # Already running

        self.background_task = asyncio.create_task(self._background_preload_loop())

    async def stop_background_preloading(self) -> None:
        """Stop background preloading task."""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass

    async def _background_preload_loop(self) -> None:
        """Background preloading loop."""
        while True:
            try:
                # Wait for next preload opportunity
                await asyncio.sleep(5)  # Check every 5 seconds

                # Process preload queue
                if self.preload_queue:
                    prediction = self.preload_queue.popleft()
                    # Load MCPs in background
                    # (Implementation would activate profiles)

                # Predict next likely command
                if len(self.command_history) >= 3:
                    next_command = self._predict_next_command()
                    if next_command:
                        prediction = await self.predict_for_command(next_command)
                        if prediction.confidence >= 0.7:
                            # Queue for preloading
                            self.preload_queue.append(prediction)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but continue
                print(f"Background preload error: {e}")
                await asyncio.sleep(1)

    def _should_preload(self, prediction: PreloadPrediction) -> bool:
        """
        Determine if preloading should occur based on strategy.

        Args:
            prediction: Preload prediction

        Returns:
            True if should preload
        """
        if self.strategy == PreloadStrategy.DISABLED:
            return False

        if self.strategy == PreloadStrategy.AGGRESSIVE:
            return prediction.confidence >= 0.5

        if self.strategy == PreloadStrategy.BALANCED:
            return prediction.confidence >= 0.7

        if self.strategy == PreloadStrategy.CONSERVATIVE:
            return prediction.confidence >= 0.85 and prediction.priority >= 4

        return False

    def _predict_from_sequence(
        self,
        recent_commands: List[Dict[str, Any]],
        current_command: str
    ) -> Set[str]:
        """
        Predict MCPs from command sequence patterns.

        Args:
            recent_commands: Recent command history
            current_command: Current command

        Returns:
            Set of predicted MCPs
        """
        # Simple pattern: if fix → test sequence is common, predict test MCPs after fix
        sequences = {
            ('fix', 'run-all-tests'): {'memory-bank'},
            ('code-review', 'fix'): {'serena', 'memory-bank'},
            ('ultra-think', 'fix'): {'serena', 'memory-bank'},
        }

        predicted_mcps: Set[str] = set()

        # Check last 2 commands
        if len(recent_commands) >= 1:
            last_cmd = recent_commands[-1]['command']
            pattern = (last_cmd, current_command)
            if pattern in sequences:
                predicted_mcps.update(sequences[pattern])

        return predicted_mcps

    def _predict_from_context(self, context: Dict[str, Any]) -> Set[str]:
        """
        Predict MCPs from context.

        Args:
            context: Execution context

        Returns:
            Set of predicted MCPs
        """
        predicted_mcps: Set[str] = set()

        # File context
        if 'file' in context or 'file_paths' in context:
            predicted_mcps.add('serena')

        # Error context
        if 'error' in context or 'error_message' in context:
            predicted_mcps.add('memory-bank')  # Error solutions cached

        # GitHub context
        if 'pr_number' in context or 'issue_number' in context:
            predicted_mcps.add('github')

        # Library context
        if 'library' in context:
            predicted_mcps.add('context7')

        return predicted_mcps

    def _calculate_priority(self, command: str, confidence: float) -> int:
        """
        Calculate preload priority.

        Args:
            command: Command name
            confidence: Prediction confidence

        Returns:
            Priority (1-5, 5=highest)
        """
        # Base priority from confidence
        if confidence >= 0.9:
            priority = 5
        elif confidence >= 0.8:
            priority = 4
        elif confidence >= 0.7:
            priority = 3
        elif confidence >= 0.6:
            priority = 2
        else:
            priority = 1

        # Boost for certain commands
        high_priority_commands = ['fix', 'run-all-tests', 'fix-commit-errors']
        if command in high_priority_commands:
            priority = min(priority + 1, 5)

        return priority

    def _get_command_sequences(self) -> Dict[Tuple[str, str], int]:
        """
        Get common command sequences from history.

        Returns:
            Dictionary of (cmd1, cmd2) → count
        """
        sequences = Counter()

        commands = [h['command'] for h in self.command_history]
        for i in range(len(commands) - 1):
            sequence = (commands[i], commands[i + 1])
            sequences[sequence] += 1

        return dict(sequences)

    def _predict_next_command(self) -> Optional[str]:
        """
        Predict next likely command based on patterns.

        Returns:
            Predicted command or None
        """
        if len(self.command_history) < 2:
            return None

        # Get last command
        last_command = self.command_history[-1]['command']

        # Get sequences
        sequences = self._get_command_sequences()

        # Find most common following command
        following_commands = Counter()
        for (cmd1, cmd2), count in sequences.items():
            if cmd1 == last_command:
                following_commands[cmd2] = count

        if following_commands:
            return following_commands.most_common(1)[0][0]

        return None

    def clear_cache(self) -> None:
        """Clear preload cache."""
        self.preload_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get preloader statistics.

        Returns:
            Statistics dictionary
        """
        success_rate = (
            self.stats['preloads_successful'] / self.stats['preloads_attempted']
            if self.stats['preloads_attempted'] > 0 else 0.0
        )

        cache_hit_rate = (
            self.stats['cache_hits'] / self.stats['preloads_attempted']
            if self.stats['preloads_attempted'] > 0 else 0.0
        )

        return {
            **self.stats,
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate,
            'cached_mcps': len(self.preload_cache),
            'queue_size': len(self.preload_queue),
            'strategy': self.strategy.value,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            'predictions_made': 0,
            'preloads_attempted': 0,
            'preloads_successful': 0,
            'cache_hits': 0,
            'total_time_saved_ms': 0,
        }
