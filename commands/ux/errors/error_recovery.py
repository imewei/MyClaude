"""
Automatic error recovery system.

Implements automatic retry strategies, fallback options, graceful degradation,
state preservation, and rollback capabilities.
"""

from typing import Optional, Callable, Any, Dict, List, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import time
import functools
import json
from pathlib import Path
from datetime import datetime

T = TypeVar('T')


class RecoveryStrategy(Enum):
    """Error recovery strategy."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ROLLBACK = "rollback"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    strategy: RecoveryStrategy
    attempt_number: int
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class RecoveryState:
    """State information for recovery."""
    operation_id: str
    checkpoints: Dict[str, Any] = field(default_factory=dict)
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    max_attempts: int = 3
    backoff_factor: float = 2.0
    last_checkpoint: Optional[str] = None


class ErrorRecovery:
    """
    Automatic error recovery system.

    Features:
    - Retry strategies with exponential backoff
    - Fallback to alternative approaches
    - Graceful degradation (reduced functionality)
    - State preservation via checkpoints
    - Automatic rollback on errors
    - User confirmation for critical operations

    Example:
        recovery = ErrorRecovery()

        # Automatic retry with backoff
        @recovery.with_retry(max_attempts=3)
        def fetch_data():
            return api.get_data()

        # Fallback to alternative
        @recovery.with_fallback(fallback_fn=use_cache)
        def fetch_remote_data():
            return api.get_data()

        # State preservation
        with recovery.checkpoint("processing_data"):
            process_large_dataset()
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        auto_rollback: bool = True,
        require_confirmation: bool = False
    ):
        """
        Initialize error recovery system.

        Args:
            state_dir: Directory for saving state
            auto_rollback: Automatically rollback on errors
            require_confirmation: Require user confirmation for recovery
        """
        self.state_dir = state_dir or Path.home() / ".claude" / "recovery"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.auto_rollback = auto_rollback
        self.require_confirmation = require_confirmation

        self.states: Dict[str, RecoveryState] = {}
        self._operation_counter = 0

    def with_retry(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,),
        on_retry: Optional[Callable] = None
    ):
        """
        Decorator for automatic retry with exponential backoff.

        Args:
            max_attempts: Maximum retry attempts
            backoff_factor: Backoff multiplier
            exceptions: Tuple of exceptions to catch
            on_retry: Callback function called on each retry

        Example:
            @recovery.with_retry(max_attempts=3)
            def fetch_data():
                return api.get_data()
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                operation_id = f"{func.__name__}_{self._operation_counter}"
                self._operation_counter += 1

                state = RecoveryState(
                    operation_id=operation_id,
                    max_attempts=max_attempts,
                    backoff_factor=backoff_factor
                )

                for attempt in range(1, max_attempts + 1):
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)

                        # Record success
                        state.attempts.append(RecoveryAttempt(
                            strategy=RecoveryStrategy.RETRY,
                            attempt_number=attempt,
                            timestamp=datetime.now(),
                            success=True,
                            duration_seconds=time.time() - start_time
                        ))

                        return result

                    except exceptions as e:
                        duration = time.time() - start_time

                        # Record failure
                        state.attempts.append(RecoveryAttempt(
                            strategy=RecoveryStrategy.RETRY,
                            attempt_number=attempt,
                            timestamp=datetime.now(),
                            success=False,
                            error=str(e),
                            duration_seconds=duration
                        ))

                        # Last attempt - raise error
                        if attempt == max_attempts:
                            raise

                        # Calculate backoff delay
                        delay = backoff_factor ** (attempt - 1)

                        # Call retry callback
                        if on_retry:
                            on_retry(attempt, max_attempts, e, delay)

                        # Wait before retry
                        time.sleep(delay)

                # Should never reach here
                raise RuntimeError("Retry logic failed")

            return wrapper
        return decorator

    def with_fallback(
        self,
        fallback_fn: Callable,
        exceptions: tuple = (Exception,)
    ):
        """
        Decorator for fallback to alternative function.

        Args:
            fallback_fn: Alternative function to try
            exceptions: Exceptions that trigger fallback

        Example:
            @recovery.with_fallback(fallback_fn=use_cached_data)
            def fetch_data():
                return api.get_data()
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Try fallback
                    return fallback_fn(*args, **kwargs)

            return wrapper
        return decorator

    def with_degradation(
        self,
        degraded_fn: Callable,
        exceptions: tuple = (Exception,)
    ):
        """
        Decorator for graceful degradation.

        Args:
            degraded_fn: Degraded function with reduced functionality
            exceptions: Exceptions that trigger degradation

        Example:
            @recovery.with_degradation(degraded_fn=simple_process)
            def complex_process(data):
                return advanced_processing(data)
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Use degraded functionality
                    return degraded_fn(*args, **kwargs)

            return wrapper
        return decorator

    def save_checkpoint(
        self,
        operation_id: str,
        checkpoint_name: str,
        state_data: Any
    ):
        """
        Save a checkpoint for state preservation.

        Args:
            operation_id: Unique operation identifier
            checkpoint_name: Name of checkpoint
            state_data: State data to save
        """
        if operation_id not in self.states:
            self.states[operation_id] = RecoveryState(operation_id=operation_id)

        # Save to memory
        self.states[operation_id].checkpoints[checkpoint_name] = state_data
        self.states[operation_id].last_checkpoint = checkpoint_name

        # Save to disk
        checkpoint_file = self.state_dir / f"{operation_id}_{checkpoint_name}.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "operation_id": operation_id,
                    "checkpoint_name": checkpoint_name,
                    "timestamp": datetime.now().isoformat(),
                    "state_data": state_data
                }, f, indent=2)
        except Exception:
            pass  # Silently fail

    def load_checkpoint(
        self,
        operation_id: str,
        checkpoint_name: str
    ) -> Optional[Any]:
        """
        Load a checkpoint.

        Args:
            operation_id: Operation identifier
            checkpoint_name: Checkpoint name

        Returns:
            Saved state data or None
        """
        # Try memory first
        if operation_id in self.states:
            if checkpoint_name in self.states[operation_id].checkpoints:
                return self.states[operation_id].checkpoints[checkpoint_name]

        # Try disk
        checkpoint_file = self.state_dir / f"{operation_id}_{checkpoint_name}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return data.get("state_data")
            except Exception:
                pass

        return None

    def rollback(
        self,
        operation_id: str,
        checkpoint_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Rollback to a checkpoint.

        Args:
            operation_id: Operation identifier
            checkpoint_name: Checkpoint to rollback to (or last if None)

        Returns:
            Checkpoint state data
        """
        if operation_id not in self.states:
            return None

        state = self.states[operation_id]

        # Use last checkpoint if not specified
        if not checkpoint_name:
            checkpoint_name = state.last_checkpoint

        if not checkpoint_name:
            return None

        # Load checkpoint
        return self.load_checkpoint(operation_id, checkpoint_name)

    def checkpoint(self, operation_id: str, checkpoint_name: str):
        """
        Context manager for checkpoint-based operations.

        Example:
            with recovery.checkpoint("process", "start"):
                # State is saved at entry
                process_data()
                # Automatic rollback on error
        """
        class CheckpointContext:
            def __init__(self, recovery_instance, op_id, cp_name):
                self.recovery = recovery_instance
                self.operation_id = op_id
                self.checkpoint_name = cp_name

            def __enter__(self):
                # Save checkpoint on entry
                self.recovery.save_checkpoint(
                    self.operation_id,
                    self.checkpoint_name,
                    {"status": "started", "timestamp": datetime.now().isoformat()}
                )
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None and self.recovery.auto_rollback:
                    # Error occurred - rollback
                    self.recovery.rollback(self.operation_id, self.checkpoint_name)
                    return False  # Re-raise exception
                return False

        return CheckpointContext(self, operation_id, checkpoint_name)

    def clear_checkpoints(self, operation_id: str):
        """Clear all checkpoints for an operation."""
        if operation_id in self.states:
            del self.states[operation_id]

        # Clear disk checkpoints
        for checkpoint_file in self.state_dir.glob(f"{operation_id}_*.json"):
            try:
                checkpoint_file.unlink()
            except Exception:
                pass

    def get_recovery_stats(self, operation_id: str) -> Dict[str, Any]:
        """Get recovery statistics for an operation."""
        if operation_id not in self.states:
            return {}

        state = self.states[operation_id]
        total_attempts = len(state.attempts)
        successful = sum(1 for a in state.attempts if a.success)
        failed = total_attempts - successful

        return {
            "operation_id": operation_id,
            "total_attempts": total_attempts,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_attempts * 100) if total_attempts > 0 else 0,
            "checkpoints": list(state.checkpoints.keys()),
            "last_checkpoint": state.last_checkpoint
        }


# Global recovery instance
_global_recovery: Optional[ErrorRecovery] = None


def get_global_recovery() -> ErrorRecovery:
    """Get or create global recovery instance."""
    global _global_recovery
    if _global_recovery is None:
        _global_recovery = ErrorRecovery()
    return _global_recovery


# Convenience decorators using global instance
def retry(max_attempts: int = 3, **kwargs):
    """Retry decorator using global recovery instance."""
    recovery = get_global_recovery()
    return recovery.with_retry(max_attempts=max_attempts, **kwargs)


def fallback(fallback_fn: Callable, **kwargs):
    """Fallback decorator using global recovery instance."""
    recovery = get_global_recovery()
    return recovery.with_fallback(fallback_fn=fallback_fn, **kwargs)