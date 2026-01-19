"""
Timer utility for profiling code blocks.

Used by load_profiler.py, activation_profiler.py, and other profiling tools.
"""

import time
from typing import Any


class Timer:
    """Context manager for timing code blocks.

    Usage:
        timer = Timer()
        with timer:
            # code to time
        print(f"Duration: {timer.duration_ms:.2f}ms")
    """

    def __init__(self) -> None:
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000
