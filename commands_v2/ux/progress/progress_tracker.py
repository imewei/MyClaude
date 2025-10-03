"""
Comprehensive progress tracking system with rich terminal output.

Provides beautiful progress bars, spinners, status indicators, and time estimates
for long-running operations in the command executor framework.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import threading
from contextlib import contextmanager

try:
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
        MofNCompleteColumn
    )
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressStatus(Enum):
    """Status indicators for progress items."""
    PENDING = "⏳"
    RUNNING = "▶"
    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    SKIPPED = "⊘"


@dataclass
class ProgressItem:
    """Individual progress tracking item."""
    id: str
    description: str
    total: Optional[int] = None
    completed: int = 0
    status: ProgressStatus = ProgressStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def progress_percentage(self) -> Optional[float]:
        """Calculate progress percentage."""
        if self.total and self.total > 0:
            return (self.completed / self.total) * 100
        return None

    @property
    def eta(self) -> Optional[timedelta]:
        """Estimate time to completion."""
        if not self.total or self.completed == 0 or not self.start_time:
            return None

        elapsed = self.elapsed_time
        if not elapsed:
            return None

        rate = self.completed / elapsed.total_seconds()
        remaining = self.total - self.completed

        if rate > 0:
            seconds_remaining = remaining / rate
            return timedelta(seconds=seconds_remaining)

        return None


class ProgressTracker:
    """
    Comprehensive progress tracking system.

    Features:
    - Rich progress bars with multiple columns
    - Multi-level hierarchical progress
    - Spinner animations for indeterminate operations
    - Status indicators (success, error, warning, info)
    - Time estimates (ETA and elapsed)
    - Percentage completion
    - Collapsible sections for detailed progress

    Example:
        tracker = ProgressTracker()

        # Create a progress item
        task_id = tracker.add_task("Processing files", total=100)

        # Update progress
        for i in range(100):
            tracker.update(task_id, completed=i+1)
            time.sleep(0.1)

        # Mark as complete
        tracker.complete(task_id, ProgressStatus.SUCCESS)
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        enabled: bool = True,
        show_time: bool = True,
        show_percentage: bool = True,
        show_eta: bool = True,
        refresh_rate: int = 10
    ):
        """
        Initialize progress tracker.

        Args:
            console: Rich console instance
            enabled: Whether tracking is enabled
            show_time: Show elapsed time
            show_percentage: Show completion percentage
            show_eta: Show estimated time remaining
            refresh_rate: Refresh rate per second
        """
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.enabled = enabled and RICH_AVAILABLE
        self.show_time = show_time
        self.show_percentage = show_percentage
        self.show_eta = show_eta
        self.refresh_rate = refresh_rate

        self.items: Dict[str, ProgressItem] = {}
        self.progress: Optional[Progress] = None
        self.task_ids: Dict[str, Any] = {}  # Maps item_id to rich task_id
        self._lock = threading.Lock()
        self._counter = 0

    def _generate_id(self) -> str:
        """Generate unique ID for progress item."""
        with self._lock:
            self._counter += 1
            return f"task_{self._counter}"

    def add_task(
        self,
        description: str,
        total: Optional[int] = None,
        parent_id: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Add a new progress task.

        Args:
            description: Task description
            total: Total number of items (None for indeterminate)
            parent_id: Parent task ID for hierarchical progress
            **metadata: Additional metadata

        Returns:
            Task ID
        """
        task_id = self._generate_id()

        item = ProgressItem(
            id=task_id,
            description=description,
            total=total,
            parent_id=parent_id,
            metadata=metadata
        )

        with self._lock:
            self.items[task_id] = item

            # Update parent's children list
            if parent_id and parent_id in self.items:
                self.items[parent_id].children.append(task_id)

        return task_id

    def update(
        self,
        task_id: str,
        completed: Optional[int] = None,
        advance: Optional[int] = None,
        description: Optional[str] = None,
        **metadata
    ):
        """
        Update task progress.

        Args:
            task_id: Task ID
            completed: New completed count
            advance: Increment completed by this amount
            description: Update description
            **metadata: Update metadata
        """
        with self._lock:
            if task_id not in self.items:
                return

            item = self.items[task_id]

            if completed is not None:
                item.completed = completed
            elif advance is not None:
                item.completed += advance

            if description is not None:
                item.description = description

            if metadata:
                item.metadata.update(metadata)

            # Start timer if not started
            if item.status == ProgressStatus.PENDING:
                item.status = ProgressStatus.RUNNING
                item.start_time = datetime.now()

    def complete(self, task_id: str, status: ProgressStatus = ProgressStatus.SUCCESS):
        """
        Mark task as complete.

        Args:
            task_id: Task ID
            status: Final status
        """
        with self._lock:
            if task_id not in self.items:
                return

            item = self.items[task_id]
            item.status = status
            item.end_time = datetime.now()

            # Set completed to total if applicable
            if item.total is not None:
                item.completed = item.total

    def get_item(self, task_id: str) -> Optional[ProgressItem]:
        """Get progress item by ID."""
        return self.items.get(task_id)

    def get_children(self, task_id: str) -> List[ProgressItem]:
        """Get all child items."""
        item = self.items.get(task_id)
        if not item:
            return []

        return [self.items[child_id] for child_id in item.children if child_id in self.items]

    @contextmanager
    def live_progress(self):
        """
        Context manager for live progress updates.

        Example:
            with tracker.live_progress():
                task = tracker.add_task("Processing", total=100)
                for i in range(100):
                    tracker.update(task, advance=1)
        """
        if not self.enabled:
            yield
            return

        # Create progress columns
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
        ]

        if self.show_percentage:
            columns.append(TaskProgressColumn())

        columns.append(MofNCompleteColumn())

        if self.show_time:
            columns.append(TimeElapsedColumn())

        if self.show_eta:
            columns.append(TimeRemainingColumn())

        self.progress = Progress(*columns, refresh_per_second=self.refresh_rate)

        with self.progress:
            yield

    def format_status(self, item: ProgressItem) -> str:
        """Format item status with icon and color."""
        status_colors = {
            ProgressStatus.PENDING: "yellow",
            ProgressStatus.RUNNING: "blue",
            ProgressStatus.SUCCESS: "green",
            ProgressStatus.ERROR: "red",
            ProgressStatus.WARNING: "yellow",
            ProgressStatus.INFO: "cyan",
            ProgressStatus.SKIPPED: "dim"
        }

        color = status_colors.get(item.status, "white")
        icon = item.status.value

        return f"[{color}]{icon}[/{color}]"

    def format_time(self, td: Optional[timedelta]) -> str:
        """Format timedelta for display."""
        if not td:
            return "--:--"

        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def render_table(self, include_children: bool = True) -> Optional[Table]:
        """
        Render progress as a rich table.

        Args:
            include_children: Include child tasks

        Returns:
            Rich Table or None
        """
        if not self.enabled or not self.console:
            return None

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", style="dim", width=6)
        table.add_column("Task", style="cyan")
        table.add_column("Progress", justify="right")

        if self.show_time:
            table.add_column("Elapsed", justify="right")

        if self.show_eta:
            table.add_column("ETA", justify="right")

        # Get root items (items without parent)
        root_items = [item for item in self.items.values() if not item.parent_id]

        for item in root_items:
            self._add_item_to_table(table, item, include_children)

        return table

    def _add_item_to_table(self, table: Table, item: ProgressItem, include_children: bool, depth: int = 0):
        """Add item and its children to table."""
        # Format task name with indentation
        indent = "  " * depth
        task_name = f"{indent}{item.description}"

        # Format progress
        if item.total:
            progress = f"{item.completed}/{item.total}"
            if self.show_percentage:
                pct = item.progress_percentage or 0
                progress += f" ({pct:.1f}%)"
        else:
            progress = str(item.completed) if item.completed > 0 else "..."

        row = [
            self.format_status(item),
            task_name,
            progress
        ]

        if self.show_time:
            row.append(self.format_time(item.elapsed_time))

        if self.show_eta:
            row.append(self.format_time(item.eta))

        table.add_row(*row)

        # Add children
        if include_children:
            for child in self.get_children(item.id):
                self._add_item_to_table(table, child, include_children, depth + 1)

    def print_summary(self):
        """Print summary of all tasks."""
        if not self.enabled or not self.console:
            return

        table = self.render_table()
        if table:
            self.console.print(table)

    def clear(self):
        """Clear all progress items."""
        with self._lock:
            self.items.clear()
            self.task_ids.clear()


# Convenience function
def create_progress_tracker(**kwargs) -> ProgressTracker:
    """Create a progress tracker with default settings."""
    return ProgressTracker(**kwargs)


# Global tracker instance
_global_tracker: Optional[ProgressTracker] = None


def get_global_tracker() -> ProgressTracker:
    """Get or create global progress tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = create_progress_tracker()
    return _global_tracker