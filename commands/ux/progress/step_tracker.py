"""
Multi-step operation tracker.

Tracks complex operations with multiple sequential steps, showing
completion status, duration, and overall progress.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class StepStatus(Enum):
    """Status of individual step."""
    PENDING = "⏳ Pending"
    RUNNING = "▶ Running"
    COMPLETED = "✓ Completed"
    FAILED = "✗ Failed"
    SKIPPED = "⊘ Skipped"
    WARNING = "⚠ Warning"


@dataclass
class Step:
    """Individual step in multi-step operation."""
    id: str
    name: str
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate step duration."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def is_complete(self) -> bool:
        """Check if step is complete (success or failure)."""
        return self.status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]


class StepTracker:
    """
    Track complex multi-step operations.

    Features:
    - Step sequence visualization (completed, current, upcoming)
    - Individual step status tracking
    - Step duration measurement
    - Overall progress calculation
    - Expandable step details
    - Error tracking per step

    Example:
        tracker = StepTracker("Code Optimization")

        # Define steps
        tracker.add_step("analyze", "Analyze code")
        tracker.add_step("optimize", "Apply optimizations")
        tracker.add_step("test", "Run tests")
        tracker.add_step("report", "Generate report")

        # Execute steps
        tracker.start_step("analyze")
        # ... do analysis ...
        tracker.complete_step("analyze")

        tracker.start_step("optimize")
        # ... optimize ...
        tracker.complete_step("optimize")

        # Print summary
        tracker.print_summary()
    """

    def __init__(
        self,
        name: str,
        console: Optional[Console] = None,
        enabled: bool = True
    ):
        """
        Initialize step tracker.

        Args:
            name: Operation name
            console: Rich console instance
            enabled: Whether tracking is enabled
        """
        self.name = name
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.enabled = enabled and RICH_AVAILABLE

        self.steps: List[Step] = []
        self.step_map: Dict[str, Step] = {}
        self.current_step: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def add_step(
        self,
        step_id: str,
        name: str,
        description: str = "",
        **metadata
    ) -> Step:
        """
        Add a step to the sequence.

        Args:
            step_id: Unique step ID
            name: Step name
            description: Step description
            **metadata: Additional metadata

        Returns:
            Created step
        """
        step = Step(
            id=step_id,
            name=name,
            description=description,
            metadata=metadata
        )

        self.steps.append(step)
        self.step_map[step_id] = step

        return step

    def start_step(self, step_id: str):
        """
        Start executing a step.

        Args:
            step_id: Step ID to start
        """
        if step_id not in self.step_map:
            return

        step = self.step_map[step_id]
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        self.current_step = step_id

        # Start overall timer if first step
        if not self.start_time:
            self.start_time = datetime.now()

    def complete_step(
        self,
        step_id: str,
        status: StepStatus = StepStatus.COMPLETED,
        error: Optional[str] = None
    ):
        """
        Mark step as complete.

        Args:
            step_id: Step ID
            status: Final status
            error: Error message if failed
        """
        if step_id not in self.step_map:
            return

        step = self.step_map[step_id]
        step.status = status
        step.end_time = datetime.now()
        step.error = error

        if self.current_step == step_id:
            self.current_step = None

        # End overall timer if last step
        if all(s.is_complete for s in self.steps):
            self.end_time = datetime.now()

    def skip_step(self, step_id: str, reason: str = ""):
        """Skip a step."""
        self.complete_step(step_id, StepStatus.SKIPPED, reason)

    def fail_step(self, step_id: str, error: str):
        """Mark step as failed."""
        self.complete_step(step_id, StepStatus.FAILED, error)

    def get_step(self, step_id: str) -> Optional[Step]:
        """Get step by ID."""
        return self.step_map.get(step_id)

    def get_completed_steps(self) -> List[Step]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    def get_failed_steps(self) -> List[Step]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def get_progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if not self.steps:
            return 0.0

        completed = sum(1 for s in self.steps if s.is_complete)
        return (completed / len(self.steps)) * 100

    def get_total_duration(self) -> Optional[timedelta]:
        """Get total operation duration."""
        if not self.start_time:
            return None

        end = self.end_time or datetime.now()
        return end - self.start_time

    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(s.is_complete for s in self.steps)

    def has_failures(self) -> bool:
        """Check if any steps failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def _format_duration(self, td: Optional[timedelta]) -> str:
        """Format duration for display."""
        if not td:
            return "--"

        total_seconds = int(td.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def _get_step_color(self, step: Step) -> str:
        """Get color for step status."""
        colors = {
            StepStatus.PENDING: "dim",
            StepStatus.RUNNING: "blue",
            StepStatus.COMPLETED: "green",
            StepStatus.FAILED: "red",
            StepStatus.SKIPPED: "yellow",
            StepStatus.WARNING: "yellow"
        }
        return colors.get(step.status, "white")

    def render_table(self, show_details: bool = False) -> Optional[Table]:
        """
        Render steps as a table.

        Args:
            show_details: Show detailed information

        Returns:
            Rich Table or None
        """
        if not self.enabled or not self.console:
            return None

        table = Table(show_header=True, header_style="bold magenta", title=f"[bold]{self.name}[/bold]")
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Status", width=12)
        table.add_column("Step", style="cyan")

        if show_details:
            table.add_column("Duration", justify="right")
            table.add_column("Details")

        for idx, step in enumerate(self.steps, 1):
            color = self._get_step_color(step)
            status_text = f"[{color}]{step.status.value}[/{color}]"

            row = [str(idx), status_text, step.name]

            if show_details:
                row.append(self._format_duration(step.duration))

                # Add details
                if step.error:
                    row.append(f"[red]{step.error}[/red]")
                elif step.description:
                    row.append(f"[dim]{step.description}[/dim]")
                else:
                    row.append("")

            table.add_row(*row)

        return table

    def print_summary(self, show_details: bool = True):
        """
        Print step summary.

        Args:
            show_details: Show detailed information
        """
        if not self.enabled or not self.console:
            return

        table = self.render_table(show_details)
        if not table:
            return

        self.console.print()
        self.console.print(table)

        # Print overall stats
        progress = self.get_progress_percentage()
        duration = self.get_total_duration()
        completed = len(self.get_completed_steps())
        failed = len(self.get_failed_steps())

        stats = f"""[bold]Progress:[/bold] {progress:.1f}% ({completed}/{len(self.steps)} steps)
[bold]Duration:[/bold] {self._format_duration(duration)}
[bold]Failed:[/bold] {failed}"""

        self.console.print(Panel(stats, title="[bold cyan]Summary[/bold cyan]", border_style="cyan"))

    def print_progress(self):
        """Print current progress inline."""
        if not self.enabled or not self.console:
            return

        progress = self.get_progress_percentage()
        completed = len(self.get_completed_steps())
        total = len(self.steps)

        current_name = ""
        if self.current_step:
            current = self.step_map[self.current_step]
            current_name = current.name

        self.console.print(
            f"[bold]{self.name}:[/bold] {progress:.0f}% ({completed}/{total}) - "
            f"[blue]{current_name or 'Complete'}[/blue]"
        )


# Convenience function
def create_step_tracker(name: str, **kwargs) -> StepTracker:
    """Create a step tracker with default settings."""
    return StepTracker(name, **kwargs)