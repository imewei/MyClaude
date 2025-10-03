"""
Live updating dashboard for real-time monitoring.

Displays command execution status, agent activity, resource usage,
cache statistics, and performance metrics in a beautiful terminal UI.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
import psutil
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class DashboardMetrics:
    """Metrics displayed on dashboard."""
    current_command: str = ""
    command_status: str = "Idle"
    active_agents: List[str] = field(default_factory=list)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0
    operations_completed: int = 0
    operations_failed: int = 0
    total_execution_time: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class LiveDashboard:
    """
    Real-time dashboard for monitoring command execution.

    Features:
    - Command execution status
    - Active agent monitoring
    - Resource usage (CPU, Memory, Disk)
    - Cache statistics with hit rate
    - Performance metrics
    - Error tracking
    - Professional table/panel layout

    Example:
        dashboard = LiveDashboard()

        with dashboard.live():
            dashboard.update_command("optimize", "Running")
            dashboard.add_agent("Scientific Agent")
            dashboard.increment_cache_hits()

            # Do work...

            dashboard.remove_agent("Scientific Agent")
            dashboard.update_command("optimize", "Complete")
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        enabled: bool = True,
        refresh_rate: int = 4
    ):
        """
        Initialize live dashboard.

        Args:
            console: Rich console instance
            enabled: Whether dashboard is enabled
            refresh_rate: Updates per second
        """
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.enabled = enabled and RICH_AVAILABLE
        self.refresh_rate = refresh_rate

        self.metrics = DashboardMetrics()
        self._lock = threading.Lock()
        self._live: Optional[Live] = None
        self._update_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def update_command(self, command: str, status: str):
        """
        Update current command information.

        Args:
            command: Command name
            status: Command status (Running, Complete, Failed, etc.)
        """
        with self._lock:
            self.metrics.current_command = command
            self.metrics.command_status = status
            self.metrics.last_update = datetime.now()

    def add_agent(self, agent_name: str):
        """Add active agent."""
        with self._lock:
            if agent_name not in self.metrics.active_agents:
                self.metrics.active_agents.append(agent_name)
                self.metrics.last_update = datetime.now()

    def remove_agent(self, agent_name: str):
        """Remove active agent."""
        with self._lock:
            if agent_name in self.metrics.active_agents:
                self.metrics.active_agents.remove(agent_name)
                self.metrics.last_update = datetime.now()

    def increment_cache_hits(self, count: int = 1):
        """Increment cache hit counter."""
        with self._lock:
            self.metrics.cache_hits += count
            self.metrics.last_update = datetime.now()

    def increment_cache_misses(self, count: int = 1):
        """Increment cache miss counter."""
        with self._lock:
            self.metrics.cache_misses += count
            self.metrics.last_update = datetime.now()

    def update_cache_size(self, size: int):
        """Update cache size."""
        with self._lock:
            self.metrics.cache_size = size
            self.metrics.last_update = datetime.now()

    def increment_operations(self, completed: int = 0, failed: int = 0):
        """Update operation counters."""
        with self._lock:
            self.metrics.operations_completed += completed
            self.metrics.operations_failed += failed
            self.metrics.last_update = datetime.now()

    def add_execution_time(self, seconds: float):
        """Add to total execution time."""
        with self._lock:
            self.metrics.total_execution_time += seconds
            self.metrics.last_update = datetime.now()

    def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU and Memory
            self.metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            self.metrics.memory_percent = mem.percent
            self.metrics.memory_used_mb = mem.used / (1024 * 1024)
            self.metrics.memory_total_mb = mem.total / (1024 * 1024)

            # Disk
            disk = psutil.disk_usage('/')
            self.metrics.disk_percent = disk.percent
        except Exception:
            pass  # Silently fail if psutil unavailable

    def _create_command_panel(self) -> Panel:
        """Create command status panel."""
        status_color = {
            "Idle": "dim",
            "Running": "blue",
            "Complete": "green",
            "Failed": "red",
            "Warning": "yellow"
        }.get(self.metrics.command_status, "white")

        content = f"""[bold]Command:[/bold] {self.metrics.current_command or 'None'}
[bold]Status:[/bold] [{status_color}]{self.metrics.command_status}[/{status_color}]"""

        return Panel(
            content,
            title="[bold cyan]Command Execution[/bold cyan]",
            border_style="cyan"
        )

    def _create_agents_panel(self) -> Panel:
        """Create active agents panel."""
        if self.metrics.active_agents:
            agents_text = "\n".join([f"▶ {agent}" for agent in self.metrics.active_agents])
        else:
            agents_text = "[dim]No active agents[/dim]"

        return Panel(
            agents_text,
            title=f"[bold magenta]Active Agents ({len(self.metrics.active_agents)})[/bold magenta]",
            border_style="magenta"
        )

    def _create_resources_table(self) -> Table:
        """Create resource usage table."""
        table = Table(show_header=True, header_style="bold yellow", box=None)
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", justify="right")
        table.add_column("Bar", width=20)

        # CPU
        cpu_bar = self._create_usage_bar(self.metrics.cpu_percent)
        table.add_row("CPU", f"{self.metrics.cpu_percent:.1f}%", cpu_bar)

        # Memory
        mem_bar = self._create_usage_bar(self.metrics.memory_percent)
        mem_text = f"{self.metrics.memory_used_mb:.0f}MB / {self.metrics.memory_total_mb:.0f}MB ({self.metrics.memory_percent:.1f}%)"
        table.add_row("Memory", mem_text, mem_bar)

        # Disk
        disk_bar = self._create_usage_bar(self.metrics.disk_percent)
        table.add_row("Disk", f"{self.metrics.disk_percent:.1f}%", disk_bar)

        return table

    def _create_usage_bar(self, percent: float) -> str:
        """Create a simple ASCII usage bar."""
        width = 20
        filled = int((percent / 100) * width)
        bar = "█" * filled + "░" * (width - filled)

        # Color based on usage
        if percent < 50:
            color = "green"
        elif percent < 80:
            color = "yellow"
        else:
            color = "red"

        return f"[{color}]{bar}[/{color}]"

    def _create_cache_panel(self) -> Panel:
        """Create cache statistics panel."""
        total = self.metrics.cache_hits + self.metrics.cache_misses
        hit_rate = (self.metrics.cache_hits / total * 100) if total > 0 else 0.0

        hit_rate_color = "green" if hit_rate > 70 else "yellow" if hit_rate > 40 else "red"

        content = f"""[bold]Hit Rate:[/bold] [{hit_rate_color}]{hit_rate:.1f}%[/{hit_rate_color}]
[bold]Hits:[/bold] {self.metrics.cache_hits:,}
[bold]Misses:[/bold] {self.metrics.cache_misses:,}
[bold]Size:[/bold] {self.metrics.cache_size:,} items"""

        return Panel(
            content,
            title="[bold green]Cache Statistics[/bold green]",
            border_style="green"
        )

    def _create_performance_panel(self) -> Panel:
        """Create performance metrics panel."""
        total_ops = self.metrics.operations_completed + self.metrics.operations_failed
        success_rate = (self.metrics.operations_completed / total_ops * 100) if total_ops > 0 else 0.0

        success_color = "green" if success_rate > 90 else "yellow" if success_rate > 70 else "red"

        content = f"""[bold]Total Operations:[/bold] {total_ops:,}
[bold]Completed:[/bold] [green]{self.metrics.operations_completed:,}[/green]
[bold]Failed:[/bold] [red]{self.metrics.operations_failed:,}[/red]
[bold]Success Rate:[/bold] [{success_color}]{success_rate:.1f}%[/{success_color}]
[bold]Total Time:[/bold] {self.metrics.total_execution_time:.2f}s"""

        return Panel(
            content,
            title="[bold blue]Performance Metrics[/bold blue]",
            border_style="blue"
        )

    def _create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )

        # Header: Command and Agents
        layout["header"].split_row(
            Layout(self._create_command_panel(), ratio=2),
            Layout(self._create_agents_panel(), ratio=1)
        )

        # Main: Resources and Metrics
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )

        resources_panel = Panel(
            self._create_resources_table(),
            title="[bold yellow]System Resources[/bold yellow]",
            border_style="yellow"
        )

        layout["main"]["left"].split_column(
            Layout(resources_panel, ratio=1),
            Layout(self._create_cache_panel(), ratio=1)
        )

        layout["main"]["right"].update(self._create_performance_panel())

        # Footer
        timestamp = self.metrics.last_update.strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"[dim]Last Update: {timestamp}[/dim]"
        layout["footer"].update(Panel(footer_text, border_style="dim"))

        return layout

    @contextmanager
    def live(self):
        """
        Context manager for live dashboard updates.

        Example:
            with dashboard.live():
                dashboard.update_command("optimize", "Running")
                # Do work...
        """
        if not self.enabled:
            yield
            return

        # Start update thread
        self._stop_flag.clear()
        self._update_thread = threading.Thread(target=self._auto_update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()

        # Start live display
        self._live = Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=False
        )

        try:
            with self._live:
                yield
        finally:
            # Stop update thread
            self._stop_flag.set()
            if self._update_thread:
                self._update_thread.join(timeout=1.0)

    def _auto_update_loop(self):
        """Background thread to auto-update system metrics."""
        while not self._stop_flag.is_set():
            try:
                with self._lock:
                    self._update_system_metrics()

                if self._live:
                    self._live.update(self._create_layout())

                time.sleep(1.0 / self.refresh_rate)
            except Exception:
                pass

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = DashboardMetrics()


# Convenience function
def create_dashboard(**kwargs) -> LiveDashboard:
    """Create a live dashboard with default settings."""
    return LiveDashboard(**kwargs)


# Global dashboard instance
_global_dashboard: Optional[LiveDashboard] = None


def get_global_dashboard() -> LiveDashboard:
    """Get or create global dashboard."""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = create_dashboard()
    return _global_dashboard