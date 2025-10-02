"""Real-Time Monitoring for Training and Optimization.

This module provides real-time monitoring capabilities:
1. Performance monitoring (time, memory, GPU)
2. Training logging and metric tracking
3. Live plotting during training
4. Progress tracking and ETA estimation

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from collections import deque
import numpy as np

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import GPUtil for GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# =============================================================================
# Performance Monitor
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics.

    Attributes:
        cpu_percent: CPU utilization percentage
        memory_mb: Memory usage in MB
        gpu_util: GPU utilization percentage (if available)
        gpu_memory_mb: GPU memory usage in MB (if available)
        timestamp: Time of measurement
    """
    cpu_percent: float
    memory_mb: float
    gpu_util: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Monitor system performance during execution.

    Tracks CPU, memory, and optionally GPU usage.
    """

    def __init__(self, track_gpu: bool = True):
        """Initialize performance monitor.

        Args:
            track_gpu: Whether to track GPU metrics
        """
        self.track_gpu = track_gpu and GPU_AVAILABLE
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()

    def capture(self) -> PerformanceMetrics:
        """Capture current performance metrics.

        Returns:
            PerformanceMetrics object
        """
        # CPU and memory
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
        else:
            cpu_percent = 0.0
            memory_mb = 0.0

        # GPU
        gpu_util = None
        gpu_memory_mb = None

        if self.track_gpu:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_util = gpu.load * 100
                    gpu_memory_mb = gpu.memoryUsed
            except:
                pass

        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_util=gpu_util,
            gpu_memory_mb=gpu_memory_mb
        )

        self.metrics_history.append(metrics)

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        if not self.metrics_history:
            return {}

        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_mb for m in self.metrics_history]

        summary = {
            'total_time': time.time() - self.start_time,
            'n_samples': len(self.metrics_history),
            'cpu': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            },
            'memory': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values)
            }
        }

        # Add GPU stats if available
        gpu_util_values = [m.gpu_util for m in self.metrics_history if m.gpu_util is not None]
        if gpu_util_values:
            summary['gpu'] = {
                'mean': np.mean(gpu_util_values),
                'max': np.max(gpu_util_values),
                'min': np.min(gpu_util_values)
            }

        return summary

    def reset(self):
        """Reset monitoring."""
        self.metrics_history.clear()
        self.start_time = time.time()


# =============================================================================
# Training Logger
# =============================================================================

class TrainingLogger:
    """Log training metrics to file and memory.

    Supports JSON and CSV output formats.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_interval: int = 1,
        save_format: str = 'json'
    ):
        """Initialize training logger.

        Args:
            log_dir: Directory for log files
            log_interval: Log every N steps
            save_format: 'json' or 'csv'
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_interval = log_interval
        self.save_format = save_format

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, List[Any]] = {}
        self.step = 0

    def log(self, **kwargs):
        """Log metrics.

        Args:
            **kwargs: Metric name-value pairs
        """
        self.step += 1

        # Only log at specified interval
        if self.step % self.log_interval != 0:
            return

        # Add step number
        kwargs['step'] = self.step
        kwargs['timestamp'] = time.time()

        # Store in memory
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        # Save to file if log_dir specified
        if self.log_dir:
            self._save_to_file(kwargs)

    def _save_to_file(self, metrics_dict: Dict[str, Any]):
        """Save metrics to file.

        Args:
            metrics_dict: Dictionary of metrics
        """
        if self.save_format == 'json':
            log_file = self.log_dir / f'metrics_step_{self.step}.json'
            with open(log_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)

        elif self.save_format == 'csv':
            log_file = self.log_dir / 'metrics.csv'

            # Create header if file doesn't exist
            if not log_file.exists():
                with open(log_file, 'w') as f:
                    f.write(','.join(metrics_dict.keys()) + '\n')

            # Append data
            with open(log_file, 'a') as f:
                values = [str(v) for v in metrics_dict.values()]
                f.write(','.join(values) + '\n')

    def get_metric(self, name: str) -> List[Any]:
        """Get metric history.

        Args:
            name: Metric name

        Returns:
            List of metric values
        """
        return self.metrics.get(name, [])

    def get_all_metrics(self) -> Dict[str, List[Any]]:
        """Get all metrics.

        Returns:
            Dictionary mapping metric names to value lists
        """
        return self.metrics.copy()

    def save_summary(self, path: Optional[Path] = None):
        """Save summary of all metrics.

        Args:
            path: Path to save summary (defaults to log_dir/summary.json)
        """
        if path is None:
            if self.log_dir is None:
                raise ValueError("Must provide path or log_dir")
            path = self.log_dir / 'summary.json'

        summary = {
            'total_steps': self.step,
            'metrics': {}
        }

        # Compute statistics for each metric
        for name, values in self.metrics.items():
            if name in ['step', 'timestamp']:
                continue

            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                summary['metrics'][name] = {
                    'final': numeric_values[-1],
                    'mean': float(np.mean(numeric_values)),
                    'std': float(np.std(numeric_values)),
                    'min': float(np.min(numeric_values)),
                    'max': float(np.max(numeric_values))
                }

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to {path}")


# =============================================================================
# Metrics Tracker
# =============================================================================

class MetricsTracker:
    """Track and compute rolling statistics for metrics.

    Maintains rolling windows for computing moving averages.
    """

    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.

        Args:
            window_size: Size of rolling window
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.counts: Dict[str, int] = {}

    def update(self, name: str, value: float):
        """Update metric with new value.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
            self.counts[name] = 0

        self.metrics[name].append(value)
        self.counts[name] += 1

    def get_mean(self, name: str) -> Optional[float]:
        """Get rolling mean of metric.

        Args:
            name: Metric name

        Returns:
            Rolling mean or None if no data
        """
        if name not in self.metrics or not self.metrics[name]:
            return None

        return float(np.mean(self.metrics[name]))

    def get_std(self, name: str) -> Optional[float]:
        """Get rolling std of metric.

        Args:
            name: Metric name

        Returns:
            Rolling std or None if no data
        """
        if name not in self.metrics or not self.metrics[name]:
            return None

        return float(np.std(self.metrics[name]))

    def get_last(self, name: str) -> Optional[float]:
        """Get last value of metric.

        Args:
            name: Metric name

        Returns:
            Last value or None if no data
        """
        if name not in self.metrics or not self.metrics[name]:
            return None

        return self.metrics[name][-1]

    def get_count(self, name: str) -> int:
        """Get total count of metric updates.

        Args:
            name: Metric name

        Returns:
            Total count
        """
        return self.counts.get(name, 0)

    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with statistics
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}

        values = list(self.metrics[name])

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'last': values[-1],
            'count': self.counts[name]
        }


# =============================================================================
# Live Plotter (Optional - requires matplotlib with interactive backend)
# =============================================================================

class LivePlotter:
    """Live plotting during training.

    Updates plots in real-time as training progresses.
    """

    def __init__(self, metrics: List[str], window_size: int = 100):
        """Initialize live plotter.

        Args:
            metrics: List of metric names to plot
            window_size: Size of display window
        """
        self.metrics = metrics
        self.window_size = window_size
        self.data: Dict[str, deque] = {m: deque(maxlen=window_size) for m in metrics}

        try:
            import matplotlib.pyplot as plt
            plt.ion()  # Interactive mode
            self.plt = plt
            self.fig, self.axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)))
            if len(metrics) == 1:
                self.axes = [self.axes]

            self.lines = []
            for i, metric in enumerate(metrics):
                line, = self.axes[i].plot([], [], 'b-', linewidth=2)
                self.lines.append(line)
                self.axes[i].set_ylabel(metric)
                self.axes[i].grid(True, alpha=0.3)

            self.axes[-1].set_xlabel('Step')
            self.fig.tight_layout()
            self.available = True

        except ImportError:
            self.available = False
            print("Warning: matplotlib not available for live plotting")

    def update(self, **kwargs):
        """Update plots with new data.

        Args:
            **kwargs: Metric name-value pairs
        """
        if not self.available:
            return

        # Update data
        for metric in self.metrics:
            if metric in kwargs:
                self.data[metric].append(kwargs[metric])

        # Update plots
        for i, metric in enumerate(self.metrics):
            if self.data[metric]:
                x = list(range(len(self.data[metric])))
                y = list(self.data[metric])

                self.lines[i].set_data(x, y)

                self.axes[i].relim()
                self.axes[i].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot window."""
        if self.available:
            self.plt.close(self.fig)


# =============================================================================
# Progress Tracker
# =============================================================================

class ProgressTracker:
    """Track progress and estimate time remaining.

    Computes ETA based on recent progress.
    """

    def __init__(self, total_steps: int, smoothing: float = 0.1):
        """Initialize progress tracker.

        Args:
            total_steps: Total number of steps
            smoothing: Smoothing factor for ETA estimation
        """
        self.total_steps = total_steps
        self.smoothing = smoothing

        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.step_times: deque = deque(maxlen=100)

    def update(self, step: Optional[int] = None):
        """Update progress.

        Args:
            step: Current step (increments by 1 if None)
        """
        current_time = time.time()

        if step is None:
            self.current_step += 1
        else:
            self.current_step = step

        # Track step time
        step_time = current_time - self.last_update_time
        self.step_times.append(step_time)
        self.last_update_time = current_time

    def get_progress(self) -> float:
        """Get progress fraction.

        Returns:
            Progress in [0, 1]
        """
        return self.current_step / self.total_steps

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Elapsed time
        """
        return time.time() - self.start_time

    def get_eta(self) -> float:
        """Get estimated time remaining in seconds.

        Returns:
            Estimated time remaining
        """
        if not self.step_times:
            return 0.0

        # Use recent average step time
        avg_step_time = np.mean(self.step_times)

        remaining_steps = self.total_steps - self.current_step
        eta = remaining_steps * avg_step_time

        return eta

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary.

        Returns:
            Dictionary with progress information
        """
        elapsed = self.get_elapsed_time()
        eta = self.get_eta()
        progress = self.get_progress()

        return {
            'step': self.current_step,
            'total_steps': self.total_steps,
            'progress': progress,
            'percent': progress * 100,
            'elapsed_time': elapsed,
            'eta': eta,
            'steps_per_sec': self.current_step / elapsed if elapsed > 0 else 0
        }

    def print_progress(self):
        """Print progress bar."""
        summary = self.get_summary()

        progress = summary['progress']
        percent = summary['percent']
        elapsed = summary['elapsed_time']
        eta = summary['eta']

        # Create progress bar
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '=' * filled + '-' * (bar_length - filled)

        print(f"\rProgress: [{bar}] {percent:.1f}% | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end='', flush=True)

        if progress >= 1.0:
            print()  # New line when complete


# =============================================================================
# Convenience Functions
# =============================================================================

def monitor_training(
    train_func: Callable,
    total_steps: int,
    log_dir: Optional[Path] = None,
    log_interval: int = 10,
    track_performance: bool = True
) -> Dict[str, Any]:
    """Monitor training function execution.

    Args:
        train_func: Training function (should call logger.log(...) internally)
        total_steps: Total training steps
        log_dir: Directory for logs
        log_interval: Logging interval
        track_performance: Whether to track system performance

    Returns:
        Dictionary with training results and logs
    """
    logger = TrainingLogger(log_dir=log_dir, log_interval=log_interval)
    progress = ProgressTracker(total_steps)

    perf_monitor = None
    if track_performance:
        perf_monitor = PerformanceMonitor()

    # Run training
    try:
        result = train_func(logger=logger, progress=progress)
    finally:
        # Save logs
        if log_dir:
            logger.save_summary()

    # Collect results
    results = {
        'training_result': result,
        'metrics': logger.get_all_metrics(),
        'progress': progress.get_summary()
    }

    if perf_monitor:
        results['performance'] = perf_monitor.get_summary()

    return results
