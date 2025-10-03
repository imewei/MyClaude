"""Visualization and Monitoring for Optimal Control.

This module provides visualization and monitoring capabilities:
1. Interactive Plotly Dash dashboards
2. Static plotting utilities (matplotlib)
3. Real-time training monitors
4. Performance profiling and analysis
5. Result visualization and comparison

Author: Nonequilibrium Physics Agents
"""

__version__ = "4.3.0-dev"

# Plotting utilities
try:
    from .plotting import (
        plot_trajectory,
        plot_control,
        plot_phase_portrait,
        plot_convergence,
        plot_quantum_state,
        plot_fidelity,
        create_animation,
        save_figure
    )
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plot_trajectory = None
    plot_control = None
    plot_phase_portrait = None
    plot_convergence = None
    plot_quantum_state = None
    plot_fidelity = None
    create_animation = None
    save_figure = None

# Dashboard
try:
    from .dashboard import (
        OptimalControlDashboard,
        TrainingMonitor,
        create_dashboard,
        run_dashboard
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    OptimalControlDashboard = None
    TrainingMonitor = None
    create_dashboard = None
    run_dashboard = None

# Monitoring
try:
    from .monitoring import (
        PerformanceMonitor,
        TrainingLogger,
        MetricsTracker,
        LivePlotter
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PerformanceMonitor = None
    TrainingLogger = None
    MetricsTracker = None
    LivePlotter = None

# Profiling
try:
    from .profiling import (
        profile_solver,
        profile_training,
        memory_profile,
        create_profile_report
    )
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    profile_solver = None
    profile_training = None
    memory_profile = None
    create_profile_report = None

__all__ = [
    # Plotting
    'plot_trajectory',
    'plot_control',
    'plot_phase_portrait',
    'plot_convergence',
    'plot_quantum_state',
    'plot_fidelity',
    'create_animation',
    'save_figure',
    'PLOTTING_AVAILABLE',

    # Dashboard
    'OptimalControlDashboard',
    'TrainingMonitor',
    'create_dashboard',
    'run_dashboard',
    'DASHBOARD_AVAILABLE',

    # Monitoring
    'PerformanceMonitor',
    'TrainingLogger',
    'MetricsTracker',
    'LivePlotter',
    'MONITORING_AVAILABLE',

    # Profiling
    'profile_solver',
    'profile_training',
    'memory_profile',
    'create_profile_report',
    'PROFILING_AVAILABLE',
]
