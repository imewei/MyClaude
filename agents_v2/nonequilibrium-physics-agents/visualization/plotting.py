"""Plotting Utilities for Optimal Control.

This module provides plotting functions for visualizing:
1. State trajectories and control signals
2. Phase portraits and limit cycles
3. Convergence plots for optimization
4. Quantum states (density matrices, Bloch sphere)
5. Fidelity and performance metrics
6. Animations of control evolution

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# =============================================================================
# Trajectory Plotting
# =============================================================================

def plot_trajectory(
    t: np.ndarray,
    x: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "State Trajectory",
    xlabel: str = "Time",
    ylabel: str = "State",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot state trajectory over time.

    Args:
        t: Time array [n_steps]
        x: State array [n_steps, n_states] or [n_steps]
        labels: State labels
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    # Handle 1D or 2D state
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_states = x.shape[1]

    # Generate labels if not provided
    if labels is None:
        if n_states == 1:
            labels = ["x"]
        else:
            labels = [f"x_{i+1}" for i in range(n_states)]

    # Plot each state
    for i in range(n_states):
        ax.plot(t, x[:, i], label=labels[i], linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_control(
    t: np.ndarray,
    u: np.ndarray,
    labels: Optional[List[str]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Control Signal",
    xlabel: str = "Time",
    ylabel: str = "Control",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot control signal over time.

    Args:
        t: Time array [n_steps]
        u: Control array [n_steps, n_controls] or [n_steps]
        labels: Control labels
        bounds: (lower, upper) control bounds
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    # Handle 1D or 2D control
    if u.ndim == 1:
        u = u.reshape(-1, 1)

    n_controls = u.shape[1]

    # Generate labels if not provided
    if labels is None:
        if n_controls == 1:
            labels = ["u"]
        else:
            labels = [f"u_{i+1}" for i in range(n_controls)]

    # Plot each control
    for i in range(n_controls):
        ax.plot(t, u[:, i], label=labels[i], linewidth=2)

    # Plot bounds if provided
    if bounds is not None:
        lower, upper = bounds
        for i in range(n_controls):
            ax.axhline(lower[i], color='r', linestyle='--', alpha=0.5, label=f'Lower bound' if i == 0 else '')
            ax.axhline(upper[i], color='r', linestyle='--', alpha=0.5, label=f'Upper bound' if i == 0 else '')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_phase_portrait(
    x: np.ndarray,
    x_idx: int = 0,
    y_idx: int = 1,
    title: str = "Phase Portrait",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot phase portrait (2D projection of state space).

    Args:
        x: State array [n_steps, n_states]
        x_idx: Index of x-axis state
        y_idx: Index of y-axis state
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    # Extract states
    x_data = x[:, x_idx]
    y_data = x[:, y_idx]

    # Plot trajectory
    ax.plot(x_data, y_data, 'b-', linewidth=2, alpha=0.7)

    # Mark start and end
    ax.plot(x_data[0], y_data[0], 'go', markersize=10, label='Start')
    ax.plot(x_data[-1], y_data[-1], 'ro', markersize=10, label='End')

    # Add direction arrows
    n_arrows = 5
    indices = np.linspace(0, len(x_data)-1, n_arrows, dtype=int)
    for i in indices[:-1]:
        dx = x_data[i+1] - x_data[i]
        dy = y_data[i+1] - y_data[i]
        ax.arrow(x_data[i], y_data[i], dx*0.3, dy*0.3,
                head_width=0.05, head_length=0.05, fc='k', ec='k', alpha=0.5)

    if xlabel is None:
        xlabel = f"x_{x_idx+1}"
    if ylabel is None:
        ylabel = f"x_{y_idx+1}"

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


# =============================================================================
# Convergence Plotting
# =============================================================================

def plot_convergence(
    iterations: np.ndarray,
    costs: np.ndarray,
    title: str = "Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Cost",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot convergence of optimization algorithm.

    Args:
        iterations: Iteration numbers
        costs: Cost values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use log scale for y-axis
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(iterations, costs, 'b-', linewidth=2, marker='o', markersize=4)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add final value annotation
    final_cost = costs[-1]
    ax.annotate(f'Final: {final_cost:.2e}',
                xy=(iterations[-1], final_cost),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


# =============================================================================
# Quantum State Visualization
# =============================================================================

def plot_quantum_state(
    rho: np.ndarray,
    title: str = "Quantum State (Density Matrix)",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot quantum density matrix.

    Args:
        rho: Density matrix [n_dim, n_dim] (complex)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot real part
    im1 = ax1.imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1)
    ax1.set_title('Real Part')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1)

    # Plot imaginary part
    im2 = ax2.imshow(np.imag(rho), cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_title('Imaginary Part')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_fidelity(
    t: np.ndarray,
    fidelity: np.ndarray,
    title: str = "Fidelity vs Time",
    xlabel: str = "Time",
    ylabel: str = "Fidelity",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot fidelity over time.

    Args:
        t: Time array
        fidelity: Fidelity values [0, 1]
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(t, fidelity, 'b-', linewidth=2)
    ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Perfect fidelity')
    ax.axhline(0.99, color='g', linestyle='--', alpha=0.5, label='99% fidelity')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate final fidelity
    final_fidelity = fidelity[-1]
    ax.annotate(f'Final: {final_fidelity:.4f}',
                xy=(t[-1], final_fidelity),
                xytext=(-50, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


# =============================================================================
# Comparison Plots
# =============================================================================

def plot_comparison(
    data_dict: Dict[str, Dict[str, np.ndarray]],
    x_key: str = "t",
    y_key: str = "cost",
    title: str = "Comparison",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    log_scale: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Plot comparison of multiple methods/runs.

    Args:
        data_dict: Dict mapping method name to data dict
        x_key: Key for x-axis data
        y_key: Key for y-axis data
        title: Plot title
        xlabel: X-axis label (defaults to x_key)
        ylabel: Y-axis label (defaults to y_key)
        log_scale: Use log scale for y-axis
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, ax = plt.subplots(figsize=figsize)

    for method_name, data in data_dict.items():
        x = data[x_key]
        y = data[y_key]
        ax.plot(x, y, linewidth=2, marker='o', markersize=4, label=method_name, alpha=0.8)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel if xlabel else x_key, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y_key, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


# =============================================================================
# Multi-Panel Plots
# =============================================================================

def plot_control_summary(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    cost: Optional[np.ndarray] = None,
    state_labels: Optional[List[str]] = None,
    control_labels: Optional[List[str]] = None,
    title: str = "Optimal Control Summary",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """Create multi-panel summary plot.

    Args:
        t: Time array
        x: State trajectory
        u: Control signal
        cost: Cost over time (optional)
        state_labels: State labels
        control_labels: Control labels
        title: Overall title
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot

    Returns:
        Figure object if show=False
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    # Determine layout
    n_plots = 3 if cost is not None else 2

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_plots, 1, figure=fig, hspace=0.3)

    # Plot states
    ax1 = fig.add_subplot(gs[0])
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_states = x.shape[1]
    if state_labels is None:
        state_labels = [f"x_{i+1}" for i in range(n_states)]
    for i in range(n_states):
        ax1.plot(t, x[:, i], label=state_labels[i], linewidth=2)
    ax1.set_ylabel("State", fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title("State Trajectory")

    # Plot control
    ax2 = fig.add_subplot(gs[1])
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    n_controls = u.shape[1]
    if control_labels is None:
        control_labels = [f"u_{i+1}" for i in range(n_controls)]
    for i in range(n_controls):
        ax2.plot(t, u[:, i], label=control_labels[i], linewidth=2)
    ax2.set_ylabel("Control", fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Control Signal")

    # Plot cost if provided
    if cost is not None:
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(t, cost, 'r-', linewidth=2)
        ax3.set_xlabel("Time", fontsize=12)
        ax3.set_ylabel("Cost", fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Instantaneous Cost")
    else:
        ax2.set_xlabel("Time", fontsize=12)

    fig.suptitle(title, fontsize=16, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
        return None
    else:
        return fig


# =============================================================================
# Animation
# =============================================================================

def create_animation(
    t: np.ndarray,
    x: np.ndarray,
    u: Optional[np.ndarray] = None,
    interval: int = 50,
    title: str = "Optimal Control Animation",
    save_path: Optional[Path] = None
) -> animation.FuncAnimation:
    """Create animation of control evolution.

    Args:
        t: Time array
        x: State trajectory [n_steps, n_states]
        u: Control signal [n_steps, n_controls] (optional)
        interval: Milliseconds between frames
        title: Animation title
        save_path: Path to save animation (MP4)

    Returns:
        Animation object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8) if u is not None else (10, 6))
    if u is None:
        axes = [axes]

    # Initialize plots
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    lines_x = []
    for i in range(x.shape[1]):
        line, = axes[0].plot([], [], linewidth=2, label=f'x_{i+1}')
        lines_x.append(line)

    axes[0].set_xlim(t[0], t[-1])
    axes[0].set_ylim(np.min(x)*1.1, np.max(x)*1.1)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("State")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    lines_u = []
    if u is not None:
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        for i in range(u.shape[1]):
            line, = axes[1].plot([], [], linewidth=2, label=f'u_{i+1}')
            lines_u.append(line)
        axes[1].set_xlim(t[0], t[-1])
        axes[1].set_ylim(np.min(u)*1.1, np.max(u)*1.1)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Control")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)

    def init():
        for line in lines_x:
            line.set_data([], [])
        for line in lines_u:
            line.set_data([], [])
        return lines_x + lines_u

    def animate(frame):
        for i, line in enumerate(lines_x):
            line.set_data(t[:frame], x[:frame, i])
        if u is not None:
            for i, line in enumerate(lines_u):
                line.set_data(t[:frame], u[:frame, i])
        return lines_x + lines_u

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t), interval=interval, blit=True
    )

    if save_path:
        anim.save(str(save_path), writer='ffmpeg', fps=20)

    return anim


# =============================================================================
# Utility Functions
# =============================================================================

def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int = 300,
    formats: List[str] = ['png', 'pdf']
):
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        dpi: Resolution
        formats: List of formats to save
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib not available")

    path = Path(path)
    for fmt in formats:
        save_path = path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
