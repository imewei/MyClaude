"""Utility Functions for ML-Based Optimal Control.

This module provides helper functions for neural network-based optimal control:
- Data generation and preprocessing
- Neural network initialization from PMP solutions
- Visualization utilities
- Performance metrics

Author: Nonequilibrium Physics Agents
Date: 2025-09-30
"""

import warnings
from typing import Callable, Dict, Tuple, Optional, List
import numpy as np

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    warnings.warn(
        "JAX not available. Some utility functions will use NumPy. "
        "For better performance, install JAX: pip install jax jaxlib"
    )


def generate_training_data(
    solver,
    x0_samples: np.ndarray,
    xf_samples: Optional[np.ndarray] = None,
    duration: float = 10.0,
    n_steps: int = 100,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """Generate training data from optimal control solver.

    Solves multiple optimal control problems and extracts trajectories
    for supervised learning of value functions and policies.

    Parameters
    ----------
    solver : PontryaginSolver or CollocationSolver
        Optimal control solver
    x0_samples : np.ndarray
        Initial state samples, shape (n_samples, state_dim)
    xf_samples : Optional[np.ndarray]
        Target state samples, shape (n_samples, state_dim)
    duration : float
        Control duration
    n_steps : int
        Number of time steps
    verbose : bool
        Print progress

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'states': State trajectories, shape (n_samples * n_steps, state_dim)
        - 'actions': Control trajectories, shape (n_samples * n_steps, action_dim)
        - 'values': Value-to-go, shape (n_samples * n_steps,)
        - 'times': Time points, shape (n_samples * n_steps,)

    Examples
    --------
    >>> from solvers import PontryaginSolver
    >>> solver = PontryaginSolver(state_dim=2, control_dim=1, ...)
    >>> x0_samples = np.random.randn(100, 2)
    >>> data = generate_training_data(solver, x0_samples)
    >>> print(data['states'].shape)  # (10000, 2)
    """
    all_states = []
    all_actions = []
    all_values = []
    all_times = []

    n_samples = len(x0_samples)

    for i, x0 in enumerate(x0_samples):
        if verbose and i % 10 == 0:
            print(f"Generating trajectory {i}/{n_samples}")

        xf = xf_samples[i] if xf_samples is not None else None

        try:
            result = solver.solve(
                x0=x0,
                xf=xf,
                duration=duration,
                n_steps=n_steps,
                verbose=False
            )

            if result['converged']:
                # Extract trajectory data
                states = result['x']
                actions = result['u']
                times = result['t']

                # Compute value-to-go (cumulative future cost)
                # Approximate using trapezoidal rule
                costs = np.array([
                    solver.L(states[j], actions[j], times[j])
                    for j in range(len(times))
                ])
                dt = times[1] - times[0]
                values_to_go = np.cumsum(costs[::-1])[::-1] * dt

                all_states.append(states)
                all_actions.append(actions)
                all_values.append(values_to_go)
                all_times.append(times)

        except Exception as e:
            if verbose:
                print(f"  Failed to solve for sample {i}: {e}")
            continue

    # Concatenate all trajectories
    data = {
        'states': np.vstack(all_states),
        'actions': np.vstack(all_actions),
        'values': np.concatenate(all_values),
        'times': np.concatenate(all_times)
    }

    if verbose:
        print(f"\nGenerated {len(all_states)} successful trajectories")
        print(f"Total data points: {len(data['states'])}")

    return data


def initialize_policy_from_pmp(
    policy_network,
    policy_state,
    pmp_data: Dict[str, np.ndarray],
    n_epochs: int = 100,
    batch_size: int = 64,
    verbose: bool = False
):
    """Initialize policy network from PMP solutions via supervised learning.

    Trains policy network to imitate optimal controls from PMP solver.

    Parameters
    ----------
    policy_network : PolicyNetwork
        Policy network to initialize
    policy_state : train_state.TrainState
        Policy training state
    pmp_data : Dict[str, np.ndarray]
        Data from generate_training_data()
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    verbose : bool
        Print progress

    Returns
    -------
    train_state.TrainState
        Initialized policy state

    Examples
    --------
    >>> policy_net, policy_state = create_policy_network(state_dim=2, action_dim=1)
    >>> pmp_data = generate_training_data(solver, x0_samples)
    >>> policy_state = initialize_policy_from_pmp(
    ...     policy_net, policy_state, pmp_data, n_epochs=100
    ... )
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required for policy initialization")

    states = jnp.array(pmp_data['states'])
    actions = jnp.array(pmp_data['actions'])

    n_data = len(states)

    for epoch in range(n_epochs):
        # Random batch
        idx = random.choice(random.PRNGKey(epoch), n_data, shape=(batch_size,))
        states_batch = states[idx]
        actions_batch = actions[idx]

        # Supervised learning loss
        def loss_fn(params):
            action_mean, action_log_std = policy_network.apply(params, states_batch)
            # MSE loss (ignore variance for now)
            return jnp.mean((action_mean - actions_batch)**2)

        loss, grads = jax.value_and_grad(loss_fn)(policy_state.params)
        policy_state = policy_state.apply_gradients(grads=grads)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}: Loss = {loss:.6f}")

    return policy_state


def initialize_value_from_pmp(
    value_network,
    value_state,
    pmp_data: Dict[str, np.ndarray],
    n_epochs: int = 100,
    batch_size: int = 64,
    verbose: bool = False
):
    """Initialize value network from PMP solutions via supervised learning.

    Trains value network to predict value-to-go from PMP solver.

    Parameters
    ----------
    value_network : ValueNetwork
        Value network to initialize
    value_state : train_state.TrainState
        Value training state
    pmp_data : Dict[str, np.ndarray]
        Data from generate_training_data()
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    verbose : bool
        Print progress

    Returns
    -------
    train_state.TrainState
        Initialized value state
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required for value initialization")

    states = jnp.array(pmp_data['states'])
    values = jnp.array(pmp_data['values'])

    n_data = len(states)

    for epoch in range(n_epochs):
        # Random batch
        idx = random.choice(random.PRNGKey(epoch), n_data, shape=(batch_size,))
        states_batch = states[idx]
        values_batch = values[idx]

        # Supervised learning loss
        def loss_fn(params):
            pred_values = value_network.apply(params, states_batch)
            return jnp.mean((pred_values.squeeze() - values_batch)**2)

        loss, grads = jax.value_and_grad(loss_fn)(value_state.params)
        value_state = value_state.apply_gradients(grads=grads)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}: Loss = {loss:.6f}")

    return value_state


def generate_pinn_training_data(
    state_bounds: Tuple[np.ndarray, np.ndarray],
    time_range: Tuple[float, float],
    n_interior: int = 1000,
    n_boundary: int = 100,
    terminal_cost: Optional[Callable] = None
) -> Dict[str, np.ndarray]:
    """Generate training data for PINN.

    Creates interior points for physics loss and boundary points for boundary conditions.

    Parameters
    ----------
    state_bounds : Tuple[np.ndarray, np.ndarray]
        State space bounds (x_min, x_max)
    time_range : Tuple[float, float]
        Time range (t_min, t_max)
    n_interior : int
        Number of interior points
    n_boundary : int
        Number of boundary points
    terminal_cost : Optional[Callable]
        Terminal cost function Φ(x) for boundary condition

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'x_interior': Interior state points
        - 't_interior': Interior time points
        - 'x_boundary': Boundary state points
        - 't_boundary': Boundary time points (all at t_max)
        - 'values_boundary': Boundary values (from terminal cost)

    Examples
    --------
    >>> x_min, x_max = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    >>> terminal_cost = lambda x: np.sum(x**2)
    >>> data = generate_pinn_training_data(
    ...     (x_min, x_max), (0.0, 10.0), terminal_cost=terminal_cost
    ... )
    """
    x_min, x_max = state_bounds
    t_min, t_max = time_range
    state_dim = len(x_min)

    # Interior points (uniform random sampling)
    x_interior = np.random.uniform(
        x_min, x_max, size=(n_interior, state_dim)
    )
    t_interior = np.random.uniform(
        t_min, t_max, size=(n_interior, 1)
    )

    # Boundary points (at final time)
    x_boundary = np.random.uniform(
        x_min, x_max, size=(n_boundary, state_dim)
    )
    t_boundary = np.full((n_boundary, 1), t_max)

    # Boundary values (terminal cost)
    if terminal_cost is not None:
        values_boundary = np.array([
            terminal_cost(x_boundary[i])
            for i in range(n_boundary)
        ]).reshape(-1, 1)
    else:
        values_boundary = np.zeros((n_boundary, 1))

    return {
        'x_interior': x_interior,
        't_interior': t_interior,
        'x_boundary': x_boundary,
        't_boundary': t_boundary,
        'values_boundary': values_boundary
    }


def compute_policy_performance(
    policy_network,
    policy_params,
    env,
    n_episodes: int = 10,
    deterministic: bool = False
) -> Dict[str, float]:
    """Evaluate policy performance on environment.

    Parameters
    ----------
    policy_network : PolicyNetwork
        Policy network
    policy_params : PyTree
        Policy parameters
    env : OptimalControlEnv
        Environment to evaluate on
    n_episodes : int
        Number of evaluation episodes
    deterministic : bool
        Use deterministic policy (mean action)

    Returns
    -------
    Dict[str, float]
        Performance metrics:
        - 'mean_return': Average episodic return
        - 'std_return': Standard deviation of returns
        - 'mean_length': Average episode length
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required for policy evaluation")

    returns = []
    lengths = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Get action from policy
            action_mean, action_log_std = policy_network.apply(
                policy_params, state[None, :]
            )

            if deterministic:
                action = action_mean[0]
            else:
                action_std = jnp.exp(action_log_std[0])
                action = action_mean[0] + action_std * random.normal(
                    random.PRNGKey(episode * 1000 + episode_length),
                    shape=action_std.shape
                )

            # Step environment
            state, reward, done, _ = env.step(np.array(action))
            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'returns': returns
    }


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """Plot training history.

    Parameters
    ----------
    history : Dict[str, List[float]]
        Training history from train_actor_critic() or train_pinn()
    save_path : Optional[str]
        Path to save plot

    Examples
    --------
    >>> history = train_actor_critic(env, actor_critic_state, trainer, ...)
    >>> plot_training_history(history, 'training_history.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("Matplotlib not available. Cannot plot training history.")
        return

    n_metrics = len([k for k in history.keys() if k != 'episode_rewards'])
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    idx = 0

    # Plot losses
    for key in ['policy_loss', 'value_loss', 'entropy', 'total_loss', 'physics_loss', 'boundary_loss']:
        if key in history:
            axes[idx].plot(history[key], label=key.replace('_', ' ').title())
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].set_title(f'{key.replace("_", " ").title()} Over Training')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            idx += 1

    # Plot episode rewards if available
    if 'episode_rewards' in history and len(history['episode_rewards']) > 0:
        if idx < len(axes):
            axes[idx].plot(history['episode_rewards'], label='Episode Reward')
            axes[idx].set_xlabel('Episode')
            axes[idx].set_ylabel('Reward')
            axes[idx].set_title('Episode Rewards Over Training')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def compare_solvers(
    pmp_solver,
    ml_policy,
    ml_policy_params,
    env,
    x0_samples: np.ndarray,
    duration: float = 10.0,
    n_steps: int = 100
) -> Dict[str, np.ndarray]:
    """Compare PMP solver and ML policy performance.

    Parameters
    ----------
    pmp_solver : PontryaginSolver
        PMP solver
    ml_policy : PolicyNetwork
        Trained ML policy
    ml_policy_params : PyTree
        ML policy parameters
    env : OptimalControlEnv
        Environment
    x0_samples : np.ndarray
        Initial states to test
    duration : float
        Control duration
    n_steps : int
        Number of steps

    Returns
    -------
    Dict[str, np.ndarray]
        Comparison results:
        - 'pmp_costs': PMP solver costs
        - 'ml_costs': ML policy costs
        - 'speedup': Time speedup (PMP time / ML time)
    """
    pmp_costs = []
    ml_costs = []
    pmp_times = []
    ml_times = []

    for x0 in x0_samples:
        # PMP solver
        import time
        start = time.time()
        try:
            pmp_result = pmp_solver.solve(x0=x0, duration=duration, n_steps=n_steps, verbose=False)
            if pmp_result['converged']:
                pmp_costs.append(pmp_result['cost'])
                pmp_times.append(time.time() - start)
            else:
                continue
        except:
            continue

        # ML policy
        start = time.time()
        env_copy = env
        state = x0.copy()
        ml_cost = 0.0

        for _ in range(n_steps):
            if JAX_AVAILABLE:
                action_mean, _ = ml_policy.apply(ml_policy_params, state[None, :])
                action = np.array(action_mean[0])
            else:
                action = np.zeros(env.action_dim)

            state, reward, done, _ = env_copy.step(action)
            ml_cost += -reward

            if done:
                break

        ml_costs.append(ml_cost)
        ml_times.append(time.time() - start)

    return {
        'pmp_costs': np.array(pmp_costs),
        'ml_costs': np.array(ml_costs),
        'pmp_times': np.array(pmp_times),
        'ml_times': np.array(ml_times),
        'speedup': np.mean(pmp_times) / np.mean(ml_times) if len(ml_times) > 0 else 0.0,
        'cost_ratio': np.mean(ml_costs) / np.mean(pmp_costs) if len(pmp_costs) > 0 else float('inf')
    }


# Convenience exports
__all__ = [
    'generate_training_data',
    'initialize_policy_from_pmp',
    'initialize_value_from_pmp',
    'generate_pinn_training_data',
    'compute_policy_performance',
    'plot_training_history',
    'compare_solvers',
    'JAX_AVAILABLE',
]
