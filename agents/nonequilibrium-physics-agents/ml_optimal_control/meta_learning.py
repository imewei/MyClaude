"""Meta-Learning for Optimal Control.

This module implements meta-learning algorithms for fast adaptation:
1. MAML (Model-Agnostic Meta-Learning)
2. Reptile (simplified meta-learning)
3. Context-based adaptation
4. Transfer learning utilities

Meta-learning enables agents to:
- Quickly adapt to new tasks
- Learn from few examples (few-shot learning)
- Generalize across task distributions

Applications in optimal control:
- Adapting to new system dynamics
- Learning control for similar but different systems
- Personalization and online adaptation

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Tuple, Callable, Optional, Sequence, List
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Provide fallbacks
    class train_state:
        class TrainState:
            pass
    class jax:
        @staticmethod
        def jit(fn):
            return fn


# =============================================================================
# Task Distribution
# =============================================================================

class Task:
    """Single task for meta-learning.

    A task consists of:
    - Dynamics function
    - Cost function
    - Initial state distribution
    """

    def __init__(
        self,
        task_id: str,
        dynamics: Callable,
        cost: Callable,
        x0_mean: np.ndarray,
        x0_std: np.ndarray,
        duration: float = 5.0,
        dt: float = 0.05
    ):
        """Initialize task.

        Args:
            task_id: Unique task identifier
            dynamics: Dynamics function dx/dt = f(x, u, t)
            cost: Cost function L(x, u, t)
            x0_mean: Mean of initial state distribution
            x0_std: Std of initial state distribution
            duration: Episode duration
            dt: Time step
        """
        self.task_id = task_id
        self.dynamics = dynamics
        self.cost = cost
        self.x0_mean = x0_mean
        self.x0_std = x0_std
        self.duration = duration
        self.dt = dt

    def sample_initial_state(self) -> np.ndarray:
        """Sample initial state from distribution."""
        return self.x0_mean + self.x0_std * np.random.randn(*self.x0_mean.shape)


class TaskDistribution:
    """Distribution over tasks for meta-learning."""

    def __init__(self, tasks: List[Task]):
        """Initialize task distribution.

        Args:
            tasks: List of tasks
        """
        self.tasks = tasks
        self.n_tasks = len(tasks)

    def sample(self, n_tasks: int = 1) -> List[Task]:
        """Sample tasks from distribution.

        Args:
            n_tasks: Number of tasks to sample

        Returns:
            List of sampled tasks
        """
        indices = np.random.choice(self.n_tasks, size=n_tasks, replace=True)
        return [self.tasks[i] for i in indices]


# =============================================================================
# MAML (Model-Agnostic Meta-Learning)
# =============================================================================

class MAMLTrainer:
    """MAML meta-learning for optimal control policies.

    MAML learns an initialization that can quickly adapt to new tasks
    with few gradient steps.

    Algorithm:
    1. Sample batch of tasks
    2. For each task:
       a. Take K gradient steps on task (inner loop)
       b. Evaluate on task
    3. Update meta-parameters based on task performance (outer loop)

    References:
        Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation
        of Deep Networks", 2017
    """

    def __init__(
        self,
        policy_network,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5,
        batch_size: int = 10
    ):
        """Initialize MAML trainer.

        Args:
            policy_network: Policy network architecture
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            n_inner_steps: Number of gradient steps for adaptation
            batch_size: Number of tasks per meta-update
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX/Flax required for MAML")

        self.policy_network = policy_network
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.batch_size = batch_size

        # Initialize meta-parameters
        rng = jax.random.PRNGKey(0)
        dummy_state = jnp.ones((1, state_dim))
        meta_params = policy_network.init(rng, dummy_state)

        # Create training state for meta-parameters
        self.meta_state = train_state.TrainState.create(
            apply_fn=policy_network.apply,
            params=meta_params,
            tx=optax.adam(outer_lr)
        )

    def adapt(
        self,
        task: Task,
        n_episodes: int = 10
    ) -> train_state.TrainState:
        """Adapt policy to new task (inner loop).

        Args:
            task: Task to adapt to
            n_episodes: Number of episodes for adaptation

        Returns:
            Adapted policy parameters
        """
        # Start from meta-parameters
        adapted_params = self.meta_state.params

        # Collect data from task
        states, actions, rewards = self._collect_data(task, adapted_params, n_episodes)

        # Inner loop: gradient descent on task
        for step in range(self.n_inner_steps):
            loss, grads = self._compute_policy_gradient(
                adapted_params, states, actions, rewards
            )

            # Manual gradient descent (not using optimizer)
            adapted_params = jax.tree_util.tree_map(
                lambda p, g: p - self.inner_lr * g,
                adapted_params, grads
            )

        # Create temporary state with adapted params
        adapted_state = train_state.TrainState.create(
            apply_fn=self.meta_state.apply_fn,
            params=adapted_params,
            tx=self.meta_state.tx
        )

        return adapted_state

    def meta_train_step(self, task_distribution: TaskDistribution) -> Dict[str, float]:
        """Perform one meta-training step (outer loop).

        Args:
            task_distribution: Distribution over tasks

        Returns:
            Dictionary with training metrics
        """
        # Sample batch of tasks
        tasks = task_distribution.sample(self.batch_size)

        # Compute meta-gradient
        def meta_loss_fn(meta_params):
            total_loss = 0.0

            for task in tasks:
                # Start from meta-parameters
                adapted_params = meta_params

                # Inner loop adaptation
                states_adapt, actions_adapt, rewards_adapt = self._collect_data(
                    task, adapted_params, n_episodes=5
                )

                for _ in range(self.n_inner_steps):
                    loss, grads = self._compute_policy_gradient(
                        adapted_params, states_adapt, actions_adapt, rewards_adapt
                    )
                    adapted_params = jax.tree_util.tree_map(
                        lambda p, g: p - self.inner_lr * g,
                        adapted_params, grads
                    )

                # Evaluate adapted parameters on new data
                states_eval, actions_eval, rewards_eval = self._collect_data(
                    task, adapted_params, n_episodes=5
                )

                task_loss, _ = self._compute_policy_gradient(
                    adapted_params, states_eval, actions_eval, rewards_eval
                )

                total_loss += task_loss

            return total_loss / len(tasks)

        loss, grads = jax.value_and_grad(meta_loss_fn)(self.meta_state.params)
        self.meta_state = self.meta_state.apply_gradients(grads=grads)

        return {'meta_loss': float(loss)}

    def _collect_data(
        self,
        task: Task,
        params,
        n_episodes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect trajectory data from task.

        Args:
            task: Task to collect data from
            params: Policy parameters
            n_episodes: Number of episodes

        Returns:
            states, actions, rewards
        """
        all_states = []
        all_actions = []
        all_rewards = []

        for _ in range(n_episodes):
            states = []
            actions = []
            rewards = []

            x = task.sample_initial_state()
            t = 0.0

            while t < task.duration:
                # Select action
                state_jax = jnp.array(x.reshape(1, -1))
                action_dist = self.policy_network.apply(params, state_jax)

                # Sample action (assuming Gaussian policy)
                if len(action_dist) == 2:  # (mean, log_std)
                    action_mean, action_log_std = action_dist
                    action = np.array(action_mean[0])
                else:
                    action = np.array(action_dist[0])

                # Store
                states.append(x.copy())
                actions.append(action.copy())

                # Compute reward
                reward = -task.cost(x, action, t) * task.dt
                rewards.append(reward)

                # Step dynamics (Euler integration)
                dx = task.dynamics(x, action, t)
                x = x + dx * task.dt
                t += task.dt

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)

        return (
            np.array(all_states),
            np.array(all_actions),
            np.array(all_rewards)
        )

    @staticmethod
    @jax.jit
    def _compute_policy_gradient(params, states, actions, rewards):
        """Compute policy gradient loss."""
        # Simple REINFORCE-style loss
        # This is a placeholder - in practice, use PPO or similar
        states_jax = jnp.array(states)
        actions_jax = jnp.array(actions)
        rewards_jax = jnp.array(rewards)

        # Compute returns (simple sum for now)
        returns = jnp.cumsum(rewards_jax[::-1])[::-1]
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

        # Simplified policy loss
        loss = -jnp.mean(returns)

        return loss, jax.grad(lambda p: loss)(params)


# =============================================================================
# Reptile (Simplified Meta-Learning)
# =============================================================================

class ReptileTrainer:
    """Reptile meta-learning algorithm.

    Reptile is a simpler alternative to MAML that directly updates
    meta-parameters toward task-adapted parameters.

    Algorithm:
    1. Sample task
    2. Adapt policy to task via SGD
    3. Move meta-parameters toward adapted parameters

    References:
        Nichol et al., "On First-Order Meta-Learning Algorithms", 2018
    """

    def __init__(
        self,
        policy_network,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 10
    ):
        """Initialize Reptile trainer.

        Args:
            policy_network: Policy network architecture
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            inner_lr: Learning rate for task adaptation
            outer_lr: Learning rate for meta-update
            n_inner_steps: Number of gradient steps for adaptation
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX/Flax required for Reptile")

        self.policy_network = policy_network
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps

        # Initialize meta-parameters
        rng = jax.random.PRNGKey(0)
        dummy_state = jnp.ones((1, state_dim))
        self.meta_params = policy_network.init(rng, dummy_state)

    def meta_train_step(self, task: Task, n_episodes: int = 10) -> Dict[str, float]:
        """Perform one Reptile meta-training step.

        Args:
            task: Task to train on
            n_episodes: Number of episodes for adaptation

        Returns:
            Dictionary with training metrics
        """
        # Start from meta-parameters
        adapted_params = self.meta_params

        # Collect data and adapt
        states, actions, rewards = self._collect_data(task, adapted_params, n_episodes)

        initial_loss = None
        for step in range(self.n_inner_steps):
            loss, grads = self._compute_policy_gradient(
                adapted_params, states, actions, rewards
            )

            if initial_loss is None:
                initial_loss = loss

            # Gradient descent
            adapted_params = jax.tree_util.tree_map(
                lambda p, g: p - self.inner_lr * g,
                adapted_params, grads
            )

        # Reptile meta-update: move toward adapted parameters
        self.meta_params = jax.tree_util.tree_map(
            lambda meta_p, adapted_p: meta_p + self.outer_lr * (adapted_p - meta_p),
            self.meta_params, adapted_params
        )

        return {
            'initial_loss': float(initial_loss),
            'final_loss': float(loss)
        }

    def _collect_data(
        self,
        task: Task,
        params,
        n_episodes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect trajectory data (same as MAML)."""
        all_states = []
        all_actions = []
        all_rewards = []

        for _ in range(n_episodes):
            states = []
            actions = []
            rewards = []

            x = task.sample_initial_state()
            t = 0.0

            while t < task.duration:
                state_jax = jnp.array(x.reshape(1, -1))
                action_dist = self.policy_network.apply(params, state_jax)

                if len(action_dist) == 2:
                    action_mean, _ = action_dist
                    action = np.array(action_mean[0])
                else:
                    action = np.array(action_dist[0])

                states.append(x.copy())
                actions.append(action.copy())

                reward = -task.cost(x, action, t) * task.dt
                rewards.append(reward)

                dx = task.dynamics(x, action, t)
                x = x + dx * task.dt
                t += task.dt

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)

        return (
            np.array(all_states),
            np.array(all_actions),
            np.array(all_rewards)
        )

    @staticmethod
    @jax.jit
    def _compute_policy_gradient(params, states, actions, rewards):
        """Compute policy gradient (placeholder)."""
        states_jax = jnp.array(states)
        rewards_jax = jnp.array(rewards)

        returns = jnp.cumsum(rewards_jax[::-1])[::-1]
        returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

        loss = -jnp.mean(returns)

        return loss, jax.grad(lambda p: loss)(params)


# =============================================================================
# Context-Based Adaptation
# =============================================================================

if JAX_AVAILABLE:

    class ContextEncoder(nn.Module):
        """Encoder for task context.

        Processes trajectories to produce task embedding that captures
        task-specific information.
        """
        hidden_dims: Sequence[int]
        context_dim: int

        @nn.compact
        def __call__(self, states, actions, rewards):
            """Encode trajectory into context vector.

            Args:
                states: [batch, seq_len, state_dim]
                actions: [batch, seq_len, action_dim]
                rewards: [batch, seq_len]

            Returns:
                context: [batch, context_dim]
            """
            # Concatenate inputs
            x = jnp.concatenate([
                states,
                actions,
                rewards.reshape(*rewards.shape, 1)
            ], axis=-1)

            # Temporal encoding (simple mean pooling for now)
            # In practice, use RNN/Transformer
            x = jnp.mean(x, axis=1)

            # MLP encoder
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)

            context = nn.Dense(self.context_dim)(x)
            return context


    class ContextConditionedPolicy(nn.Module):
        """Policy conditioned on task context.

        Takes both state and context as input to adapt to current task.
        """
        hidden_dims: Sequence[int]
        action_dim: int
        context_dim: int

        @nn.compact
        def __call__(self, state, context):
            """Forward pass.

            Args:
                state: Current state [batch, state_dim]
                context: Task context [batch, context_dim]

            Returns:
                action_mean: [batch, action_dim]
                action_log_std: [batch, action_dim]
            """
            # Concatenate state and context
            x = jnp.concatenate([state, context], axis=-1)

            # MLP
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)

            # Output
            action_mean = nn.Dense(self.action_dim)(x)
            action_log_std = nn.Dense(self.action_dim)(x)
            action_log_std = jnp.clip(action_log_std, -20, 2)

            return action_mean, action_log_std


class ContextBasedAdapter:
    """Context-based adaptation using trajectory encoding.

    Instead of gradient-based adaptation, this approach:
    1. Encodes recent experience into context vector
    2. Conditions policy on context
    3. Learns to adapt through context alone
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (128, 128),
        learning_rate: float = 3e-4
    ):
        """Initialize context-based adapter.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            context_dim: Dimension of context vector
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX/Flax required")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim

        # Create networks
        self.encoder = ContextEncoder(
            hidden_dims=hidden_dims,
            context_dim=context_dim
        )

        self.policy = ContextConditionedPolicy(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            context_dim=context_dim
        )

        # Initialize
        rng = jax.random.PRNGKey(0)
        rng_enc, rng_pol = jax.random.split(rng)

        dummy_states = jnp.ones((1, 10, state_dim))
        dummy_actions = jnp.ones((1, 10, action_dim))
        dummy_rewards = jnp.ones((1, 10))

        encoder_params = self.encoder.init(
            rng_enc, dummy_states, dummy_actions, dummy_rewards
        )

        dummy_context = jnp.ones((1, context_dim))
        dummy_state = jnp.ones((1, state_dim))

        policy_params = self.policy.init(rng_pol, dummy_state, dummy_context)

        # Combine parameters
        params = {'encoder': encoder_params, 'policy': policy_params}

        # Create training state
        self.train_state = train_state.TrainState.create(
            apply_fn=None,  # Not used
            params=params,
            tx=optax.adam(learning_rate)
        )

        # Context buffer for current task
        self.context_buffer = {
            'states': [],
            'actions': [],
            'rewards': []
        }

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action conditioned on current context.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Encode current context
        if len(self.context_buffer['states']) > 0:
            context = self._encode_context()
        else:
            # Zero context if no data yet
            context = np.zeros((1, self.context_dim))

        # Get action
        state_jax = jnp.array(state.reshape(1, -1))
        context_jax = jnp.array(context)

        action_mean, action_log_std = self.policy.apply(
            self.train_state.params['policy'],
            state_jax,
            context_jax
        )

        # Sample action
        action = np.array(action_mean[0])

        return action

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float):
        """Add experience to context buffer.

        Args:
            state: State
            action: Action
            reward: Reward
        """
        self.context_buffer['states'].append(state)
        self.context_buffer['actions'].append(action)
        self.context_buffer['rewards'].append(reward)

    def reset_context(self):
        """Reset context buffer (new task/episode)."""
        self.context_buffer = {
            'states': [],
            'actions': [],
            'rewards': []
        }

    def _encode_context(self) -> np.ndarray:
        """Encode current buffer into context vector.

        Returns:
            Context vector
        """
        # Take last K steps (e.g., 50)
        K = 50
        states = np.array(self.context_buffer['states'][-K:])
        actions = np.array(self.context_buffer['actions'][-K:])
        rewards = np.array(self.context_buffer['rewards'][-K:])

        # Pad if necessary
        if len(states) < K:
            pad_len = K - len(states)
            states = np.pad(states, ((0, pad_len), (0, 0)), mode='constant')
            actions = np.pad(actions, ((0, pad_len), (0, 0)), mode='constant')
            rewards = np.pad(rewards, (0, pad_len), mode='constant')

        # Add batch dimension
        states = states.reshape(1, K, -1)
        actions = actions.reshape(1, K, -1)
        rewards = rewards.reshape(1, K)

        # Encode
        context = self.encoder.apply(
            self.train_state.params['encoder'],
            jnp.array(states),
            jnp.array(actions),
            jnp.array(rewards)
        )

        return np.array(context)


# =============================================================================
# Helper Functions
# =============================================================================

def create_task_distribution(
    base_dynamics: Callable,
    base_cost: Callable,
    state_dim: int,
    n_tasks: int = 10,
    perturbation_scale: float = 0.1
) -> TaskDistribution:
    """Create task distribution with perturbed dynamics/cost.

    Args:
        base_dynamics: Base dynamics function
        base_cost: Base cost function
        state_dim: Dimension of state space
        n_tasks: Number of tasks
        perturbation_scale: Scale of perturbations

    Returns:
        Task distribution
    """
    tasks = []

    for i in range(n_tasks):
        # Create perturbed task
        task_id = f"task_{i}"

        # Add random perturbation (placeholder - customize per application)
        def perturbed_dynamics(x, u, t, base_fn=base_dynamics, seed=i):
            return base_fn(x, u, t)

        def perturbed_cost(x, u, t, base_fn=base_cost, seed=i):
            return base_fn(x, u, t)

        x0_mean = np.zeros(state_dim)
        x0_std = 0.1 * np.ones(state_dim)

        task = Task(
            task_id=task_id,
            dynamics=perturbed_dynamics,
            cost=perturbed_cost,
            x0_mean=x0_mean,
            x0_std=x0_std
        )

        tasks.append(task)

    return TaskDistribution(tasks)
