"""Advanced Reinforcement Learning Algorithms.

This module implements state-of-the-art RL algorithms for continuous control:
1. SAC (Soft Actor-Critic) - Maximum entropy RL
2. TD3 (Twin Delayed DDPG) - Improved DDPG with reduced overestimation
3. DDPG (Deep Deterministic Policy Gradient) - Deterministic actor-critic

These algorithms are particularly effective for optimal control problems.

Key Features:
- Off-policy learning (sample efficient)
- Continuous action spaces
- Replay buffers for experience replay
- Target networks for stability
- Entropy regularization (SAC)
- Clipped double Q-learning (TD3)

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Tuple, Callable, Optional, Sequence
from dataclasses import dataclass
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
    # Provide fallback for decorator
    class jax:
        @staticmethod
        def jit(fn):
            return fn


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy RL.

    Stores transitions (s, a, r, s', done) and samples random batches
    for training.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary with keys: states, actions, rewards, next_states, dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }

    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size


# =============================================================================
# Network Architectures for Advanced RL
# =============================================================================

if JAX_AVAILABLE:

    class DeterministicPolicy(nn.Module):
        """Deterministic policy network for DDPG/TD3.

        Outputs deterministic actions in [-1, 1] using tanh activation.
        """
        hidden_dims: Sequence[int]
        action_dim: int

        @nn.compact
        def __call__(self, x):
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)
            action = nn.Dense(self.action_dim)(x)
            action = nn.tanh(action)  # Bound to [-1, 1]
            return action


    class QNetwork(nn.Module):
        """Q-value network (critic) for off-policy RL.

        Estimates Q(s, a) for state-action pairs.
        """
        hidden_dims: Sequence[int]

        @nn.compact
        def __call__(self, state, action):
            x = jnp.concatenate([state, action], axis=-1)
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)
            q_value = nn.Dense(1)(x)
            return q_value


    class DoubleQNetwork(nn.Module):
        """Twin Q-networks for TD3.

        Maintains two Q-networks to reduce overestimation bias.
        """
        hidden_dims: Sequence[int]

        @nn.compact
        def __call__(self, state, action):
            # Q1
            x1 = jnp.concatenate([state, action], axis=-1)
            for hidden_dim in self.hidden_dims:
                x1 = nn.Dense(hidden_dim)(x1)
                x1 = nn.relu(x1)
            q1 = nn.Dense(1)(x1)

            # Q2
            x2 = jnp.concatenate([state, action], axis=-1)
            for hidden_dim in self.hidden_dims:
                x2 = nn.Dense(hidden_dim)(x2)
                x2 = nn.relu(x2)
            q2 = nn.Dense(1)(x2)

            return q1, q2


    class GaussianPolicy(nn.Module):
        """Gaussian policy network for SAC.

        Outputs mean and log_std for a Gaussian distribution.
        Uses reparameterization trick for gradient estimation.
        """
        hidden_dims: Sequence[int]
        action_dim: int
        log_std_min: float = -20
        log_std_max: float = 2

        @nn.compact
        def __call__(self, x):
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)

            mean = nn.Dense(self.action_dim)(x)
            log_std = nn.Dense(self.action_dim)(x)
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

            return mean, log_std

        def sample(self, params, state, rng_key):
            """Sample action with reparameterization trick.

            Args:
                params: Network parameters
                state: Current state
                rng_key: JAX random key

            Returns:
                action: Sampled action
                log_prob: Log probability of action
            """
            mean, log_std = self.apply(params, state)
            std = jnp.exp(log_std)

            # Reparameterization trick
            eps = jax.random.normal(rng_key, shape=mean.shape)
            action_unbounded = mean + eps * std

            # Apply tanh squashing
            action = jnp.tanh(action_unbounded)

            # Compute log probability with change of variables
            log_prob = -0.5 * jnp.sum(eps**2 + jnp.log(2 * jnp.pi), axis=-1)
            log_prob -= jnp.sum(log_std, axis=-1)
            log_prob -= jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)

            return action, log_prob


# =============================================================================
# DDPG (Deep Deterministic Policy Gradient)
# =============================================================================

class DDPGTrainer:
    """DDPG trainer for continuous control.

    DDPG is an off-policy actor-critic algorithm that learns a deterministic
    policy. It uses experience replay and target networks for stability.

    References:
        Lillicrap et al., "Continuous control with deep reinforcement learning", 2015
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize DDPG trainer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Target network update rate (soft update)
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            exploration_noise: Noise std for exploration
            action_bounds: (lower, upper) bounds for actions
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX/Flax required for DDPG")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise

        # Action bounds
        if action_bounds is None:
            self.action_lower = -np.ones(action_dim)
            self.action_upper = np.ones(action_dim)
        else:
            self.action_lower, self.action_upper = action_bounds

        # Create networks
        self.actor = DeterministicPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
        self.critic = QNetwork(hidden_dims=hidden_dims)

        # Initialize networks
        rng = jax.random.PRNGKey(0)
        rng_actor, rng_critic = jax.random.split(rng)

        dummy_state = jnp.ones((1, state_dim))
        dummy_action = jnp.ones((1, action_dim))

        actor_params = self.actor.init(rng_actor, dummy_state)
        critic_params = self.critic.init(rng_critic, dummy_state, dummy_action)

        # Create training states
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(actor_lr)
        )

        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(critic_lr)
        )

        # Target networks (initialized with same parameters)
        self.actor_target_params = actor_params
        self.critic_target_params = critic_params

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

        self.total_steps = 0

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using current policy.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise

        Returns:
            Selected action
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX required")

        state_jax = jnp.array(state.reshape(1, -1))
        action = self.actor.apply(self.actor_state.params, state_jax)
        action = np.array(action[0])

        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise

        # Clip to action bounds
        action = np.clip(action, self.action_lower, self.action_upper)

        return action

    @staticmethod
    @jax.jit
    def _update_critic(critic_state, actor_target_params, critic_target_params,
                       states, actions, rewards, next_states, dones, gamma):
        """Update critic network (JIT compiled)."""
        def critic_loss_fn(params):
            # Current Q-values
            q_values = critic_state.apply_fn(params, states, actions).squeeze()

            # Target Q-values
            next_actions = DeterministicPolicy(hidden_dims=(256, 256), action_dim=actions.shape[-1]).apply(
                actor_target_params, next_states
            )
            next_q_values = QNetwork(hidden_dims=(256, 256)).apply(
                critic_target_params, next_states, next_actions
            ).squeeze()

            target_q_values = rewards + gamma * (1 - dones) * next_q_values
            target_q_values = jax.lax.stop_gradient(target_q_values)

            # MSE loss
            loss = jnp.mean((q_values - target_q_values) ** 2)
            return loss, {'critic_loss': loss, 'q_values': jnp.mean(q_values)}

        (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)

        return critic_state, info

    @staticmethod
    @jax.jit
    def _update_actor(actor_state, critic_params, states):
        """Update actor network (JIT compiled)."""
        def actor_loss_fn(params):
            actions = DeterministicPolicy(hidden_dims=(256, 256), action_dim=states.shape[-1]).apply(
                params, states
            )
            q_values = QNetwork(hidden_dims=(256, 256)).apply(critic_params, states, actions)
            # Negative because we want to maximize Q
            loss = -jnp.mean(q_values)
            return loss, {'actor_loss': loss}

        (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, info

    def train_step(self) -> Dict[str, float]:
        """Perform one training step.

        Returns:
            Dictionary with training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = jnp.array(batch['states'])
        actions = jnp.array(batch['actions'])
        rewards = jnp.array(batch['rewards'])
        next_states = jnp.array(batch['next_states'])
        dones = jnp.array(batch['dones'])

        # Update critic
        self.critic_state, critic_info = self._update_critic(
            self.critic_state, self.actor_target_params, self.critic_target_params,
            states, actions, rewards, next_states, dones, self.gamma
        )

        # Update actor
        self.actor_state, actor_info = self._update_actor(
            self.actor_state, self.critic_state.params, states
        )

        # Soft update of target networks
        self.actor_target_params = jax.tree_util.tree_map(
            lambda p, tp: self.tau * p + (1 - self.tau) * tp,
            self.actor_state.params, self.actor_target_params
        )

        self.critic_target_params = jax.tree_util.tree_map(
            lambda p, tp: self.tau * p + (1 - self.tau) * tp,
            self.critic_state.params, self.critic_target_params
        )

        self.total_steps += 1

        return {**critic_info, **actor_info}


# =============================================================================
# TD3 (Twin Delayed DDPG)
# =============================================================================

class TD3Trainer(DDPGTrainer):
    """TD3 trainer for continuous control.

    TD3 improves upon DDPG with three key tricks:
    1. Twin Q-networks to reduce overestimation
    2. Delayed policy updates
    3. Target policy smoothing

    References:
        Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods", 2018
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize TD3 trainer.

        Args:
            policy_noise: Noise added to target policy
            noise_clip: Range to clip target policy noise
            policy_freq: Frequency of delayed policy updates
            (Other args same as DDPG)
        """
        # Initialize parent (DDPG)
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            exploration_noise=exploration_noise,
            action_bounds=action_bounds
        )

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # Replace critic with double Q-network
        rng = jax.random.PRNGKey(1)
        dummy_state = jnp.ones((1, state_dim))
        dummy_action = jnp.ones((1, action_dim))

        self.critic = DoubleQNetwork(hidden_dims=hidden_dims)
        critic_params = self.critic.init(rng, dummy_state, dummy_action)

        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(critic_lr)
        )

        self.critic_target_params = critic_params

    @staticmethod
    @jax.jit
    def _update_critic_td3(critic_state, actor_target_params, critic_target_params,
                            states, actions, rewards, next_states, dones, gamma,
                            policy_noise, noise_clip, action_dim):
        """Update critic with twin Q-learning (JIT compiled)."""
        def critic_loss_fn(params):
            # Current Q-values from both critics
            q1, q2 = critic_state.apply_fn(params, states, actions)
            q1 = q1.squeeze()
            q2 = q2.squeeze()

            # Target actions with smoothing noise
            next_actions = DeterministicPolicy(hidden_dims=(256, 256), action_dim=action_dim).apply(
                actor_target_params, next_states
            )

            # Add clipped noise
            noise = jax.random.normal(jax.random.PRNGKey(0), shape=next_actions.shape) * policy_noise
            noise = jnp.clip(noise, -noise_clip, noise_clip)
            next_actions = jnp.clip(next_actions + noise, -1, 1)

            # Target Q-values (minimum of two critics)
            target_q1, target_q2 = DoubleQNetwork(hidden_dims=(256, 256)).apply(
                critic_target_params, next_states, next_actions
            )
            target_q = jnp.minimum(target_q1.squeeze(), target_q2.squeeze())

            target_q_values = rewards + gamma * (1 - dones) * target_q
            target_q_values = jax.lax.stop_gradient(target_q_values)

            # MSE loss for both critics
            loss = jnp.mean((q1 - target_q_values) ** 2) + jnp.mean((q2 - target_q_values) ** 2)

            return loss, {
                'critic_loss': loss,
                'q1_mean': jnp.mean(q1),
                'q2_mean': jnp.mean(q2)
            }

        (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)

        return critic_state, info

    def train_step(self) -> Dict[str, float]:
        """Perform one training step with TD3 improvements."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = jnp.array(batch['states'])
        actions = jnp.array(batch['actions'])
        rewards = jnp.array(batch['rewards'])
        next_states = jnp.array(batch['next_states'])
        dones = jnp.array(batch['dones'])

        # Update critic
        self.critic_state, critic_info = self._update_critic_td3(
            self.critic_state, self.actor_target_params, self.critic_target_params,
            states, actions, rewards, next_states, dones, self.gamma,
            self.policy_noise, self.noise_clip, self.action_dim
        )

        info = critic_info

        # Delayed policy update
        if self.total_steps % self.policy_freq == 0:
            # Update actor (use first Q-network for gradients)
            @jax.jit
            def update_actor_td3(actor_state, critic_params, states, action_dim):
                def actor_loss_fn(params):
                    actions = DeterministicPolicy(hidden_dims=(256, 256), action_dim=action_dim).apply(
                        params, states
                    )
                    q1, _ = DoubleQNetwork(hidden_dims=(256, 256)).apply(critic_params, states, actions)
                    loss = -jnp.mean(q1)
                    return loss, {'actor_loss': loss}

                (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
                actor_state = actor_state.apply_gradients(grads=grads)
                return actor_state, info

            self.actor_state, actor_info = update_actor_td3(
                self.actor_state, self.critic_state.params, states, self.action_dim
            )
            info.update(actor_info)

            # Soft update of target networks
            self.actor_target_params = jax.tree_util.tree_map(
                lambda p, tp: self.tau * p + (1 - self.tau) * tp,
                self.actor_state.params, self.actor_target_params
            )

            self.critic_target_params = jax.tree_util.tree_map(
                lambda p, tp: self.tau * p + (1 - self.tau) * tp,
                self.critic_state.params, self.critic_target_params
            )

        self.total_steps += 1

        return info


# =============================================================================
# SAC (Soft Actor-Critic)
# =============================================================================

class SACTrainer:
    """SAC trainer for continuous control.

    SAC is a maximum entropy RL algorithm that learns a stochastic policy
    while maximizing both reward and entropy. This leads to more robust
    and exploratory policies.

    References:
        Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize SAC trainer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            alpha_lr: Temperature parameter learning rate
            gamma: Discount factor
            tau: Target network update rate
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            auto_entropy_tuning: Whether to automatically tune entropy temperature
            target_entropy: Target entropy (default: -action_dim)
            action_bounds: (lower, upper) bounds for actions
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX/Flax required for SAC")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning

        # Action bounds
        if action_bounds is None:
            self.action_lower = -np.ones(action_dim)
            self.action_upper = np.ones(action_dim)
        else:
            self.action_lower, self.action_upper = action_bounds

        # Create networks
        self.actor = GaussianPolicy(hidden_dims=hidden_dims, action_dim=action_dim)
        self.critic = DoubleQNetwork(hidden_dims=hidden_dims)

        # Initialize networks
        rng = jax.random.PRNGKey(0)
        rng_actor, rng_critic = jax.random.split(rng)

        dummy_state = jnp.ones((1, state_dim))
        dummy_action = jnp.ones((1, action_dim))

        actor_params = self.actor.init(rng_actor, dummy_state)
        critic_params = self.critic.init(rng_critic, dummy_state, dummy_action)

        # Create training states
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=optax.adam(actor_lr)
        )

        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(critic_lr)
        )

        # Target critic (no target for actor in SAC)
        self.critic_target_params = critic_params

        # Entropy temperature
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        if auto_entropy_tuning:
            self.log_alpha = jnp.array(0.0)
            self.alpha_optimizer = optax.adam(alpha_lr)
            self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)
        else:
            self.log_alpha = jnp.array(0.0)  # alpha = 1

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)

        self.rng = jax.random.PRNGKey(42)
        self.total_steps = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy.

        Args:
            state: Current state
            deterministic: If True, use mean action (no sampling)

        Returns:
            Selected action
        """
        state_jax = jnp.array(state.reshape(1, -1))

        if deterministic:
            mean, _ = self.actor.apply(self.actor_state.params, state_jax)
            action = jnp.tanh(mean)
            action = np.array(action[0])
        else:
            self.rng, subkey = jax.random.split(self.rng)
            action, _ = self.actor.sample(self.actor_state.params, state_jax, subkey)
            action = np.array(action[0])

        # Scale to action bounds
        action = (action + 1) / 2  # [0, 1]
        action = self.action_lower + action * (self.action_upper - self.action_lower)

        return action

    def train_step(self) -> Dict[str, float]:
        """Perform one training step.

        Returns:
            Dictionary with training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = jnp.array(batch['states'])
        actions = jnp.array(batch['actions'])
        rewards = jnp.array(batch['rewards'])
        next_states = jnp.array(batch['next_states'])
        dones = jnp.array(batch['dones'])

        # Update networks
        self.rng, subkey = jax.random.split(self.rng)
        results = self._train_step_jit(
            self.actor_state, self.critic_state, self.critic_target_params,
            self.log_alpha, self.alpha_opt_state if self.auto_entropy_tuning else None,
            states, actions, rewards, next_states, dones,
            subkey, self.gamma, self.target_entropy, self.auto_entropy_tuning
        )

        if self.auto_entropy_tuning:
            (self.actor_state, self.critic_state, self.log_alpha,
             self.alpha_opt_state, info) = results
        else:
            self.actor_state, self.critic_state, info = results

        # Soft update target critic
        self.critic_target_params = jax.tree_util.tree_map(
            lambda p, tp: self.tau * p + (1 - self.tau) * tp,
            self.critic_state.params, self.critic_target_params
        )

        self.total_steps += 1

        return info

    @staticmethod
    @jax.jit
    def _train_step_jit(actor_state, critic_state, critic_target_params,
                        log_alpha, alpha_opt_state, states, actions, rewards,
                        next_states, dones, rng_key, gamma, target_entropy,
                        auto_entropy_tuning):
        """JIT-compiled training step."""
        alpha = jnp.exp(log_alpha)

        # Update critic
        rng_key, subkey = jax.random.split(rng_key)

        def critic_loss_fn(params):
            # Current Q-values
            q1, q2 = critic_state.apply_fn(params, states, actions)

            # Next actions and their log probs
            next_action, next_log_prob = GaussianPolicy(
                hidden_dims=(256, 256), action_dim=actions.shape[-1]
            ).sample(actor_state.params, next_states, subkey)

            # Target Q-values
            target_q1, target_q2 = DoubleQNetwork(hidden_dims=(256, 256)).apply(
                critic_target_params, next_states, next_action
            )
            target_q = jnp.minimum(target_q1, target_q2).squeeze()

            # Add entropy bonus
            target_q = target_q - alpha * next_log_prob

            target_q_values = rewards + gamma * (1 - dones) * target_q
            target_q_values = jax.lax.stop_gradient(target_q_values)

            # MSE loss
            loss = jnp.mean((q1.squeeze() - target_q_values) ** 2) + \
                   jnp.mean((q2.squeeze() - target_q_values) ** 2)

            return loss, {
                'critic_loss': loss,
                'q1_mean': jnp.mean(q1),
                'q2_mean': jnp.mean(q2)
            }

        (_, critic_info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)

        # Update actor
        rng_key, subkey = jax.random.split(rng_key)

        def actor_loss_fn(params):
            action, log_prob = GaussianPolicy(
                hidden_dims=(256, 256), action_dim=actions.shape[-1]
            ).sample(params, states, subkey)

            q1, q2 = DoubleQNetwork(hidden_dims=(256, 256)).apply(
                critic_state.params, states, action
            )
            q = jnp.minimum(q1, q2).squeeze()

            # Maximize Q - alpha * log_prob
            loss = jnp.mean(alpha * log_prob - q)

            return loss, {
                'actor_loss': loss,
                'entropy': -jnp.mean(log_prob)
            }

        (_, actor_info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        info = {**critic_info, **actor_info, 'alpha': alpha}

        # Update alpha (temperature)
        if auto_entropy_tuning and alpha_opt_state is not None:
            def alpha_loss_fn(log_alpha):
                alpha = jnp.exp(log_alpha)
                loss = -jnp.mean(log_alpha * (actor_info['entropy'] + target_entropy))
                return loss

            alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
            updates, alpha_opt_state = optax.adam(3e-4).update(alpha_grads, alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, updates)

            info['alpha_loss'] = alpha_loss

            return actor_state, critic_state, log_alpha, alpha_opt_state, info
        else:
            return actor_state, critic_state, info


# =============================================================================
# Helper Functions
# =============================================================================

def create_ddpg_trainer(
    state_dim: int,
    action_dim: int,
    **kwargs
) -> DDPGTrainer:
    """Create DDPG trainer with default hyperparameters.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        **kwargs: Additional arguments for DDPGTrainer

    Returns:
        Initialized DDPG trainer
    """
    return DDPGTrainer(state_dim=state_dim, action_dim=action_dim, **kwargs)


def create_td3_trainer(
    state_dim: int,
    action_dim: int,
    **kwargs
) -> TD3Trainer:
    """Create TD3 trainer with default hyperparameters.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        **kwargs: Additional arguments for TD3Trainer

    Returns:
        Initialized TD3 trainer
    """
    return TD3Trainer(state_dim=state_dim, action_dim=action_dim, **kwargs)


def create_sac_trainer(
    state_dim: int,
    action_dim: int,
    **kwargs
) -> SACTrainer:
    """Create SAC trainer with default hyperparameters.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        **kwargs: Additional arguments for SACTrainer

    Returns:
        Initialized SAC trainer
    """
    return SACTrainer(state_dim=state_dim, action_dim=action_dim, **kwargs)
