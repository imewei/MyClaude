"""Model-Based Reinforcement Learning.

This module implements model-based RL approaches for optimal control:
1. World models (dynamics learning)
2. Model Predictive Control (MPC) with learned models
3. Dyna-style planning
4. Model-based value expansion (MVE)

Model-based RL learns a model of the environment dynamics, which can be
used for:
- Planning (finding optimal actions)
- Simulation (generating synthetic experience)
- Improving sample efficiency

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
    # Provide fallback for decorator
    class jax:
        @staticmethod
        def jit(fn):
            return fn


# =============================================================================
# World Model Networks
# =============================================================================

if JAX_AVAILABLE:

    class DeterministicDynamicsModel(nn.Module):
        """Deterministic dynamics model: s_{t+1} = f(s_t, a_t).

        Predicts next state given current state and action.
        """
        hidden_dims: Sequence[int]
        state_dim: int

        @nn.compact
        def __call__(self, state, action):
            x = jnp.concatenate([state, action], axis=-1)
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)
            # Predict state delta (residual learning)
            delta = nn.Dense(self.state_dim)(x)
            next_state = state + delta
            return next_state


    class ProbabilisticDynamicsModel(nn.Module):
        """Probabilistic dynamics model: s_{t+1} ~ N(μ(s_t, a_t), Σ(s_t, a_t)).

        Predicts mean and variance of next state distribution.
        """
        hidden_dims: Sequence[int]
        state_dim: int
        min_log_var: float = -10
        max_log_var: float = 0.5

        @nn.compact
        def __call__(self, state, action):
            x = jnp.concatenate([state, action], axis=-1)
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)

            # Predict mean (residual)
            delta_mean = nn.Dense(self.state_dim)(x)
            mean = state + delta_mean

            # Predict log variance
            log_var = nn.Dense(self.state_dim)(x)
            log_var = jnp.clip(log_var, self.min_log_var, self.max_log_var)

            return mean, log_var


    class EnsembleDynamicsModel(nn.Module):
        """Ensemble of dynamics models for uncertainty estimation.

        Uses bootstrap aggregating for better uncertainty quantification.
        """
        hidden_dims: Sequence[int]
        state_dim: int
        n_models: int

        @nn.compact
        def __call__(self, state, action):
            """Forward pass through all models in ensemble.

            Returns:
                means: [n_models, batch, state_dim]
                log_vars: [n_models, batch, state_dim]
            """
            means = []
            log_vars = []

            for i in range(self.n_models):
                x = jnp.concatenate([state, action], axis=-1)
                for hidden_dim in self.hidden_dims:
                    x = nn.Dense(hidden_dim, name=f'model{i}_dense')(x)
                    x = nn.relu(x)

                delta_mean = nn.Dense(self.state_dim, name=f'model{i}_mean')(x)
                mean = state + delta_mean

                log_var = nn.Dense(self.state_dim, name=f'model{i}_logvar')(x)
                log_var = jnp.clip(log_var, -10, 0.5)

                means.append(mean)
                log_vars.append(log_var)

            return jnp.stack(means), jnp.stack(log_vars)


    class RewardModel(nn.Module):
        """Reward model: r = R(s, a).

        Predicts immediate reward given state and action.
        """
        hidden_dims: Sequence[int]

        @nn.compact
        def __call__(self, state, action):
            x = jnp.concatenate([state, action], axis=-1)
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = nn.relu(x)
            reward = nn.Dense(1)(x)
            return reward


# =============================================================================
# Dynamics Model Trainer
# =============================================================================

class DynamicsModelTrainer:
    """Trainer for learning dynamics models from data.

    Supports both deterministic and probabilistic models.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        model_type: str = 'probabilistic',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """Initialize dynamics model trainer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            model_type: 'deterministic' or 'probabilistic'
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX/Flax required for model-based RL")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = model_type

        # Create model
        if model_type == 'deterministic':
            self.model = DeterministicDynamicsModel(
                hidden_dims=hidden_dims,
                state_dim=state_dim
            )
        elif model_type == 'probabilistic':
            self.model = ProbabilisticDynamicsModel(
                hidden_dims=hidden_dims,
                state_dim=state_dim
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize model
        rng = jax.random.PRNGKey(0)
        dummy_state = jnp.ones((1, state_dim))
        dummy_action = jnp.ones((1, action_dim))

        params = self.model.init(rng, dummy_state, dummy_action)

        # Create training state
        tx = optax.adamw(learning_rate, weight_decay=weight_decay)
        self.train_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx
        )

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next state.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Predicted next state (mean if probabilistic)
        """
        state_jax = jnp.array(state)
        action_jax = jnp.array(action)

        if self.model_type == 'deterministic':
            next_state = self.model.apply(self.train_state.params, state_jax, action_jax)
        else:
            next_state, _ = self.model.apply(self.train_state.params, state_jax, action_jax)

        return np.array(next_state)

    @staticmethod
    @jax.jit
    def _train_step_deterministic(train_state, states, actions, next_states):
        """Training step for deterministic model."""
        def loss_fn(params):
            pred_next_states = DeterministicDynamicsModel(
                hidden_dims=(256, 256), state_dim=states.shape[-1]
            ).apply(params, states, actions)

            # MSE loss
            loss = jnp.mean((pred_next_states - next_states) ** 2)

            return loss, {'dynamics_loss': loss}

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

        return train_state, info

    @staticmethod
    @jax.jit
    def _train_step_probabilistic(train_state, states, actions, next_states):
        """Training step for probabilistic model."""
        def loss_fn(params):
            mean, log_var = ProbabilisticDynamicsModel(
                hidden_dims=(256, 256), state_dim=states.shape[-1]
            ).apply(params, states, actions)

            # Negative log likelihood
            var = jnp.exp(log_var)
            loss = 0.5 * jnp.mean((next_states - mean) ** 2 / var + log_var)

            return loss, {
                'dynamics_loss': loss,
                'mean_log_var': jnp.mean(log_var)
            }

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

        return train_state, info

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray
    ) -> Dict[str, float]:
        """Perform one training step.

        Args:
            states: Current states [batch, state_dim]
            actions: Actions taken [batch, action_dim]
            next_states: Next states [batch, state_dim]

        Returns:
            Dictionary with training metrics
        """
        states_jax = jnp.array(states)
        actions_jax = jnp.array(actions)
        next_states_jax = jnp.array(next_states)

        if self.model_type == 'deterministic':
            self.train_state, info = self._train_step_deterministic(
                self.train_state, states_jax, actions_jax, next_states_jax
            )
        else:
            self.train_state, info = self._train_step_probabilistic(
                self.train_state, states_jax, actions_jax, next_states_jax
            )

        return info


# =============================================================================
# Model Predictive Control (MPC)
# =============================================================================

class ModelPredictiveControl:
    """Model Predictive Control with learned dynamics.

    Uses learned dynamics model to plan optimal actions over a horizon.
    """

    def __init__(
        self,
        dynamics_model: DynamicsModelTrainer,
        cost_fn: Callable[[np.ndarray, np.ndarray], float],
        horizon: int = 10,
        n_samples: int = 1000,
        n_elite: int = 100,
        n_iterations: int = 5,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        """Initialize MPC controller.

        Args:
            dynamics_model: Learned dynamics model
            cost_fn: Cost function cost(state, action)
            horizon: Planning horizon
            n_samples: Number of action sequences to sample
            n_elite: Number of elite sequences to keep
            n_iterations: Number of CEM iterations
            action_bounds: (lower, upper) bounds for actions
        """
        self.dynamics_model = dynamics_model
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations

        self.action_dim = dynamics_model.action_dim

        if action_bounds is None:
            self.action_lower = -np.ones(self.action_dim)
            self.action_upper = np.ones(self.action_dim)
        else:
            self.action_lower, self.action_upper = action_bounds

        # Initialize action distribution (mean and std)
        self.action_mean = np.zeros((horizon, self.action_dim))
        self.action_std = np.ones((horizon, self.action_dim))

    def plan(self, state: np.ndarray) -> np.ndarray:
        """Plan optimal action using CEM (Cross-Entropy Method).

        Args:
            state: Current state

        Returns:
            Optimal action (first action in sequence)
        """
        # Cross-Entropy Method for trajectory optimization
        for iteration in range(self.n_iterations):
            # Sample action sequences
            action_sequences = np.random.randn(
                self.n_samples, self.horizon, self.action_dim
            )
            action_sequences = (
                self.action_mean + self.action_std * action_sequences
            )

            # Clip to bounds
            action_sequences = np.clip(
                action_sequences, self.action_lower, self.action_upper
            )

            # Evaluate costs
            costs = self._evaluate_sequences(state, action_sequences)

            # Select elite sequences
            elite_indices = np.argsort(costs)[:self.n_elite]
            elite_sequences = action_sequences[elite_indices]

            # Update distribution
            self.action_mean = elite_sequences.mean(axis=0)
            self.action_std = elite_sequences.std(axis=0) + 1e-6

        # Return first action
        return self.action_mean[0]

    def _evaluate_sequences(
        self,
        initial_state: np.ndarray,
        action_sequences: np.ndarray
    ) -> np.ndarray:
        """Evaluate cost of action sequences.

        Args:
            initial_state: Initial state
            action_sequences: [n_samples, horizon, action_dim]

        Returns:
            costs: [n_samples]
        """
        n_samples = action_sequences.shape[0]
        costs = np.zeros(n_samples)

        # Rollout each sequence
        for i in range(n_samples):
            state = initial_state.copy()
            total_cost = 0.0

            for t in range(self.horizon):
                action = action_sequences[i, t]

                # Accumulate cost
                total_cost += self.cost_fn(state, action)

                # Predict next state
                state = self.dynamics_model.predict(state, action)

            costs[i] = total_cost

        return costs

    def reset(self):
        """Reset action distribution."""
        self.action_mean = np.zeros((self.horizon, self.action_dim))
        self.action_std = np.ones((self.horizon, self.action_dim))


# =============================================================================
# Dyna-Style Algorithm
# =============================================================================

class DynaAgent:
    """Dyna-style model-based RL agent.

    Combines real experience with simulated experience from learned model.

    Algorithm:
    1. Interact with environment and store transitions
    2. Learn dynamics model from stored transitions
    3. Use model to generate simulated transitions
    4. Train policy on both real and simulated data
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_trainer,  # Any RL trainer (SAC, TD3, etc.)
        dynamics_trainer: DynamicsModelTrainer,
        n_model_updates_per_step: int = 5,
        n_simulated_steps: int = 10,
        real_ratio: float = 0.5
    ):
        """Initialize Dyna agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            policy_trainer: RL algorithm trainer (SAC, TD3, etc.)
            dynamics_trainer: Dynamics model trainer
            n_model_updates_per_step: Model updates per environment step
            n_simulated_steps: Number of simulated steps per real step
            real_ratio: Ratio of real to simulated data in training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_trainer = policy_trainer
        self.dynamics_trainer = dynamics_trainer
        self.n_model_updates_per_step = n_model_updates_per_step
        self.n_simulated_steps = n_simulated_steps
        self.real_ratio = real_ratio

    def train_step(self) -> Dict[str, float]:
        """Perform one Dyna training step.

        Returns:
            Dictionary with training metrics
        """
        info = {}

        # 1. Train dynamics model on real data
        if len(self.policy_trainer.replay_buffer) >= self.policy_trainer.batch_size:
            for _ in range(self.n_model_updates_per_step):
                batch = self.policy_trainer.replay_buffer.sample(
                    self.policy_trainer.batch_size
                )
                model_info = self.dynamics_trainer.train_step(
                    batch['states'], batch['actions'], batch['next_states']
                )
            info.update(model_info)

        # 2. Generate simulated experience
        if len(self.policy_trainer.replay_buffer) >= self.n_simulated_steps:
            # Sample starting states from replay buffer
            batch = self.policy_trainer.replay_buffer.sample(self.n_simulated_steps)
            states = batch['states']

            # Simulate trajectories
            for state in states:
                # Select action using current policy
                action = self.policy_trainer.select_action(state, add_noise=True)

                # Predict next state
                next_state = self.dynamics_trainer.predict(state, action)

                # Compute reward (using true reward function if available)
                # For now, we don't add simulated data to buffer
                # In practice, you'd compute reward and add to buffer

        # 3. Train policy on real data (and simulated if added to buffer)
        policy_info = self.policy_trainer.train_step()
        info.update(policy_info)

        return info


# =============================================================================
# Model-Based Value Expansion (MVE)
# =============================================================================

class ModelBasedValueExpansion:
    """Model-Based Value Expansion for improved value estimates.

    Uses learned dynamics model to perform multi-step rollouts,
    improving value function estimates with synthetic data.

    V(s) ≈ r_0 + γ r_1 + ... + γ^k r_{k-1} + γ^k V(s_k)

    where s_1, ..., s_k are generated by the learned model.
    """

    def __init__(
        self,
        dynamics_model: DynamicsModelTrainer,
        reward_model: Optional[Callable] = None,
        expansion_steps: int = 5,
        gamma: float = 0.99
    ):
        """Initialize MVE.

        Args:
            dynamics_model: Learned dynamics model
            reward_model: Reward model (if None, use environment rewards)
            expansion_steps: Number of model rollout steps
            gamma: Discount factor
        """
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.expansion_steps = expansion_steps
        self.gamma = gamma

    def expand_value(
        self,
        state: np.ndarray,
        policy: Callable[[np.ndarray], np.ndarray],
        value_fn: Callable[[np.ndarray], float],
        reward_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> float:
        """Compute expanded value estimate.

        Args:
            state: Current state
            policy: Policy function
            value_fn: Value function
            reward_fn: Reward function

        Returns:
            Expanded value estimate
        """
        total_value = 0.0
        current_state = state.copy()
        discount = 1.0

        # Rollout using model
        for step in range(self.expansion_steps):
            # Get action from policy
            action = policy(current_state)

            # Compute reward
            reward = reward_fn(current_state, action)
            total_value += discount * reward

            # Predict next state
            current_state = self.dynamics_model.predict(current_state, action)

            discount *= self.gamma

        # Add terminal value
        total_value += discount * value_fn(current_state)

        return total_value


# =============================================================================
# Helper Functions
# =============================================================================

def create_dynamics_model(
    state_dim: int,
    action_dim: int,
    model_type: str = 'probabilistic',
    **kwargs
) -> DynamicsModelTrainer:
    """Create dynamics model trainer.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        model_type: 'deterministic' or 'probabilistic'
        **kwargs: Additional arguments

    Returns:
        Initialized dynamics model trainer
    """
    return DynamicsModelTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        model_type=model_type,
        **kwargs
    )


def create_mpc_controller(
    dynamics_model: DynamicsModelTrainer,
    cost_fn: Callable[[np.ndarray, np.ndarray], float],
    **kwargs
) -> ModelPredictiveControl:
    """Create MPC controller.

    Args:
        dynamics_model: Learned dynamics model
        cost_fn: Cost function
        **kwargs: Additional arguments

    Returns:
        Initialized MPC controller
    """
    return ModelPredictiveControl(
        dynamics_model=dynamics_model,
        cost_fn=cost_fn,
        **kwargs
    )
