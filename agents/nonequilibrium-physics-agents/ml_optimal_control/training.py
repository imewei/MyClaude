"""Training Algorithms for ML-Based Optimal Control.

This module implements training algorithms for neural network-based optimal control:
- PPO (Proximal Policy Optimization) for reinforcement learning
- PINN training for physics-informed value functions

Author: Nonequilibrium Physics Agents
Date: 2025-09-30
"""

import warnings
from typing import Callable, Dict, Tuple, Optional
import numpy as np

# Check if JAX/Flax are available
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn(
        "JAX/Flax not available. ML training features disabled. "
        "Install with: pip install jax jaxlib flax optax"
    )


if JAX_AVAILABLE:

    class PPOTrainer:
        """Proximal Policy Optimization (PPO) Trainer.

        Implements PPO algorithm for continuous control with clipped objective.

        Parameters
        ----------
        actor_critic : ActorCriticNetwork
            Combined actor-critic network
        learning_rate : float
            Learning rate for optimizer
        clip_epsilon : float
            PPO clipping parameter (default: 0.2)
        value_coef : float
            Value loss coefficient (default: 0.5)
        entropy_coef : float
            Entropy bonus coefficient (default: 0.01)
        gamma : float
            Discount factor (default: 0.99)
        gae_lambda : float
            GAE lambda parameter (default: 0.95)

        Methods
        -------
        compute_gae(rewards, values, dones)
            Compute Generalized Advantage Estimation
        compute_ppo_loss(params, states, actions, old_log_probs, advantages, returns)
            Compute PPO loss
        train_step(state, batch)
            Perform one training step
        train(env, n_steps, n_epochs)
            Train policy on environment
        """

        def __init__(
            self,
            actor_critic,
            learning_rate: float = 3e-4,
            clip_epsilon: float = 0.2,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
            gamma: float = 0.99,
            gae_lambda: float = 0.95
        ):
            """Initialize PPO trainer."""
            self.actor_critic = actor_critic
            self.clip_epsilon = clip_epsilon
            self.value_coef = value_coef
            self.entropy_coef = entropy_coef
            self.gamma = gamma
            self.gae_lambda = gae_lambda

        @staticmethod
        @jit
        def compute_gae(
            rewards: jnp.ndarray,
            values: jnp.ndarray,
            dones: jnp.ndarray,
            gamma: float,
            gae_lambda: float
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Compute Generalized Advantage Estimation.

            Parameters
            ----------
            rewards : jnp.ndarray
                Rewards, shape (T,)
            values : jnp.ndarray
                State values, shape (T+1,)
            dones : jnp.ndarray
                Done flags, shape (T,)
            gamma : float
                Discount factor
            gae_lambda : float
                GAE lambda parameter

            Returns
            -------
            Tuple[jnp.ndarray, jnp.ndarray]
                (advantages, returns), each shape (T,)
            """
            T = len(rewards)
            advantages = jnp.zeros(T)
            returns = jnp.zeros(T)

            gae = 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
                advantages = advantages.at[t].set(gae)
                returns = returns.at[t].set(gae + values[t])

            return advantages, returns

        def compute_ppo_loss(
            self,
            params,
            states: jnp.ndarray,
            actions: jnp.ndarray,
            old_log_probs: jnp.ndarray,
            advantages: jnp.ndarray,
            returns: jnp.ndarray
        ) -> Tuple[float, Dict]:
            """Compute PPO loss.

            Parameters
            ----------
            params : PyTree
                Network parameters
            states : jnp.ndarray
                States, shape (batch, state_dim)
            actions : jnp.ndarray
                Actions taken, shape (batch, action_dim)
            old_log_probs : jnp.ndarray
                Old action log probabilities, shape (batch,)
            advantages : jnp.ndarray
                Advantages, shape (batch,)
            returns : jnp.ndarray
                Returns, shape (batch,)

            Returns
            -------
            Tuple[float, Dict]
                (total_loss, info_dict)
            """
            # Forward pass
            (action_mean, action_log_std), values = self.actor_critic.apply(params, states)

            # Compute new log probabilities
            action_std = jnp.exp(action_log_std)
            action_dist_new = jnp.sum(
                -0.5 * ((actions - action_mean) / action_std) ** 2
                - action_log_std
                - 0.5 * jnp.log(2 * jnp.pi),
                axis=-1
            )

            # PPO ratio
            ratio = jnp.exp(action_dist_new - old_log_probs)

            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # Value loss
            value_loss = jnp.mean((values.squeeze() - returns) ** 2)

            # Entropy bonus
            entropy = jnp.mean(jnp.sum(action_log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1))

            # Total loss
            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            info = {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'entropy': entropy,
                'ratio_mean': jnp.mean(ratio),
                'ratio_std': jnp.std(ratio)
            }

            return total_loss, info

        @jit
        def train_step(
            self,
            state: train_state.TrainState,
            batch: Dict
        ) -> Tuple[train_state.TrainState, Dict]:
            """Perform one PPO training step.

            Parameters
            ----------
            state : train_state.TrainState
                Current training state
            batch : Dict
                Training batch with keys: states, actions, old_log_probs, advantages, returns

            Returns
            -------
            Tuple[train_state.TrainState, Dict]
                (updated_state, info_dict)
            """
            def loss_fn(params):
                return self.compute_ppo_loss(
                    params,
                    batch['states'],
                    batch['actions'],
                    batch['old_log_probs'],
                    batch['advantages'],
                    batch['returns']
                )

            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)

            info['loss'] = loss
            return state, info


    class PINNTrainer:
        """Physics-Informed Neural Network (PINN) Trainer.

        Trains neural network to satisfy Hamilton-Jacobi-Bellman equation:
            -∂V/∂t = min_u [L(x, u) + ∇V·f(x, u)]

        Parameters
        ----------
        pinn_network : PINNNetwork
            Physics-informed neural network
        dynamics : Callable
            System dynamics f(x, u, t)
        running_cost : Callable
            Running cost L(x, u, t)
        learning_rate : float
            Learning rate for optimizer
        physics_weight : float
            Weight for physics loss term (default: 1.0)
        boundary_weight : float
            Weight for boundary condition loss (default: 1.0)

        Methods
        -------
        compute_physics_loss(params, x, t)
            Compute HJB equation residual
        compute_boundary_loss(params, x_boundary, t_boundary, values_boundary)
            Compute boundary condition loss
        train_step(state, batch)
            Perform one training step
        train(x_data, t_data, x_boundary, t_boundary, values_boundary, n_epochs)
            Train PINN
        """

        def __init__(
            self,
            pinn_network,
            dynamics: Callable,
            running_cost: Callable,
            learning_rate: float = 1e-3,
            physics_weight: float = 1.0,
            boundary_weight: float = 1.0
        ):
            """Initialize PINN trainer."""
            self.pinn_network = pinn_network
            self.dynamics = dynamics
            self.running_cost = running_cost
            self.physics_weight = physics_weight
            self.boundary_weight = boundary_weight

        def compute_physics_loss(
            self,
            params,
            x: jnp.ndarray,
            t: jnp.ndarray,
            control_bounds: Optional[Tuple[float, float]] = None
        ) -> float:
            """Compute physics loss from HJB equation residual.

            Parameters
            ----------
            params : PyTree
                Network parameters
            x : jnp.ndarray
                State points, shape (batch, state_dim)
            t : jnp.ndarray
                Time points, shape (batch, 1)
            control_bounds : Optional[Tuple[float, float]]
                Control bounds (u_min, u_max)

            Returns
            -------
            float
                Physics loss
            """
            # Define value function
            def value_fn(x, t):
                return self.pinn_network.apply(params, x[None, :], t[None, :])[0, 0]

            # Compute gradients
            def compute_gradients(x_single, t_single):
                # ∇V w.r.t. x
                grad_V_x = grad(value_fn, argnums=0)(x_single, t_single)
                # ∂V/∂t
                grad_V_t = grad(value_fn, argnums=1)(x_single, t_single)
                return grad_V_x, grad_V_t

            # Vectorize over batch
            grad_V_x_batch, grad_V_t_batch = vmap(compute_gradients)(x, t.squeeze())

            # Compute optimal control (simplified: gradient descent)
            # For now, use zero control (can be improved)
            u_opt = jnp.zeros((x.shape[0], 1))

            # Evaluate HJB: -∂V/∂t - min_u [L + ∇V·f]
            L_batch = vmap(lambda x, u, t: self.running_cost(x, u, t))(x, u_opt, t.squeeze())
            f_batch = vmap(lambda x, u, t: self.dynamics(x, u, t))(x, u_opt, t.squeeze())

            hamiltonian = L_batch + jnp.sum(grad_V_x_batch * f_batch, axis=-1)
            hjb_residual = -grad_V_t_batch - hamiltonian

            physics_loss = jnp.mean(hjb_residual ** 2)
            return physics_loss

        def compute_boundary_loss(
            self,
            params,
            x_boundary: jnp.ndarray,
            t_boundary: jnp.ndarray,
            values_boundary: jnp.ndarray
        ) -> float:
            """Compute boundary condition loss.

            Parameters
            ----------
            params : PyTree
                Network parameters
            x_boundary : jnp.ndarray
                Boundary state points, shape (batch, state_dim)
            t_boundary : jnp.ndarray
                Boundary time points, shape (batch, 1)
            values_boundary : jnp.ndarray
                Target boundary values, shape (batch, 1)

            Returns
            -------
            float
                Boundary loss
            """
            pred_values = self.pinn_network.apply(params, x_boundary, t_boundary)
            boundary_loss = jnp.mean((pred_values - values_boundary) ** 2)
            return boundary_loss

        @jit
        def train_step(
            self,
            state: train_state.TrainState,
            batch: Dict
        ) -> Tuple[train_state.TrainState, Dict]:
            """Perform one PINN training step.

            Parameters
            ----------
            state : train_state.TrainState
                Current training state
            batch : Dict
                Training batch with keys: x, t, x_boundary, t_boundary, values_boundary

            Returns
            -------
            Tuple[train_state.TrainState, Dict]
                (updated_state, info_dict)
            """
            def loss_fn(params):
                physics_loss = self.compute_physics_loss(params, batch['x'], batch['t'])
                boundary_loss = self.compute_boundary_loss(
                    params, batch['x_boundary'], batch['t_boundary'], batch['values_boundary']
                )
                total_loss = (
                    self.physics_weight * physics_loss
                    + self.boundary_weight * boundary_loss
                )
                return total_loss, {'physics_loss': physics_loss, 'boundary_loss': boundary_loss}

            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)

            info['total_loss'] = loss
            return state, info


    def train_actor_critic(
        env,
        actor_critic_state: train_state.TrainState,
        trainer: PPOTrainer,
        n_steps: int = 1000,
        n_epochs: int = 10,
        batch_size: int = 64,
        verbose: bool = True
    ) -> Tuple[train_state.TrainState, Dict]:
        """Train actor-critic network with PPO.

        Parameters
        ----------
        env : OptimalControlEnv
            RL environment
        actor_critic_state : train_state.TrainState
            Actor-critic training state
        trainer : PPOTrainer
            PPO trainer
        n_steps : int
            Number of environment steps per epoch
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : bool
            Print progress

        Returns
        -------
        Tuple[train_state.TrainState, Dict]
            (trained_state, training_history)
        """
        history = {
            'episode_rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }

        for epoch in range(n_epochs):
            # Collect rollouts
            states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []

            state = env.reset()
            episode_reward = 0

            for step in range(n_steps):
                # Get action from policy
                (action_mean, action_log_std), value = actor_critic_state.apply_fn(
                    actor_critic_state.params, state[None, :]
                )
                action_std = jnp.exp(action_log_std[0])
                action = action_mean[0] + action_std * jax.random.normal(jax.random.PRNGKey(step), shape=action_std.shape)

                # Step environment
                next_state, reward, done, _ = env.step(action)

                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                old_log_probs.append(jnp.sum(
                    -0.5 * ((action - action_mean[0]) / action_std) ** 2
                    - action_log_std[0]
                    - 0.5 * jnp.log(2 * jnp.pi)
                ))
                values.append(value[0, 0])

                episode_reward += reward
                state = next_state

                if done:
                    state = env.reset()
                    history['episode_rewards'].append(episode_reward)
                    episode_reward = 0

            # Convert to arrays
            states = jnp.array(states)
            actions = jnp.array(actions)
            rewards = jnp.array(rewards)
            dones = jnp.array(dones)
            old_log_probs = jnp.array(old_log_probs)
            values = jnp.array(values)

            # Compute advantages
            advantages, returns = trainer.compute_gae(rewards, values, dones, trainer.gamma, trainer.gae_lambda)

            # Train on batch
            batch = {
                'states': states,
                'actions': actions,
                'old_log_probs': old_log_probs,
                'advantages': advantages,
                'returns': returns
            }

            actor_critic_state, info = trainer.train_step(actor_critic_state, batch)

            history['policy_loss'].append(float(info['policy_loss']))
            history['value_loss'].append(float(info['value_loss']))
            history['entropy'].append(float(info['entropy']))

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{n_epochs}: "
                      f"Policy Loss = {info['policy_loss']:.4f}, "
                      f"Value Loss = {info['value_loss']:.4f}, "
                      f"Entropy = {info['entropy']:.4f}")

        return actor_critic_state, history


    def train_pinn(
        pinn_state: train_state.TrainState,
        trainer: PINNTrainer,
        x_data: jnp.ndarray,
        t_data: jnp.ndarray,
        x_boundary: jnp.ndarray,
        t_boundary: jnp.ndarray,
        values_boundary: jnp.ndarray,
        n_epochs: int = 1000,
        batch_size: int = 64,
        verbose: bool = True
    ) -> Tuple[train_state.TrainState, Dict]:
        """Train PINN network.

        Parameters
        ----------
        pinn_state : train_state.TrainState
            PINN training state
        trainer : PINNTrainer
            PINN trainer
        x_data : jnp.ndarray
            Interior state points
        t_data : jnp.ndarray
            Interior time points
        x_boundary : jnp.ndarray
            Boundary state points
        t_boundary : jnp.ndarray
            Boundary time points
        values_boundary : jnp.ndarray
            Boundary values
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        verbose : bool
            Print progress

        Returns
        -------
        Tuple[train_state.TrainState, Dict]
            (trained_state, training_history)
        """
        history = {
            'total_loss': [],
            'physics_loss': [],
            'boundary_loss': []
        }

        n_data = len(x_data)

        for epoch in range(n_epochs):
            # Random batch
            idx = jax.random.choice(jax.random.PRNGKey(epoch), n_data, shape=(batch_size,))
            x_batch = x_data[idx]
            t_batch = t_data[idx]

            batch = {
                'x': x_batch,
                't': t_batch,
                'x_boundary': x_boundary,
                't_boundary': t_boundary,
                'values_boundary': values_boundary
            }

            pinn_state, info = trainer.train_step(pinn_state, batch)

            history['total_loss'].append(float(info['total_loss']))
            history['physics_loss'].append(float(info['physics_loss']))
            history['boundary_loss'].append(float(info['boundary_loss']))

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs}: "
                      f"Total Loss = {info['total_loss']:.6f}, "
                      f"Physics Loss = {info['physics_loss']:.6f}, "
                      f"Boundary Loss = {info['boundary_loss']:.6f}")

        return pinn_state, history


else:
    # Dummy implementations when JAX not available
    class PPOTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    class PINNTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def train_actor_critic(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def train_pinn(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")


# Convenience exports
__all__ = [
    'PPOTrainer',
    'PINNTrainer',
    'train_actor_critic',
    'train_pinn',
    'JAX_AVAILABLE',
]
