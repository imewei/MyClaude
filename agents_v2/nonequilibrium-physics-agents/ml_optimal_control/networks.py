"""Neural Network Architectures for Optimal Control.

This module implements various neural network architectures using Flax/JAX
for optimal control and reinforcement learning applications.

Architectures:
- Actor-Critic: Policy and value function networks
- PINN: Physics-Informed Neural Networks for HJB equation
- Value Network: State value function approximation
- Policy Network: Stochastic policy for continuous control

Author: Nonequilibrium Physics Agents
Date: 2025-09-30
"""

import warnings
from typing import Sequence, Callable, Tuple, Optional
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
        "JAX/Flax not available. ML optimal control features disabled. "
        "Install with: pip install jax jaxlib flax optax"
    )
    # Dummy base class for when JAX is not available
    class nn:
        class Module:
            pass


if JAX_AVAILABLE:

    class PolicyNetwork(nn.Module):
        """Policy network for continuous control (Actor).

        Outputs mean and log standard deviation of Gaussian policy.

        Parameters
        ----------
        hidden_dims : Sequence[int]
            Hidden layer dimensions
        action_dim : int
            Action space dimension
        activation : Callable
            Activation function (default: tanh)

        Attributes
        ----------
        hidden_layers : list
            Dense hidden layers
        mean_layer : Dense
            Output layer for action mean
        log_std_layer : Dense
            Output layer for action log std

        Methods
        -------
        __call__(x)
            Forward pass, returns (action_mean, action_log_std)
        """
        hidden_dims: Sequence[int]
        action_dim: int
        activation: Callable = nn.tanh

        @nn.compact
        def __call__(self, x):
            """Forward pass through policy network.

            Parameters
            ----------
            x : jnp.ndarray
                State input, shape (batch, state_dim)

            Returns
            -------
            Tuple[jnp.ndarray, jnp.ndarray]
                (action_mean, action_log_std), each shape (batch, action_dim)
            """
            # Hidden layers
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = self.activation(x)

            # Action mean (unbounded)
            action_mean = nn.Dense(self.action_dim)(x)

            # Action log std (bounded for stability)
            action_log_std = nn.Dense(self.action_dim)(x)
            action_log_std = jnp.clip(action_log_std, -20, 2)

            return action_mean, action_log_std


    class ValueNetwork(nn.Module):
        """Value network for state value function (Critic).

        Outputs scalar value V(s) for given state.

        Parameters
        ----------
        hidden_dims : Sequence[int]
            Hidden layer dimensions
        activation : Callable
            Activation function (default: tanh)

        Attributes
        ----------
        hidden_layers : list
            Dense hidden layers
        value_layer : Dense
            Output layer for value

        Methods
        -------
        __call__(x)
            Forward pass, returns value
        """
        hidden_dims: Sequence[int]
        activation: Callable = nn.tanh

        @nn.compact
        def __call__(self, x):
            """Forward pass through value network.

            Parameters
            ----------
            x : jnp.ndarray
                State input, shape (batch, state_dim)

            Returns
            -------
            jnp.ndarray
                State value, shape (batch, 1)
            """
            # Hidden layers
            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = self.activation(x)

            # Value output
            value = nn.Dense(1)(x)

            return value


    class ActorCriticNetwork(nn.Module):
        """Combined Actor-Critic network.

        Shared representation with separate policy and value heads.

        Parameters
        ----------
        hidden_dims : Sequence[int]
            Shared hidden layer dimensions
        policy_dims : Sequence[int]
            Policy-specific layer dimensions
        value_dims : Sequence[int]
            Value-specific layer dimensions
        action_dim : int
            Action space dimension
        activation : Callable
            Activation function (default: tanh)

        Attributes
        ----------
        shared_layers : list
            Shared dense layers
        policy_head : PolicyNetwork
            Policy network head
        value_head : ValueNetwork
            Value network head

        Methods
        -------
        __call__(x)
            Forward pass, returns ((action_mean, action_log_std), value)
        """
        hidden_dims: Sequence[int]
        policy_dims: Sequence[int]
        value_dims: Sequence[int]
        action_dim: int
        activation: Callable = nn.tanh

        @nn.compact
        def __call__(self, x):
            """Forward pass through actor-critic network.

            Parameters
            ----------
            x : jnp.ndarray
                State input, shape (batch, state_dim)

            Returns
            -------
            Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]
                ((action_mean, action_log_std), value)
            """
            # Shared layers
            shared = x
            for hidden_dim in self.hidden_dims:
                shared = nn.Dense(hidden_dim)(shared)
                shared = self.activation(shared)

            # Policy head
            policy = shared
            for policy_dim in self.policy_dims:
                policy = nn.Dense(policy_dim)(policy)
                policy = self.activation(policy)
            action_mean = nn.Dense(self.action_dim)(policy)
            action_log_std = nn.Dense(self.action_dim)(policy)
            action_log_std = jnp.clip(action_log_std, -20, 2)

            # Value head
            value = shared
            for value_dim in self.value_dims:
                value = nn.Dense(value_dim)(value)
                value = self.activation(value)
            value = nn.Dense(1)(value)

            return (action_mean, action_log_std), value


    class PINNNetwork(nn.Module):
        """Physics-Informed Neural Network for Hamilton-Jacobi-Bellman equation.

        Learns value function V(x, t) that satisfies HJB PDE:
            -∂V/∂t = min_u [L(x, u) + ∇V·f(x, u)]

        Parameters
        ----------
        hidden_dims : Sequence[int]
            Hidden layer dimensions
        state_dim : int
            State space dimension
        activation : Callable
            Activation function (default: tanh)

        Attributes
        ----------
        hidden_layers : list
            Dense hidden layers
        output_layer : Dense
            Output layer for value

        Methods
        -------
        __call__(x, t)
            Forward pass, returns value V(x, t)
        compute_gradients(x, t)
            Compute ∇V and ∂V/∂t
        compute_hjb_residual(x, t, dynamics, cost)
            Compute HJB equation residual for physics loss
        """
        hidden_dims: Sequence[int]
        state_dim: int
        activation: Callable = nn.tanh

        @nn.compact
        def __call__(self, x, t):
            """Forward pass through PINN.

            Parameters
            ----------
            x : jnp.ndarray
                State input, shape (batch, state_dim)
            t : jnp.ndarray
                Time input, shape (batch, 1)

            Returns
            -------
            jnp.ndarray
                Value V(x, t), shape (batch, 1)
            """
            # Concatenate state and time
            inputs = jnp.concatenate([x, t], axis=-1)

            # Hidden layers
            h = inputs
            for hidden_dim in self.hidden_dims:
                h = nn.Dense(hidden_dim)(h)
                h = self.activation(h)

            # Value output
            value = nn.Dense(1)(h)

            return value


    def create_policy_network(
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        learning_rate: float = 3e-4
    ):
        """Create and initialize a policy network.

        Parameters
        ----------
        state_dim : int
            State space dimension
        action_dim : int
            Action space dimension
        hidden_dims : Sequence[int]
            Hidden layer dimensions
        learning_rate : float
            Learning rate for optimizer

        Returns
        -------
        Tuple[PolicyNetwork, train_state.TrainState]
            Network module and training state
        """
        network = PolicyNetwork(
            hidden_dims=hidden_dims,
            action_dim=action_dim
        )

        # Initialize
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, state_dim))
        params = network.init(rng, dummy_x)

        # Create training state
        tx = optax.adam(learning_rate)
        state = train_state.TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx
        )

        return network, state


    def create_value_network(
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        learning_rate: float = 3e-4
    ):
        """Create and initialize a value network.

        Parameters
        ----------
        state_dim : int
            State space dimension
        hidden_dims : Sequence[int]
            Hidden layer dimensions
        learning_rate : float
            Learning rate for optimizer

        Returns
        -------
        Tuple[ValueNetwork, train_state.TrainState]
            Network module and training state
        """
        network = ValueNetwork(hidden_dims=hidden_dims)

        # Initialize
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, state_dim))
        params = network.init(rng, dummy_x)

        # Create training state
        tx = optax.adam(learning_rate)
        state = train_state.TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx
        )

        return network, state


    def create_actor_critic_network(
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        policy_dims: Sequence[int] = (32,),
        value_dims: Sequence[int] = (32,),
        learning_rate: float = 3e-4
    ):
        """Create and initialize an actor-critic network.

        Parameters
        ----------
        state_dim : int
            State space dimension
        action_dim : int
            Action space dimension
        hidden_dims : Sequence[int]
            Shared hidden layer dimensions
        policy_dims : Sequence[int]
            Policy-specific layer dimensions
        value_dims : Sequence[int]
            Value-specific layer dimensions
        learning_rate : float
            Learning rate for optimizer

        Returns
        -------
        Tuple[ActorCriticNetwork, train_state.TrainState]
            Network module and training state
        """
        network = ActorCriticNetwork(
            hidden_dims=hidden_dims,
            policy_dims=policy_dims,
            value_dims=value_dims,
            action_dim=action_dim
        )

        # Initialize
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, state_dim))
        params = network.init(rng, dummy_x)

        # Create training state
        tx = optax.adam(learning_rate)
        state = train_state.TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx
        )

        return network, state


    def create_pinn_network(
        state_dim: int,
        hidden_dims: Sequence[int] = (64, 64, 64),
        learning_rate: float = 1e-3
    ):
        """Create and initialize a PINN network.

        Parameters
        ----------
        state_dim : int
            State space dimension
        hidden_dims : Sequence[int]
            Hidden layer dimensions
        learning_rate : float
            Learning rate for optimizer

        Returns
        -------
        Tuple[PINNNetwork, train_state.TrainState]
            Network module and training state
        """
        network = PINNNetwork(
            hidden_dims=hidden_dims,
            state_dim=state_dim
        )

        # Initialize
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, state_dim))
        dummy_t = jnp.ones((1, 1))
        params = network.init(rng, dummy_x, dummy_t)

        # Create training state
        tx = optax.adam(learning_rate)
        state = train_state.TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx
        )

        return network, state


else:
    # Dummy implementations when JAX not available
    def create_policy_network(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def create_value_network(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def create_actor_critic_network(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def create_pinn_network(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")


# Backward compatibility aliases and helper functions
if JAX_AVAILABLE:
    # Alias for backward compatibility
    NeuralController = PolicyNetwork

    # Helper function for creating MLPs
    def create_mlp(input_dim: int, output_dim: int, hidden_sizes: Sequence[int], activation=nn.tanh):
        """Create a simple MLP network.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function

        Returns:
            Flax Module
        """
        class MLP(nn.Module):
            @nn.compact
            def __call__(self, x):
                for size in hidden_sizes:
                    x = nn.Dense(size)(x)
                    x = activation(x)
                return nn.Dense(output_dim)(x)
        return MLP()

    # Helper aliases
    def value_function_network(*args, **kwargs):
        """Alias for create_value_network for backward compatibility."""
        return create_value_network(*args, **kwargs)

    def policy_network(*args, **kwargs):
        """Alias for create_policy_network for backward compatibility."""
        return create_policy_network(*args, **kwargs)
else:
    # Dummy implementations
    def NeuralController(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def create_mlp(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def value_function_network(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")

    def policy_network(*args, **kwargs):
        raise ImportError("JAX/Flax required. Install with: pip install jax jaxlib flax optax")


# Convenience exports
__all__ = [
    'PolicyNetwork',
    'ValueNetwork',
    'ActorCriticNetwork',
    'PINNNetwork',
    'NeuralController',  # Alias for PolicyNetwork
    'create_policy_network',
    'create_value_network',
    'create_actor_critic_network',
    'create_pinn_network',
    'create_mlp',
    'value_function_network',
    'policy_network',
    'JAX_AVAILABLE',
]
