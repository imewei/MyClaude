"""Tests for ML Neural Network Architectures.

This test suite validates neural network architectures for optimal control.

Test Categories:
1. Network creation and initialization
2. Forward pass functionality
3. Training state management
4. Integration with JAX/Flax

Author: Nonequilibrium Physics Agents
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np

# Check if JAX/Flax are available
try:
    import jax
    import jax.numpy as jnp
    from ml_optimal_control.networks import (
        PolicyNetwork,
        ValueNetwork,
        ActorCriticNetwork,
        PINNNetwork,
        create_policy_network,
        create_value_network,
        create_actor_critic_network,
        create_pinn_network,
        JAX_AVAILABLE
    )
    JAX_AVAILABLE_LOCAL = True
except ImportError:
    JAX_AVAILABLE_LOCAL = False
    pytestmark = pytest.mark.skip(reason="JAX/Flax not available")


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestPolicyNetwork:
    """Test Policy Network."""

    def test_1_creation(self):
        """Test 1: Policy network creation."""
        print("\n  Test 1: Policy network creation")

        network, state = create_policy_network(
            state_dim=4,
            action_dim=2,
            hidden_dims=(64, 64),
            learning_rate=3e-4
        )

        assert network is not None
        assert state is not None
        print(f"    Created policy network: state_dim=4, action_dim=2")

    def test_2_forward_pass(self):
        """Test 2: Forward pass."""
        print("\n  Test 2: Policy network forward pass")

        network, state = create_policy_network(
            state_dim=4,
            action_dim=2
        )

        # Test forward pass
        x = jnp.ones((8, 4))  # Batch of 8
        action_mean, action_log_std = network.apply(state.params, x)

        print(f"    Input shape: {x.shape}")
        print(f"    Output mean shape: {action_mean.shape}")
        print(f"    Output log_std shape: {action_log_std.shape}")

        assert action_mean.shape == (8, 2)
        assert action_log_std.shape == (8, 2)

    def test_3_output_bounds(self):
        """Test 3: Check output bounds."""
        print("\n  Test 3: Policy output bounds")

        network, state = create_policy_network(state_dim=2, action_dim=1)

        x = jnp.array([[1.0, -1.0]])
        action_mean, action_log_std = network.apply(state.params, x)

        # Log std should be clipped
        assert jnp.all(action_log_std >= -20)
        assert jnp.all(action_log_std <= 2)
        print(f"    Log std bounds satisfied: [{-20}, {2}]")


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestValueNetwork:
    """Test Value Network."""

    def test_4_creation(self):
        """Test 4: Value network creation."""
        print("\n  Test 4: Value network creation")

        network, state = create_value_network(
            state_dim=4,
            hidden_dims=(64, 64)
        )

        assert network is not None
        assert state is not None
        print(f"    Created value network: state_dim=4")

    def test_5_forward_pass(self):
        """Test 5: Forward pass."""
        print("\n  Test 5: Value network forward pass")

        network, state = create_value_network(state_dim=4)

        x = jnp.ones((8, 4))
        value = network.apply(state.params, x)

        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {value.shape}")

        assert value.shape == (8, 1)


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestActorCriticNetwork:
    """Test Actor-Critic Network."""

    def test_6_creation(self):
        """Test 6: Actor-Critic network creation."""
        print("\n  Test 6: Actor-Critic network creation")

        network, state = create_actor_critic_network(
            state_dim=4,
            action_dim=2,
            hidden_dims=(64, 64),
            policy_dims=(32,),
            value_dims=(32,)
        )

        assert network is not None
        assert state is not None
        print(f"    Created actor-critic network")

    def test_7_forward_pass(self):
        """Test 7: Forward pass."""
        print("\n  Test 7: Actor-Critic forward pass")

        network, state = create_actor_critic_network(
            state_dim=4,
            action_dim=2
        )

        x = jnp.ones((8, 4))
        (action_mean, action_log_std), value = network.apply(state.params, x)

        print(f"    Input shape: {x.shape}")
        print(f"    Action mean shape: {action_mean.shape}")
        print(f"    Action log_std shape: {action_log_std.shape}")
        print(f"    Value shape: {value.shape}")

        assert action_mean.shape == (8, 2)
        assert action_log_std.shape == (8, 2)
        assert value.shape == (8, 1)


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestPINNNetwork:
    """Test Physics-Informed Neural Network."""

    def test_8_creation(self):
        """Test 8: PINN network creation."""
        print("\n  Test 8: PINN network creation")

        network, state = create_pinn_network(
            state_dim=2,
            hidden_dims=(64, 64, 64)
        )

        assert network is not None
        assert state is not None
        print(f"    Created PINN network: state_dim=2")

    def test_9_forward_pass(self):
        """Test 9: Forward pass."""
        print("\n  Test 9: PINN forward pass")

        network, state = create_pinn_network(state_dim=2)

        x = jnp.ones((8, 2))
        t = jnp.linspace(0, 1, 8).reshape(-1, 1)
        value = network.apply(state.params, x, t)

        print(f"    State shape: {x.shape}")
        print(f"    Time shape: {t.shape}")
        print(f"    Output shape: {value.shape}")

        assert value.shape == (8, 1)


def run_all_tests():
    """Run all ML network tests."""
    print("\n" + "="*70)
    print("ML Neural Network Tests")
    print("="*70)

    if not JAX_AVAILABLE_LOCAL:
        print("\n✗ JAX/Flax not available - tests skipped")
        print("  Install with: pip install jax jaxlib flax optax")
        return False

    test_classes = [
        TestPolicyNetwork,
        TestValueNetwork,
        TestActorCriticNetwork,
        TestPINNNetwork
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)

        test_obj = test_class()
        methods = [m for m in dir(test_obj) if m.startswith('test_')]

        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_obj, method_name)
                method()
                passed_tests += 1
                print(f"  ✓ {method_name}")
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
            except Exception as e:
                print(f"  ✗ {method_name}: Unexpected error: {e}")

    print("\n" + "="*70)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("="*70)

    return passed_tests == total_tests


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
