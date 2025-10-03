"""Tests for Advanced RL Algorithms.

This test suite validates advanced reinforcement learning algorithms:
1. DDPG (Deep Deterministic Policy Gradient)
2. TD3 (Twin Delayed DDPG)
3. SAC (Soft Actor-Critic)
4. Replay Buffer
5. Network architectures

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
    from ml_optimal_control.advanced_rl import (
        ReplayBuffer,
        DDPGTrainer,
        TD3Trainer,
        SACTrainer,
        DeterministicPolicy,
        QNetwork,
        DoubleQNetwork,
        GaussianPolicy,
        create_ddpg_trainer,
        create_td3_trainer,
        create_sac_trainer,
        JAX_AVAILABLE
    )
    JAX_AVAILABLE_LOCAL = True
except ImportError:
    JAX_AVAILABLE_LOCAL = False
    pytestmark = pytest.mark.skip(reason="JAX/Flax not available")


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestReplayBuffer:
    """Test Replay Buffer."""

    def test_1_creation(self):
        """Test 1: Create replay buffer."""
        print("\n  Test 1: Replay buffer creation")

        buffer = ReplayBuffer(capacity=1000, state_dim=4, action_dim=2)

        assert buffer.capacity == 1000
        assert buffer.state_dim == 4
        assert buffer.action_dim == 2
        assert len(buffer) == 0
        print(f"    Created buffer: capacity=1000, state_dim=4, action_dim=2")

    def test_2_add_transitions(self):
        """Test 2: Add transitions to buffer."""
        print("\n  Test 2: Add transitions")

        buffer = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)

        # Add some transitions
        for i in range(50):
            state = np.random.randn(2)
            action = np.random.randn(1)
            reward = np.random.randn()
            next_state = np.random.randn(2)
            done = False

            buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 50
        print(f"    Added 50 transitions, buffer size: {len(buffer)}")

    def test_3_circular_buffer(self):
        """Test 3: Circular buffer behavior."""
        print("\n  Test 3: Circular buffer (overflow)")

        buffer = ReplayBuffer(capacity=10, state_dim=2, action_dim=1)

        # Add more than capacity
        for i in range(15):
            buffer.add(
                np.random.randn(2),
                np.random.randn(1),
                0.0,
                np.random.randn(2),
                False
            )

        assert len(buffer) == 10  # Should not exceed capacity
        print(f"    Added 15 transitions to buffer with capacity 10")
        print(f"    Buffer size: {len(buffer)} (capped at capacity)")

    def test_4_sample_batch(self):
        """Test 4: Sample batches."""
        print("\n  Test 4: Sample batches")

        buffer = ReplayBuffer(capacity=100, state_dim=2, action_dim=1)

        # Fill buffer
        for i in range(50):
            buffer.add(
                np.random.randn(2),
                np.random.randn(1),
                float(i),
                np.random.randn(2),
                False
            )

        # Sample batch
        batch = buffer.sample(batch_size=32)

        assert batch['states'].shape == (32, 2)
        assert batch['actions'].shape == (32, 1)
        assert batch['rewards'].shape == (32,)
        assert batch['next_states'].shape == (32, 2)
        assert batch['dones'].shape == (32,)

        print(f"    Sampled batch of 32 transitions")
        print(f"    States shape: {batch['states'].shape}")


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestNetworkArchitectures:
    """Test network architectures for advanced RL."""

    def test_5_deterministic_policy(self):
        """Test 5: Deterministic policy network."""
        print("\n  Test 5: Deterministic policy network")

        policy = DeterministicPolicy(hidden_dims=(64, 64), action_dim=2)

        rng = jax.random.PRNGKey(0)
        state = jnp.ones((8, 4))

        params = policy.init(rng, state)
        action = policy.apply(params, state)

        print(f"    Input shape: {state.shape}")
        print(f"    Output shape: {action.shape}")
        print(f"    Output range: [{jnp.min(action):.3f}, {jnp.max(action):.3f}]")

        assert action.shape == (8, 2)
        assert jnp.all(action >= -1) and jnp.all(action <= 1)  # tanh bounds

    def test_6_q_network(self):
        """Test 6: Q-value network."""
        print("\n  Test 6: Q-value network")

        q_net = QNetwork(hidden_dims=(64, 64))

        rng = jax.random.PRNGKey(0)
        state = jnp.ones((8, 4))
        action = jnp.ones((8, 2))

        params = q_net.init(rng, state, action)
        q_value = q_net.apply(params, state, action)

        print(f"    State shape: {state.shape}")
        print(f"    Action shape: {action.shape}")
        print(f"    Q-value shape: {q_value.shape}")

        assert q_value.shape == (8, 1)

    def test_7_double_q_network(self):
        """Test 7: Double Q-network (TD3)."""
        print("\n  Test 7: Double Q-network")

        q_net = DoubleQNetwork(hidden_dims=(64, 64))

        rng = jax.random.PRNGKey(0)
        state = jnp.ones((8, 4))
        action = jnp.ones((8, 2))

        params = q_net.init(rng, state, action)
        q1, q2 = q_net.apply(params, state, action)

        print(f"    Q1 shape: {q1.shape}")
        print(f"    Q2 shape: {q2.shape}")

        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_8_gaussian_policy(self):
        """Test 8: Gaussian policy (SAC)."""
        print("\n  Test 8: Gaussian policy network")

        policy = GaussianPolicy(hidden_dims=(64, 64), action_dim=2)

        rng = jax.random.PRNGKey(0)
        state = jnp.ones((8, 4))

        params = policy.init(rng, state)
        mean, log_std = policy.apply(params, state)

        print(f"    Mean shape: {mean.shape}")
        print(f"    Log_std shape: {log_std.shape}")
        print(f"    Log_std range: [{jnp.min(log_std):.3f}, {jnp.max(log_std):.3f}]")

        assert mean.shape == (8, 2)
        assert log_std.shape == (8, 2)
        assert jnp.all(log_std >= -20) and jnp.all(log_std <= 2)  # Clipped

    def test_9_gaussian_policy_sample(self):
        """Test 9: Sample from Gaussian policy."""
        print("\n  Test 9: Sample actions from policy")

        policy = GaussianPolicy(hidden_dims=(64, 64), action_dim=2)

        rng = jax.random.PRNGKey(0)
        state = jnp.ones((1, 4))

        params = policy.init(rng, state)

        rng, subkey = jax.random.split(rng)
        action, log_prob = policy.sample(params, state, subkey)

        print(f"    Action shape: {action.shape}")
        print(f"    Action range: [{jnp.min(action):.3f}, {jnp.max(action):.3f}]")
        print(f"    Log prob: {log_prob[0]:.3f}")

        assert action.shape == (1, 2)
        assert jnp.all(action >= -1) and jnp.all(action <= 1)  # tanh squashed


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestDDPG:
    """Test DDPG trainer."""

    def test_10_creation(self):
        """Test 10: Create DDPG trainer."""
        print("\n  Test 10: DDPG trainer creation")

        trainer = create_ddpg_trainer(state_dim=4, action_dim=2)

        assert trainer.state_dim == 4
        assert trainer.action_dim == 2
        assert trainer.actor is not None
        assert trainer.critic is not None
        print(f"    Created DDPG trainer: state_dim=4, action_dim=2")

    def test_11_select_action(self):
        """Test 11: Select action."""
        print("\n  Test 11: Select action")

        trainer = create_ddpg_trainer(state_dim=2, action_dim=1)
        state = np.random.randn(2)

        # Without noise
        action_det = trainer.select_action(state, add_noise=False)
        print(f"    Deterministic action: {action_det}")

        # With noise
        action_noisy = trainer.select_action(state, add_noise=True)
        print(f"    Noisy action: {action_noisy}")

        assert action_det.shape == (1,)
        assert action_noisy.shape == (1,)

    def test_12_train_step(self):
        """Test 12: Training step."""
        print("\n  Test 12: DDPG training step")

        trainer = create_ddpg_trainer(
            state_dim=2,
            action_dim=1,
            batch_size=32,
            buffer_capacity=1000
        )

        # Fill replay buffer
        for i in range(100):
            trainer.replay_buffer.add(
                np.random.randn(2),
                np.random.randn(1),
                np.random.randn(),
                np.random.randn(2),
                False
            )

        # Train
        info = trainer.train_step()

        print(f"    Critic loss: {info.get('critic_loss', 'N/A')}")
        print(f"    Actor loss: {info.get('actor_loss', 'N/A')}")

        assert 'critic_loss' in info
        assert 'actor_loss' in info


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestTD3:
    """Test TD3 trainer."""

    def test_13_creation(self):
        """Test 13: Create TD3 trainer."""
        print("\n  Test 13: TD3 trainer creation")

        trainer = create_td3_trainer(state_dim=4, action_dim=2)

        assert trainer.state_dim == 4
        assert trainer.action_dim == 2
        assert trainer.policy_freq == 2  # Delayed updates
        print(f"    Created TD3 trainer with delayed policy updates")

    def test_14_train_step(self):
        """Test 14: TD3 training step."""
        print("\n  Test 14: TD3 training step")

        trainer = create_td3_trainer(
            state_dim=2,
            action_dim=1,
            batch_size=32,
            policy_freq=2
        )

        # Fill buffer
        for i in range(100):
            trainer.replay_buffer.add(
                np.random.randn(2),
                np.random.randn(1),
                np.random.randn(),
                np.random.randn(2),
                False
            )

        # Train (first step - only critic update)
        info1 = trainer.train_step()
        print(f"    Step 1 (critic only): {list(info1.keys())}")

        # Train (second step - critic + actor update)
        info2 = trainer.train_step()
        print(f"    Step 2 (critic + actor): {list(info2.keys())}")

        assert 'critic_loss' in info1
        # Actor update happens every policy_freq steps
        if 'actor_loss' in info2:
            print(f"    Actor loss: {info2['actor_loss']}")


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestSAC:
    """Test SAC trainer."""

    def test_15_creation(self):
        """Test 15: Create SAC trainer."""
        print("\n  Test 15: SAC trainer creation")

        trainer = create_sac_trainer(
            state_dim=4,
            action_dim=2,
            auto_entropy_tuning=True
        )

        assert trainer.state_dim == 4
        assert trainer.action_dim == 2
        assert trainer.auto_entropy_tuning is True
        print(f"    Created SAC trainer with automatic entropy tuning")

    def test_16_select_action_stochastic(self):
        """Test 16: Select stochastic action."""
        print("\n  Test 16: Select action (stochastic)")

        trainer = create_sac_trainer(state_dim=2, action_dim=1)
        state = np.random.randn(2)

        # Stochastic
        action1 = trainer.select_action(state, deterministic=False)
        action2 = trainer.select_action(state, deterministic=False)

        print(f"    Action 1: {action1}")
        print(f"    Action 2: {action2}")

        # Should be different due to sampling
        # (Note: might rarely be equal, but very unlikely)

        assert action1.shape == (1,)
        assert action2.shape == (1,)

    def test_17_select_action_deterministic(self):
        """Test 17: Select deterministic action."""
        print("\n  Test 17: Select action (deterministic)")

        trainer = create_sac_trainer(state_dim=2, action_dim=1)
        state = np.random.randn(2)

        # Deterministic
        action1 = trainer.select_action(state, deterministic=True)
        action2 = trainer.select_action(state, deterministic=True)

        print(f"    Action 1: {action1}")
        print(f"    Action 2: {action2}")

        # Should be identical
        assert np.allclose(action1, action2)

    def test_18_train_step(self):
        """Test 18: SAC training step."""
        print("\n  Test 18: SAC training step")

        trainer = create_sac_trainer(
            state_dim=2,
            action_dim=1,
            batch_size=32,
            auto_entropy_tuning=True
        )

        # Fill buffer
        for i in range(100):
            trainer.replay_buffer.add(
                np.random.randn(2),
                np.random.randn(1),
                np.random.randn(),
                np.random.randn(2),
                False
            )

        # Train
        info = trainer.train_step()

        print(f"    Critic loss: {info.get('critic_loss', 'N/A')}")
        print(f"    Actor loss: {info.get('actor_loss', 'N/A')}")
        print(f"    Entropy: {info.get('entropy', 'N/A')}")
        print(f"    Alpha: {info.get('alpha', 'N/A')}")

        assert 'critic_loss' in info
        assert 'actor_loss' in info
        assert 'entropy' in info
        assert 'alpha' in info

        if 'alpha_loss' in info:
            print(f"    Alpha loss: {info['alpha_loss']}")


@pytest.mark.skipif(not JAX_AVAILABLE_LOCAL, reason="JAX/Flax not available")
class TestIntegration:
    """Integration tests for advanced RL."""

    def test_19_simple_control_task(self):
        """Test 19: Simple control task with DDPG."""
        print("\n  Test 19: Simple LQR-like task")

        # Simple task: x_dot = u, cost = x^2 + u^2
        # Optimal policy: u = -k*x for some k

        trainer = create_ddpg_trainer(
            state_dim=1,
            action_dim=1,
            batch_size=32,
            exploration_noise=0.1
        )

        # Collect some experience
        x = np.array([1.0])
        total_reward = 0.0

        for step in range(50):
            # Select action
            action = trainer.select_action(x, add_noise=True)

            # Simple dynamics: x_next = x + u * dt
            dt = 0.1
            x_next = x + action * dt

            # Reward: -(x^2 + u^2)
            reward = -(x**2 + action**2)[0] * dt

            # Store transition
            trainer.replay_buffer.add(x, action, reward, x_next, False)

            total_reward += reward
            x = x_next

            # Train if enough data
            if len(trainer.replay_buffer) >= trainer.batch_size:
                trainer.train_step()

        print(f"    Collected 50 steps, total reward: {total_reward:.3f}")
        print(f"    Buffer size: {len(trainer.replay_buffer)}")
        print(f"    Training steps: {trainer.total_steps}")

        assert len(trainer.replay_buffer) == 50


def run_all_tests():
    """Run all advanced RL tests."""
    print("\n" + "="*70)
    print("Advanced RL Tests")
    print("="*70)

    if not JAX_AVAILABLE_LOCAL:
        print("\n✗ JAX/Flax not available - tests skipped")
        print("  Install with: pip install jax jaxlib flax optax")
        return False

    test_classes = [
        TestReplayBuffer,
        TestNetworkArchitectures,
        TestDDPG,
        TestTD3,
        TestSAC,
        TestIntegration
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
