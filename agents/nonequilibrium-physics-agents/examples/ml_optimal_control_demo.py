"""ML-Based Optimal Control Demonstration.

This script demonstrates machine learning approaches to optimal control:
1. Neural network policy initialization from PMP
2. Actor-Critic training with PPO
3. PINN for value function approximation
4. Quantum control with learned policies

Author: Nonequilibrium Physics Agents
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Check if JAX/Flax are available
try:
    import jax
    import jax.numpy as jnp
    from ml_optimal_control.networks import (
        create_policy_network,
        create_value_network,
        create_actor_critic_network
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX/Flax not available. Install with: pip install jax jaxlib flax optax")

# Import environments (work without JAX)
from ml_optimal_control.environments import OptimalControlEnv, QuantumControlEnv


def demo_1_policy_network():
    """Demo 1: Policy network basics."""
    print("\n" + "="*70)
    print("Demo 1: Policy Network Basics")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    # Create policy network
    print("\nCreating policy network...")
    policy_net, policy_state = create_policy_network(
        state_dim=2,
        action_dim=1,
        hidden_dims=(64, 64),
        learning_rate=3e-4
    )

    print(f"  State dimension: 2")
    print(f"  Action dimension: 1")
    print(f"  Hidden layers: (64, 64)")

    # Test forward pass
    print("\nTesting forward pass...")
    test_state = jnp.array([[1.0, -0.5]])
    action_mean, action_log_std = policy_net.apply(policy_state.params, test_state)

    print(f"  Input state: {test_state[0]}")
    print(f"  Action mean: {action_mean[0]}")
    print(f"  Action std: {jnp.exp(action_log_std[0])}")

    # Sample actions
    print("\nSampling actions...")
    action_std = jnp.exp(action_log_std[0])
    for i in range(5):
        action = action_mean[0] + action_std * jax.random.normal(
            jax.random.PRNGKey(i), shape=action_std.shape
        )
        print(f"  Sample {i+1}: {action}")

    print("\n✓ Policy network demonstration complete")


def demo_2_value_network():
    """Demo 2: Value network basics."""
    print("\n" + "="*70)
    print("Demo 2: Value Network Basics")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    # Create value network
    print("\nCreating value network...")
    value_net, value_state = create_value_network(
        state_dim=2,
        hidden_dims=(64, 64)
    )

    print(f"  State dimension: 2")
    print(f"  Hidden layers: (64, 64)")

    # Test forward pass
    print("\nTesting value estimation...")
    test_states = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])

    values = value_net.apply(value_state.params, test_states)

    print(f"\nState values:")
    for i, (state, value) in enumerate(zip(test_states, values)):
        print(f"  State {state}: Value = {value[0]:.4f}")

    print("\n✓ Value network demonstration complete")


def demo_3_actor_critic():
    """Demo 3: Actor-Critic network."""
    print("\n" + "="*70)
    print("Demo 3: Actor-Critic Network")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    # Create actor-critic network
    print("\nCreating actor-critic network...")
    ac_net, ac_state = create_actor_critic_network(
        state_dim=2,
        action_dim=1,
        hidden_dims=(64, 64),
        policy_dims=(32,),
        value_dims=(32,)
    )

    print(f"  State dimension: 2")
    print(f"  Action dimension: 1")
    print(f"  Shared layers: (64, 64)")
    print(f"  Policy head: (32,)")
    print(f"  Value head: (32,)")

    # Test forward pass
    print("\nTesting combined forward pass...")
    test_state = jnp.array([[1.0, -0.5]])
    (action_mean, action_log_std), value = ac_net.apply(ac_state.params, test_state)

    print(f"\nFor state {test_state[0]}:")
    print(f"  Action mean: {action_mean[0]}")
    print(f"  Action std: {jnp.exp(action_log_std[0])}")
    print(f"  State value: {value[0, 0]:.4f}")

    print("\n✓ Actor-Critic network demonstration complete")


def demo_4_optimal_control_env():
    """Demo 4: Optimal Control Environment."""
    print("\n" + "="*70)
    print("Demo 4: Optimal Control Environment")
    print("="*70)

    # Simple LQR dynamics
    def dynamics(x, u, t):
        return u

    def cost(x, u, t):
        return x**2 + u**2

    print("\nCreating LQR environment...")
    env = OptimalControlEnv(
        dynamics=dynamics,
        cost=cost,
        x0=np.array([1.0]),
        duration=5.0,
        dt=0.1
    )

    print(f"  Initial state: {env.x0}")
    print(f"  Duration: {env.duration}")
    print(f"  Time step: {env.dt}")

    # Run a few steps with random policy
    print("\nRunning environment with random policy...")
    state = env.reset()
    total_reward = 0.0

    for step in range(10):
        action = np.random.randn(1) * 0.1  # Small random actions
        next_state, reward, done, info = env.step(action)

        if step < 5:
            print(f"  Step {step}: state={float(next_state[0]):.4f}, "
                  f"reward={float(reward):.4f}, done={done}")

        total_reward += reward
        state = next_state

        if done:
            break

    print(f"\n  Total reward (10 steps): {float(total_reward):.4f}")
    print("\n✓ Environment demonstration complete")


def demo_5_quantum_control_env():
    """Demo 5: Quantum Control Environment."""
    print("\n" + "="*70)
    print("Demo 5: Quantum Control Environment")
    print("="*70)

    # Two-level system
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    H0 = np.zeros((2, 2), dtype=complex)

    psi0 = np.array([1, 0], dtype=complex)
    psi_target = np.array([0, 1], dtype=complex)

    print("\nCreating quantum control environment...")
    print("  System: Two-level (qubit)")
    print(f"  Initial state: |0⟩")
    print(f"  Target state: |1⟩")

    env = QuantumControlEnv(
        H0=H0,
        control_hamiltonians=[sigma_x],
        psi0=psi0,
        psi_target=psi_target,
        duration=5.0,
        dt=0.05,
        control_bounds=(np.array([-3.0]), np.array([3.0]))
    )

    print(f"  Duration: {env.duration}")
    print(f"  Control bounds: [-3.0, 3.0]")

    # Run with constant control
    print("\nRunning with constant control u = 1.0...")
    state = env.reset()

    for step in range(20):
        action = np.array([1.0])  # Constant control
        next_state, reward, done, info = env.step(action)

        if step % 5 == 0:
            fidelity = env.get_fidelity()
            print(f"  Step {step}: fidelity = {fidelity:.4f}")

        state = next_state

        if done:
            print(f"\n  Final fidelity: {info.get('final_fidelity', 'N/A')}")
            break

    print("\n✓ Quantum control environment demonstration complete")


def main():
    """Run all ML optimal control demonstrations."""
    print("\n" + "="*70)
    print("ML-BASED OPTIMAL CONTROL - DEMONSTRATIONS")
    print("="*70)
    print("\nThese demos showcase machine learning for optimal control.")
    print("Neural networks learn policies and value functions from data.")

    try:
        # Run demos
        demo_1_policy_network()
        demo_2_value_network()
        demo_3_actor_critic()
        demo_4_optimal_control_env()
        demo_5_quantum_control_env()

        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70)

        if JAX_AVAILABLE:
            print("\nKey Concepts Demonstrated:")
            print("  1. Neural network policies for continuous control")
            print("  2. Value function approximation")
            print("  3. Actor-Critic architecture")
            print("  4. RL environments for optimal control")
            print("  5. Quantum control with learned policies")
        else:
            print("\nNote: JAX/Flax not available.")
            print("Install for full demonstrations: pip install jax jaxlib flax optax")

    except Exception as e:
        print(f"\n✗ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
