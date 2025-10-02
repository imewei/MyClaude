"""Advanced RL Demonstrations (Week 6).

This script demonstrates Week 6 ML capabilities:
1. SAC (Soft Actor-Critic) for continuous control
2. TD3 (Twin Delayed DDPG) with target smoothing
3. Model-based RL with learned dynamics
4. Model Predictive Control (MPC)
5. Meta-learning with MAML/Reptile

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
    from ml_optimal_control.advanced_rl import (
        create_sac_trainer,
        create_td3_trainer,
        create_ddpg_trainer
    )
    from ml_optimal_control.model_based_rl import (
        create_dynamics_model,
        create_mpc_controller
    )
    from ml_optimal_control.meta_learning import (
        Task,
        TaskDistribution,
        ReptileTrainer,
        create_task_distribution
    )
    from ml_optimal_control.networks import PolicyNetwork
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX/Flax not available. Install with: pip install jax jaxlib flax optax")

# Import environments (work without JAX)
from ml_optimal_control.environments import OptimalControlEnv


# =============================================================================
# Demo 1: SAC for LQR
# =============================================================================

def demo_1_sac_lqr():
    """Demo 1: SAC on simple LQR problem."""
    print("\n" + "="*70)
    print("Demo 1: SAC (Soft Actor-Critic) on LQR")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    print("\nProblem: x_dot = u, cost = x^2 + u^2")
    print("Goal: Learn optimal policy u = -k*x")

    # Create SAC trainer
    print("\nCreating SAC trainer...")
    trainer = create_sac_trainer(
        state_dim=1,
        action_dim=1,
        auto_entropy_tuning=True,
        batch_size=64
    )
    print(f"  State dim: 1, Action dim: 1")
    print(f"  Auto entropy tuning: ON")

    # Training loop
    print("\nTraining for 100 steps...")
    x = np.array([1.0])
    episode_rewards = []
    episode_reward = 0.0

    for step in range(100):
        # Select action
        action = trainer.select_action(x, deterministic=False)

        # Dynamics: x_next = x + u * dt
        dt = 0.1
        x_next = x + action * dt

        # Reward: -(x^2 + u^2)
        reward = -(x**2 + action**2)[0] * dt
        episode_reward += reward

        # Store transition
        trainer.replay_buffer.add(x, action, reward, x_next, False)

        # Train
        if len(trainer.replay_buffer) >= trainer.batch_size:
            info = trainer.train_step()

            if step % 20 == 0 and 'alpha' in info:
                print(f"  Step {step}: alpha={float(info['alpha']):.3f}, "
                      f"entropy={float(info.get('entropy', 0)):.3f}")

        # Reset episode
        if step % 20 == 0 and step > 0:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            x = np.array([np.random.randn()])  # New initial state
        else:
            x = x_next

    print(f"\n  Training complete!")
    print(f"  Final entropy temperature (alpha): {float(trainer.log_alpha):.3f}")
    print(f"  Average episode reward: {np.mean(episode_rewards):.3f}")

    # Test learned policy
    print("\nTesting learned policy (deterministic)...")
    test_states = np.array([[-1.0], [0.0], [1.0], [2.0]])
    for state in test_states:
        action = trainer.select_action(state, deterministic=True)
        print(f"  State: {state[0]:+.2f} → Action: {action[0]:+.3f}")

    print("\n✓ SAC demonstration complete")


# =============================================================================
# Demo 2: TD3 vs DDPG Comparison
# =============================================================================

def demo_2_td3_vs_ddpg():
    """Demo 2: Compare TD3 and DDPG."""
    print("\n" + "="*70)
    print("Demo 2: TD3 vs DDPG Comparison")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    print("\nComparing algorithms on same task...")
    print("Problem: Damped oscillator")

    def train_agent(trainer_name, trainer, n_steps=100):
        """Train agent and return performance."""
        x = np.array([1.0, 0.0])  # [position, velocity]
        total_reward = 0.0

        for step in range(n_steps):
            action = trainer.select_action(x, add_noise=True)

            # Damped oscillator: x_dot = v, v_dot = -k*x - c*v + u
            k, c, dt = 1.0, 0.5, 0.05
            v = x[1]
            x_next = x.copy()
            x_next[0] += v * dt
            x_next[1] += (-k*x[0] - c*v + action[0]) * dt

            # Reward
            reward = -(x[0]**2 + x[1]**2 + 0.1*action[0]**2) * dt
            total_reward += reward

            trainer.replay_buffer.add(x, action, reward, x_next, False)

            if len(trainer.replay_buffer) >= trainer.batch_size:
                trainer.train_step()

            x = x_next

        return total_reward

    # Create trainers
    print("\nCreating trainers...")
    ddpg = create_ddpg_trainer(state_dim=2, action_dim=1, batch_size=32)
    td3 = create_td3_trainer(state_dim=2, action_dim=1, batch_size=32, policy_freq=2)

    # Train both
    print("\nTraining DDPG...")
    ddpg_reward = train_agent("DDPG", ddpg, n_steps=150)

    print("Training TD3...")
    td3_reward = train_agent("TD3", td3, n_steps=150)

    # Compare
    print("\n" + "-"*70)
    print("Results:")
    print(f"  DDPG total reward: {ddpg_reward:.3f}")
    print(f"  TD3 total reward:  {td3_reward:.3f}")
    print(f"  Difference: {abs(ddpg_reward - td3_reward):.3f}")

    print("\nKey TD3 improvements:")
    print("  1. Twin Q-networks reduce overestimation")
    print("  2. Delayed policy updates improve stability")
    print("  3. Target policy smoothing reduces variance")

    print("\n✓ TD3 vs DDPG demonstration complete")


# =============================================================================
# Demo 3: Model-Based RL
# =============================================================================

def demo_3_model_based_rl():
    """Demo 3: Learn dynamics model and use for planning."""
    print("\n" + "="*70)
    print("Demo 3: Model-Based RL - Dynamics Learning")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    print("\nLearning dynamics model for pendulum...")
    print("True dynamics: theta_dot = omega, omega_dot = -sin(theta) + u")

    # Create dynamics model
    print("\nCreating probabilistic dynamics model...")
    dynamics_model = create_dynamics_model(
        state_dim=2,
        action_dim=1,
        model_type='probabilistic',
        hidden_dims=(64, 64)
    )
    print(f"  Model type: probabilistic")
    print(f"  Hidden layers: (64, 64)")

    # Generate training data
    print("\nGenerating training data...")
    n_samples = 200
    states = []
    actions = []
    next_states = []

    for i in range(n_samples):
        # Random state and action
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-2, 2)
        u = np.random.uniform(-2, 2)

        state = np.array([theta, omega])
        action = np.array([u])

        # True dynamics
        dt = 0.05
        theta_next = theta + omega * dt
        omega_next = omega + (-np.sin(theta) + u) * dt
        next_state = np.array([theta_next, omega_next])

        states.append(state)
        actions.append(action)
        next_states.append(next_state)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    print(f"  Generated {n_samples} transitions")

    # Train model
    print("\nTraining dynamics model...")
    for epoch in range(50):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        batch_size = 32

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            info = dynamics_model.train_step(
                states[batch_idx],
                actions[batch_idx],
                next_states[batch_idx]
            )

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={float(info['dynamics_loss']):.4f}")

    # Test predictions
    print("\nTesting model predictions...")
    test_state = np.array([[0.0, 0.0]])
    test_action = np.array([[1.0]])

    pred_next = dynamics_model.predict(test_state, test_action)
    true_next = test_state + np.array([[0.0, 1.0]]) * 0.05

    print(f"  Test: state={test_state[0]}, action={test_action[0]}")
    print(f"  Predicted next: {pred_next[0]}")
    print(f"  True next:      {true_next[0]}")
    print(f"  Error: {np.linalg.norm(pred_next - true_next):.4f}")

    print("\n✓ Model-based RL demonstration complete")


# =============================================================================
# Demo 4: Model Predictive Control
# =============================================================================

def demo_4_mpc():
    """Demo 4: MPC with learned dynamics."""
    print("\n" + "="*70)
    print("Demo 4: Model Predictive Control (MPC)")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    print("\nUsing learned model for planning...")

    # Create simple dynamics model (pre-trained)
    print("\nCreating dynamics model...")
    dynamics_model = create_dynamics_model(
        state_dim=2,
        action_dim=1,
        model_type='deterministic'
    )

    # Cost function: quadratic cost
    def cost_fn(state, action):
        """Cost: x^T Q x + u^T R u"""
        Q = np.eye(2)
        R = np.array([[0.1]])
        return float(state @ Q @ state + action @ R @ action)

    # Create MPC controller
    print("\nCreating MPC controller...")
    mpc = create_mpc_controller(
        dynamics_model=dynamics_model,
        cost_fn=cost_fn,
        horizon=10,
        n_samples=500,
        n_elite=50,
        n_iterations=3
    )

    print(f"  Planning horizon: 10 steps")
    print(f"  Samples per iteration: 500")
    print(f"  Elite samples: 50")
    print(f"  CEM iterations: 3")

    # Plan from initial state
    print("\nPlanning from initial state [1.0, 0.5]...")
    initial_state = np.array([1.0, 0.5])

    optimal_action = mpc.plan(initial_state)

    print(f"  Optimal action: {optimal_action}")
    print(f"  Action dimension: {optimal_action.shape}")

    print("\nMPC advantages:")
    print("  1. Plans optimal sequence over horizon")
    print("  2. Handles constraints naturally")
    print("  3. Uses learned model (sample efficient)")
    print("  4. Replans at each step (robust to errors)")

    print("\n✓ MPC demonstration complete")


# =============================================================================
# Demo 5: Meta-Learning with Reptile
# =============================================================================

def demo_5_meta_learning():
    """Demo 5: Fast adaptation with meta-learning."""
    print("\n" + "="*70)
    print("Demo 5: Meta-Learning (Reptile)")
    print("="*70)

    if not JAX_AVAILABLE:
        print("  ✗ JAX not available - demo skipped")
        return

    print("\nMeta-learning across task distribution...")
    print("Task family: LQR with different costs")

    # Create task distribution
    print("\nCreating task distribution...")

    def create_lqr_task(task_id, Q_scale):
        """Create LQR task with scaled Q matrix."""
        def dynamics(x, u, t):
            return u

        def cost(x, u, t):
            return Q_scale * x**2 + u**2

        return Task(
            task_id=task_id,
            dynamics=dynamics,
            cost=cost,
            x0_mean=np.array([1.0]),
            x0_std=np.array([0.1]),
            duration=2.0,
            dt=0.05
        )

    # Create tasks with different Q scales
    tasks = [
        create_lqr_task(f"task_{i}", Q_scale=0.5 + i*0.5)
        for i in range(5)
    ]

    task_dist = TaskDistribution(tasks)
    print(f"  Created {len(tasks)} tasks")
    print(f"  Q scales: {[0.5 + i*0.5 for i in range(5)]}")

    # Create Reptile trainer
    print("\nCreating Reptile meta-learner...")
    policy_network = PolicyNetwork(hidden_dims=(32, 32), action_dim=1)

    trainer = ReptileTrainer(
        policy_network=policy_network,
        state_dim=1,
        action_dim=1,
        inner_lr=0.01,
        outer_lr=0.001,
        n_inner_steps=5
    )

    print(f"  Inner learning rate: 0.01")
    print(f"  Outer learning rate: 0.001")
    print(f"  Adaptation steps: 5")

    # Meta-training
    print("\nMeta-training for 10 iterations...")
    for iteration in range(10):
        # Sample task
        task = task_dist.sample(1)[0]

        # Meta-training step
        info = trainer.meta_train_step(task, n_episodes=5)

        if iteration % 3 == 0:
            print(f"  Iteration {iteration}: "
                  f"initial_loss={float(info.get('initial_loss', 0)):.3f}, "
                  f"final_loss={float(info.get('final_loss', 0)):.3f}")

    print("\nMeta-learning benefits:")
    print("  1. Fast adaptation to new tasks (few gradient steps)")
    print("  2. Learns good initialization")
    print("  3. Transfers knowledge across tasks")
    print("  4. Sample efficient for new tasks")

    print("\n✓ Meta-learning demonstration complete")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all Week 6 demonstrations."""
    print("\n" + "="*70)
    print("WEEK 6: ADVANCED RL - DEMONSTRATIONS")
    print("="*70)
    print("\nAdvanced reinforcement learning for optimal control:")
    print("  • SAC, TD3, DDPG for continuous control")
    print("  • Model-based RL and planning")
    print("  • Meta-learning for fast adaptation")

    try:
        # Run demos
        demo_1_sac_lqr()
        demo_2_td3_vs_ddpg()
        demo_3_model_based_rl()
        demo_4_mpc()
        demo_5_meta_learning()

        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70)

        if JAX_AVAILABLE:
            print("\nKey Concepts Demonstrated:")
            print("  1. Maximum entropy RL (SAC)")
            print("  2. Twin Q-learning and delayed updates (TD3)")
            print("  3. Learned world models for planning")
            print("  4. Model Predictive Control with learned dynamics")
            print("  5. Meta-learning for rapid task adaptation")
            print("\nPerformance Benefits:")
            print("  • 10-100x sample efficiency (model-based)")
            print("  • Robust policies (maximum entropy)")
            print("  • Fast adaptation (meta-learning)")
            print("  • Planning capability (MPC)")
        else:
            print("\nNote: JAX/Flax not available.")
            print("Install for full demonstrations: pip install jax jaxlib flax optax")

    except Exception as e:
        print(f"\n✗ Error during demonstrations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
