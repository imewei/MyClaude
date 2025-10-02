"""Demonstrations of transfer learning and curriculum learning.

Shows practical applications for optimal control problems.

Author: Nonequilibrium Physics Agents
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.transfer_learning import (
    TransferLearningManager,
    TransferConfig,
    TransferStrategy,
    DomainAdaptation,
    create_transfer_learning_example,
    domain_adaptation_example
)

from ml_optimal_control.curriculum_learning import (
    CurriculumLearning,
    CurriculumConfig,
    CurriculumStrategy,
    DifficultyMetric,
    TaskGraph,
    ReverseCurriculum,
    create_lqr_curriculum,
    create_pendulum_curriculum,
    create_task_graph_example
)


def demo_1_transfer_learning_basics():
    """Demo 1: Transfer learning from LQR to nonlinear control."""
    print("\n" + "="*60)
    print("DEMO 1: Transfer Learning Basics")
    print("="*60)

    # Create transfer manager
    manager = TransferLearningManager(TransferConfig(
        strategy=TransferStrategy.FINE_TUNE.value,
        learning_rate=1e-4
    ))

    # Register source task: Simple LQR
    print("\n1. Registering source task (LQR 2D)...")
    manager.register_source_task(
        name='lqr_2d',
        model_params={'layer1': np.random.randn(2, 64), 'layer2': np.random.randn(64, 1)},
        task_config={
            'n_states': 2,
            'n_controls': 1,
            'problem_type': 'lqr',
            'dynamics_type': 'linear'
        },
        performance=0.92,
        metadata={'training_episodes': 1000, 'training_time': 120}
    )

    # Register another source task: Higher dimensional LQR
    print("2. Registering another source task (LQR 4D)...")
    manager.register_source_task(
        name='lqr_4d',
        model_params={'layer1': np.random.randn(4, 64), 'layer2': np.random.randn(64, 2)},
        task_config={
            'n_states': 4,
            'n_controls': 2,
            'problem_type': 'lqr',
            'dynamics_type': 'linear'
        },
        performance=0.87,
        metadata={'training_episodes': 2000, 'training_time': 300}
    )

    # Target task: Nonlinear pendulum
    print("\n3. Defining target task (Nonlinear Pendulum)...")
    target_config = {
        'n_states': 2,
        'n_controls': 1,
        'problem_type': 'pendulum',
        'dynamics_type': 'nonlinear'
    }

    # Select best source task
    print("4. Selecting best source task for transfer...")
    best_source = manager.select_source_task(target_config)
    print(f"   → Selected: {best_source}")
    print(f"   → Source performance: {manager.source_tasks[best_source].performance:.2%}")

    # Transfer knowledge
    print("\n5. Transferring knowledge to target task...")
    transferred_params = manager.transfer(
        source_name=best_source,
        target_model=None,  # Would be actual model
        strategy=TransferStrategy.FINE_TUNE.value
    )
    print("   → Transfer complete!")

    # Show statistics
    source_task = manager.source_tasks[best_source]
    print(f"\n6. Transfer summary:")
    print(f"   Source task: {best_source}")
    print(f"   Source training: {source_task.metadata.get('training_episodes', 'N/A')} episodes")
    print(f"   Source time: {source_task.metadata.get('training_time', 'N/A')} seconds")
    print(f"   Expected speedup: 3-10x (typical for transfer learning)")


def demo_2_domain_adaptation():
    """Demo 2: Domain adaptation between different dynamics."""
    print("\n" + "="*60)
    print("DEMO 2: Domain Adaptation")
    print("="*60)

    # Simulate features from source and target domains
    np.random.seed(42)

    print("\n1. Simulating source domain (Linear dynamics)...")
    n_samples = 200
    n_features = 16
    source_features = np.random.randn(n_samples, n_features)
    print(f"   Source: {n_samples} samples, {n_features} features")

    print("\n2. Simulating target domain (Nonlinear dynamics)...")
    # Target has shifted mean and different covariance
    target_features = np.random.randn(n_samples, n_features) * 1.2 + 0.3
    print(f"   Target: {n_samples} samples, {n_features} features")

    # Create domain adapter
    print("\n3. Computing domain discrepancy...")
    adapter = DomainAdaptation(
        source_domain='linear_lqr',
        target_domain='nonlinear_pendulum',
        adaptation_weight=0.1
    )

    # Compute MMD
    mmd_loss = adapter.compute_domain_loss(
        source_features,
        target_features,
        method='mmd'
    )
    print(f"   MMD loss: {mmd_loss:.4f}")
    print(f"   → {'High' if mmd_loss > 0.5 else 'Low'} domain discrepancy")

    # Compute CORAL
    coral_loss = adapter.compute_domain_loss(
        source_features,
        target_features,
        method='coral'
    )
    print(f"   CORAL loss: {coral_loss:.4f}")
    print(f"   → Covariance difference: {'Significant' if coral_loss > 0.1 else 'Moderate'}")

    print("\n4. Domain adaptation strategy:")
    print(f"   → Use MMD/CORAL loss to align domains during training")
    print(f"   → Total loss = task_loss + {adapter.adaptation_weight} * domain_loss")


def demo_3_curriculum_learning_lqr():
    """Demo 3: Curriculum learning for LQR with increasing time horizon."""
    print("\n" + "="*60)
    print("DEMO 3: Curriculum Learning - LQR")
    print("="*60)

    # Create curriculum
    print("\n1. Creating adaptive curriculum...")
    curriculum = create_lqr_curriculum()

    print(f"\n2. Generated {len(curriculum.tasks)} tasks:")
    for i, task in enumerate(curriculum.tasks):
        horizon = task.config['time_horizon']
        print(f"   Task {i}: Difficulty={task.difficulty:.2f}, "
              f"Horizon=[{horizon[0]:.1f}, {horizon[1]:.1f}]")

    # Simulate training
    print("\n3. Simulating training with adaptive curriculum...")
    episode = 0
    max_episodes = 100

    while episode < max_episodes and curriculum.current_task_idx < len(curriculum.tasks):
        current_task = curriculum.get_current_task()

        # Simulate performance (improves with practice on each task)
        base_performance = 0.5
        task_practice = curriculum.stage_episodes
        performance = min(base_performance + task_practice * 0.05, 0.95)

        # Add noise
        performance += np.random.randn() * 0.1
        performance = np.clip(performance, 0, 1)

        # Update curriculum
        advanced = curriculum.update(performance)

        if advanced:
            stats = curriculum.get_statistics()
            print(f"   Episode {episode}: Advanced to Task {stats['current_task_idx']} "
                  f"(difficulty={stats['current_difficulty']:.2f})")

        episode += 1

    # Final statistics
    print("\n4. Training complete!")
    stats = curriculum.get_statistics()
    print(f"   Final task: {stats['current_task_idx']}/{stats['total_tasks']-1}")
    print(f"   Completion: {stats['completion_progress']:.1%}")
    print(f"   Total episodes: {stats['total_episodes']}")


def demo_4_curriculum_pendulum():
    """Demo 4: Curriculum for pendulum with tightening constraints."""
    print("\n" + "="*60)
    print("DEMO 4: Curriculum Learning - Pendulum Swing-Up")
    print("="*60)

    # Create pendulum curriculum
    print("\n1. Creating curriculum with control constraint progression...")
    curriculum = create_pendulum_curriculum()

    print(f"\n2. Generated {len(curriculum.tasks)} tasks:")
    for i, task in enumerate(curriculum.tasks):
        bounds = task.config['control_bounds']
        print(f"   Task {i}: Difficulty={task.difficulty:.2f}, "
              f"Control bounds=[{bounds[0]:.2f}, {bounds[1]:.2f}]")

    print("\n3. Training strategy:")
    print("   → Start with relaxed control bounds (easy to swing up)")
    print("   → Gradually tighten bounds as agent learns")
    print("   → Final task has tight bounds (realistic constraints)")

    # Simulate a few updates
    print("\n4. Simulating curriculum progression...")
    for ep in range(15):
        # Simulate improving performance
        performance = 0.6 + ep * 0.02 + np.random.randn() * 0.05
        performance = np.clip(performance, 0, 1)

        advanced = curriculum.update(performance)

        if advanced:
            current_task = curriculum.get_current_task()
            print(f"   Episode {ep}: Advanced to difficulty {current_task.difficulty:.2f} "
                  f"(bounds: {current_task.config['control_bounds']})")


def demo_5_task_graph():
    """Demo 5: Task graph with prerequisites."""
    print("\n" + "="*60)
    print("DEMO 5: Task Graph with Prerequisites")
    print("="*60)

    # Create task graph
    print("\n1. Creating task graph with dependencies...")
    graph = create_task_graph_example()

    print("\n2. Task structure:")
    for task_id, task in graph.tasks.items():
        prereqs = graph.dependencies[task_id]
        prereq_str = ', '.join(prereqs) if prereqs else "None"
        print(f"   {task_id}: difficulty={task.difficulty:.1f}, "
              f"prerequisites=[{prereq_str}]")

    # Simulate completing tasks
    print("\n3. Simulating task progression...")

    step = 0
    while len(graph.completed_tasks) < len(graph.tasks):
        # Get available tasks
        available = graph.get_available_tasks()

        if not available:
            break

        # Select next task (easiest)
        next_task = graph.get_next_task(strategy='easiest')

        print(f"\n   Step {step}:")
        print(f"   Available tasks: {available}")
        print(f"   Selected: {next_task}")

        # Simulate completing task
        graph.complete_task(next_task)
        print(f"   Completed: {next_task}")

        step += 1

    print(f"\n4. All tasks completed!")
    print(f"   Total steps: {step}")
    print(f"   Completed tasks: {list(graph.completed_tasks)}")


def demo_6_reverse_curriculum():
    """Demo 6: Reverse curriculum learning."""
    print("\n" + "="*60)
    print("DEMO 6: Reverse Curriculum Learning")
    print("="*60)

    # Define goal and initial states
    print("\n1. Defining goal and initial configurations...")

    goal_config = {
        'angle': np.pi,  # Upright position
        'angular_velocity': 0.0,
        'time_horizon': 5.0
    }
    print(f"   Goal: angle={goal_config['angle']:.2f} rad (upright)")

    initial_config = {
        'angle': 0.0,  # Downward position
        'angular_velocity': 0.0,
        'time_horizon': 5.0
    }
    print(f"   Initial: angle={initial_config['angle']:.2f} rad (downward)")

    # Create reverse curriculum
    print("\n2. Creating reverse curriculum...")
    reverse_curriculum = ReverseCurriculum(
        goal_config,
        initial_config,
        num_stages=5
    )

    print(f"\n3. Generated {len(reverse_curriculum.tasks)} tasks:")
    print("   (Progresses from near-goal to near-initial)")
    for i, task in enumerate(reverse_curriculum.tasks):
        angle = task.config['angle']
        print(f"   Stage {i}: difficulty={task.difficulty:.2f}, "
              f"start_angle={angle:.2f} rad ({np.degrees(angle):.1f}°)")

    print("\n4. Reverse curriculum strategy:")
    print("   → Start with easy task (already near upright)")
    print("   → Gradually start from lower angles")
    print("   → End with full swing-up from bottom")
    print("   → Makes learning easier by working backwards from goal!")


def demo_7_combined_transfer_curriculum():
    """Demo 7: Combining transfer learning with curriculum."""
    print("\n" + "="*60)
    print("DEMO 7: Transfer Learning + Curriculum Learning")
    print("="*60)

    print("\n1. Scenario: Learning cart-pole control")
    print("   → Source task: Trained on simple LQR (2D linear system)")
    print("   → Target task: Cart-pole balancing (4D nonlinear system)")

    # Create transfer manager
    print("\n2. Setting up transfer learning...")
    manager = TransferLearningManager()

    manager.register_source_task(
        name='lqr_2d',
        model_params={'weights': 'pretrained'},
        task_config={'n_states': 2, 'problem_type': 'lqr'},
        performance=0.90
    )

    # Create curriculum for cart-pole
    print("\n3. Creating curriculum for cart-pole...")
    curriculum = CurriculumLearning(CurriculumConfig(
        strategy=CurriculumStrategy.ADAPTIVE.value,
        performance_threshold=0.75
    ))

    # Generate curriculum stages
    base_config = {
        'n_states': 4,
        'n_controls': 1,
        'problem_type': 'cartpole',
        'time_horizon': [0, 5],
        'pole_length': 1.0
    }

    curriculum.generate_curriculum(
        base_config,
        difficulty_metric=DifficultyMetric.TIME_HORIZON.value,
        num_tasks=4
    )

    print(f"   Generated {len(curriculum.tasks)} curriculum stages")

    # Workflow
    print("\n4. Combined training workflow:")
    print("   a) Transfer from LQR to cart-pole (warm start)")
    print("   b) Fine-tune on easiest curriculum stage")
    print("   c) Progressively advance through curriculum")
    print("   d) Final model handles full cart-pole task")

    print("\n5. Expected benefits:")
    print("   → Transfer: 3-10x faster initial learning")
    print("   → Curriculum: 2-5x better final performance")
    print("   → Combined: 6-50x improvement over from-scratch training!")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*10 + "TRANSFER & CURRICULUM LEARNING DEMONSTRATIONS")
    print("="*70)

    # Run demos
    demo_1_transfer_learning_basics()
    demo_2_domain_adaptation()
    demo_3_curriculum_learning_lqr()
    demo_4_curriculum_pendulum()
    demo_5_task_graph()
    demo_6_reverse_curriculum()
    demo_7_combined_transfer_curriculum()

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
