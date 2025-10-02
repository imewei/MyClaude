"""Tests for transfer learning and curriculum learning.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.transfer_learning import (
    TransferLearningManager,
    TransferConfig,
    TransferStrategy,
    DomainAdaptation,
    MultiTaskTransfer,
    SourceTask
)

from ml_optimal_control.curriculum_learning import (
    CurriculumLearning,
    CurriculumConfig,
    CurriculumStrategy,
    DifficultyMetric,
    TaskGraph,
    ReverseCurriculum,
    Task
)


class TestTransferLearning:
    """Tests for transfer learning framework."""

    def test_transfer_manager_initialization(self):
        """Test: TransferLearningManager initializes correctly."""
        manager = TransferLearningManager()
        assert len(manager.source_tasks) == 0
        assert len(manager.transfer_history) == 0

    def test_register_source_task(self):
        """Test: Can register source tasks."""
        manager = TransferLearningManager()

        manager.register_source_task(
            name='lqr_2d',
            model_params={'dummy': 'params'},
            task_config={'n_states': 2, 'n_controls': 1},
            performance=0.95
        )

        assert 'lqr_2d' in manager.source_tasks
        assert manager.source_tasks['lqr_2d'].performance == 0.95

    def test_similarity_metric(self):
        """Test: Similarity metric computes reasonable scores."""
        manager = TransferLearningManager()

        # Register two source tasks
        manager.register_source_task(
            name='lqr_2d',
            model_params={},
            task_config={
                'n_states': 2,
                'n_controls': 1,
                'problem_type': 'lqr'
            },
            performance=0.9
        )

        manager.register_source_task(
            name='lqr_4d',
            model_params={},
            task_config={
                'n_states': 4,
                'n_controls': 2,
                'problem_type': 'lqr'
            },
            performance=0.85
        )

        # Target similar to 2d
        target_config = {
            'n_states': 2,
            'n_controls': 1,
            'problem_type': 'lqr'
        }

        best_source = manager.select_source_task(target_config)
        assert best_source == 'lqr_2d', "Should select most similar source"

    def test_transfer_strategies(self):
        """Test: Different transfer strategies work."""
        manager = TransferLearningManager()

        manager.register_source_task(
            name='source',
            model_params={'layer1': np.array([1, 2, 3])},
            task_config={'n_states': 2},
            performance=0.9
        )

        strategies = [
            TransferStrategy.FINE_TUNE.value,
            TransferStrategy.FEATURE_EXTRACTION.value,
            TransferStrategy.PROGRESSIVE.value,
            TransferStrategy.SELECTIVE.value
        ]

        for strategy in strategies:
            params = manager.transfer(
                source_name='source',
                target_model=None,
                strategy=strategy
            )
            assert params is not None, f"Strategy {strategy} should return params"

    def test_save_load_source_task(self, tmp_path):
        """Test: Can save and load source tasks."""
        manager = TransferLearningManager()

        manager.register_source_task(
            name='test_task',
            model_params={'weights': np.array([1, 2, 3])},
            task_config={'n_states': 2},
            performance=0.88
        )

        # Save
        filepath = tmp_path / "source_task.pkl"
        manager.save_source_task('test_task', filepath)

        # Load into new manager
        manager2 = TransferLearningManager()
        loaded_name = manager2.load_source_task(filepath)

        assert loaded_name == 'test_task'
        assert 'test_task' in manager2.source_tasks
        assert manager2.source_tasks['test_task'].performance == 0.88


class TestDomainAdaptation:
    """Tests for domain adaptation."""

    def test_mmd_computation(self):
        """Test: MMD computes reasonable values."""
        adapter = DomainAdaptation(
            source_domain='source',
            target_domain='target',
            adaptation_weight=0.1
        )

        # Identical distributions
        np.random.seed(42)
        source_features = np.random.randn(50, 10)
        target_features = np.random.randn(50, 10)

        mmd = adapter.compute_domain_loss(
            source_features,
            target_features,
            method='mmd'
        )

        assert mmd >= 0, "MMD should be non-negative"
        assert mmd < 10, "MMD should be reasonable for similar distributions"

    def test_mmd_detects_shift(self):
        """Test: MMD detects distribution shift."""
        adapter = DomainAdaptation('source', 'target')

        np.random.seed(42)
        source_features = np.random.randn(100, 10)
        target_features = np.random.randn(100, 10) + 2.0  # Shifted

        mmd = adapter.compute_domain_loss(
            source_features,
            target_features,
            method='mmd'
        )

        # Should detect shift
        assert mmd > 1.0, "MMD should be large for shifted distributions"

    def test_coral_computation(self):
        """Test: CORAL computes covariance alignment loss."""
        adapter = DomainAdaptation('source', 'target')

        np.random.seed(42)
        source_features = np.random.randn(100, 10)
        target_features = np.random.randn(100, 10)

        coral = adapter.compute_domain_loss(
            source_features,
            target_features,
            method='coral'
        )

        assert coral >= 0, "CORAL should be non-negative"
        assert np.isfinite(coral), "CORAL should be finite"

    def test_coral_detects_covariance_difference(self):
        """Test: CORAL detects covariance differences."""
        adapter = DomainAdaptation('source', 'target')

        np.random.seed(42)
        source_features = np.random.randn(100, 10)

        # Target with different covariance
        target_features = np.random.randn(100, 10) * 2.0

        coral = adapter.compute_domain_loss(
            source_features,
            target_features,
            method='coral'
        )

        # Should detect covariance difference
        assert coral > 0.1, "CORAL should detect covariance differences"


class TestMultiTaskTransfer:
    """Tests for multi-task transfer learning."""

    def test_multi_task_initialization(self):
        """Test: MultiTaskTransfer initializes correctly."""
        mt = MultiTaskTransfer(
            shared_layer_sizes=[64, 32],
            task_specific_sizes=[16, 8]
        )

        assert mt.shared_layer_sizes == [64, 32]
        assert mt.task_specific_sizes == [16, 8]
        assert len(mt.task_models) == 0

    def test_add_task(self):
        """Test: Can add tasks to multi-task learning."""
        mt = MultiTaskTransfer([64, 32], [16, 8])

        mt.add_task('task1', {'model': 'placeholder'})
        mt.add_task('task2', {'model': 'placeholder'})

        assert len(mt.task_models) == 2
        assert 'task1' in mt.task_models
        assert 'task2' in mt.task_models

    def test_shared_representation(self):
        """Test: Shared representation computation."""
        mt = MultiTaskTransfer([64, 32], [16])

        state = np.random.randn(10)
        shared_params = None  # Placeholder

        features = mt.compute_shared_representation(state, shared_params)

        assert features is not None
        assert features.shape[-1] == 32, "Should match last shared layer size"


class TestCurriculumLearning:
    """Tests for curriculum learning."""

    def test_curriculum_initialization(self):
        """Test: CurriculumLearning initializes correctly."""
        curriculum = CurriculumLearning()

        assert len(curriculum.tasks) == 0
        assert curriculum.current_task_idx == 0
        assert len(curriculum.performance_history) == 0

    def test_add_task(self):
        """Test: Can add tasks to curriculum."""
        curriculum = CurriculumLearning()

        curriculum.add_task(
            task_id='easy',
            difficulty=0.2,
            config={'time_horizon': [0, 1]}
        )

        curriculum.add_task(
            task_id='hard',
            difficulty=0.8,
            config={'time_horizon': [0, 5]}
        )

        assert len(curriculum.tasks) == 2
        # Should be sorted by difficulty
        assert curriculum.tasks[0].difficulty < curriculum.tasks[1].difficulty

    def test_automatic_curriculum_generation(self):
        """Test: Automatic curriculum generation."""
        curriculum = CurriculumLearning()

        base_config = {
            'n_states': 2,
            'n_controls': 1,
            'time_horizon': [0.0, 1.0]
        }

        curriculum.generate_curriculum(
            base_config,
            difficulty_metric=DifficultyMetric.TIME_HORIZON.value,
            num_tasks=5
        )

        assert len(curriculum.tasks) == 5
        # Difficulties should be increasing
        for i in range(len(curriculum.tasks) - 1):
            assert curriculum.tasks[i].difficulty <= curriculum.tasks[i+1].difficulty

    def test_fixed_curriculum_update(self):
        """Test: Fixed curriculum advances on schedule."""
        config = CurriculumConfig(
            strategy=CurriculumStrategy.FIXED.value,
            stage_duration=10
        )

        curriculum = CurriculumLearning(config)
        curriculum.add_task('task1', 0.2, {})
        curriculum.add_task('task2', 0.5, {})

        # Simulate episodes
        for i in range(9):
            advanced = curriculum.update(performance=0.5)
            assert not advanced, "Should not advance before stage_duration"

        advanced = curriculum.update(performance=0.5)
        assert advanced, "Should advance after stage_duration"
        assert curriculum.current_task_idx == 1

    def test_adaptive_curriculum_update(self):
        """Test: Adaptive curriculum advances on performance."""
        config = CurriculumConfig(
            strategy=CurriculumStrategy.ADAPTIVE.value,
            performance_threshold=0.7,
            patience=5
        )

        curriculum = CurriculumLearning(config)
        curriculum.add_task('task1', 0.2, {})
        curriculum.add_task('task2', 0.5, {})

        # Poor performance - should not advance
        for i in range(3):
            advanced = curriculum.update(performance=0.3)
            assert not advanced

        # Good performance - should advance
        for i in range(5):
            advanced = curriculum.update(performance=0.8)

        # Should have advanced by now
        assert curriculum.current_task_idx == 1

    def test_self_paced_curriculum(self):
        """Test: Self-paced curriculum adjusts difficulty."""
        config = CurriculumConfig(
            strategy=CurriculumStrategy.SELF_PACED.value,
            self_paced_window=10,
            difficulty_increment=0.2
        )

        curriculum = CurriculumLearning(config)

        # Create tasks at different difficulties
        for i, diff in enumerate([0.2, 0.4, 0.6, 0.8]):
            curriculum.add_task(f'task{i}', diff, {})

        # Start at easiest
        assert curriculum.current_task_idx == 0

        # High performance should increase difficulty
        for i in range(15):
            curriculum.update(performance=0.9)

        # Should have moved to harder task
        assert curriculum.current_task_idx > 0

    def test_get_statistics(self):
        """Test: Curriculum statistics."""
        curriculum = CurriculumLearning()

        curriculum.add_task('easy', 0.2, {})
        curriculum.add_task('medium', 0.5, {})
        curriculum.add_task('hard', 0.8, {})

        curriculum.update(performance=0.7)
        curriculum.update(performance=0.8)

        stats = curriculum.get_statistics()

        assert 'current_task_idx' in stats
        assert 'current_difficulty' in stats
        assert 'total_tasks' in stats
        assert stats['total_tasks'] == 3
        assert stats['total_episodes'] == 2


class TestTaskGraph:
    """Tests for task graph with prerequisites."""

    def test_task_graph_initialization(self):
        """Test: TaskGraph initializes correctly."""
        graph = TaskGraph()

        assert len(graph.tasks) == 0
        assert len(graph.dependencies) == 0
        assert len(graph.completed_tasks) == 0

    def test_add_task_with_prerequisites(self):
        """Test: Can add tasks with prerequisites."""
        graph = TaskGraph()

        graph.add_task('task_a', 0.2, {}, prerequisites=[])
        graph.add_task('task_b', 0.5, {}, prerequisites=['task_a'])

        assert len(graph.tasks) == 2
        assert graph.dependencies['task_b'] == ['task_a']

    def test_available_tasks(self):
        """Test: Correctly identifies available tasks."""
        graph = TaskGraph()

        graph.add_task('task_a', 0.2, {}, prerequisites=[])
        graph.add_task('task_b', 0.5, {}, prerequisites=['task_a'])
        graph.add_task('task_c', 0.7, {}, prerequisites=['task_a', 'task_b'])

        # Initially, only task_a is available
        available = graph.get_available_tasks()
        assert available == ['task_a']

        # Complete task_a
        graph.complete_task('task_a')
        available = graph.get_available_tasks()
        assert 'task_b' in available
        assert 'task_c' not in available  # Still needs task_b

        # Complete task_b
        graph.complete_task('task_b')
        available = graph.get_available_tasks()
        assert 'task_c' in available

    def test_get_next_task_easiest(self):
        """Test: get_next_task selects easiest available."""
        graph = TaskGraph()

        graph.add_task('easy', 0.2, {}, prerequisites=[])
        graph.add_task('medium', 0.5, {}, prerequisites=[])
        graph.add_task('hard', 0.8, {}, prerequisites=[])

        next_task = graph.get_next_task(strategy='easiest')
        assert next_task == 'easy'

    def test_get_next_task_hardest(self):
        """Test: get_next_task selects hardest available."""
        graph = TaskGraph()

        graph.add_task('easy', 0.2, {}, prerequisites=[])
        graph.add_task('medium', 0.5, {}, prerequisites=[])
        graph.add_task('hard', 0.8, {}, prerequisites=[])

        next_task = graph.get_next_task(strategy='hardest')
        assert next_task == 'hard'


class TestReverseCurriculum:
    """Tests for reverse curriculum learning."""

    def test_reverse_curriculum_initialization(self):
        """Test: ReverseCurriculum initializes correctly."""
        goal_config = {'position': 1.0, 'velocity': 0.0}
        initial_config = {'position': 0.0, 'velocity': 0.0}

        reverse_curriculum = ReverseCurriculum(
            goal_config,
            initial_config,
            num_stages=5
        )

        assert len(reverse_curriculum.tasks) == 5

    def test_reverse_curriculum_difficulty_order(self):
        """Test: Reverse curriculum has correct difficulty order."""
        goal_config = {'position': 1.0}
        initial_config = {'position': 0.0}

        reverse_curriculum = ReverseCurriculum(
            goal_config,
            initial_config,
            num_stages=5
        )

        # First task should be hardest (closest to goal)
        # Last task should be easiest (closest to initial)
        assert reverse_curriculum.tasks[0].difficulty > reverse_curriculum.tasks[-1].difficulty

    def test_reverse_curriculum_interpolation(self):
        """Test: Reverse curriculum interpolates between goal and initial."""
        goal_config = {'position': 10.0}
        initial_config = {'position': 0.0}

        reverse_curriculum = ReverseCurriculum(
            goal_config,
            initial_config,
            num_stages=3
        )

        # Check interpolation
        positions = [task.config['position'] for task in reverse_curriculum.tasks]

        # Should go from close to goal to close to initial
        assert positions[0] > positions[1] > positions[2]


class TestCurriculumIntegration:
    """Integration tests for curriculum learning."""

    def test_full_curriculum_workflow(self):
        """Test: Complete curriculum workflow."""
        # Create curriculum
        curriculum = CurriculumLearning(CurriculumConfig(
            strategy=CurriculumStrategy.ADAPTIVE.value,
            performance_threshold=0.75
        ))

        # Generate tasks
        base_config = {'time_horizon': [0, 1]}
        curriculum.generate_curriculum(
            base_config,
            difficulty_metric=DifficultyMetric.TIME_HORIZON.value,
            num_tasks=3
        )

        # Simulate training
        for episode in range(50):
            current_task = curriculum.get_current_task()

            # Simulate performance (improves over time)
            performance = min(0.5 + episode * 0.01, 0.9)

            advanced = curriculum.update(performance)

            if advanced:
                print(f"Episode {episode}: Advanced to difficulty {current_task.difficulty:.2f}")

        # Should have progressed through curriculum
        stats = curriculum.get_statistics()
        assert stats['current_task_idx'] > 0, "Should have advanced through tasks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
