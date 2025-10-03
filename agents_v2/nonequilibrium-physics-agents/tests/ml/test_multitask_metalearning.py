"""Tests for Enhanced Multi-Task and Meta-Learning.

Author: Nonequilibrium Physics Agents
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_optimal_control.multitask_metalearning import (
    MultiTaskLearning,
    MultiTaskConfig,
    MultiTaskArchitecture,
    EnhancedMAML,
    Reptile,
    ANIL,
    TaskConditionalMetaLearning,
    TaskEmbedding,
    AdaptiveInnerSteps,
    MetaLearningConfig,
    MetaLearningAlgorithm
)

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestMultiTaskConfig:
    """Tests for multi-task configuration."""

    def test_default_config(self):
        """Test: Default multi-task configuration."""
        config = MultiTaskConfig()

        assert config.architecture == MultiTaskArchitecture.HARD_SHARING.value
        assert len(config.shared_layers) > 0
        assert len(config.task_specific_layers) > 0
        assert config.learning_rate > 0

    def test_custom_config(self):
        """Test: Custom multi-task configuration."""
        config = MultiTaskConfig(
            architecture=MultiTaskArchitecture.SOFT_SHARING.value,
            shared_layers=[256, 256],
            num_clusters=3
        )

        assert config.architecture == MultiTaskArchitecture.SOFT_SHARING.value
        assert config.shared_layers == [256, 256]
        assert config.num_clusters == 3


class TestMetaLearningConfig:
    """Tests for meta-learning configuration."""

    def test_default_config(self):
        """Test: Default meta-learning configuration."""
        config = MetaLearningConfig()

        assert config.algorithm == MetaLearningAlgorithm.MAML.value
        assert config.meta_learning_rate > 0
        assert config.inner_learning_rate > 0
        assert config.num_inner_steps > 0

    def test_custom_config(self):
        """Test: Custom meta-learning configuration."""
        config = MetaLearningConfig(
            algorithm=MetaLearningAlgorithm.REPTILE.value,
            num_inner_steps=10,
            shots_per_task=20
        )

        assert config.algorithm == MetaLearningAlgorithm.REPTILE.value
        assert config.num_inner_steps == 10
        assert config.shots_per_task == 20


class TestMultiTaskLearning:
    """Tests for multi-task learning."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_register_task(self):
        """Test: Register tasks."""
        config = MultiTaskConfig()
        mtl = MultiTaskLearning(config)

        mtl.register_task("task1", input_dim=4, output_dim=2)
        mtl.register_task("task2", input_dim=4, output_dim=3)

        assert len(mtl.tasks) == 2
        assert "task1" in mtl.tasks
        assert "task2" in mtl.tasks
        assert mtl.tasks["task1"]["output_dim"] == 2
        assert mtl.tasks["task2"]["output_dim"] == 3

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_hard_sharing_model(self):
        """Test: Create hard sharing multi-task model."""
        config = MultiTaskConfig(architecture=MultiTaskArchitecture.HARD_SHARING.value)
        mtl = MultiTaskLearning(config)

        mtl.register_task("task1", input_dim=4, output_dim=2)
        mtl.register_task("task2", input_dim=4, output_dim=3)

        model = mtl.create_model()

        assert model is not None
        assert mtl.model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_soft_sharing_model(self):
        """Test: Create soft sharing multi-task model."""
        config = MultiTaskConfig(architecture=MultiTaskArchitecture.SOFT_SHARING.value)
        mtl = MultiTaskLearning(config)

        mtl.register_task("task1", input_dim=4, output_dim=2)
        mtl.register_task("task2", input_dim=4, output_dim=2)

        model = mtl.create_model()

        assert model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_hard_sharing_forward_pass(self):
        """Test: Hard sharing model forward pass."""
        config = MultiTaskConfig()
        mtl = MultiTaskLearning(config)

        mtl.register_task("task1", input_dim=4, output_dim=2)
        mtl.register_task("task2", input_dim=4, output_dim=3)

        model = mtl.create_model()

        # Initialize and test
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 4))

        # Task 0
        params = model.init(key, x, task_idx=0)
        output = model.apply(params, x, task_idx=0)

        assert output.shape == (1, 2)
        assert jnp.isfinite(output).all()

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_task_similarity(self):
        """Test: Compute task similarity."""
        config = MultiTaskConfig()
        mtl = MultiTaskLearning(config)

        # Create synthetic task data
        key = jax.random.PRNGKey(42)
        X1 = jax.random.normal(key, (50, 4))
        y1 = jax.random.normal(key, (50, 2))

        X2 = X1 + 0.1 * jax.random.normal(key, (50, 4))  # Similar
        y2 = y1 + 0.1 * jax.random.normal(key, (50, 2))

        X3 = jax.random.normal(key, (50, 4))  # Different
        y3 = jax.random.normal(key, (50, 2))

        sim_12 = mtl.compute_task_similarity((X1, y1), (X2, y2))
        sim_13 = mtl.compute_task_similarity((X1, y1), (X3, y3))

        # Similar tasks should have higher similarity
        assert sim_12 > sim_13
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_cluster_tasks(self):
        """Test: Cluster tasks based on similarity."""
        config = MultiTaskConfig(num_clusters=2)
        mtl = MultiTaskLearning(config)

        mtl.register_task("task1", input_dim=4, output_dim=2)
        mtl.register_task("task2", input_dim=4, output_dim=2)
        mtl.register_task("task3", input_dim=4, output_dim=2)

        # Create task data (task1 and task2 similar, task3 different)
        key = jax.random.PRNGKey(42)
        X_base = jax.random.normal(key, (50, 4))
        y_base = jax.random.normal(key, (50, 2))

        task_data = {
            "task1": (X_base, y_base),
            "task2": (X_base + 0.01, y_base + 0.01),  # Very similar
            "task3": (jax.random.normal(key, (50, 4)), jax.random.normal(key, (50, 2)))  # Different
        }

        clusters = mtl.cluster_tasks(task_data)

        assert len(clusters) <= 2  # Should create at most 2 clusters
        # task1 and task2 should be in same cluster
        task1_cluster = None
        task2_cluster = None
        for cluster_id, tasks in clusters.items():
            if "task1" in tasks:
                task1_cluster = cluster_id
            if "task2" in tasks:
                task2_cluster = cluster_id

        # They should be in the same cluster (high similarity)
        assert task1_cluster == task2_cluster

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_negative_transfer_detection(self):
        """Test: Detect negative transfer."""
        config = MultiTaskConfig(detect_negative_transfer=True)
        mtl = MultiTaskLearning(config)

        # Significant performance drop
        is_negative = mtl.detect_negative_transfer(
            task_name="task1",
            current_performance=0.5,
            baseline_performance=1.0
        )

        assert is_negative  # 50% drop should trigger

        # Small performance drop
        is_negative = mtl.detect_negative_transfer(
            task_name="task1",
            current_performance=0.95,
            baseline_performance=1.0
        )

        assert not is_negative  # 5% drop should not trigger


class TestEnhancedMAML:
    """Tests for enhanced MAML."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_model(self):
        """Test: Create MAML model."""
        config = MetaLearningConfig()
        maml = EnhancedMAML(config)

        model = maml.create_model(input_dim=4, output_dim=2)

        assert model is not None
        assert maml.model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_inner_loop(self):
        """Test: MAML inner loop adaptation."""
        config = MetaLearningConfig(num_inner_steps=3)
        maml = EnhancedMAML(config)

        model = maml.create_model(input_dim=4, output_dim=2)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))
        params = model.init(key, x_dummy)

        # Create support set
        support_x = jax.random.normal(key, (10, 4))
        support_y = jax.random.normal(key, (10, 2))

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        # Adapt
        adapted_params = maml.inner_loop(params, support_x, support_y, mse_loss)

        # Parameters should have changed
        assert adapted_params is not params
        # Check that at least one parameter changed
        params_changed = any(
            not jnp.allclose(p1, p2)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(adapted_params)
            )
        )
        assert params_changed

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_outer_loop_loss(self):
        """Test: MAML outer loop (meta-loss)."""
        config = MetaLearningConfig(num_inner_steps=2)
        maml = EnhancedMAML(config)

        model = maml.create_model(input_dim=4, output_dim=2)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))
        meta_params = model.init(key, x_dummy)

        # Create tasks batch
        tasks_batch = []
        for i in range(3):
            key, subkey = jax.random.split(key)
            support_x = jax.random.normal(subkey, (10, 4))
            support_y = jax.random.normal(subkey, (10, 2))
            query_x = jax.random.normal(subkey, (5, 4))
            query_y = jax.random.normal(subkey, (5, 2))
            tasks_batch.append((support_x, support_y, query_x, query_y))

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        meta_loss = maml.outer_loop_loss(meta_params, tasks_batch, mse_loss)

        assert jnp.isfinite(meta_loss)
        assert meta_loss >= 0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_meta_overfitting_detection(self):
        """Test: Detect meta-overfitting."""
        config = MetaLearningConfig(early_stopping_patience=3)
        maml = EnhancedMAML(config)

        # Simulate increasing validation loss
        should_stop = False
        for val_loss in [1.0, 0.9, 0.8, 0.85, 0.9, 0.95, 1.0]:
            should_stop = maml.detect_meta_overfitting(val_loss)
            if should_stop:
                break

        assert should_stop  # Should detect overfitting


class TestReptile:
    """Tests for Reptile meta-learning."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_model(self):
        """Test: Create Reptile model."""
        config = MetaLearningConfig()
        reptile = Reptile(config)

        model = reptile.create_model(input_dim=4, output_dim=2)

        assert model is not None
        assert reptile.model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_adapt_to_task(self):
        """Test: Reptile task adaptation."""
        config = MetaLearningConfig(num_inner_steps=5)
        reptile = Reptile(config)

        model = reptile.create_model(input_dim=4, output_dim=2)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))
        initial_params = model.init(key, x_dummy)

        # Task data
        X = jax.random.normal(key, (20, 4))
        y = jax.random.normal(key, (20, 2))

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        adapted_params = reptile.adapt_to_task(initial_params, (X, y), mse_loss)

        # Parameters should change
        params_changed = any(
            not jnp.allclose(p1, p2)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(initial_params),
                jax.tree_util.tree_leaves(adapted_params)
            )
        )
        assert params_changed

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_meta_train_step(self):
        """Test: Reptile meta-training step."""
        config = MetaLearningConfig(num_inner_steps=3)
        reptile = Reptile(config)

        model = reptile.create_model(input_dim=4, output_dim=2)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))
        meta_params = model.init(key, x_dummy)

        # Create tasks batch
        tasks_batch = []
        for i in range(3):
            key, subkey = jax.random.split(key)
            X = jax.random.normal(subkey, (20, 4))
            y = jax.random.normal(subkey, (20, 2))
            tasks_batch.append((X, y))

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        updated_meta_params = reptile.meta_train_step(meta_params, tasks_batch, mse_loss)

        # Meta-parameters should update
        params_changed = any(
            not jnp.allclose(p1, p2)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(meta_params),
                jax.tree_util.tree_leaves(updated_meta_params)
            )
        )
        assert params_changed


class TestANIL:
    """Tests for ANIL (Almost No Inner Loop)."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_model(self):
        """Test: Create ANIL model (body + head)."""
        config = MetaLearningConfig()
        anil = ANIL(config)

        body, head = anil.create_model(input_dim=4, output_dim=2)

        assert body is not None
        assert head is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_inner_loop_head_only(self):
        """Test: ANIL inner loop (adapt head only)."""
        config = MetaLearningConfig(num_inner_steps=3)
        anil = ANIL(config)

        body_model, head_model = anil.create_model(input_dim=4, output_dim=2)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))

        body_params = body_model.init(key, x_dummy)

        # Get feature dimension
        features = body_model.apply(body_params, x_dummy)
        feature_dim = features.shape[-1]

        head_params = head_model.init(key, jnp.ones((1, feature_dim)))

        # Support set
        support_x = jax.random.normal(key, (10, 4))
        support_y = jax.random.normal(key, (10, 2))

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        adapted_head_params = anil.inner_loop(
            body_params,
            head_params,
            body_model,
            head_model,
            support_x,
            support_y,
            mse_loss
        )

        # Head parameters should change
        head_changed = any(
            not jnp.allclose(p1, p2)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(head_params),
                jax.tree_util.tree_leaves(adapted_head_params)
            )
        )
        assert head_changed


class TestTaskEmbedding:
    """Tests for task embedding."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_register_tasks(self):
        """Test: Register tasks for embedding."""
        task_emb = TaskEmbedding(embedding_dim=32)

        task_emb.register_task("task1")
        task_emb.register_task("task2")
        task_emb.register_task("task3")

        assert len(task_emb.task_to_id) == 3
        assert "task1" in task_emb.task_to_id
        assert task_emb.task_to_id["task1"] == 0
        assert task_emb.task_to_id["task2"] == 1

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_initialize_embeddings(self):
        """Test: Initialize task embeddings."""
        task_emb = TaskEmbedding(embedding_dim=16)

        task_emb.register_task("task1")
        task_emb.register_task("task2")

        key = jax.random.PRNGKey(42)
        task_emb.initialize_embeddings(key)

        assert task_emb.embeddings is not None
        assert task_emb.embeddings.shape == (2, 16)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_get_embedding(self):
        """Test: Get task embedding."""
        task_emb = TaskEmbedding(embedding_dim=16)

        task_emb.register_task("task1")
        task_emb.register_task("task2")

        key = jax.random.PRNGKey(42)
        task_emb.initialize_embeddings(key)

        emb1 = task_emb.get_embedding("task1")
        emb2 = task_emb.get_embedding("task2")

        assert emb1.shape == (16,)
        assert emb2.shape == (16,)
        assert not jnp.allclose(emb1, emb2)  # Should be different

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_update_embedding(self):
        """Test: Update task embedding with gradient."""
        task_emb = TaskEmbedding(embedding_dim=16)

        task_emb.register_task("task1")

        key = jax.random.PRNGKey(42)
        task_emb.initialize_embeddings(key)

        emb_before = task_emb.get_embedding("task1").copy()

        # Update with gradient
        gradient = jnp.ones(16) * 0.1
        task_emb.update_embedding("task1", gradient, learning_rate=0.01)

        emb_after = task_emb.get_embedding("task1")

        # Should have changed
        assert not jnp.allclose(emb_before, emb_after)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_task_similarity_from_embeddings(self):
        """Test: Compute task similarity from embeddings."""
        task_emb = TaskEmbedding(embedding_dim=16)

        task_emb.register_task("task1")
        task_emb.register_task("task2")
        task_emb.register_task("task3")

        # Initialize with specific embeddings
        task_emb.embeddings = jnp.array([
            jnp.ones(16),  # task1
            jnp.ones(16) * 0.9,  # task2 (similar to task1)
            -jnp.ones(16)  # task3 (opposite to task1)
        ])

        sim_12 = task_emb.compute_task_similarity_from_embeddings("task1", "task2")
        sim_13 = task_emb.compute_task_similarity_from_embeddings("task1", "task3")

        # task1 and task2 should be more similar than task1 and task3
        assert sim_12 > sim_13
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0


class TestTaskConditionalMetaLearning:
    """Tests for task-conditional meta-learning."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_create_model(self):
        """Test: Create task-conditional model."""
        config = MetaLearningConfig(task_embedding_dim=32)
        tc_meta = TaskConditionalMetaLearning(config)

        model = tc_meta.create_model(input_dim=4, output_dim=2)

        assert model is not None
        assert tc_meta.model is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_task_specific_initialization(self):
        """Test: Get task-specific initialization."""
        config = MetaLearningConfig(task_embedding_dim=16)
        tc_meta = TaskConditionalMetaLearning(config)

        model = tc_meta.create_model(input_dim=4, output_dim=2)

        # Initialize base parameters
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))
        base_params = model.init(key, x_dummy)

        # Register tasks
        tc_meta.task_embedding.register_task("task1")
        tc_meta.task_embedding.register_task("task2")
        tc_meta.task_embedding.initialize_embeddings(key)

        # Get task-specific initializations
        task1_params = tc_meta.get_task_initialization("task1", base_params)
        task2_params = tc_meta.get_task_initialization("task2", base_params)

        # Should be different from base and from each other
        params_diff = any(
            not jnp.allclose(p1, p2)
            for p1, p2 in zip(
                jax.tree_util.tree_leaves(task1_params),
                jax.tree_util.tree_leaves(task2_params)
            )
        )
        assert params_diff


class TestAdaptiveInnerSteps:
    """Tests for adaptive inner steps."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_compute_adaptive_steps(self):
        """Test: Compute adaptive number of inner steps."""
        adaptive = AdaptiveInnerSteps(min_steps=1, max_steps=10, tolerance=1e-4)

        # Create simple model
        config = MetaLearningConfig()
        maml = EnhancedMAML(config)
        model = maml.create_model(input_dim=4, output_dim=2)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))
        params = model.init(key, x_dummy)

        # Create support set
        support_x = jax.random.normal(key, (20, 4))
        support_y = jax.random.normal(key, (20, 2))

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        num_steps, adapted_params = adaptive.compute_adaptive_steps(
            model,
            params,
            support_x,
            support_y,
            mse_loss,
            learning_rate=0.01
        )

        # Should determine some number of steps
        assert 1 <= num_steps <= 10
        assert adapted_params is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_early_stopping_plateau(self):
        """Test: Adaptive steps stops early on plateau."""
        adaptive = AdaptiveInnerSteps(min_steps=2, max_steps=20, tolerance=1e-3)

        # Create simple model
        config = MetaLearningConfig()
        maml = EnhancedMAML(config)
        model = maml.create_model(input_dim=2, output_dim=1)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 2))
        params = model.init(key, x_dummy)

        # Create easy support set (will plateau quickly)
        support_x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        support_y = jnp.array([[1.0], [0.0]])

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        num_steps, _ = adaptive.compute_adaptive_steps(
            model,
            params,
            support_x,
            support_y,
            mse_loss,
            learning_rate=0.1
        )

        # Should stop before max_steps due to plateau
        assert num_steps < adaptive.max_steps


class TestIntegration:
    """Integration tests for multi-task and meta-learning."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_multitask_complete_workflow(self):
        """Test: Complete multi-task learning workflow."""
        # Configuration
        config = MultiTaskConfig(
            architecture=MultiTaskArchitecture.HARD_SHARING.value,
            shared_layers=[32, 32],
            task_specific_layers=[16]
        )

        mtl = MultiTaskLearning(config)

        # Register tasks
        mtl.register_task("task1", input_dim=4, output_dim=2)
        mtl.register_task("task2", input_dim=4, output_dim=3)

        # Create model
        model = mtl.create_model()

        # Test forward pass for both tasks
        key = jax.random.PRNGKey(42)
        x = jnp.ones((1, 4))

        params = model.init(key, x, task_idx=0)

        output1 = model.apply(params, x, task_idx=0)
        output2 = model.apply(params, x, task_idx=1)

        assert output1.shape == (1, 2)
        assert output2.shape == (1, 3)

        print("Multi-task learning workflow completed successfully")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_metalearning_complete_workflow(self):
        """Test: Complete meta-learning workflow."""
        # Configuration
        config = MetaLearningConfig(
            algorithm=MetaLearningAlgorithm.MAML.value,
            num_inner_steps=2,
            model_layers=[32, 32]
        )

        maml = EnhancedMAML(config)

        # Create model
        model = maml.create_model(input_dim=4, output_dim=2)

        # Initialize
        key = jax.random.PRNGKey(42)
        x_dummy = jnp.ones((1, 4))
        meta_params = model.init(key, x_dummy)

        # Create task
        support_x = jax.random.normal(key, (10, 4))
        support_y = jax.random.normal(key, (10, 2))

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        # Adapt to task
        adapted_params = maml.inner_loop(meta_params, support_x, support_y, mse_loss)

        # Test on query point
        query_x = jax.random.normal(key, (1, 4))
        prediction = model.apply(adapted_params, query_x)

        assert prediction.shape == (1, 2)
        assert jnp.isfinite(prediction).all()

        print("Meta-learning workflow completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
