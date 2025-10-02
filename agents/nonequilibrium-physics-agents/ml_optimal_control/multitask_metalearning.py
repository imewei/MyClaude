"""Enhanced Multi-Task and Meta-Learning for Optimal Control.

This module provides advanced multi-task learning architectures and improved
meta-learning algorithms for optimal control, building on foundations from
Week 5-6 and integrating with transfer learning from Week 17-18.

Features:
- Multi-task architectures with task clustering
- Enhanced gradient-based meta-learning (MAML, Reptile)
- Task relationship discovery and embedding learning
- Negative transfer detection and mitigation
- Task-conditional meta-learning

Author: Nonequilibrium Physics Agents
Week: 21-22 of Phase 4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    import flax.linen as nn
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Provide comprehensive fallbacks
    jnp = np

    class nn:
        class Module:
            pass
        @staticmethod
        def compact(fn):
            return fn
        @staticmethod
        def Dense(*args, **kwargs):
            pass
        @staticmethod
        def relu(x):
            return np.maximum(0, x)

        class initializers:
            @staticmethod
            def lecun_normal():
                return lambda *args: None

    class jax:
        class random:
            @staticmethod
            def PRNGKey(seed):
                return None
            @staticmethod
            def normal(key, shape):
                return np.random.normal(size=shape)
            @staticmethod
            def split(key):
                return None, None

        class tree_util:
            @staticmethod
            def tree_leaves(tree):
                return []
            @staticmethod
            def tree_map(fn, *trees):
                return None

        class tree_map:
            @staticmethod
            def __call__(*args, **kwargs):
                return None

        @staticmethod
        def grad(fn):
            return fn

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def vmap(fn):
            return fn

        class nn:
            @staticmethod
            def softmax(x, axis=None):
                return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class MultiTaskArchitecture(Enum):
    """Multi-task learning architecture types."""
    HARD_SHARING = "hard_sharing"  # Shared bottom layers, task-specific heads
    SOFT_SHARING = "soft_sharing"  # Cross-stitch networks, task-specific columns
    TASK_CLUSTERING = "task_clustering"  # Cluster tasks, shared within clusters
    HIERARCHICAL = "hierarchical"  # Hierarchical multi-task structure


class MetaLearningAlgorithm(Enum):
    """Meta-learning algorithm types."""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # First-order MAML approximation
    FOMAML = "fomaml"  # First-Order MAML
    ANIL = "anil"  # Almost No Inner Loop (meta-learn head only)
    TASK_CONDITIONAL = "task_conditional"  # Task-conditional adaptation


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning."""

    # Architecture
    architecture: str = MultiTaskArchitecture.HARD_SHARING.value
    shared_layers: List[int] = field(default_factory=lambda: [128, 128])
    task_specific_layers: List[int] = field(default_factory=lambda: [64, 64])

    # Task clustering
    num_clusters: Optional[int] = None  # Auto-detect if None
    cluster_threshold: float = 0.5  # Similarity threshold for clustering

    # Training
    learning_rate: float = 1e-3
    task_weights: Optional[Dict[str, float]] = None  # Per-task loss weights

    # Negative transfer detection
    detect_negative_transfer: bool = True
    negative_transfer_threshold: float = 0.1  # Relative performance drop

    # Regularization
    task_diversity_weight: float = 0.1  # Encourage task-specific differences


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""

    # Algorithm
    algorithm: str = MetaLearningAlgorithm.MAML.value

    # Meta-learning hyperparameters
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 1e-2
    num_inner_steps: int = 5
    num_meta_iterations: int = 1000

    # Task sampling
    tasks_per_batch: int = 4
    shots_per_task: int = 10  # K-shot learning
    query_points_per_task: int = 15

    # Architecture
    model_layers: List[int] = field(default_factory=lambda: [128, 128, 64])

    # Meta-overfitting prevention
    meta_validation_split: float = 0.2  # Split meta-training tasks
    early_stopping_patience: int = 50

    # Task conditioning
    task_embedding_dim: int = 32  # For task-conditional meta-learning
    learn_task_embeddings: bool = True


class HardSharingMTL(nn.Module):
    """Hard parameter sharing multi-task network.

    Architecture: Shared bottom layers + task-specific heads
    """
    shared_layers: List[int]
    task_specific_layers: List[int]
    num_tasks: int
    output_dims: List[int]  # Output dimension for each task

    @nn.compact
    def __call__(self, x, task_idx: int):
        """Forward pass for specific task.

        Args:
            x: Input features
            task_idx: Which task to compute (0 to num_tasks-1)

        Returns:
            Task-specific output
        """
        # Shared layers
        for hidden_dim in self.shared_layers:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)

        # Task-specific layers
        for hidden_dim in self.task_specific_layers:
            x = nn.Dense(hidden_dim, name=f'task_{task_idx}_dense')(x)
            x = nn.relu(x)

        # Task-specific output
        output = nn.Dense(self.output_dims[task_idx], name=f'task_{task_idx}_output')(x)

        return output


class SoftSharingMTL(nn.Module):
    """Soft parameter sharing with cross-stitch connections.

    Architecture: Task-specific columns with cross-stitch units
    """
    num_layers: int
    layer_dims: List[int]
    num_tasks: int
    output_dims: List[int]

    @nn.compact
    def __call__(self, x, task_idx: int):
        """Forward pass with cross-stitch connections."""
        # Initialize task-specific representations
        task_reps = [x for _ in range(self.num_tasks)]

        # Process through layers with cross-stitch
        for layer_idx, hidden_dim in enumerate(self.layer_dims):
            # Task-specific transformations
            new_task_reps = []
            for t in range(self.num_tasks):
                rep = nn.Dense(hidden_dim, name=f'layer_{layer_idx}_task_{t}')(task_reps[t])
                new_task_reps.append(nn.relu(rep))

            # Cross-stitch: Linear combination of task representations
            # α[i,j] controls how much task i uses information from task j
            if layer_idx < len(self.layer_dims) - 1:  # Don't cross-stitch at last layer
                alpha = self.param(
                    f'cross_stitch_{layer_idx}',
                    nn.initializers.lecun_normal(),
                    (self.num_tasks, self.num_tasks)
                )

                # Normalize to sum to 1 per row
                alpha = jax.nn.softmax(alpha, axis=1)

                # Linear combination
                combined_reps = []
                for i in range(self.num_tasks):
                    combined = sum(alpha[i, j] * new_task_reps[j] for j in range(self.num_tasks))
                    combined_reps.append(combined)

                task_reps = combined_reps
            else:
                task_reps = new_task_reps

        # Task-specific output
        output = nn.Dense(self.output_dims[task_idx], name=f'output_task_{task_idx}')(
            task_reps[task_idx]
        )

        return output


class MultiTaskLearning:
    """Enhanced multi-task learning for optimal control."""

    def __init__(self, config: MultiTaskConfig):
        """Initialize multi-task learning.

        Args:
            config: Multi-task learning configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for MultiTaskLearning")

        self.config = config
        self.tasks: Dict[str, Dict] = {}  # task_name -> task_info
        self.task_clusters: Optional[Dict[int, List[str]]] = None
        self.model = None
        self.params = None

    def register_task(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        loss_fn: Optional[Callable] = None
    ):
        """Register a task for multi-task learning.

        Args:
            name: Task name
            input_dim: Input dimension
            output_dim: Output dimension
            loss_fn: Optional custom loss function
        """
        self.tasks[name] = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'loss_fn': loss_fn or self._default_loss,
            'performance_history': []
        }

    def create_model(self) -> nn.Module:
        """Create multi-task model based on architecture type.

        Returns:
            Flax neural network module
        """
        num_tasks = len(self.tasks)
        output_dims = [info['output_dim'] for info in self.tasks.values()]

        if self.config.architecture == MultiTaskArchitecture.HARD_SHARING.value:
            self.model = HardSharingMTL(
                shared_layers=self.config.shared_layers,
                task_specific_layers=self.config.task_specific_layers,
                num_tasks=num_tasks,
                output_dims=output_dims
            )

        elif self.config.architecture == MultiTaskArchitecture.SOFT_SHARING.value:
            layer_dims = self.config.shared_layers + self.config.task_specific_layers
            self.model = SoftSharingMTL(
                num_layers=len(layer_dims),
                layer_dims=layer_dims,
                num_tasks=num_tasks,
                output_dims=output_dims
            )

        else:
            raise ValueError(f"Architecture {self.config.architecture} not implemented")

        return self.model

    def compute_task_similarity(
        self,
        task1_data: Tuple[jnp.ndarray, jnp.ndarray],
        task2_data: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> float:
        """Compute similarity between two tasks.

        Uses feature distribution distance and output correlation.

        Args:
            task1_data: (X1, y1) for task 1
            task2_data: (X2, y2) for task 2

        Returns:
            Similarity score in [0, 1]
        """
        X1, y1 = task1_data
        X2, y2 = task2_data

        # Feature distribution similarity (inverse of Wasserstein distance approximation)
        feature_sim = 1.0 / (1.0 + jnp.linalg.norm(jnp.mean(X1, axis=0) - jnp.mean(X2, axis=0)))

        # Output correlation (if dimensions match)
        if y1.shape[1] == y2.shape[1]:
            min_samples = min(len(y1), len(y2))
            corr = jnp.corrcoef(y1[:min_samples].flatten(), y2[:min_samples].flatten())[0, 1]
            output_sim = (corr + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        else:
            output_sim = 0.5  # Neutral similarity if dimensions don't match

        # Combined similarity
        similarity = 0.5 * feature_sim + 0.5 * output_sim

        return float(similarity)

    def cluster_tasks(
        self,
        task_data: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Dict[int, List[str]]:
        """Cluster tasks based on similarity.

        Args:
            task_data: Dict mapping task names to (X, y) data

        Returns:
            Dict mapping cluster_id to list of task names
        """
        task_names = list(task_data.keys())
        n_tasks = len(task_names)

        # Compute pairwise similarities
        similarity_matrix = np.zeros((n_tasks, n_tasks))
        for i, task1 in enumerate(task_names):
            for j, task2 in enumerate(task_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_task_similarity(task_data[task1], task_data[task2])
                    similarity_matrix[i, j] = sim

        # Simple agglomerative clustering
        clusters = {i: [task_names[i]] for i in range(n_tasks)}
        cluster_id = n_tasks

        while len(clusters) > (self.config.num_clusters or 1):
            # Find most similar pair of clusters
            max_sim = -1
            best_pair = None

            for i in list(clusters.keys()):
                for j in list(clusters.keys()):
                    if i >= j:
                        continue

                    # Average similarity between clusters
                    sims = []
                    for task_i in clusters[i]:
                        for task_j in clusters[j]:
                            idx_i = task_names.index(task_i)
                            idx_j = task_names.index(task_j)
                            sims.append(similarity_matrix[idx_i, idx_j])

                    avg_sim = np.mean(sims)

                    if avg_sim > max_sim:
                        max_sim = avg_sim
                        best_pair = (i, j)

            # Merge if similarity above threshold
            if max_sim < self.config.cluster_threshold:
                break

            i, j = best_pair
            clusters[cluster_id] = clusters[i] + clusters[j]
            del clusters[i]
            del clusters[j]
            cluster_id += 1

        # Renumber clusters
        self.task_clusters = {i: cluster for i, cluster in enumerate(clusters.values())}

        return self.task_clusters

    def detect_negative_transfer(
        self,
        task_name: str,
        current_performance: float,
        baseline_performance: float
    ) -> bool:
        """Detect if multi-task learning is causing negative transfer.

        Args:
            task_name: Name of task to check
            current_performance: Current multi-task performance
            baseline_performance: Single-task baseline performance

        Returns:
            True if negative transfer detected
        """
        if not self.config.detect_negative_transfer:
            return False

        # Performance drop compared to baseline
        relative_drop = (baseline_performance - current_performance) / (baseline_performance + 1e-8)

        is_negative = relative_drop > self.config.negative_transfer_threshold

        if is_negative:
            print(f"⚠️  Negative transfer detected for task '{task_name}': "
                  f"{relative_drop*100:.1f}% performance drop")

        return is_negative

    @staticmethod
    def _default_loss(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
        """Default MSE loss."""
        return jnp.mean((predictions - targets) ** 2)

    def task_diversity_regularization(self, params) -> float:
        """Compute task diversity regularization.

        Encourages task-specific parameters to be different from each other.

        Args:
            params: Model parameters

        Returns:
            Diversity loss (lower = more similar tasks)
        """
        # Extract task-specific parameters
        task_params = []

        for key in params['params']:
            if 'task_' in key:
                task_params.append(params['params'][key])

        if len(task_params) < 2:
            return 0.0

        # Compute pairwise differences
        diversity = 0.0
        count = 0

        for i in range(len(task_params)):
            for j in range(i + 1, len(task_params)):
                # Negative of parameter similarity (we want diversity)
                for param_i, param_j in zip(
                    jax.tree_util.tree_leaves(task_params[i]),
                    jax.tree_util.tree_leaves(task_params[j])
                ):
                    if param_i.shape == param_j.shape:
                        diversity -= jnp.sum((param_i - param_j) ** 2)
                        count += 1

        return diversity / (count + 1e-8) if count > 0 else 0.0


class EnhancedMAML:
    """Enhanced Model-Agnostic Meta-Learning with improvements.

    Improvements over basic MAML:
    - Meta-overfitting detection
    - Task-conditional adaptation
    - Automatic inner step tuning
    """

    def __init__(self, config: MetaLearningConfig):
        """Initialize enhanced MAML.

        Args:
            config: Meta-learning configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for EnhancedMAML")

        self.config = config
        self.model = None
        self.meta_params = None
        self.task_embeddings: Optional[jnp.ndarray] = None

        # Meta-validation for overfitting detection
        self.meta_train_performance = []
        self.meta_val_performance = []
        self.best_meta_params = None
        self.patience_counter = 0

    def create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create base model for meta-learning.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            Flax neural network module
        """
        class BaseModel(nn.Module):
            layer_dims: List[int]
            output_dim: int

            @nn.compact
            def __call__(self, x):
                for hidden_dim in self.layer_dims:
                    x = nn.Dense(hidden_dim)(x)
                    x = nn.relu(x)
                output = nn.Dense(self.output_dim)(x)
                return output

        self.model = BaseModel(
            layer_dims=self.config.model_layers,
            output_dim=output_dim
        )

        return self.model

    def inner_loop(
        self,
        params,
        support_x: jnp.ndarray,
        support_y: jnp.ndarray,
        loss_fn: Callable
    ):
        """Inner loop: Adapt to task using support set.

        Args:
            params: Initial parameters (meta-parameters)
            support_x: Support set inputs
            support_y: Support set targets
            loss_fn: Loss function

        Returns:
            Adapted parameters
        """
        adapted_params = params

        for step in range(self.config.num_inner_steps):
            # Compute gradient on support set
            def support_loss(p):
                predictions = self.model.apply(p, support_x)
                return loss_fn(predictions, support_y)

            grads = grad(support_loss)(adapted_params)

            # Gradient descent update
            adapted_params = jax.tree_map(
                lambda p, g: p - self.config.inner_learning_rate * g,
                adapted_params,
                grads
            )

        return adapted_params

    def outer_loop_loss(
        self,
        meta_params,
        tasks_batch: List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        loss_fn: Callable
    ) -> float:
        """Outer loop: Compute meta-loss across tasks.

        Args:
            meta_params: Meta-parameters
            tasks_batch: List of (support_x, support_y, query_x, query_y)
            loss_fn: Loss function

        Returns:
            Meta-loss (average query loss after adaptation)
        """
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks_batch:
            # Adapt to task
            adapted_params = self.inner_loop(meta_params, support_x, support_y, loss_fn)

            # Evaluate on query set
            query_predictions = self.model.apply(adapted_params, query_x)
            task_loss = loss_fn(query_predictions, query_y)

            meta_loss += task_loss

        return meta_loss / len(tasks_batch)

    def meta_train_step(
        self,
        meta_params,
        tasks_batch: List[Tuple],
        loss_fn: Callable,
        optimizer_state
    ):
        """Single meta-training step.

        Args:
            meta_params: Current meta-parameters
            tasks_batch: Batch of tasks
            loss_fn: Loss function
            optimizer_state: Optimizer state

        Returns:
            Updated meta_params, optimizer_state, meta_loss
        """
        # Compute meta-gradient
        def meta_loss_fn(params):
            return self.outer_loop_loss(params, tasks_batch, loss_fn)

        meta_loss, meta_grads = jax.value_and_grad(meta_loss_fn)(meta_params)

        # Meta-update
        # (Optimizer update logic would go here - simplified for demo)
        meta_params = jax.tree_map(
            lambda p, g: p - self.config.meta_learning_rate * g,
            meta_params,
            meta_grads
        )

        return meta_params, optimizer_state, meta_loss

    def detect_meta_overfitting(self, val_loss: float) -> bool:
        """Detect meta-overfitting using validation performance.

        Args:
            val_loss: Current meta-validation loss

        Returns:
            True if should stop training (meta-overfitting detected)
        """
        self.meta_val_performance.append(val_loss)

        if len(self.meta_val_performance) < 2:
            return False

        # Check if validation loss is increasing
        if val_loss > min(self.meta_val_performance[:-1]):
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            self.best_meta_params = self.meta_params

        if self.patience_counter >= self.config.early_stopping_patience:
            print(f"Meta-overfitting detected. Stopping at iteration "
                  f"{len(self.meta_val_performance)}")
            return True

        return False


class TaskEmbedding:
    """Learn task embeddings for task-conditional meta-learning."""

    def __init__(self, embedding_dim: int = 32):
        """Initialize task embedding.

        Args:
            embedding_dim: Dimension of task embeddings
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for TaskEmbedding")

        self.embedding_dim = embedding_dim
        self.task_to_id: Dict[str, int] = {}
        self.embeddings: Optional[jnp.ndarray] = None

    def register_task(self, task_name: str):
        """Register a new task.

        Args:
            task_name: Name of task
        """
        if task_name not in self.task_to_id:
            task_id = len(self.task_to_id)
            self.task_to_id[task_name] = task_id

    def initialize_embeddings(self, key: jax.random.PRNGKey):
        """Initialize task embeddings randomly.

        Args:
            key: JAX random key
        """
        num_tasks = len(self.task_to_id)
        self.embeddings = jax.random.normal(key, (num_tasks, self.embedding_dim))

    def get_embedding(self, task_name: str) -> jnp.ndarray:
        """Get embedding for task.

        Args:
            task_name: Name of task

        Returns:
            Task embedding vector
        """
        task_id = self.task_to_id[task_name]
        return self.embeddings[task_id]

    def update_embedding(
        self,
        task_name: str,
        gradient: jnp.ndarray,
        learning_rate: float = 1e-3
    ):
        """Update task embedding using gradient.

        Args:
            task_name: Name of task
            gradient: Gradient with respect to embedding
            learning_rate: Learning rate
        """
        task_id = self.task_to_id[task_name]
        self.embeddings = self.embeddings.at[task_id].set(
            self.embeddings[task_id] - learning_rate * gradient
        )

    def compute_task_similarity_from_embeddings(
        self,
        task1: str,
        task2: str
    ) -> float:
        """Compute similarity between tasks using embeddings.

        Args:
            task1: First task name
            task2: Second task name

        Returns:
            Cosine similarity in [-1, 1]
        """
        emb1 = self.get_embedding(task1)
        emb2 = self.get_embedding(task2)

        cosine_sim = jnp.dot(emb1, emb2) / (
            jnp.linalg.norm(emb1) * jnp.linalg.norm(emb2) + 1e-8
        )

        return float(cosine_sim)


class Reptile:
    """Reptile: A simpler first-order meta-learning algorithm.

    Reptile is a first-order approximation to MAML that doesn't require
    computing gradients through the inner loop optimization.

    Key idea: Move meta-parameters toward task-adapted parameters.
    """

    def __init__(self, config: MetaLearningConfig):
        """Initialize Reptile.

        Args:
            config: Meta-learning configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for Reptile")

        self.config = config
        self.model = None
        self.meta_params = None

    def create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create base model for meta-learning.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            Flax neural network module
        """
        class BaseModel(nn.Module):
            layer_dims: List[int]
            output_dim: int

            @nn.compact
            def __call__(self, x):
                for hidden_dim in self.layer_dims:
                    x = nn.Dense(hidden_dim)(x)
                    x = nn.relu(x)
                output = nn.Dense(self.output_dim)(x)
                return output

        self.model = BaseModel(
            layer_dims=self.config.model_layers,
            output_dim=output_dim
        )

        return self.model

    def adapt_to_task(
        self,
        initial_params,
        task_data: Tuple[jnp.ndarray, jnp.ndarray],
        loss_fn: Callable
    ):
        """Adapt parameters to a task using SGD.

        Args:
            initial_params: Starting parameters (meta-parameters)
            task_data: (X, y) for task
            loss_fn: Loss function

        Returns:
            Task-adapted parameters
        """
        X, y = task_data
        adapted_params = initial_params

        for step in range(self.config.num_inner_steps):
            def task_loss(p):
                predictions = self.model.apply(p, X)
                return loss_fn(predictions, y)

            grads = grad(task_loss)(adapted_params)

            # SGD update
            adapted_params = jax.tree_map(
                lambda p, g: p - self.config.inner_learning_rate * g,
                adapted_params,
                grads
            )

        return adapted_params

    def meta_train_step(
        self,
        meta_params,
        tasks_batch: List[Tuple[jnp.ndarray, jnp.ndarray]],
        loss_fn: Callable
    ):
        """Single Reptile meta-training step.

        Args:
            meta_params: Current meta-parameters
            tasks_batch: List of (X, y) task data
            loss_fn: Loss function

        Returns:
            Updated meta_params
        """
        # Adapt to each task in batch
        adapted_params_list = []

        for task_data in tasks_batch:
            adapted_params = self.adapt_to_task(meta_params, task_data, loss_fn)
            adapted_params_list.append(adapted_params)

        # Compute average of adapted parameters
        avg_adapted_params = jax.tree_map(
            lambda *params: jnp.mean(jnp.stack(params), axis=0),
            *adapted_params_list
        )

        # Move meta-parameters toward average adapted parameters
        meta_params = jax.tree_map(
            lambda meta_p, adapted_p: meta_p + self.config.meta_learning_rate * (adapted_p - meta_p),
            meta_params,
            avg_adapted_params
        )

        return meta_params


class ANIL:
    """Almost No Inner Loop (ANIL).

    Key insight: Only meta-learn the head (final layer), not the body.
    Body is trained normally, head is meta-learned.

    Advantages:
    - Faster than MAML (fewer parameters to meta-learn)
    - Often similar performance
    """

    def __init__(self, config: MetaLearningConfig):
        """Initialize ANIL.

        Args:
            config: Meta-learning configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for ANIL")

        self.config = config
        self.model = None
        self.body_params = None  # Shared body (not meta-learned in inner loop)
        self.head_params = None  # Task-specific head (meta-learned)

    def create_model(self, input_dim: int, output_dim: int):
        """Create model with separate body and head.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            (body_model, head_model)
        """
        class Body(nn.Module):
            layer_dims: List[int]

            @nn.compact
            def __call__(self, x):
                for hidden_dim in self.layer_dims:
                    x = nn.Dense(hidden_dim)(x)
                    x = nn.relu(x)
                return x

        class Head(nn.Module):
            output_dim: int

            @nn.compact
            def __call__(self, x):
                return nn.Dense(self.output_dim)(x)

        body = Body(layer_dims=self.config.model_layers)
        head = Head(output_dim=output_dim)

        return body, head

    def inner_loop(
        self,
        body_params,
        initial_head_params,
        body_model,
        head_model,
        support_x: jnp.ndarray,
        support_y: jnp.ndarray,
        loss_fn: Callable
    ):
        """Inner loop: Only adapt head parameters.

        Args:
            body_params: Fixed body parameters
            initial_head_params: Initial head parameters
            body_model: Body network
            head_model: Head network
            support_x: Support set inputs
            support_y: Support set targets
            loss_fn: Loss function

        Returns:
            Adapted head parameters
        """
        adapted_head_params = initial_head_params

        for step in range(self.config.num_inner_steps):
            def support_loss(head_p):
                # Fixed body, adapted head
                features = body_model.apply(body_params, support_x)
                predictions = head_model.apply(head_p, features)
                return loss_fn(predictions, support_y)

            grads = grad(support_loss)(adapted_head_params)

            # Update only head
            adapted_head_params = jax.tree_map(
                lambda p, g: p - self.config.inner_learning_rate * g,
                adapted_head_params,
                grads
            )

        return adapted_head_params


class TaskConditionalMetaLearning:
    """Task-conditional meta-learning with learned task embeddings.

    Instead of same initialization for all tasks, use task-specific
    initialization based on learned task embeddings.
    """

    def __init__(self, config: MetaLearningConfig):
        """Initialize task-conditional meta-learning.

        Args:
            config: Meta-learning configuration
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for TaskConditionalMetaLearning")

        self.config = config
        self.task_embedding = TaskEmbedding(config.task_embedding_dim)
        self.model = None

        # Hypernetwork: Maps task embedding to initial parameters
        self.hypernetwork = None

    def create_model(self, input_dim: int, output_dim: int):
        """Create base model and hypernetwork.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            (base_model, hypernetwork)
        """
        class BaseModel(nn.Module):
            layer_dims: List[int]
            output_dim: int

            @nn.compact
            def __call__(self, x):
                for hidden_dim in self.layer_dims:
                    x = nn.Dense(hidden_dim)(x)
                    x = nn.relu(x)
                output = nn.Dense(self.output_dim)(x)
                return output

        class HyperNetwork(nn.Module):
            """Generates initial parameters conditioned on task embedding."""
            target_param_shape: Tuple[int, ...]

            @nn.compact
            def __call__(self, task_embedding):
                # Simple hypernetwork: embedding -> parameters
                x = nn.Dense(128)(task_embedding)
                x = nn.relu(x)
                x = nn.Dense(64)(x)
                x = nn.relu(x)

                # Generate parameters
                param_size = int(np.prod(self.target_param_shape))
                params_flat = nn.Dense(param_size)(x)

                return params_flat.reshape(self.target_param_shape)

        self.model = BaseModel(
            layer_dims=self.config.model_layers,
            output_dim=output_dim
        )

        return self.model

    def get_task_initialization(
        self,
        task_name: str,
        base_params
    ):
        """Get task-specific parameter initialization.

        Args:
            task_name: Name of task
            base_params: Base parameters (can be modified by task embedding)

        Returns:
            Task-conditioned initial parameters
        """
        task_emb = self.task_embedding.get_embedding(task_name)

        # Simple approach: Add task-specific offset to base parameters
        # (More sophisticated: Use hypernetwork to generate full params)

        task_offset = jax.tree_map(
            lambda p: 0.01 * jnp.tanh(task_emb[:p.size].reshape(p.shape)),
            base_params
        )

        task_init_params = jax.tree_map(
            lambda base, offset: base + offset,
            base_params,
            task_offset
        )

        return task_init_params


class AdaptiveInnerSteps:
    """Automatically determine optimal number of inner gradient steps.

    Key idea: Stop inner loop when performance plateaus.
    """

    def __init__(
        self,
        min_steps: int = 1,
        max_steps: int = 20,
        tolerance: float = 1e-4
    ):
        """Initialize adaptive inner steps.

        Args:
            min_steps: Minimum inner steps
            max_steps: Maximum inner steps
            tolerance: Stop if loss improvement < tolerance
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.tolerance = tolerance

    def compute_adaptive_steps(
        self,
        model,
        params,
        support_x: jnp.ndarray,
        support_y: jnp.ndarray,
        loss_fn: Callable,
        learning_rate: float
    ) -> Tuple[int, Any]:
        """Compute optimal number of inner steps for task.

        Args:
            model: Neural network model
            params: Initial parameters
            support_x: Support set inputs
            support_y: Support set targets
            loss_fn: Loss function
            learning_rate: Inner learning rate

        Returns:
            (optimal_steps, adapted_params)
        """
        adapted_params = params
        prev_loss = float('inf')

        for step in range(self.max_steps):
            # Compute loss
            def task_loss(p):
                predictions = model.apply(p, support_x)
                return loss_fn(predictions, support_y)

            current_loss = task_loss(adapted_params)

            # Check for plateau
            improvement = prev_loss - current_loss

            if step >= self.min_steps and improvement < self.tolerance:
                return step, adapted_params

            # Update parameters
            grads = grad(task_loss)(adapted_params)
            adapted_params = jax.tree_map(
                lambda p, g: p - learning_rate * g,
                adapted_params,
                grads
            )

            prev_loss = current_loss

        return self.max_steps, adapted_params
