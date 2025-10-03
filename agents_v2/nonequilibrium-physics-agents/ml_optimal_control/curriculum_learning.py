"""Curriculum learning for optimal control training.

Progressively increases task difficulty to improve learning efficiency
and final performance. Inspired by human learning - start simple,
gradually increase complexity.

Key techniques:
- Task difficulty progression
- Automatic curriculum generation
- Performance-based advancement
- Self-paced learning
- Teacher-student curriculum

Author: Nonequilibrium Physics Agents
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class CurriculumStrategy(Enum):
    """Curriculum learning strategies."""
    FIXED = "fixed"  # Pre-defined difficulty progression
    ADAPTIVE = "adaptive"  # Adjust based on performance
    SELF_PACED = "self_paced"  # Agent chooses difficulty
    TEACHER_STUDENT = "teacher_student"  # Teacher network guides
    REVERSE = "reverse"  # Start from goal, work backwards


class DifficultyMetric(Enum):
    """Metrics for measuring task difficulty."""
    TIME_HORIZON = "time_horizon"  # Longer horizon = harder
    STATE_DIMENSION = "state_dimension"  # More states = harder
    CONTROL_CONSTRAINTS = "control_constraints"  # Tighter = harder
    NONLINEARITY = "nonlinearity"  # More nonlinear = harder
    DISTURBANCE = "disturbance"  # More noise = harder
    SPARSE_REWARD = "sparse_reward"  # Sparser = harder


@dataclass
class Task:
    """Single task in curriculum."""
    task_id: str
    difficulty: float  # Difficulty score in [0, 1]
    config: Dict[str, Any]  # Task configuration
    prerequisites: List[str] = field(default_factory=list)  # Required prior tasks
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: str = CurriculumStrategy.ADAPTIVE.value
    difficulty_metric: str = DifficultyMetric.TIME_HORIZON.value

    # Fixed curriculum
    num_stages: int = 5
    stage_duration: int = 1000  # Episodes per stage

    # Adaptive curriculum
    performance_threshold: float = 0.7  # Advance when > threshold
    patience: int = 5  # Episodes before giving up on current stage
    backtrack_on_failure: bool = True  # Return to easier task if struggling

    # Self-paced
    self_paced_window: int = 100  # Consider last N episodes
    difficulty_increment: float = 0.1  # How much to increase

    # General
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    initial_difficulty: float = 0.1


class CurriculumLearning:
    """Manages curriculum for optimal control training."""

    def __init__(self, config: Optional[CurriculumConfig] = None):
        """Initialize curriculum learning.

        Args:
            config: Curriculum configuration
        """
        self.config = config or CurriculumConfig()
        self.tasks: List[Task] = []
        self.current_task_idx: int = 0
        self.performance_history: List[float] = []
        self.difficulty_history: List[float] = []
        self.stage_episodes: int = 0

    def add_task(
        self,
        task_id: str,
        difficulty: float,
        config: Dict[str, Any],
        prerequisites: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add task to curriculum.

        Args:
            task_id: Task identifier
            difficulty: Difficulty score [0, 1]
            config: Task configuration
            prerequisites: Required prior tasks
            metadata: Additional task information
        """
        task = Task(
            task_id=task_id,
            difficulty=difficulty,
            config=config,
            prerequisites=prerequisites or [],
            metadata=metadata or {}
        )
        self.tasks.append(task)

        # Sort tasks by difficulty
        self.tasks.sort(key=lambda t: t.difficulty)

    def generate_curriculum(
        self,
        base_config: Dict[str, Any],
        difficulty_metric: Optional[str] = None,
        num_tasks: int = 5
    ) -> None:
        """Automatically generate curriculum from base configuration.

        Args:
            base_config: Base task configuration
            difficulty_metric: Metric to vary (default: from config)
            num_tasks: Number of curriculum stages
        """
        difficulty_metric = difficulty_metric or self.config.difficulty_metric
        self.tasks = []

        for i in range(num_tasks):
            difficulty = i / (num_tasks - 1)  # 0 to 1
            task_config = base_config.copy()

            # Adjust configuration based on difficulty metric
            if difficulty_metric == DifficultyMetric.TIME_HORIZON.value:
                # Increase time horizon with difficulty
                base_horizon = base_config.get('time_horizon', [0.0, 1.0])
                task_config['time_horizon'] = [
                    base_horizon[0],
                    base_horizon[1] * (1 + 4 * difficulty)  # 1x to 5x
                ]

            elif difficulty_metric == DifficultyMetric.CONTROL_CONSTRAINTS.value:
                # Tighten control constraints with difficulty
                base_bounds = base_config.get('control_bounds', (-1.0, 1.0))
                scale = 1.0 - 0.8 * difficulty  # 1.0 to 0.2
                task_config['control_bounds'] = (
                    base_bounds[0] * scale,
                    base_bounds[1] * scale
                )

            elif difficulty_metric == DifficultyMetric.DISTURBANCE.value:
                # Increase disturbance with difficulty
                task_config['disturbance_std'] = 0.1 * difficulty

            elif difficulty_metric == DifficultyMetric.SPARSE_REWARD.value:
                # Make reward sparser with difficulty
                task_config['reward_sparsity'] = difficulty

            self.add_task(
                task_id=f'task_{i}',
                difficulty=difficulty,
                config=task_config,
                metadata={'stage': i}
            )

    def get_current_task(self) -> Task:
        """Get current task in curriculum.

        Returns:
            Current task
        """
        if not self.tasks:
            raise ValueError("No tasks in curriculum")

        return self.tasks[self.current_task_idx]

    def update(self, performance: float) -> bool:
        """Update curriculum based on performance.

        Args:
            performance: Performance metric (e.g., success rate, return)

        Returns:
            True if advanced to next task, False otherwise
        """
        self.performance_history.append(performance)
        self.difficulty_history.append(self.tasks[self.current_task_idx].difficulty)
        self.stage_episodes += 1

        strategy = self.config.strategy

        if strategy == CurriculumStrategy.FIXED.value:
            return self._update_fixed()

        elif strategy == CurriculumStrategy.ADAPTIVE.value:
            return self._update_adaptive(performance)

        elif strategy == CurriculumStrategy.SELF_PACED.value:
            return self._update_self_paced(performance)

        else:
            return False

    def _update_fixed(self) -> bool:
        """Update with fixed curriculum strategy.

        Returns:
            True if advanced
        """
        if self.stage_episodes >= self.config.stage_duration:
            # Advance to next task
            if self.current_task_idx < len(self.tasks) - 1:
                self.current_task_idx += 1
                self.stage_episodes = 0
                return True
        return False

    def _update_adaptive(self, performance: float) -> bool:
        """Update with adaptive curriculum strategy.

        Args:
            performance: Current performance

        Returns:
            True if advanced
        """
        # Check recent performance
        window = min(self.config.patience, len(self.performance_history))
        recent_performance = np.mean(self.performance_history[-window:])

        # Advance if performing well
        if recent_performance >= self.config.performance_threshold:
            if self.current_task_idx < len(self.tasks) - 1:
                self.current_task_idx += 1
                self.stage_episodes = 0
                self.performance_history = []  # Reset for new task
                return True

        # Backtrack if struggling for too long
        elif self.config.backtrack_on_failure:
            if self.stage_episodes >= self.config.patience * 10:
                if self.current_task_idx > 0:
                    self.current_task_idx -= 1
                    self.stage_episodes = 0
                    self.performance_history = []
                    return False

        return False

    def _update_self_paced(self, performance: float) -> bool:
        """Update with self-paced learning.

        Agent chooses difficulty based on recent success.

        Args:
            performance: Current performance

        Returns:
            True if difficulty changed
        """
        # Compute recent success rate
        window = min(self.config.self_paced_window, len(self.performance_history))
        recent_performance = np.mean(self.performance_history[-window:])

        current_difficulty = self.tasks[self.current_task_idx].difficulty

        # If doing well, increase difficulty
        if recent_performance >= 0.8:
            new_difficulty = min(
                current_difficulty + self.config.difficulty_increment,
                self.config.max_difficulty
            )
        # If struggling, decrease difficulty
        elif recent_performance <= 0.3:
            new_difficulty = max(
                current_difficulty - self.config.difficulty_increment,
                self.config.min_difficulty
            )
        else:
            return False

        # Find task closest to new difficulty
        for i, task in enumerate(self.tasks):
            if abs(task.difficulty - new_difficulty) < self.config.difficulty_increment / 2:
                if i != self.current_task_idx:
                    self.current_task_idx = i
                    self.stage_episodes = 0
                    return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'current_task_idx': self.current_task_idx,
            'current_difficulty': self.tasks[self.current_task_idx].difficulty,
            'total_tasks': len(self.tasks),
            'completion_progress': self.current_task_idx / max(len(self.tasks) - 1, 1),
            'avg_recent_performance': np.mean(self.performance_history[-10:]) if self.performance_history else 0.0,
            'stage_episodes': self.stage_episodes,
            'total_episodes': len(self.performance_history)
        }


class TaskGraph:
    """Represents curriculum as directed acyclic graph of tasks.

    Enables complex curricula with multiple paths and prerequisites.
    """

    def __init__(self):
        """Initialize task graph."""
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, List[str]] = {}  # task_id -> prerequisites
        self.completed_tasks: set = set()

    def add_task(
        self,
        task_id: str,
        difficulty: float,
        config: Dict[str, Any],
        prerequisites: Optional[List[str]] = None
    ) -> None:
        """Add task to graph.

        Args:
            task_id: Task identifier
            difficulty: Difficulty score
            config: Task configuration
            prerequisites: Required prior tasks
        """
        task = Task(
            task_id=task_id,
            difficulty=difficulty,
            config=config,
            prerequisites=prerequisites or []
        )
        self.tasks[task_id] = task
        self.dependencies[task_id] = prerequisites or []

    def get_available_tasks(self) -> List[str]:
        """Get tasks whose prerequisites are satisfied.

        Returns:
            List of available task IDs
        """
        available = []
        for task_id, prerequisites in self.dependencies.items():
            if task_id not in self.completed_tasks:
                if all(prereq in self.completed_tasks for prereq in prerequisites):
                    available.append(task_id)

        return available

    def complete_task(self, task_id: str) -> None:
        """Mark task as completed.

        Args:
            task_id: Task identifier
        """
        self.completed_tasks.add(task_id)

    def get_next_task(self, strategy: str = 'easiest') -> Optional[str]:
        """Get next task to attempt.

        Args:
            strategy: Selection strategy ('easiest', 'hardest', 'random')

        Returns:
            Task ID or None if no tasks available
        """
        available = self.get_available_tasks()
        if not available:
            return None

        if strategy == 'easiest':
            # Select easiest available task
            return min(available, key=lambda tid: self.tasks[tid].difficulty)
        elif strategy == 'hardest':
            # Select hardest available task
            return max(available, key=lambda tid: self.tasks[tid].difficulty)
        elif strategy == 'random':
            # Random selection
            return np.random.choice(available)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class ReverseCurriculum:
    """Reverse curriculum learning - start from goal, work backwards.

    Useful for tasks with clear goal states (e.g., inverted pendulum).
    """

    def __init__(
        self,
        goal_config: Dict[str, Any],
        initial_config: Dict[str, Any],
        num_stages: int = 5
    ):
        """Initialize reverse curriculum.

        Args:
            goal_config: Final goal task configuration
            initial_config: Initial (easiest) task configuration
            num_stages: Number of curriculum stages
        """
        self.goal_config = goal_config
        self.initial_config = initial_config
        self.num_stages = num_stages
        self.tasks: List[Task] = []

        self._generate_reverse_tasks()

    def _generate_reverse_tasks(self) -> None:
        """Generate tasks from goal backwards to initial state."""
        for i in range(self.num_stages):
            # Interpolate between goal and initial
            alpha = i / (self.num_stages - 1)  # 0 to 1
            task_config = {}

            for key in self.goal_config:
                if key in self.initial_config:
                    # Linear interpolation
                    goal_val = self.goal_config[key]
                    init_val = self.initial_config[key]

                    if isinstance(goal_val, (int, float)):
                        task_config[key] = alpha * goal_val + (1 - alpha) * init_val
                    else:
                        # For non-numeric, use goal config
                        task_config[key] = goal_val

            difficulty = 1.0 - alpha  # Reverse: goal is hardest (alpha=1 → diff=0)

            task = Task(
                task_id=f'reverse_stage_{i}',
                difficulty=difficulty,
                config=task_config
            )
            self.tasks.append(task)

        # Reverse order: start with goal, end with initial
        self.tasks.reverse()


def create_lqr_curriculum() -> CurriculumLearning:
    """Create example curriculum for LQR problems.

    Progressively increases time horizon.
    """
    base_config = {
        'n_states': 2,
        'n_controls': 1,
        'problem_type': 'lqr',
        'time_horizon': [0.0, 1.0],
        'Q': [[1.0, 0.0], [0.0, 1.0]],
        'R': [[0.1]]
    }

    config = CurriculumConfig(
        strategy=CurriculumStrategy.ADAPTIVE.value,
        difficulty_metric=DifficultyMetric.TIME_HORIZON.value,
        num_stages=5,
        performance_threshold=0.75
    )

    curriculum = CurriculumLearning(config)
    curriculum.generate_curriculum(base_config, num_tasks=5)

    return curriculum


def create_pendulum_curriculum() -> CurriculumLearning:
    """Create example curriculum for pendulum swing-up.

    Progressively tightens control constraints.
    """
    base_config = {
        'n_states': 2,
        'n_controls': 1,
        'problem_type': 'pendulum',
        'time_horizon': [0.0, 3.0],
        'control_bounds': (-5.0, 5.0),  # Start with relaxed bounds
        'target_angle': np.pi  # Upright position
    }

    config = CurriculumConfig(
        strategy=CurriculumStrategy.ADAPTIVE.value,
        difficulty_metric=DifficultyMetric.CONTROL_CONSTRAINTS.value,
        num_stages=5,
        performance_threshold=0.7
    )

    curriculum = CurriculumLearning(config)
    curriculum.generate_curriculum(base_config, num_tasks=5)

    return curriculum


def create_task_graph_example() -> TaskGraph:
    """Create example task graph with dependencies.

    Task structure:
    - Task A (easiest, no prerequisites)
    - Task B (depends on A)
    - Task C (depends on A)
    - Task D (depends on B and C)
    """
    graph = TaskGraph()

    # Task A: Simple stabilization
    graph.add_task(
        task_id='stabilize_simple',
        difficulty=0.2,
        config={'n_states': 2, 'n_controls': 1, 'time_horizon': [0, 1]},
        prerequisites=[]
    )

    # Task B: Longer horizon
    graph.add_task(
        task_id='stabilize_long',
        difficulty=0.5,
        config={'n_states': 2, 'n_controls': 1, 'time_horizon': [0, 5]},
        prerequisites=['stabilize_simple']
    )

    # Task C: More states
    graph.add_task(
        task_id='stabilize_highdim',
        difficulty=0.6,
        config={'n_states': 4, 'n_controls': 2, 'time_horizon': [0, 1]},
        prerequisites=['stabilize_simple']
    )

    # Task D: Combined challenge
    graph.add_task(
        task_id='stabilize_advanced',
        difficulty=0.9,
        config={'n_states': 4, 'n_controls': 2, 'time_horizon': [0, 5]},
        prerequisites=['stabilize_long', 'stabilize_highdim']
    )

    return graph


if __name__ == "__main__":
    print("=== LQR Curriculum Example ===")
    lqr_curriculum = create_lqr_curriculum()
    print(f"Number of tasks: {len(lqr_curriculum.tasks)}")
    for task in lqr_curriculum.tasks:
        print(f"  {task.task_id}: difficulty={task.difficulty:.2f}, horizon={task.config['time_horizon']}")

    print("\n=== Pendulum Curriculum Example ===")
    pendulum_curriculum = create_pendulum_curriculum()
    print(f"Number of tasks: {len(pendulum_curriculum.tasks)}")
    for task in pendulum_curriculum.tasks:
        print(f"  {task.task_id}: difficulty={task.difficulty:.2f}, bounds={task.config['control_bounds']}")

    print("\n=== Task Graph Example ===")
    graph = create_task_graph_example()
    print(f"Total tasks: {len(graph.tasks)}")
    print(f"Available tasks (initially): {graph.get_available_tasks()}")

    # Simulate completing tasks
    graph.complete_task('stabilize_simple')
    print(f"Available after completing 'stabilize_simple': {graph.get_available_tasks()}")

    graph.complete_task('stabilize_long')
    graph.complete_task('stabilize_highdim')
    print(f"Available after completing prerequisites: {graph.get_available_tasks()}")

    print("\nCurriculum learning framework ready!")
