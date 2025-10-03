"""Standard Data Formats for Optimal Control.

Defines standardized data structures for:
- Solver inputs and outputs
- Training data for ML models
- Optimization results
- HPC job specifications
- API request/response formats

Ensures consistency across all Phase 4 components.

Author: Nonequilibrium Physics Agents
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np


class SolverType(Enum):
    """Enumeration of available solver types."""
    PMP = "pmp"
    COLLOCATION = "collocation"
    MAGNUS = "magnus"
    JAX_PMP = "jax_pmp"
    RL_PPO = "rl_ppo"
    RL_SAC = "rl_sac"
    RL_TD3 = "rl_td3"
    MULTI_OBJECTIVE = "multi_objective"
    ROBUST = "robust"
    STOCHASTIC = "stochastic"


class ProblemType(Enum):
    """Enumeration of optimal control problem types."""
    LQR = "lqr"
    QUANTUM_CONTROL = "quantum_control"
    TRAJECTORY_TRACKING = "trajectory_tracking"
    ENERGY_OPTIMIZATION = "energy_optimization"
    THERMODYNAMIC_PROCESS = "thermodynamic_process"
    CUSTOM = "custom"


@dataclass
class StandardDataFormat:
    """Base class for all standard data formats."""
    version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime to ISO format
        if isinstance(data.get('timestamp'), datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary."""
        # Parse datetime if present
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class SolverInput(StandardDataFormat):
    """Standard input format for all solvers.

    Attributes:
        solver_type: Type of solver to use
        problem_type: Type of optimal control problem
        n_states: Number of state variables
        n_controls: Number of control variables
        initial_state: Initial state vector
        target_state: Target state vector (optional)
        time_horizon: Time horizon [t0, tf]
        dynamics: System dynamics specification
        cost: Cost function specification
        constraints: Constraints specification
        solver_config: Solver-specific configuration
    """
    solver_type: str = SolverType.PMP.value
    problem_type: str = ProblemType.CUSTOM.value
    n_states: int = 2
    n_controls: int = 1
    initial_state: List[float] = field(default_factory=lambda: [0.0, 0.0])
    target_state: Optional[List[float]] = None
    time_horizon: List[float] = field(default_factory=lambda: [0.0, 1.0])

    # System specification
    dynamics: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Solver configuration
    solver_config: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate solver input."""
        # Check dimensions
        if len(self.initial_state) != self.n_states:
            raise ValueError(f"initial_state length {len(self.initial_state)} != n_states {self.n_states}")

        if self.target_state is not None and len(self.target_state) != self.n_states:
            raise ValueError(f"target_state length {len(self.target_state)} != n_states {self.n_states}")

        # Check time horizon
        if len(self.time_horizon) != 2:
            raise ValueError("time_horizon must have 2 elements [t0, tf]")

        if self.time_horizon[1] <= self.time_horizon[0]:
            raise ValueError("tf must be greater than t0")

        # Validate solver type
        try:
            SolverType(self.solver_type)
        except ValueError:
            raise ValueError(f"Invalid solver_type: {self.solver_type}")

        return True


@dataclass
class SolverOutput(StandardDataFormat):
    """Standard output format for all solvers.

    Attributes:
        success: Whether solver succeeded
        solver_type: Type of solver used
        optimal_control: Optimal control trajectory
        optimal_state: Optimal state trajectory
        optimal_cost: Optimal cost value
        convergence: Convergence information
        computation_time: Solver computation time (seconds)
        iterations: Number of iterations
        error_message: Error message if failed
    """
    success: bool = True
    solver_type: str = SolverType.PMP.value
    optimal_control: Optional[np.ndarray] = None
    optimal_state: Optional[np.ndarray] = None
    optimal_cost: Optional[float] = None
    convergence: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    iterations: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with numpy array handling."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary with numpy array handling."""
        # Convert lists back to numpy arrays
        if 'optimal_control' in data and data['optimal_control'] is not None:
            data['optimal_control'] = np.array(data['optimal_control'])
        if 'optimal_state' in data and data['optimal_state'] is not None:
            data['optimal_state'] = np.array(data['optimal_state'])

        # Parse datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        return cls(**data)


@dataclass
class TrainingData(StandardDataFormat):
    """Standard format for ML training data.

    Attributes:
        problem_type: Type of problem
        states: State samples (N x n_states)
        controls: Control samples (N x n_controls)
        values: Value function samples (N,)
        advantages: Advantage function samples (N,) for RL
        rewards: Reward samples (N,) for RL
        next_states: Next state samples (N x n_states) for RL
        dones: Episode termination flags (N,) for RL
        n_samples: Number of samples
        n_states: State dimension
        n_controls: Control dimension
        generation_method: How data was generated
    """
    problem_type: str = ProblemType.CUSTOM.value
    states: np.ndarray = field(default_factory=lambda: np.array([]))
    controls: np.ndarray = field(default_factory=lambda: np.array([]))
    values: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    rewards: Optional[np.ndarray] = None
    next_states: Optional[np.ndarray] = None
    dones: Optional[np.ndarray] = None
    n_samples: int = 0
    n_states: int = 0
    n_controls: int = 0
    generation_method: str = "unknown"

    def __post_init__(self):
        """Set dimensions from data."""
        if self.states.size > 0:
            self.n_samples = self.states.shape[0]
            self.n_states = self.states.shape[1] if self.states.ndim > 1 else 1
        if self.controls.size > 0:
            self.n_controls = self.controls.shape[1] if self.controls.ndim > 1 else 1

    def validate(self) -> bool:
        """Validate training data."""
        if self.states.size == 0:
            raise ValueError("states cannot be empty")

        if self.controls.size == 0:
            raise ValueError("controls cannot be empty")

        if self.states.shape[0] != self.controls.shape[0]:
            raise ValueError("states and controls must have same number of samples")

        if self.values is not None and len(self.values) != self.n_samples:
            raise ValueError("values must have same length as states")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data


@dataclass
class OptimizationResult(StandardDataFormat):
    """Standard format for optimization results.

    Used for multi-objective, robust, and stochastic optimization.

    Attributes:
        success: Whether optimization succeeded
        objective_values: Objective function values
        optimal_parameters: Optimal parameters
        pareto_front: Pareto front for multi-objective (optional)
        uncertainty_bounds: Uncertainty quantification (optional)
        risk_metrics: Risk metrics for stochastic optimization
        computation_time: Total computation time
        n_evaluations: Number of function evaluations
        convergence_history: Convergence history
    """
    success: bool = True
    objective_values: Union[float, List[float]] = 0.0
    optimal_parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    pareto_front: Optional[np.ndarray] = None
    uncertainty_bounds: Optional[Dict[str, Any]] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    computation_time: float = 0.0
    n_evaluations: int = 0
    convergence_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            else:
                data[key] = value
        return data


@dataclass
class HPCJobSpec(StandardDataFormat):
    """Standard format for HPC job specification.

    Attributes:
        job_name: Job name
        job_type: Type of job (solver, training, parameter_sweep)
        input_data: Input data (SolverInput or other)
        resources: Resource requirements
        scheduler: Scheduler type (slurm, pbs, dask)
        priority: Job priority
        dependencies: Job dependencies
    """
    job_name: str = "optimal_control_job"
    job_type: str = "solver"
    input_data: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=lambda: {
        "nodes": 1,
        "cpus": 4,
        "memory_gb": 16,
        "gpus": 0,
        "time_hours": 24
    })
    scheduler: str = "slurm"
    priority: str = "normal"
    dependencies: List[str] = field(default_factory=list)


@dataclass
class APIRequest(StandardDataFormat):
    """Standard format for API requests.

    Attributes:
        endpoint: API endpoint
        method: HTTP method
        data: Request data
        headers: Request headers
        timeout: Request timeout (seconds)
    """
    endpoint: str = "/api/solve"
    method: str = "POST"
    data: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})
    timeout: float = 300.0


@dataclass
class APIResponse(StandardDataFormat):
    """Standard format for API responses.

    Attributes:
        status_code: HTTP status code
        success: Whether request succeeded
        data: Response data
        error: Error message if failed
        execution_time: Server-side execution time
    """
    status_code: int = 200
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


# Conversion utilities

def convert_to_standard_format(data: Dict[str, Any], format_type: str) -> StandardDataFormat:
    """Convert dictionary to standard format.

    Args:
        data: Input dictionary
        format_type: Target format type ('solver_input', 'solver_output', etc.)

    Returns:
        Standard format object
    """
    format_map = {
        'solver_input': SolverInput,
        'solver_output': SolverOutput,
        'training_data': TrainingData,
        'optimization_result': OptimizationResult,
        'hpc_job_spec': HPCJobSpec,
        'api_request': APIRequest,
        'api_response': APIResponse,
    }

    if format_type not in format_map:
        raise ValueError(f"Unknown format type: {format_type}")

    return format_map[format_type].from_dict(data)


def validate_standard_format(obj: StandardDataFormat) -> bool:
    """Validate standard format object.

    Args:
        obj: Standard format object

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check version
    if not hasattr(obj, 'version'):
        raise ValueError("Missing version field")

    # Check timestamp
    if not hasattr(obj, 'timestamp'):
        raise ValueError("Missing timestamp field")

    # Format-specific validation
    if isinstance(obj, SolverInput):
        return obj.validate()
    elif isinstance(obj, TrainingData):
        return obj.validate()

    return True


def merge_metadata(obj1: StandardDataFormat, obj2: StandardDataFormat) -> Dict[str, Any]:
    """Merge metadata from two objects.

    Args:
        obj1: First object
        obj2: Second object

    Returns:
        Merged metadata dictionary
    """
    metadata = obj1.metadata.copy()
    metadata.update(obj2.metadata)
    metadata['merged_timestamp'] = datetime.now().isoformat()
    return metadata


# Example usage and factory functions

def create_solver_input(
    solver_type: str,
    n_states: int,
    n_controls: int,
    initial_state: List[float],
    target_state: Optional[List[float]] = None,
    time_horizon: Optional[List[float]] = None,
    **kwargs
) -> SolverInput:
    """Factory function for creating solver input.

    Args:
        solver_type: Type of solver
        n_states: Number of states
        n_controls: Number of controls
        initial_state: Initial state
        target_state: Target state (optional)
        time_horizon: Time horizon (default [0.0, 1.0])
        **kwargs: Additional arguments

    Returns:
        SolverInput object
    """
    if time_horizon is None:
        time_horizon = [0.0, 1.0]

    return SolverInput(
        solver_type=solver_type,
        n_states=n_states,
        n_controls=n_controls,
        initial_state=initial_state,
        target_state=target_state,
        time_horizon=time_horizon,
        **kwargs
    )


def create_training_data_from_solver_outputs(
    outputs: List[SolverOutput],
    problem_type: str = ProblemType.CUSTOM.value
) -> TrainingData:
    """Create training data from solver outputs.

    Args:
        outputs: List of solver outputs
        problem_type: Type of problem

    Returns:
        TrainingData object
    """
    # Collect all states and controls
    all_states = []
    all_controls = []

    for output in outputs:
        if output.optimal_state is not None:
            all_states.append(output.optimal_state)
        if output.optimal_control is not None:
            all_controls.append(output.optimal_control)

    # Concatenate
    states = np.vstack(all_states) if all_states else np.array([])
    controls = np.vstack(all_controls) if all_controls else np.array([])

    return TrainingData(
        problem_type=problem_type,
        states=states,
        controls=controls,
        generation_method="solver_outputs",
        metadata={'n_trajectories': len(outputs)}
    )
