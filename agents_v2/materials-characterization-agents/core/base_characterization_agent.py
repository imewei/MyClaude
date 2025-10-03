"""Base agent interface for materials science agents.

This module defines the abstract base class and data models that all
materials science agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib
import json


class ExecutionEnvironment(Enum):
    """Computational environment for agent execution."""
    LOCAL = "local"
    HPC = "hpc"
    CLOUD = "cloud"


class AgentStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class ResourceRequirement:
    """Computational resource requirements for agent execution."""
    cpu_cores: int = 1
    memory_gb: float = 1.0
    gpu_count: int = 0
    estimated_time_sec: float = 60.0
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL

    def exceeds(self, available: 'ResourceRequirement') -> bool:
        """Check if requirements exceed available resources."""
        return (
            self.cpu_cores > available.cpu_cores or
            self.memory_gb > available.memory_gb or
            self.gpu_count > available.gpu_count
        )


@dataclass
class Capability:
    """Agent capability specification."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    typical_use_cases: List[str]


@dataclass
class Provenance:
    """Track how results were generated (reproducibility)."""
    agent_name: str
    agent_version: str
    timestamp: datetime
    input_hash: str
    parameters: Dict[str, Any]
    execution_time_sec: float
    environment: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_name': self.agent_name,
            'agent_version': self.agent_version,
            'timestamp': self.timestamp.isoformat(),
            'input_hash': self.input_hash,
            'parameters': self.parameters,
            'execution_time_sec': self.execution_time_sec,
            'environment': self.environment
        }


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


@dataclass
class AgentResult:
    """Standardized agent output."""
    agent_name: str
    status: AgentStatus
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: Optional[Provenance] = None

    @property
    def success(self) -> bool:
        return self.status in [AgentStatus.SUCCESS, AgentStatus.CACHED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_name': self.agent_name,
            'status': self.status.value,
            'data': self.data,
            'metadata': self.metadata,
            'errors': self.errors,
            'warnings': self.warnings,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class AgentMetadata:
    """Agent metadata and configuration."""
    name: str
    version: str
    description: str
    author: str
    capabilities: List[Capability]
    dependencies: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all materials science agents.

    All agents must implement this interface to ensure consistency
    and interoperability within the multi-agent system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize agent with optional configuration.

        Args:
            config: Agent-specific configuration dictionary
        """
        self.config = config or {}
        self.metadata = self.get_metadata()
        self._cache = {}

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute agent's primary function.

        Args:
            input_data: Input data for agent execution

        Returns:
            AgentResult containing outputs, metadata, and provenance

        Raises:
            ValidationError: If input validation fails
            ExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data before execution.

        Args:
            data: Input data to validate

        Returns:
            ValidationResult with status and any errors/warnings
        """
        pass

    @abstractmethod
    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed.

        Args:
            data: Input data for resource estimation

        Returns:
            ResourceRequirement specifying needed resources
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[Capability]:
        """Return list of agent capabilities.

        Returns:
            List of Capability objects describing what agent can do
        """
        pass

    @abstractmethod
    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata.

        Returns:
            AgentMetadata with name, version, description, etc.
        """
        pass

    def execute_with_caching(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute with result caching.

        Args:
            input_data: Input data for execution

        Returns:
            AgentResult (from cache if available, otherwise computed)
        """
        cache_key = self._compute_cache_key(input_data)

        if cache_key in self._cache:
            result = self._cache[cache_key]
            result.status = AgentStatus.CACHED
            result.metadata['cached'] = True
            return result

        result = self.execute(input_data)
        if result.success:
            self._cache[cache_key] = result

        return result

    def _compute_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Compute cache key from input data.

        Args:
            input_data: Input data to hash

        Returns:
            SHA256 hash of input data as cache key
        """
        # Serialize input data to JSON (sorted keys for consistency)
        data_str = json.dumps(input_data, sort_keys=True)
        # Add agent version to cache key (invalidate on version change)
        key_str = f"{self.metadata.name}:{self.metadata.version}:{data_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def clear_cache(self):
        """Clear agent result cache."""
        self._cache.clear()

    def validate_and_execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Validate input then execute.

        Args:
            input_data: Input data for execution

        Returns:
            AgentResult (validation errors if validation fails)
        """
        validation = self.validate_input(input_data)

        if not validation.valid:
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                errors=validation.errors,
                warnings=validation.warnings
            )

        return self.execute(input_data)


class ExperimentalAgent(BaseAgent):
    """Base class for experimental characterization agents.

    Examples: Light scattering, electron microscopy, spectroscopy
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.instrument_config = config.get('instrument', {}) if config else {}

    @abstractmethod
    def connect_instrument(self) -> bool:
        """Connect to experimental instrument (if applicable).

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def process_experimental_data(self, raw_data: Any) -> Dict[str, Any]:
        """Process raw experimental data.

        Args:
            raw_data: Raw data from instrument

        Returns:
            Processed data dictionary
        """
        pass


class ComputationalAgent(BaseAgent):
    """Base class for computational agents.

    Examples: DFT, MD simulation, ML prediction
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.compute_backend = config.get('backend', 'local') if config else 'local'

    @abstractmethod
    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit calculation to compute backend.

        Args:
            input_data: Calculation input

        Returns:
            Job ID for tracking
        """
        pass

    @abstractmethod
    def check_status(self, job_id: str) -> AgentStatus:
        """Check calculation status.

        Args:
            job_id: Job identifier

        Returns:
            AgentStatus (RUNNING, SUCCESS, FAILED, etc.)
        """
        pass

    @abstractmethod
    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve calculation results.

        Args:
            job_id: Job identifier

        Returns:
            Calculation results dictionary
        """
        pass


class CoordinationAgent(BaseAgent):
    """Base class for coordination agents.

    Examples: Characterization Master (multi-technique coordinator)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, agent_registry=None):
        super().__init__(config)
        self.agent_registry = agent_registry

    @abstractmethod
    def design_workflow(self, goal: Dict[str, Any]) -> 'Workflow':
        """Design multi-agent workflow to achieve goal.

        Args:
            goal: Characterization goal specification

        Returns:
            Workflow object (DAG of agent tasks)
        """
        pass

    @abstractmethod
    def optimize_technique_selection(self, goal: Dict[str, Any]) -> List[str]:
        """Select optimal techniques for characterization goal.

        Args:
            goal: Characterization goal

        Returns:
            List of recommended agent names
        """
        pass


# Error classes

class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class ValidationError(AgentError):
    """Input validation error."""
    pass


class ExecutionError(AgentError):
    """Execution error."""
    pass


class ResourceError(AgentError):
    """Insufficient resources error."""
    pass


class IntegrationError(AgentError):
    """Agent integration error."""
    pass