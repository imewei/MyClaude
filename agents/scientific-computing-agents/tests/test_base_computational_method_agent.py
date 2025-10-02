"""Tests for base computational method agent.

Tests cover:
- Initialization and configuration
- Numerical output validation (NaN/Inf checks)
- Convergence checking
- Performance profiling
- Result wrapping and provenance
- Kernel registration

Total: 28 tests
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_computational_method_agent import ComputationalMethodAgent
from base_agent import (
    AgentResult, AgentStatus, ValidationResult, ResourceRequirement,
    AgentMetadata, Capability
)
from computational_models import (
    ComputationalResult, ConvergenceReport, PerformanceMetrics,
    ValidationReport, NumericalKernel
)


# ============================================================================
# Concrete Test Agent
# ============================================================================

class TestConcreteAgent(ComputationalMethodAgent):
    """Concrete agent for testing base class functionality."""

    def execute(self, input_data):
        return AgentResult(
            agent_name="test",
            status=AgentStatus.SUCCESS,
            data={'result': 'test'}
        )

    def validate_input(self, data):
        return ValidationResult(valid=True)

    def estimate_resources(self, data):
        return ResourceRequirement()

    def get_capabilities(self):
        return [Capability(
            name="test",
            description="Test capability",
            input_types=["dict"],
            output_types=["dict"],
            typical_use_cases=["testing"]
        )]

    def get_metadata(self):
        return AgentMetadata(
            name="TestAgent",
            version="1.0.0",
            description="Test agent",
            author="Test",
            capabilities=self.get_capabilities()
        )

    def submit_calculation(self, input_data):
        return "test_job_id"

    def check_status(self, job_id):
        return AgentStatus.SUCCESS

    def retrieve_results(self, job_id):
        return {'result': 'test'}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create a test agent."""
    return TestConcreteAgent()


@pytest.fixture
def agent_with_config():
    """Create agent with custom configuration."""
    return TestConcreteAgent({
        'backend': 'hpc',
        'tolerance': 1e-8,
        'max_iterations': 5000,
        'enable_profiling': True
    })


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization_default(agent):
    """Test agent initialization with default config."""
    assert agent.DOMAIN == "computational_methods"
    assert agent.VERSION == "1.0.0"
    assert agent.tolerance == 1e-6
    assert agent.max_iterations == 10000
    assert agent.enable_profiling is True
    assert agent.compute_backend == 'local'


def test_initialization_custom_config(agent_with_config):
    """Test agent initialization with custom config."""
    assert agent_with_config.tolerance == 1e-8
    assert agent_with_config.max_iterations == 5000
    assert agent_with_config.compute_backend == 'hpc'


def test_metadata(agent):
    """Test agent metadata."""
    metadata = agent.metadata
    assert metadata.name == "TestAgent"
    assert metadata.version == "1.0.0"
    assert len(metadata.capabilities) > 0


# ============================================================================
# Numerical Validation Tests
# ============================================================================

def test_validate_scalar_valid(agent):
    """Test validation of valid scalar output."""
    output = 3.14
    validation = agent.validate_numerical_output(output)

    assert validation.valid
    assert 'nan_check' in validation.checks_passed
    assert 'inf_check' in validation.checks_passed
    assert len(validation.checks_failed) == 0


def test_validate_array_valid(agent):
    """Test validation of valid array output."""
    output = np.array([1.0, 2.0, 3.0, 4.0])
    validation = agent.validate_numerical_output(output)

    assert validation.valid
    assert 'nan_check' in validation.checks_passed
    assert 'inf_check' in validation.checks_passed


def test_validate_array_with_nan(agent):
    """Test validation detects NaN in array."""
    output = np.array([1.0, 2.0, np.nan, 4.0])
    validation = agent.validate_numerical_output(output)

    assert not validation.valid
    assert 'nan_check' in validation.checks_failed
    assert len(validation.warnings) > 0
    assert 'NaN' in validation.warnings[0]


def test_validate_array_with_inf(agent):
    """Test validation detects Inf in array."""
    output = np.array([1.0, 2.0, np.inf, 4.0])
    validation = agent.validate_numerical_output(output)

    assert not validation.valid
    assert 'inf_check' in validation.checks_failed
    assert 'Inf' in validation.warnings[0]


def test_validate_scalar_nan(agent):
    """Test validation detects NaN scalar."""
    output = np.nan
    validation = agent.validate_numerical_output(output)

    assert not validation.valid
    assert 'nan_check' in validation.checks_failed


def test_validate_scalar_inf(agent):
    """Test validation detects Inf scalar."""
    output = np.inf
    validation = agent.validate_numerical_output(output)

    assert not validation.valid
    assert 'inf_check' in validation.checks_failed


def test_validate_large_values_warning(agent):
    """Test validation warns about large values."""
    output = np.array([1e12, 2e12, 3e12])
    validation = agent.validate_numerical_output(output)

    assert validation.valid  # Still valid, just warning
    assert len(validation.warnings) > 0
    assert 'very large' in validation.warnings[0]


def test_validate_dict_output(agent):
    """Test validation of dictionary output."""
    output = {
        'solution': np.array([1.0, 2.0, 3.0]),
        'residual': 1e-8
    }
    validation = agent.validate_numerical_output(output)

    # Should check the arrays in the dict
    assert validation.valid


# ============================================================================
# Convergence Checking Tests
# ============================================================================

def test_check_convergence_success(agent):
    """Test convergence check for converged solution."""
    residual = 1e-8
    report = agent.check_convergence(residual=residual, iteration=10)

    assert report.converged
    assert report.final_residual == residual
    assert report.iterations == 10
    assert report.tolerance == agent.tolerance
    assert report.failure_reason is None


def test_check_convergence_failure_max_iter(agent):
    """Test convergence check when max iterations exceeded."""
    residual = 1e-3  # Not converged
    report = agent.check_convergence(
        residual=residual,
        iteration=agent.max_iterations
    )

    assert not report.converged
    assert report.failure_reason is not None
    assert 'Maximum iterations' in report.failure_reason


def test_check_convergence_nan(agent):
    """Test convergence check when residual is NaN."""
    residual = np.nan
    report = agent.check_convergence(residual=residual, iteration=5)

    assert not report.converged
    assert 'NaN' in report.failure_reason


def test_check_convergence_inf(agent):
    """Test convergence check when residual is Inf (diverged)."""
    residual = np.inf
    report = agent.check_convergence(residual=residual, iteration=5)

    assert not report.converged
    assert 'Inf' in report.failure_reason or 'diverged' in report.failure_reason


def test_check_convergence_with_history(agent):
    """Test convergence rate estimation from history."""
    history = [1e-2, 1e-4, 1e-6, 1e-8]
    residual = history[-1]

    report = agent.check_convergence(
        residual=residual,
        iteration=len(history),
        residual_history=history
    )

    assert report.converged
    assert report.convergence_rate is not None
    assert report.convergence_rate > 0  # Should be positive (convergence)


def test_check_convergence_custom_tolerance(agent):
    """Test convergence with custom tolerance."""
    residual = 1e-4
    custom_tol = 1e-3

    report = agent.check_convergence(residual=residual, tolerance=custom_tol)

    assert report.converged  # Should converge with looser tolerance
    assert report.tolerance == custom_tol


# ============================================================================
# Performance Profiling Tests
# ============================================================================

def test_profile_performance(agent):
    """Test performance profiling."""
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=2.5)

    metrics = agent.profile_performance(
        start_time=start_time,
        end_time=end_time,
        memory_mb=128.5,
        flops=1e9
    )

    assert metrics.wall_time_sec == pytest.approx(2.5, rel=0.01)
    assert metrics.memory_peak_mb == 128.5
    assert metrics.flops == 1e9


# ============================================================================
# Kernel Registration Tests
# ============================================================================

def test_register_and_retrieve_kernel(agent):
    """Test kernel registration and retrieval."""
    def test_func(x):
        return x ** 2

    kernel = NumericalKernel(
        name="square",
        function=test_func,
        complexity="O(1)",
        description="Square function"
    )

    agent.register_kernel(kernel)
    retrieved = agent.get_kernel("square")

    assert retrieved is not None
    assert retrieved.name == "square"
    assert retrieved(5) == 25


def test_get_nonexistent_kernel(agent):
    """Test retrieval of non-existent kernel."""
    kernel = agent.get_kernel("nonexistent")
    assert kernel is None


# ============================================================================
# Computational Result Creation Tests
# ============================================================================

def test_create_computational_result_array(agent):
    """Test creating computational result with array solution."""
    solution = np.array([1.0, 2.0, 3.0, 4.0])
    metadata = {'grid_points': 4, 'method': 'test'}

    result = agent.create_computational_result(
        solution=solution,
        metadata=metadata
    )

    assert isinstance(result, ComputationalResult)
    assert np.allclose(result.solution, solution)
    assert result.metadata == metadata
    assert 'solution_norm' in result.diagnostics
    assert 'solution_min' in result.diagnostics
    assert 'solution_max' in result.diagnostics


def test_create_computational_result_scalar(agent):
    """Test creating computational result with scalar solution."""
    solution = 42.0
    metadata = {'iterations': 10}

    result = agent.create_computational_result(
        solution=solution,
        metadata=metadata
    )

    assert result.solution == 42.0
    assert 'solution_value' in result.diagnostics


def test_create_computational_result_with_convergence(agent):
    """Test creating result with convergence info."""
    solution = np.array([1.0, 2.0])
    metadata = {}
    convergence_info = ConvergenceReport(
        converged=True,
        iterations=10,
        final_residual=1e-8,
        tolerance=1e-6
    )

    result = agent.create_computational_result(
        solution=solution,
        metadata=metadata,
        convergence_info=convergence_info
    )

    assert result.convergence_info is not None
    assert result.convergence_info.converged


def test_create_computational_result_validation(agent):
    """Test creating result with automatic validation."""
    solution = np.array([1.0, np.nan, 3.0])  # Contains NaN
    metadata = {}

    result = agent.create_computational_result(
        solution=solution,
        metadata=metadata,
        validate_output=True
    )

    assert result.validation is not None
    assert not result.validation.valid  # Should detect NaN


# ============================================================================
# Agent Result Wrapping Tests
# ============================================================================

def test_wrap_result_success(agent):
    """Test wrapping computational result in agent result."""
    comp_result = ComputationalResult(
        solution=np.array([1.0, 2.0]),
        metadata={'test': True},
        diagnostics={}
    )
    input_data = {'method': 'test'}
    start_time = datetime.now() - timedelta(seconds=1)

    agent_result = agent.wrap_result_in_agent_result(
        comp_result, input_data, start_time
    )

    assert isinstance(agent_result, AgentResult)
    assert agent_result.status == AgentStatus.SUCCESS
    assert agent_result.provenance is not None
    assert agent_result.provenance.agent_name == "TestAgent"


def test_wrap_result_with_warnings(agent):
    """Test wrapping result with warnings."""
    comp_result = ComputationalResult(
        solution=np.array([1.0, 2.0]),
        metadata={},
        diagnostics={}
    )
    input_data = {}
    start_time = datetime.now()
    warnings = ["Test warning 1", "Test warning 2"]

    agent_result = agent.wrap_result_in_agent_result(
        comp_result, input_data, start_time, warnings=warnings
    )

    assert len(agent_result.warnings) == 2
    assert "Test warning 1" in agent_result.warnings


def test_wrap_result_failed_validation(agent):
    """Test wrapping result that failed validation."""
    validation = ValidationReport(
        valid=False,
        checks_performed=['test'],
        checks_passed=[],
        checks_failed=['test']
    )
    comp_result = ComputationalResult(
        solution=np.nan,
        metadata={},
        diagnostics={},
        validation=validation
    )
    input_data = {}
    start_time = datetime.now()

    agent_result = agent.wrap_result_in_agent_result(
        comp_result, input_data, start_time
    )

    assert agent_result.status == AgentStatus.FAILED


def test_wrap_result_not_converged(agent):
    """Test wrapping result that didn't converge."""
    convergence_info = ConvergenceReport(
        converged=False,
        iterations=100,
        final_residual=0.1,
        tolerance=1e-6,
        failure_reason="Max iterations"
    )
    comp_result = ComputationalResult(
        solution=np.array([1.0]),
        metadata={},
        diagnostics={},
        convergence_info=convergence_info
    )
    input_data = {}
    start_time = datetime.now()

    agent_result = agent.wrap_result_in_agent_result(
        comp_result, input_data, start_time
    )

    assert agent_result.status == AgentStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
