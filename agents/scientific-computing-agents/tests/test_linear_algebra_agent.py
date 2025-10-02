"""Tests for LinearAlgebraAgent.

Test categories:
1. Initialization and metadata
2. Input validation
3. Resource estimation
4. Linear system solving (dense/sparse)
5. Eigenvalue computation
6. Matrix factorization
7. Matrix analysis
8. Caching and provenance
"""

import pytest
import numpy as np
from scipy import sparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.linear_algebra_agent import LinearAlgebraAgent
from base_agent import AgentStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def agent():
    """Create Linear Algebra agent for testing."""
    return LinearAlgebraAgent(config={'tolerance': 1e-6})


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization_default():
    """Test default initialization."""
    agent = LinearAlgebraAgent()
    assert agent.metadata.name == "LinearAlgebraAgent"
    assert agent.VERSION == "1.0.0"
    assert agent.default_method == 'auto'


def test_initialization_custom_config():
    """Test custom configuration."""
    config = {
        'backend': 'hpc',
        'tolerance': 1e-8,
        'default_method': 'lu'
    }
    agent = LinearAlgebraAgent(config)
    assert agent.default_method == 'lu'
    assert agent.tolerance == 1e-8


def test_metadata(agent):
    """Test agent metadata."""
    metadata = agent.get_metadata()
    assert metadata.name == "LinearAlgebraAgent"
    assert metadata.version == "1.0.0"
    assert 'numpy' in metadata.dependencies
    assert 'scipy' in metadata.dependencies


def test_capabilities(agent):
    """Test capabilities list."""
    capabilities = agent.get_capabilities()
    assert len(capabilities) == 4
    names = [cap.name for cap in capabilities]
    assert 'solve_linear_system' in names
    assert 'compute_eigenvalues' in names
    assert 'matrix_factorization' in names
    assert 'matrix_analysis' in names


# ============================================================================
# Validation Tests
# ============================================================================

def test_validate_linear_system_valid(agent):
    """Test validation of valid linear system."""
    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': np.array([[1, 2], [3, 4]]),
        'vector_b': np.array([5, 6])
    }
    validation = agent.validate_input(data)
    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_missing_matrix(agent):
    """Test validation with missing matrix."""
    data = {
        'problem_type': 'linear_system_dense',
        'vector_b': np.array([1, 2])
    }
    validation = agent.validate_input(data)
    assert not validation.valid
    assert any('matrix_A' in err for err in validation.errors)


def test_validate_missing_vector_b(agent):
    """Test validation with missing b vector."""
    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': np.array([[1, 2], [3, 4]])
    }
    validation = agent.validate_input(data)
    assert not validation.valid
    assert any('vector_b' in err or 'b' in err for err in validation.errors)


def test_validate_incompatible_dimensions(agent):
    """Test validation with incompatible dimensions."""
    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': np.array([[1, 2], [3, 4]]),
        'vector_b': np.array([1, 2, 3])  # Wrong size
    }
    validation = agent.validate_input(data)
    assert not validation.valid
    assert any('Incompatible' in err for err in validation.errors)


def test_validate_eigenvalue_valid(agent):
    """Test validation of eigenvalue problem."""
    data = {
        'problem_type': 'eigenvalue_problem',
        'matrix_A': np.array([[1, 2], [3, 4]])
    }
    validation = agent.validate_input(data)
    assert validation.valid


def test_validate_eigenvalue_nonsquare(agent):
    """Test validation of eigenvalue with non-square matrix."""
    data = {
        'problem_type': 'eigenvalue_problem',
        'matrix_A': np.array([[1, 2, 3], [4, 5, 6]])
    }
    validation = agent.validate_input(data)
    assert not validation.valid
    assert any('square' in err.lower() for err in validation.errors)


def test_validate_invalid_problem_type(agent):
    """Test validation with invalid problem type."""
    data = {
        'problem_type': 'invalid_type',
        'matrix_A': np.array([[1, 2], [3, 4]])
    }
    validation = agent.validate_input(data)
    assert not validation.valid


# ============================================================================
# Resource Estimation Tests
# ============================================================================

def test_estimate_resources_small_dense(agent):
    """Test resource estimation for small dense system."""
    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': np.random.rand(100, 100),
        'vector_b': np.random.rand(100)
    }
    resources = agent.estimate_resources(data)
    assert resources.cpu_cores >= 1
    assert resources.memory_gb >= 0.1
    assert resources.estimated_time_sec >= 0


def test_estimate_resources_large_dense(agent):
    """Test resource estimation for large dense system."""
    # Create metadata representing large matrix
    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': type('MockMatrix', (), {'shape': (10000, 10000)})()
    }
    resources = agent.estimate_resources(data)
    assert resources.cpu_cores >= 4  # Should request more cores
    assert resources.memory_gb >= 0.1  # Should estimate some memory


def test_estimate_resources_eigenvalue(agent):
    """Test resource estimation for eigenvalue problem."""
    data = {
        'problem_type': 'eigenvalue_problem',
        'matrix_A': np.random.rand(500, 500),
        'num_eigenvalues': 10
    }
    resources = agent.estimate_resources(data)
    assert resources.estimated_time_sec > 0


# ============================================================================
# Linear System Tests
# ============================================================================

def test_solve_simple_system(agent):
    """Test solving simple 2x2 system."""
    # System: 2x + y = 5, x + 3y = 11
    # Solution: x = 0.8, y = 3.4
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 11.0])

    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': A,
        'vector_b': b,
        'method': 'lu'
    }

    result = agent.execute(data)

    assert result.success
    assert 'x' in result.data['solution']
    x = result.data['solution']['x']
    # Verify the solution satisfies Ax = b
    assert np.allclose(A @ x, b, atol=1e-6)


def test_solve_identity_system(agent):
    """Test solving with identity matrix."""
    A = np.eye(3)
    b = np.array([1.0, 2.0, 3.0])

    data = {
        'problem_type': 'linear_system_dense',
        'A': A,
        'b': b
    }

    result = agent.execute(data)

    assert result.success
    x = result.data['solution']['x']
    assert np.allclose(x, b)


def test_solve_with_different_methods(agent):
    """Test solving with different methods."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])  # SPD matrix
    b = np.array([1.0, 2.0])

    methods = ['lu', 'qr']

    for method in methods:
        data = {
            'problem_type': 'linear_system_dense',
            'matrix_A': A,
            'vector_b': b,
            'method': method
        }
        result = agent.execute(data)
        assert result.success, f"Method {method} failed"
        x = result.data['solution']['x']
        # Verify solution
        assert np.allclose(A @ x, b, atol=1e-6)


def test_solve_symmetric_positive_definite(agent):
    """Test solving SPD system with Cholesky."""
    # Create SPD matrix
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    b = np.array([2.0, 3.0])

    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': A,
        'vector_b': b,
        'method': 'cholesky'
    }

    result = agent.execute(data)

    assert result.success
    x = result.data['solution']['x']
    assert np.allclose(A @ x, b, atol=1e-6)


def test_solve_iterative_cg(agent):
    """Test CG iterative solver."""
    # Create SPD matrix
    n = 10
    A = np.random.rand(n, n)
    A = A.T @ A + np.eye(n)  # Make SPD
    b = np.random.rand(n)

    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': A,
        'vector_b': b,
        'method': 'cg'
    }

    result = agent.execute(data)

    assert result.success
    x = result.data['solution']['x']
    assert np.allclose(A @ x, b, atol=1e-4)  # Slightly looser tolerance for iterative


# ============================================================================
# Eigenvalue Tests
# ============================================================================

def test_compute_eigenvalues_2x2(agent):
    """Test eigenvalue computation for 2x2 matrix."""
    # Matrix with known eigenvalues: [5, 1]
    A = np.array([[3.0, 2.0], [2.0, 3.0]])

    data = {
        'problem_type': 'eigenvalue_problem',
        'matrix_A': A
    }

    result = agent.execute(data)

    assert result.success
    assert 'eigenvalues' in result.data['solution']
    eigenvalues = result.data['solution']['eigenvalues']

    # Expected eigenvalues: 5 and 1
    assert len(eigenvalues) == 2
    assert np.allclose(sorted(np.abs(eigenvalues)), [1.0, 5.0], atol=1e-6)


def test_compute_eigenvalues_identity(agent):
    """Test eigenvalues of identity matrix."""
    A = np.eye(3)

    data = {
        'problem_type': 'eigenvalue_problem',
        'matrix_A': A
    }

    result = agent.execute(data)

    assert result.success
    eigenvalues = result.data['solution']['eigenvalues']
    assert np.allclose(eigenvalues, [1.0, 1.0, 1.0])


def test_compute_partial_eigenvalues(agent):
    """Test computing partial eigenvalue spectrum."""
    n = 10
    A = np.diag(np.arange(1, n+1, dtype=float))  # Eigenvalues 1, 2, ..., 10

    data = {
        'problem_type': 'eigenvalue_problem',
        'matrix_A': A,
        'num_eigenvalues': 3,
        'which': 'largest'
    }

    result = agent.execute(data)

    assert result.success
    eigenvalues = result.data['solution']['eigenvalues']
    # Should get largest 3: [10, 9, 8]
    assert len(eigenvalues) >= 3


# ============================================================================
# Matrix Factorization Tests
# ============================================================================

def test_lu_factorization(agent):
    """Test LU factorization."""
    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    data = {
        'problem_type': 'matrix_factorization',
        'matrix_A': A,
        'factorization_type': 'lu'
    }

    result = agent.execute(data)

    assert result.success
    assert 'L' in result.data['solution']
    assert 'U' in result.data['solution']
    assert 'P' in result.data['solution']

    L = result.data['solution']['L']
    U = result.data['solution']['U']
    P = result.data['solution']['P']

    # Verify P @ A = L @ U
    assert np.allclose(P @ A, L @ U, atol=1e-10)


def test_qr_factorization(agent):
    """Test QR factorization."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    data = {
        'problem_type': 'matrix_factorization',
        'matrix_A': A,
        'factorization_type': 'qr'
    }

    result = agent.execute(data)

    assert result.success
    assert 'Q' in result.data['solution']
    assert 'R' in result.data['solution']

    Q = result.data['solution']['Q']
    R = result.data['solution']['R']

    # Verify A = Q @ R
    assert np.allclose(A, Q @ R, atol=1e-10)
    # Verify Q is orthogonal
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)


def test_svd_factorization(agent):
    """Test SVD factorization."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])

    data = {
        'problem_type': 'matrix_factorization',
        'matrix_A': A,
        'factorization_type': 'svd'
    }

    result = agent.execute(data)

    assert result.success
    assert 'U' in result.data['solution']
    assert 's' in result.data['solution']
    assert 'Vt' in result.data['solution']

    U = result.data['solution']['U']
    s = result.data['solution']['s']
    Vt = result.data['solution']['Vt']

    # Verify A = U @ S @ Vt
    S = np.diag(s)
    assert np.allclose(A, U @ S @ Vt, atol=1e-10)


# ============================================================================
# Matrix Analysis Tests
# ============================================================================

def test_matrix_analysis(agent):
    """Test matrix analysis capabilities."""
    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    data = {
        'problem_type': 'matrix_analysis',
        'matrix_A': A
    }

    result = agent.execute(data)

    assert result.success
    analysis = result.data['solution']

    assert 'condition_number' in analysis
    assert 'rank' in analysis
    assert 'determinant' in analysis
    assert 'frobenius_norm' in analysis
    assert 'is_symmetric' in analysis

    # Verify some properties
    assert analysis['rank'] == 2
    assert analysis['is_symmetric'] == True
    assert analysis['condition_number'] > 0


def test_matrix_analysis_singular(agent):
    """Test analysis of singular matrix."""
    A = np.array([[1.0, 2.0], [2.0, 4.0]])  # Singular

    data = {
        'problem_type': 'matrix_analysis',
        'matrix_A': A
    }

    result = agent.execute(data)

    assert result.success
    analysis = result.data['solution']

    assert analysis['rank'] == 1  # Rank deficient
    assert np.abs(analysis['determinant']) < 1e-10  # Near zero


# ============================================================================
# Execution Tests
# ============================================================================

def test_execute_with_invalid_input(agent):
    """Test execution with invalid input."""
    data = {
        'problem_type': 'linear_system_dense',
        # Missing matrix_A
        'vector_b': np.array([1, 2])
    }

    result = agent.execute(data)

    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0


# ============================================================================
# Caching Tests
# ============================================================================

def test_caching_same_input(agent):
    """Test that same input uses cache."""
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': A,
        'vector_b': b
    }

    # First execution
    result1 = agent.execute_with_caching(data)
    assert result1.status == AgentStatus.SUCCESS

    # Second execution (should use cache)
    result2 = agent.execute_with_caching(data)
    assert result2.status == AgentStatus.CACHED


# ============================================================================
# Job Management Tests
# ============================================================================

def test_submit_and_check_job(agent):
    """Test job submission and status checking."""
    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': np.eye(2),
        'vector_b': np.array([1, 2])
    }

    job_id = agent.submit_calculation(data)
    assert isinstance(job_id, str)
    assert job_id.startswith('linalg_')

    status = agent.check_status(job_id)
    assert status == AgentStatus.PENDING


def test_retrieve_job_results(agent):
    """Test retrieving job results."""
    data = {
        'problem_type': 'matrix_analysis',
        'matrix_A': np.eye(3)
    }

    job_id = agent.submit_calculation(data)
    results = agent.retrieve_results(job_id)

    assert 'input' in results
    assert results['input'] == data


# ============================================================================
# Provenance Tests
# ============================================================================

def test_provenance_tracking(agent):
    """Test that provenance is properly tracked."""
    data = {
        'problem_type': 'linear_system_dense',
        'matrix_A': np.eye(2),
        'vector_b': np.array([1.0, 2.0])
    }

    result = agent.execute(data)

    assert result.provenance is not None
    assert result.provenance.agent_name == "LinearAlgebraAgent"
    assert result.provenance.agent_version == "1.0.0"
    assert result.provenance.execution_time_sec > 0
    assert len(result.provenance.input_hash) == 64  # SHA256 hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
