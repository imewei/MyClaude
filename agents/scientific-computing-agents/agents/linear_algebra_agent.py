"""Linear Algebra Agent - Matrix Computation Expert.

This agent specializes in solving linear algebra problems using
direct and iterative methods.

Capabilities:
- Linear systems: Dense/sparse, direct/iterative solvers
- Eigenvalue problems: Standard/generalized, partial/full spectrum
- Matrix factorizations: LU, QR, Cholesky, SVD
- Matrix analysis: Condition number, rank, norms
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve, eigs, eigsh

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_computational_method_agent import ComputationalMethodAgent
from base_agent import (
    AgentResult, AgentStatus, ValidationResult, ResourceRequirement,
    AgentMetadata, Capability, ExecutionEnvironment
)
from computational_models import (
    ProblemSpecification, ProblemType, AlgorithmRecommendation,
    ComputationalResult, ConvergenceReport, PerformanceMetrics,
    MethodCategory, NumericalKernel
)
from numerical_kernels.linear_algebra import (
    solve_linear_system, conjugate_gradient, gmres_solver,
    compute_eigenvalues, is_symmetric, condition_number
)


class LinearAlgebraAgent(ComputationalMethodAgent):
    """Agent for solving linear algebra problems.

    Supports multiple problem types and solution methods:
    - Linear systems: LU, QR, Cholesky, CG, GMRES, BiCGSTAB
    - Eigenvalue problems: Power iteration, QR algorithm, Arnoldi/Lanczos
    - Matrix factorizations: LU, QR, Cholesky, SVD, eigendecomposition
    - Matrix analysis: Condition number, rank, determinant, norms
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Linear Algebra agent.

        Args:
            config: Configuration including:
                - backend: 'local', 'hpc', 'cloud'
                - tolerance: relative tolerance (default: 1e-6)
                - max_iterations: max iterations (default: 10000)
                - default_method: default solver method (default: 'auto')
        """
        super().__init__(config)
        self.default_method = config.get('default_method', 'auto') if config else 'auto'
        self.supported_dense_methods = ['lu', 'qr', 'cholesky', 'svd']
        self.supported_iterative_methods = ['cg', 'gmres', 'bicgstab']
        self.supported_eigen_methods = ['dense', 'arnoldi', 'lanczos']

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            name="LinearAlgebraAgent",
            version=self.VERSION,
            description="Solve linear algebra problems: systems, eigenvalues, factorizations",
            author="Scientific Computing Agents Team",
            capabilities=self.get_capabilities(),
            dependencies=['numpy', 'scipy'],
            supported_formats=['dict', 'numpy.ndarray', 'scipy.sparse']
        )

    def get_capabilities(self) -> List[Capability]:
        """Return list of agent capabilities."""
        return [
            Capability(
                name="solve_linear_system",
                description="Solve Ax = b using direct or iterative methods",
                input_types=["matrix_A", "vector_b", "method"],
                output_types=["solution_x", "residual", "convergence_info"],
                typical_use_cases=[
                    "Finite difference/element discretizations",
                    "Least squares problems",
                    "Network flow problems",
                    "Circuit analysis"
                ]
            ),
            Capability(
                name="compute_eigenvalues",
                description="Compute eigenvalues and eigenvectors",
                input_types=["matrix_A", "num_eigenvalues", "which"],
                output_types=["eigenvalues", "eigenvectors", "convergence_info"],
                typical_use_cases=[
                    "Stability analysis",
                    "Principal component analysis",
                    "Vibration modes",
                    "Quantum mechanics"
                ]
            ),
            Capability(
                name="matrix_factorization",
                description="Compute matrix factorizations (LU, QR, Cholesky, SVD)",
                input_types=["matrix_A", "factorization_type"],
                output_types=["factors", "metadata"],
                typical_use_cases=[
                    "Solving multiple systems",
                    "Least squares fitting",
                    "Data compression",
                    "Pseudoinverse computation"
                ]
            ),
            Capability(
                name="matrix_analysis",
                description="Analyze matrix properties",
                input_types=["matrix_A"],
                output_types=["condition_number", "rank", "norms", "determinant"],
                typical_use_cases=[
                    "Numerical stability assessment",
                    "Ill-conditioning detection",
                    "System analysis"
                ]
            )
        ]

    def validate_input(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate input data before execution.

        Args:
            data: Input data containing:
                - problem_type: 'linear_system_dense', 'linear_system_sparse',
                               'eigenvalue_problem', 'matrix_factorization', 'matrix_analysis'
                - matrix_A: coefficient matrix
                - For linear systems: vector_b
                - For eigenvalue: num_eigenvalues (optional), which (optional)

        Returns:
            ValidationResult with status and errors/warnings
        """
        errors = []
        warnings = []

        # Check problem type
        problem_type = data.get('problem_type')
        if not problem_type:
            errors.append("Missing required field: 'problem_type'")
        elif problem_type not in ['linear_system_dense', 'linear_system_sparse',
                                   'eigenvalue_problem', 'matrix_factorization',
                                   'matrix_analysis']:
            errors.append(f"Invalid problem_type: {problem_type}")

        # Check matrix
        if 'matrix_A' not in data and 'A' not in data:
            errors.append("Missing required field: 'matrix_A' or 'A'")
        else:
            A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')
            # Check if it's a valid matrix (array-like)
            if not hasattr(A, 'shape'):
                errors.append("matrix_A must be array-like with shape attribute")
            elif len(A.shape) != 2:
                errors.append(f"matrix_A must be 2D, got shape {A.shape}")
            elif A.shape[0] != A.shape[1] and problem_type in ['eigenvalue_problem']:
                errors.append(f"Eigenvalue problems require square matrix, got shape {A.shape}")

        # Problem-specific validation
        if problem_type in ['linear_system_dense', 'linear_system_sparse']:
            if 'vector_b' not in data and 'b' not in data:
                errors.append("Linear system requires 'vector_b' or 'b'")
            else:
                b = data.get('vector_b') if 'vector_b' in data else data.get('b')
                A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')
                if hasattr(A, 'shape') and hasattr(b, 'shape'):
                    if len(b.shape) == 1:
                        if A.shape[0] != b.shape[0]:
                            errors.append(f"Incompatible dimensions: A {A.shape}, b {b.shape}")
                    elif len(b.shape) == 2:
                        if A.shape[0] != b.shape[0]:
                            errors.append(f"Incompatible dimensions: A {A.shape}, b {b.shape}")

        elif problem_type == 'eigenvalue_problem':
            # Check if requesting partial spectrum
            k = data.get('num_eigenvalues') if 'num_eigenvalues' in data else data.get('k')
            A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')
            if k and hasattr(A, 'shape'):
                if k >= min(A.shape):
                    warnings.append(f"Requesting {k} eigenvalues from {min(A.shape)}x{min(A.shape)} matrix - will compute all")

        # Check method if specified
        method = data.get('method')
        if method and problem_type in ['linear_system_dense', 'linear_system_sparse']:
            all_methods = self.supported_dense_methods + self.supported_iterative_methods
            if method not in all_methods and method != 'auto':
                warnings.append(f"Method '{method}' not in recommended list: {all_methods}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def estimate_resources(self, data: Dict[str, Any]) -> ResourceRequirement:
        """Estimate computational resources needed.

        Args:
            data: Input data

        Returns:
            ResourceRequirement with estimated resources
        """
        problem_type = data.get('problem_type', 'linear_system_dense')
        A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')

        # Default estimates
        cpu_cores = 1
        memory_gb = 1.0
        estimated_time_sec = 1.0
        environment = ExecutionEnvironment.LOCAL

        if hasattr(A, 'shape'):
            n = A.shape[0]

            # Memory estimate: matrix storage + workspace
            if sparse.issparse(A):
                # Sparse matrix
                nnz = A.nnz
                memory_gb = max(0.1, (nnz * 16 + n * 8) / 1e9)  # 16 bytes per entry + vector
            else:
                # Dense matrix
                memory_gb = max(0.1, (n * n * 8) / 1e9)  # 8 bytes per float64

            # Time estimate based on problem type and size
            if problem_type in ['linear_system_dense']:
                # Direct solve: O(n^3)
                estimated_time_sec = max(0.1, n**3 / 1e9)  # Rough estimate
                if n > 5000:
                    environment = ExecutionEnvironment.HPC
                    cpu_cores = 4

            elif problem_type in ['linear_system_sparse']:
                # Iterative solve: depends on sparsity and condition number
                if sparse.issparse(A):
                    nnz = A.nnz
                    estimated_time_sec = max(0.1, nnz * 100 / 1e9)  # Very rough
                if n > 100000:
                    environment = ExecutionEnvironment.HPC

            elif problem_type == 'eigenvalue_problem':
                k = data.get('num_eigenvalues') or data.get('k') or n
                if k == n or k > n // 2:
                    # Full eigendecomposition: O(n^3)
                    estimated_time_sec = max(0.1, n**3 / 1e9)
                else:
                    # Partial eigenvalue: Arnoldi/Lanczos O(k*n^2)
                    estimated_time_sec = max(0.1, k * n**2 / 1e9)

                if n > 5000:
                    environment = ExecutionEnvironment.HPC

            elif problem_type == 'matrix_factorization':
                # Similar to direct solve
                estimated_time_sec = max(0.1, n**3 / 1e9)

            # Increase resources for large problems
            if n > 10000:
                memory_gb *= 2
                cpu_cores = max(cpu_cores, 8)

        return ResourceRequirement(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            estimated_time_sec=estimated_time_sec,
            execution_environment=environment
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute linear algebra solver.

        Args:
            input_data: Input data containing:
                - problem_type: 'linear_system_dense', 'eigenvalue_problem', etc.
                - matrix_A: coefficient matrix
                - For linear systems: vector_b
                - method: solution method (optional)
                - tolerance: numerical tolerance (optional)

        Returns:
            AgentResult with solution and diagnostics

        Example:
            >>> agent = LinearAlgebraAgent()
            >>> result = agent.execute({
            ...     'problem_type': 'linear_system_dense',
            ...     'matrix_A': np.array([[2, 1], [1, 3]]),
            ...     'vector_b': np.array([1, 2]),
            ...     'method': 'lu'
            ... })
        """
        start_time = datetime.now()

        try:
            # Validate input
            validation = self.validate_input(input_data)
            if not validation.valid:
                return AgentResult(
                    agent_name=self.metadata.name,
                    status=AgentStatus.FAILED,
                    data={},
                    errors=validation.errors,
                    warnings=validation.warnings
                )

            # Route to appropriate solver
            problem_type = input_data['problem_type']

            if problem_type in ['linear_system_dense', 'linear_system_sparse']:
                comp_result = self._solve_linear_system(input_data)
            elif problem_type == 'eigenvalue_problem':
                comp_result = self._compute_eigenvalues(input_data)
            elif problem_type == 'matrix_factorization':
                comp_result = self._matrix_factorization(input_data)
            elif problem_type == 'matrix_analysis':
                comp_result = self._matrix_analysis(input_data)
            else:
                raise ValueError(f"Unsupported problem_type: {problem_type}")

            # Wrap in agent result
            return self.wrap_result_in_agent_result(
                comp_result,
                input_data,
                start_time,
                warnings=validation.warnings
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_name=self.metadata.name,
                status=AgentStatus.FAILED,
                data={},
                metadata={'execution_time_sec': execution_time},
                errors=[f"Execution failed: {str(e)}"]
            )

    def _solve_linear_system(self, data: Dict[str, Any]) -> ComputationalResult:
        """Solve linear system Ax = b.

        Args:
            data: Input data with matrix_A, vector_b, method

        Returns:
            ComputationalResult with solution
        """
        A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')
        b = data.get('vector_b') if 'vector_b' in data else data.get('b')
        method = data.get('method', self.default_method)
        tol = data.get('tolerance', self.tolerance)

        # Convert to numpy arrays if needed
        if sparse.issparse(A):
            is_sparse = True
            if method == 'auto':
                method = 'gmres'
        else:
            is_sparse = False
            A = np.asarray(A)
            b = np.asarray(b)

        # Solve system
        x, info = solve_linear_system(A, b, method=method, tol=tol, maxiter=self.max_iterations)

        # Create convergence report
        iterations = info.get('iterations', 0)
        residual = info.get('residual', np.linalg.norm(A @ x - b) / np.linalg.norm(b))

        converged = residual < tol or (iterations == 0)  # Direct methods converge in 1 step
        convergence_info = ConvergenceReport(
            converged=converged,
            iterations=iterations if iterations > 0 else 1,
            final_residual=float(residual),
            tolerance=tol,
            failure_reason=None if converged else f"Residual {residual:.2e} exceeds tolerance {tol:.2e}"
        )

        # Create metadata
        metadata = {
            'method': info.get('method', method),
            'matrix_size': A.shape,
            'is_sparse': is_sparse,
            'iterations': iterations,
            'residual': float(residual)
        }

        return self.create_computational_result(
            solution={'x': x, 'residual': residual},
            metadata=metadata,
            convergence_info=convergence_info
        )

    def _compute_eigenvalues(self, data: Dict[str, Any]) -> ComputationalResult:
        """Compute eigenvalues and eigenvectors.

        Args:
            data: Input data with matrix_A, num_eigenvalues, which

        Returns:
            ComputationalResult with eigenvalues/vectors
        """
        A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')
        k = data.get('num_eigenvalues') if 'num_eigenvalues' in data else data.get('k')
        which = data.get('which', 'largest')

        # Convert to numpy array
        A = np.asarray(A)
        n = A.shape[0]

        # Compute eigenvalues
        eigenvalues, eigenvectors = compute_eigenvalues(A, k=k, which=which)

        # Create metadata
        metadata = {
            'matrix_size': A.shape,
            'num_eigenvalues': len(eigenvalues),
            'which': which,
            'is_symmetric': is_symmetric(A)
        }

        # No convergence info for dense eigensolvers (direct method)
        return self.create_computational_result(
            solution={
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors
            },
            metadata=metadata
        )

    def _matrix_factorization(self, data: Dict[str, Any]) -> ComputationalResult:
        """Compute matrix factorization.

        Args:
            data: Input data with matrix_A, factorization_type

        Returns:
            ComputationalResult with factors
        """
        A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')
        factorization_type = data.get('factorization_type', 'lu')

        A = np.asarray(A)

        if factorization_type == 'lu':
            P, L, U = linalg.lu(A)
            factors = {'P': P, 'L': L, 'U': U}
        elif factorization_type == 'qr':
            Q, R = linalg.qr(A)
            factors = {'Q': Q, 'R': R}
        elif factorization_type == 'cholesky':
            L = linalg.cholesky(A, lower=True)
            factors = {'L': L}
        elif factorization_type == 'svd':
            U, s, Vt = linalg.svd(A)
            factors = {'U': U, 's': s, 'Vt': Vt}
        else:
            raise ValueError(f"Unknown factorization type: {factorization_type}")

        metadata = {
            'factorization_type': factorization_type,
            'matrix_size': A.shape
        }

        return self.create_computational_result(
            solution=factors,
            metadata=metadata
        )

    def _matrix_analysis(self, data: Dict[str, Any]) -> ComputationalResult:
        """Analyze matrix properties.

        Args:
            data: Input data with matrix_A

        Returns:
            ComputationalResult with analysis
        """
        A = data.get('matrix_A') if 'matrix_A' in data else data.get('A')
        A = np.asarray(A)

        analysis = {
            'condition_number': float(condition_number(A)),
            'rank': int(np.linalg.matrix_rank(A)),
            'determinant': float(np.linalg.det(A)) if A.shape[0] == A.shape[1] else None,
            'frobenius_norm': float(np.linalg.norm(A, 'fro')),
            'spectral_norm': float(np.linalg.norm(A, 2)),
            'is_symmetric': bool(is_symmetric(A))
        }

        metadata = {
            'matrix_size': A.shape
        }

        return self.create_computational_result(
            solution=analysis,
            metadata=metadata
        )

    def submit_calculation(self, input_data: Dict[str, Any]) -> str:
        """Submit calculation to compute backend.

        Args:
            input_data: Calculation input

        Returns:
            Job ID for tracking
        """
        import uuid
        job_id = f"linalg_{uuid.uuid4().hex[:8]}"
        if not hasattr(self, '_jobs'):
            self._jobs = {}
        self._jobs[job_id] = {'input': input_data, 'status': 'submitted'}
        return job_id

    def check_status(self, job_id: str) -> AgentStatus:
        """Check calculation status.

        Args:
            job_id: Job identifier

        Returns:
            AgentStatus
        """
        if hasattr(self, '_jobs') and job_id in self._jobs:
            return AgentStatus.PENDING
        return AgentStatus.FAILED

    def retrieve_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve calculation results.

        Args:
            job_id: Job identifier

        Returns:
            Calculation results
        """
        if hasattr(self, '_jobs') and job_id in self._jobs:
            return self._jobs[job_id]
        return {}
